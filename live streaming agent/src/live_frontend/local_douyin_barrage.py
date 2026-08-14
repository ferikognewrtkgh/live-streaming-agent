from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import ctypes
from collections import Counter
from pathlib import Path
from typing import Any

import websocket

try:
    from open_llm_vtuber.douyin_link_payload import (
        extract_link_anchor_candidates_from_payload_base64,
    )
except ModuleNotFoundError:
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from open_llm_vtuber.douyin_link_payload import (
        extract_link_anchor_candidates_from_payload_base64,
    )


DEFAULT_LOCAL_BARRAGE_WS_URL = "ws://127.0.0.1:8888"
RAW_BARRAGE_MSG_TYPE = "201"
CONNECT_TIMEOUT_SECONDS = 1.8
RECV_TIMEOUT_SECONDS = 0.5
AUTOSTART_WAIT_SECONDS = 25.0
AUTOSTART_POLL_SECONDS = 0.5
MAX_RAW_SAMPLES = 8
MAX_SAMPLE_CHARS = 1200


def detect_local_link_anchor_candidate(
    *,
    ws_url: str = DEFAULT_LOCAL_BARRAGE_WS_URL,
    duration_seconds: float = 8.0,
    request_id: str | None = None,
    stop_event: Any = None,
    search_roots: list[Path] | None = None,
    autostart: bool = True,
) -> dict[str, Any]:
    started_at = time.time()
    deadline = time.monotonic() + max(1.0, duration_seconds)
    result: dict[str, Any] = {
        "request_id": request_id,
        "found": False,
        "candidate": None,
        "source": "local_barrage_raw_protobuf",
        "ws_url": ws_url,
        "messages": 0,
        "raw_payload_messages": 0,
        "methods": {},
        "errors": [],
        "sources": [],
        "autostart_attempted": False,
        "autostart_elevated": False,
        "autostart_pid": None,
        "started_at": started_at,
        "done": False,
    }
    samples: list[str] = []
    method_counter: Counter[str] = Counter()

    ws = _connect(ws_url, result)
    if ws is None and autostart:
        _autostart_local_barrage_service(search_roots or [], result)
        ws = _wait_for_connection(ws_url, deadline, stop_event, result)

    if ws is None:
        result["done"] = True
        result["elapsed_seconds"] = round(time.time() - started_at, 3)
        result["sources"].append(
            {
                "source": "local_barrage_raw_ws",
                "status": "connection_failed",
                "ws_url": ws_url,
                "candidate_count": 0,
                "text_sample": "\n".join(samples),
                "text_full": "\n".join(samples),
            }
        )
        return result

    try:
        ws.settimeout(RECV_TIMEOUT_SECONDS)
        while time.monotonic() < deadline and not _is_stopped(stop_event):
            try:
                raw = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except websocket.WebSocketConnectionClosedException as exc:
                result["errors"].append(f"local ws closed: {exc}")
                break
            except Exception as exc:
                result["errors"].append(f"local ws receive failed: {exc}")
                break

            text = _raw_to_text(raw)
            if len(samples) < MAX_RAW_SAMPLES:
                samples.append(_truncate_text(text))

            parsed = _parse_message(text)
            result["messages"] += 1
            method = _message_method(parsed)
            if method:
                method_counter[method] += 1

            candidates = list(_iter_raw_payload_candidates(parsed))
            if candidates:
                result["raw_payload_messages"] += 1

            for candidate in candidates:
                if candidate.get("is_host"):
                    continue
                name = str(candidate.get("name") or "").strip()
                if not name:
                    continue
                accepted = dict(candidate)
                accepted["source"] = "local_barrage_raw_protobuf"
                accepted["path"] = (
                    f"local.Data.PayloadBase64.{method}.{accepted.get('path')}"
                )
                result.update(
                    {
                        "found": True,
                        "candidate": accepted,
                        "done": True,
                        "elapsed_seconds": round(time.time() - started_at, 3),
                        "methods": dict(method_counter),
                    }
                )
                result["sources"].append(
                    {
                        "source": "local_barrage_raw_ws",
                        "status": "found",
                        "ws_url": ws_url,
                        "candidate_count": len(candidates),
                        "window_title": "",
                        "window_hwnd": "",
                        "text_sample": "\n".join(samples),
                        "text_full": "\n".join(samples),
                    }
                )
                return result
    finally:
        try:
            ws.close()
        except Exception:
            pass

    result["done"] = True
    result["elapsed_seconds"] = round(time.time() - started_at, 3)
    result["methods"] = dict(method_counter)
    result["sources"].append(
        {
            "source": "local_barrage_raw_ws",
            "status": "empty",
            "ws_url": ws_url,
            "candidate_count": 0,
            "window_title": "",
            "window_hwnd": "",
            "text_sample": "\n".join(samples),
            "text_full": "\n".join(samples),
        }
    )
    return result


def _connect(ws_url: str, result: dict[str, Any]) -> websocket.WebSocket | None:
    try:
        return websocket.create_connection(ws_url, timeout=CONNECT_TIMEOUT_SECONDS)
    except Exception as exc:
        result["errors"].append(f"local ws connect failed: {exc}")
        return None


def _wait_for_connection(
    ws_url: str,
    deadline: float,
    stop_event: Any,
    result: dict[str, Any],
) -> websocket.WebSocket | None:
    wait_until = min(deadline, time.monotonic() + AUTOSTART_WAIT_SECONDS)
    while time.monotonic() < wait_until and not _is_stopped(stop_event):
        ws = _connect(ws_url, result)
        if ws is not None:
            return ws
        time.sleep(AUTOSTART_POLL_SECONDS)
    return None


def _autostart_local_barrage_service(
    search_roots: list[Path],
    result: dict[str, Any],
) -> None:
    exe_path = _find_barrage_server_exe(search_roots)
    if exe_path is None:
        result["errors"].append("bundled WssBarrageServer.exe not found")
        return

    result["autostart_attempted"] = True
    try:
        creationflags = 0
        if sys.platform == "win32" and _hide_barrage_console():
            creationflags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            [str(exe_path)],
            cwd=str(exe_path.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        result["autostart_pid"] = process.pid
    except OSError as exc:
        if getattr(exc, "winerror", None) == 740 and sys.platform == "win32":
            result["errors"].append(
                f"{exe_path} requires elevation; requesting UAC runas"
            )
            if _start_barrage_service_elevated(exe_path, result):
                result["autostart_elevated"] = True
                return
        result["errors"].append(f"failed to start {exe_path}: {exc}")
    except Exception as exc:
        result["errors"].append(f"failed to start {exe_path}: {exc}")


def _start_barrage_service_elevated(
    exe_path: Path,
    result: dict[str, Any],
) -> bool:
    try:
        rc = ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",
            str(exe_path),
            None,
            str(exe_path.parent),
            1,
        )
    except Exception as exc:
        result["errors"].append(f"failed to request elevated start {exe_path}: {exc}")
        return False

    if int(rc) <= 32:
        result["errors"].append(
            f"elevated start rejected or failed for {exe_path}: code={int(rc)}"
        )
        return False
    return True


def _find_barrage_server_exe(search_roots: list[Path]) -> Path | None:
    candidates: list[Path] = []
    for root in search_roots:
        if not root:
            continue
        root = Path(root)
        candidates.extend(
            [
                root / "barrage_grab" / "WssBarrageServer.exe",
                root / "WssBarrageServer.exe",
                root / "Release_V2.8.0" / "WssBarrageServer.exe",
            ]
        )
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()
        except OSError:
            continue
    return None


def _hide_barrage_console() -> bool:
    value = os.environ.get("LOCAL_DOUYIN_BARRAGE_SHOW_WINDOW", "").strip().lower()
    return value not in {"1", "true", "yes", "on"}


def _is_stopped(stop_event: Any) -> bool:
    return bool(stop_event and getattr(stop_event, "is_set", lambda: False)())


def _raw_to_text(raw: Any) -> str:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="ignore")
    return str(raw)


def _try_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _parse_message(raw: Any) -> Any:
    parsed = _try_json_loads(raw)
    if isinstance(parsed, dict):
        data = parsed.get("Data")
        if isinstance(data, str):
            parsed = dict(parsed)
            parsed["Data"] = _try_json_loads(data)
    return parsed


def _message_method(parsed: Any) -> str:
    if not isinstance(parsed, dict):
        return ""
    data = parsed.get("Data") or parsed.get("data")
    if isinstance(data, str):
        data = _try_json_loads(data)
    if isinstance(data, dict):
        method = (
            data.get("Method")
            or data.get("method")
            or data.get("Common", {}).get("Method")
            or data.get("common", {}).get("method")
        )
        if method:
            return str(method)
    return str(parsed.get("Method") or parsed.get("method") or "")


def _iter_raw_payload_candidates(parsed: Any):
    if not isinstance(parsed, dict):
        return

    msg_type = str(parsed.get("Type") or parsed.get("type") or "")
    if msg_type and msg_type != RAW_BARRAGE_MSG_TYPE:
        return

    data = parsed.get("Data") or parsed.get("data")
    if isinstance(data, str):
        data = _try_json_loads(data)
    if not isinstance(data, dict):
        return

    payload_base64 = data.get("PayloadBase64") or data.get("payload_base64")
    if not payload_base64:
        return

    method = _message_method(parsed)
    yield from extract_link_anchor_candidates_from_payload_base64(
        payload_base64,
        method=method,
    )


def _truncate_text(text: str, limit: int = MAX_SAMPLE_CHARS) -> str:
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"
