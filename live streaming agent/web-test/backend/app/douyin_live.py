"""Douyin live-room barrage capture adapted from DouyinBarrageGrab's Python demo."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import ctypes
import gzip
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import websockets

LIVE_URL = "https://live.douyin.com/{web_rid}"
DOUYIN_HOME_URL = "https://live.douyin.com/"
WSS_PATH = "/webcast/im/push/v2/"
DOUYIN_AUTH_COOKIE_NAMES = {
    "sessionid",
    "sessionid_ss",
    "sid_guard",
    "sid_tt",
}
DISPOSABLE_PROFILE_PATHS = (
    "BrowserMetrics",
    "BrowserMetrics-spare.pma",
    "Default/Cache",
    "Default/Code Cache",
    "Default/DawnGraphiteCache",
    "Default/DawnWebGPUCache",
    "Default/GPUCache",
    "Default/Shared Dictionary",
    "Edge Entity Extraction",
    "Edge Shopping",
    "Edge Sidebar",
    "Edge Signal Triggers",
    "Edge Wallet",
    "EdgeLanguageDetectionModel",
    "GrShaderCache",
    "ProvenanceData",
    "ProvenanceDataTensors",
    "Safe Browsing",
    "ShaderCache",
    "Speech Recognition",
    "Subresource Filter",
    "Typosquatting",
    "WidevineCdm",
    "Well Known Domains",
    "ZxcvbnData",
    "component_crx_cache",
    "hyphen-data",
)
METHOD_TO_EVENT = {
    "WebcastChatMessage": "chat",
    "WebcastGiftMessage": "gift",
}

PAGE_KEEPALIVE_SCRIPT = r"""
(() => {
    if (globalThis.__LIVE_STREAMING_AGENT_DOUYIN_KEEPALIVE__) return;
    globalThis.__LIVE_STREAMING_AGENT_DOUYIN_KEEPALIVE__ = true;
    const nativeClose = WebSocket.prototype.close;
    Object.defineProperty(WebSocket.prototype, "close", {
        configurable: true,
        writable: true,
        value: function (...args) {
            if (String(this.url || "").includes("/webcast/im/push/v2/")) return;
            return nativeClose.apply(this, args);
        },
    });
    const visibleProperties = {
        hidden: false,
        webkitHidden: false,
        visibilityState: "visible",
        webkitVisibilityState: "visible",
    };
    for (const [name, value] of Object.entries(visibleProperties)) {
        try {
            Object.defineProperty(document, name, {
                configurable: true,
                get: () => value,
            });
        } catch (_) {}
    }
    const signalActivity = () => {
        try {
            window.dispatchEvent(new Event("focus"));
            document.dispatchEvent(new MouseEvent("mousemove", {
                bubbles: true, clientX: 100, clientY: 100,
            }));
        } catch (_) {}
    };
    signalActivity();
    setInterval(signalActivity, 25_000);
})();
"""

EventCallback = Callable[[dict[str, Any]], Awaitable[None]]
ConnectionCallback = Callable[[], Awaitable[None]]
StatusCallback = Callable[[str], Awaitable[None]]

logger = logging.getLogger(__name__)


def cleanup_browser_profile(profile_path: Path) -> tuple[int, int]:
    """Remove reproducible browser data without touching authentication storage."""
    removed_paths = 0
    removed_bytes = 0
    profile_root = profile_path.resolve()
    for relative_path in DISPOSABLE_PROFILE_PATHS:
        candidate = profile_root.joinpath(*relative_path.split("/"))
        if not candidate.exists():
            continue
        try:
            if candidate.is_file() or candidate.is_symlink():
                removed_bytes += candidate.stat().st_size
                candidate.unlink()
            else:
                removed_bytes += sum(
                    item.stat().st_size
                    for item in candidate.rglob("*")
                    if item.is_file()
                )
                shutil.rmtree(candidate)
            removed_paths += 1
        except OSError as exc:
            logger.warning(
                "Failed to clean disposable Douyin profile path=%s error=%s",
                candidate,
                exc,
            )
    if removed_paths:
        logger.info(
            "Cleaned Douyin browser cache profile=%s paths=%s size_mb=%.2f",
            profile_root.name,
            removed_paths,
            removed_bytes / (1024 * 1024),
        )
    return removed_paths, removed_bytes


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def extract_web_rid(value: str) -> str:
    value = value.strip()
    if value.isdigit():
        return value
    match = re.search(r"live\.douyin\.com/([0-9]+)", value)
    if match:
        return match.group(1)
    raise ValueError("请输入正确的抖音直播间房间号")


def find_browser(explicit: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    for env_name in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA"):
        root = os.environ.get(env_name)
        if root:
            candidates.extend(
                [
                    Path(root) / "Microsoft/Edge/Application/msedge.exe",
                    Path(root) / "Google/Chrome/Application/chrome.exe",
                ]
            )
    for command in ("msedge", "chrome", "google-chrome", "chromium"):
        if found := shutil.which(command):
            candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("未找到 Microsoft Edge 或 Google Chrome，无法抓取直播间")


def free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_json(url: str, *, method: str = "GET") -> Any:
    request = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(request, timeout=3) as response:
        return json.load(response)


def hide_process_windows(process_id: int) -> None:
    if os.name != "nt":
        return
    user32 = ctypes.windll.user32
    window_process_id = ctypes.c_ulong()

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def hide_window(hwnd: int, _lparam: int) -> bool:
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(window_process_id))
        if window_process_id.value == process_id:
            user32.ShowWindow(hwnd, 0)
        return True

    user32.EnumWindows(hide_window, 0)


class BrowserBootstrap:
    def __init__(
        self,
        executable: Path,
        *,
        profile_path: Path | None = None,
        show_browser: bool = False,
    ) -> None:
        self.executable = executable
        self.show_browser = show_browser
        self.port = free_tcp_port()
        self.temporary_profile: tempfile.TemporaryDirectory[str] | None = None
        if profile_path is None:
            self.temporary_profile = tempfile.TemporaryDirectory(
                prefix="live_streaming_agent-douyin-live-", ignore_cleanup_errors=True
            )
            self.profile_path = Path(self.temporary_profile.name)
        else:
            self.profile_path = profile_path.resolve()
            self.profile_path.mkdir(parents=True, exist_ok=True)
        self.process: subprocess.Popen[bytes] | None = None
        self.window_hider_task: asyncio.Task[None] | None = None

    @property
    def devtools_base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> None:
        if self.temporary_profile is None:
            await asyncio.to_thread(cleanup_browser_profile, self.profile_path)
        args = [
            str(self.executable),
            "--disable-gpu",
            "--disable-component-update",
            f"--remote-debugging-port={self.port}",
            "--remote-allow-origins=*",
            f"--user-data-dir={self.profile_path}",
            "--no-first-run",
            "--no-default-browser-check",
            "--mute-audio",
            "--autoplay-policy=no-user-gesture-required",
        ]
        if self.show_browser:
            args.extend(["--window-size=1280,820"])
        else:
            args.extend(["--window-position=-32000,-32000", "--window-size=1280,720"])
        args.append("about:blank")
        startupinfo = None
        creationflags = 0
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 1 if self.show_browser else 0
            creationflags = subprocess.CREATE_NO_WINDOW
        self.process = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            startupinfo=startupinfo,
            creationflags=creationflags,
        )
        if os.name == "nt" and not self.show_browser:
            self.window_hider_task = asyncio.create_task(self._keep_windows_hidden())
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("浏览器进程提前退出")
            try:
                await asyncio.to_thread(http_json, f"{self.devtools_base}/json/version")
                logger.info(
                    "Shared Douyin browser ready port=%s profile=%s",
                    self.port,
                    self.profile_path.name,
                )
                return
            except Exception:
                await asyncio.sleep(0.25)
        raise TimeoutError("等待浏览器调试端口超时")

    async def new_target(self, url: str = "about:blank") -> dict[str, Any]:
        target_url = urllib.parse.quote(url, safe=":")
        url = f"{self.devtools_base}/json/new?{target_url}"
        return await asyncio.to_thread(http_json, url, method="PUT")

    async def close_target(self, target_id: str | None) -> None:
        if not target_id:
            return
        with contextlib.suppress(Exception):
            await asyncio.to_thread(
                http_json,
                f"{self.devtools_base}/json/close/{target_id}",
            )

    async def has_douyin_session(self) -> bool:
        version = await asyncio.to_thread(
            http_json, f"{self.devtools_base}/json/version"
        )
        debugger_url = version.get("webSocketDebuggerUrl")
        if not debugger_url:
            return False
        async with CdpConnection(debugger_url) as cdp:
            result = await cdp.command("Storage.getCookies")
        return any(
            cookie.get("name") in DOUYIN_AUTH_COOKIE_NAMES
            and "douyin.com" in str(cookie.get("domain") or "")
            for cookie in result.get("cookies", [])
        )

    async def open_url(self, url: str) -> dict[str, Any]:
        target = await self.new_target()
        debugger_url = target.get("webSocketDebuggerUrl")
        if not debugger_url:
            raise RuntimeError("浏览器未提供页面调试连接")
        async with CdpConnection(debugger_url) as cdp:
            await cdp.command("Page.enable")
            navigation = await cdp.command("Page.navigate", {"url": url})
            if navigation.get("errorText"):
                raise RuntimeError(f"打开抖音登录页失败：{navigation['errorText']}")
        return target

    async def _keep_windows_hidden(self) -> None:
        while self.process and self.process.poll() is None:
            hide_process_windows(self.process.pid)
            await asyncio.sleep(0.2)

    async def close(self) -> None:
        if self.window_hider_task:
            self.window_hider_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.window_hider_task
            self.window_hider_task = None
        process = self.process
        self.process = None
        if process and process.poll() is None:
            with contextlib.suppress(Exception):
                targets = await asyncio.to_thread(
                    http_json, f"{self.devtools_base}/json/list"
                )
                for target in targets:
                    target_id = target.get("id")
                    if target_id:
                        with contextlib.suppress(Exception):
                            await asyncio.to_thread(
                                http_json,
                                f"{self.devtools_base}/json/close/{target_id}",
                            )
                await asyncio.sleep(0.5)
            if os.name == "nt":
                await asyncio.to_thread(
                    subprocess.run,
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                await asyncio.sleep(0.5)
            else:
                process.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    await asyncio.to_thread(process.wait, 3)
                if process.poll() is None:
                    process.kill()
        if self.temporary_profile:
            for _ in range(5):
                try:
                    self.temporary_profile.cleanup()
                    break
                except (PermissionError, NotADirectoryError, OSError):
                    await asyncio.sleep(0.2)
        else:
            await asyncio.to_thread(cleanup_browser_profile, self.profile_path)


class CdpConnection:
    def __init__(self, websocket_url: str) -> None:
        self.websocket_url = websocket_url
        self.websocket: Any = None
        self.reader_task: asyncio.Task[None] | None = None
        self.events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self.next_id = 0

    async def __aenter__(self) -> CdpConnection:
        parsed_url = urllib.parse.urlsplit(self.websocket_url)
        if parsed_url.hostname in {"localhost", "::1"}:
            netloc = f"127.0.0.1:{parsed_url.port}"
            self.websocket_url = urllib.parse.urlunsplit(
                (
                    parsed_url.scheme,
                    netloc,
                    parsed_url.path,
                    parsed_url.query,
                    parsed_url.fragment,
                )
            )
        self.websocket = await websockets.connect(
            self.websocket_url,
            origin=None,
            proxy=None,
            open_timeout=15,
            close_timeout=3,
            ping_interval=None,
            max_size=32 * 1024 * 1024,
        )
        self.reader_task = asyncio.create_task(self._reader())
        return self

    async def __aexit__(self, *_: object) -> None:
        if self.reader_task:
            self.reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.reader_task
        if self.websocket:
            await self.websocket.close()

    async def _reader(self) -> None:
        try:
            async for raw in self.websocket:
                message = json.loads(raw)
                message_id = message.get("id")
                if message_id is not None:
                    future = self.pending.pop(message_id, None)
                    if future and not future.done():
                        future.set_result(message)
                elif "method" in message:
                    await self.events.put(message)
        except Exception as exc:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(exc)
            self.pending.clear()

    async def command(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float = 15,
    ) -> dict[str, Any]:
        self.next_id += 1
        message_id = self.next_id
        future = asyncio.get_running_loop().create_future()
        self.pending[message_id] = future
        await self.websocket.send(
            json.dumps({"id": message_id, "method": method, "params": params or {}})
        )
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError as exc:
            self.pending.pop(message_id, None)
            raise TimeoutError(f"浏览器调试命令 {method} 超时") from exc
        if "error" in response:
            raise RuntimeError(f"浏览器调试命令 {method} 失败")
        return response.get("result", {})


def read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
        if shift > 70:
            break
    raise ValueError("无效的 protobuf varint")


def protobuf_fields(data: bytes) -> list[tuple[int, int, int | bytes]]:
    result: list[tuple[int, int, int | bytes]] = []
    offset = 0
    while offset < len(data):
        key, offset = read_varint(data, offset)
        field_number, wire_type = key >> 3, key & 7
        if wire_type == 0:
            value, offset = read_varint(data, offset)
        elif wire_type == 1:
            value = data[offset : offset + 8]
            offset += 8
        elif wire_type == 2:
            length, offset = read_varint(data, offset)
            value = data[offset : offset + length]
            offset += length
        elif wire_type == 5:
            value = data[offset : offset + 4]
            offset += 4
        else:
            break
        result.append((field_number, wire_type, value))
    return result


def field_values(data: bytes, number: int, wire_type: int | None = None) -> list[Any]:
    return [
        value
        for field_number, field_wire_type, value in protobuf_fields(data)
        if field_number == number and (wire_type is None or field_wire_type == wire_type)
    ]


def first_value(data: bytes, number: int, wire_type: int, default: Any = None) -> Any:
    values = field_values(data, number, wire_type)
    return values[0] if values else default


def decode_text(value: bytes | None) -> str:
    return value.decode("utf-8", errors="replace") if value else ""


def decode_user(data: bytes | None) -> dict[str, Any]:
    if not data:
        return {}
    return {
        "id": first_value(data, 1, 0),
        "nickname": decode_text(first_value(data, 3, 2)),
    }


def decode_message(method: str, payload: bytes) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": METHOD_TO_EVENT.get(method, "other"),
        "method": method,
    }
    if method == "WebcastChatMessage":
        event["user"] = decode_user(first_value(payload, 2, 2))
        event["content"] = decode_text(first_value(payload, 3, 2))
    elif method == "WebcastGiftMessage":
        event["user"] = decode_user(first_value(payload, 7, 2))
        gift = first_value(payload, 15, 2)
        event["gift"] = {
            "id": first_value(gift, 5, 0) if gift else None,
            "name": decode_text(first_value(gift, 16, 2)) if gift else "",
            "diamond_count": first_value(gift, 12, 0) if gift else None,
            "count": first_value(payload, 5, 0) or first_value(payload, 6, 0),
            "repeat_end": first_value(payload, 9, 0),
        }
    elif method == "WebcastGiftSortMessage":
        event["_raw_payload"] = payload
    return event


def summarize_protobuf(data: bytes, depth: int = 0) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for number, wire_type, value in protobuf_fields(data):
        item: dict[str, Any] = {"field": number, "wire": wire_type}
        if isinstance(value, bytes):
            item["length"] = len(value)
            if depth < 2 and value:
                with contextlib.suppress(ValueError):
                    nested = summarize_protobuf(value, depth + 1)
                    if nested:
                        item["nested"] = nested[:20]
            decoded = value.decode("utf-8", errors="ignore").strip()
            if decoded and sum(character.isprintable() for character in decoded) >= len(
                decoded
            ) * 0.8:
                item["text"] = decoded[:120]
        else:
            item["value"] = value
        summary.append(item)
    return summary


def decode_wss_frame(frame: bytes) -> list[dict[str, Any]]:
    payload = first_value(frame, 8, 2)
    if not payload:
        return []
    with contextlib.suppress(gzip.BadGzipFile, EOFError, OSError):
        payload = gzip.decompress(payload)
    events: list[dict[str, Any]] = []
    for message_data in field_values(payload, 1, 2):
        method = decode_text(first_value(message_data, 1, 2))
        message_payload = first_value(message_data, 2, 2)
        if method and message_payload:
            event = decode_message(method, message_payload)
            event["msg_id"] = first_value(message_data, 3, 0)
            events.append(event)
    return events


def decode_cdp_payload(payload: str) -> bytes:
    try:
        return base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error):
        return payload.encode("latin1", errors="ignore")


async def monitor_room(
    target: dict[str, Any],
    web_rid: str,
    stop_event: asyncio.Event,
    on_event: EventCallback,
    on_connected: ConnectionCallback,
    on_status: StatusCallback,
) -> None:
    debugger_url = target.get("webSocketDebuggerUrl")
    if not debugger_url:
        raise RuntimeError("浏览器未提供直播间调试连接")
    async with CdpConnection(debugger_url) as cdp:
        await cdp.command(
            "Network.enable",
            {"maxTotalBufferSize": 100_000_000, "maxResourceBufferSize": 50_000_000},
        )
        await cdp.command("Page.enable")
        await cdp.command(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": PAGE_KEEPALIVE_SCRIPT},
        )
        try:
            navigation = await cdp.command(
                "Page.navigate", {"url": LIVE_URL.format(web_rid=web_rid)}
            )
        except TimeoutError:
            # Edge can keep loading the page after Page.navigate has been accepted,
            # especially when its disposable cache was just rebuilt. Network events
            # below are the authoritative signal that the live room is connected.
            logger.warning(
                "Douyin page navigation acknowledgement timed out room_id=%s; "
                "continuing to wait for barrage connection",
                web_rid,
            )
            navigation = {}
        if navigation.get("errorText"):
            raise RuntimeError(f"打开直播间失败：{navigation['errorText']}")
        await on_status("直播间页面已打开，正在等待弹幕连接")
        connected = False
        connect_deadline = time.monotonic() + 30
        reload_due_at: float | None = None
        reload_count = 0
        last_frame_at = time.monotonic()
        webcast_request_ids: set[str] = set()
        seen_ids: set[int] = set()
        seen_methods: set[str] = set()
        received_frame_count = 0

        while not stop_event.is_set():
            now = time.monotonic()
            if connected and now - last_frame_at > 45:
                connected = False
                reload_due_at = now
            elif not connected and now > connect_deadline and reload_due_at is None:
                if reload_count >= 1:
                    page = await cdp.command(
                        "Runtime.evaluate",
                        {
                            "expression": (
                                "JSON.stringify({href:location.href,title:document.title,"
                                "text:(document.body?.innerText||'').slice(0,500)})"
                            ),
                            "returnByValue": True,
                        },
                    )
                    raw_page = (
                        page.get("result", {}).get("value")
                        if isinstance(page, dict)
                        else None
                    )
                    page_info: dict[str, Any] = {}
                    if isinstance(raw_page, str):
                        with contextlib.suppress(json.JSONDecodeError):
                            page_info = json.loads(raw_page)
                    page_text = str(page_info.get("text") or "")
                    page_title = str(page_info.get("title") or "")
                    page_url = str(page_info.get("href") or "")
                    if any(
                        marker in page_text
                        for marker in ("直播已结束", "直播结束", "房间不存在")
                    ):
                        raise RuntimeError("该直播间当前未开播或已经结束")
                    detail = f"（页面：{page_title or page_url or '无法读取'}）"
                    if any(
                        marker in page_text
                        for marker in ("验证", "验证码", "安全验证", "登录")
                    ):
                        detail = "（抖音页面要求登录或安全验证）"
                    raise RuntimeError(
                        "等待 60 秒仍未建立弹幕连接"
                        f"{detail}，请确认房间正在直播且当前网络可正常打开抖音"
                    )
                await on_status("暂未建立弹幕连接，正在重新加载直播间")
                reload_due_at = now
            if reload_due_at is not None and now >= reload_due_at:
                reload_count += 1
                webcast_request_ids.clear()
                try:
                    await cdp.command("Page.reload", {"ignoreCache": False})
                except TimeoutError:
                    logger.warning(
                        "Douyin page reload acknowledgement timed out room_id=%s",
                        web_rid,
                    )
                connected = False
                connect_deadline = time.monotonic() + 30
                last_frame_at = time.monotonic()
                reload_due_at = None
                continue
            try:
                message = await asyncio.wait_for(cdp.events.get(), timeout=1)
            except TimeoutError:
                continue
            method = message.get("method")
            params = message.get("params", {})
            if method == "Network.webSocketCreated":
                url = params.get("url", "")
                if WSS_PATH in url:
                    webcast_request_ids.add(params["requestId"])
                    connected = True
                    last_frame_at = time.monotonic()
                    reload_due_at = None
                    reload_count = 0
                    logger.info(
                        "Douyin barrage WebSocket connected room_id=%s",
                        web_rid,
                    )
                    await on_connected()
            elif method == "Network.webSocketFrameReceived":
                if params.get("requestId") not in webcast_request_ids:
                    continue
                response = params.get("response", {})
                if response.get("opcode") != 2 or not response.get("payloadData"):
                    continue
                last_frame_at = time.monotonic()
                received_frame_count += 1
                decoded_events = decode_wss_frame(
                    decode_cdp_payload(response["payloadData"])
                )
                if received_frame_count == 1:
                    logger.info(
                        "Douyin first barrage frame received room_id=%s decoded_events=%s",
                        web_rid,
                        len(decoded_events),
                    )
                for event in decoded_events:
                    method_name = str(event.get("method") or "unknown")
                    if method_name not in seen_methods:
                        seen_methods.add(method_name)
                        logger.info(
                            "Douyin message method observed room_id=%s method=%s type=%s",
                            web_rid,
                            method_name,
                            event.get("type"),
                        )
                        if method_name == "WebcastGiftSortMessage":
                            raw_payload = event.get("_raw_payload")
                            if isinstance(raw_payload, bytes):
                                logger.info(
                                    "Douyin GiftSort protobuf room_id=%s fields=%s",
                                    web_rid,
                                    json.dumps(
                                        summarize_protobuf(raw_payload),
                                        ensure_ascii=False,
                                        separators=(",", ":"),
                                    ),
                                )
                    if event.get("type") not in {"chat", "gift"}:
                        continue
                    msg_id = event.get("msg_id")
                    if msg_id and msg_id in seen_ids:
                        continue
                    if msg_id:
                        seen_ids.add(msg_id)
                        if len(seen_ids) > 5_000:
                            seen_ids.clear()
                    await on_event(event)
            elif method == "Network.webSocketClosed":
                request_id = params.get("requestId")
                if request_id in webcast_request_ids:
                    webcast_request_ids.discard(request_id)
                    if not webcast_request_ids:
                        connected = False
                        connect_deadline = time.monotonic() + 10
                        reload_due_at = time.monotonic() + 1


class DouyinCaptureSession:
    def __init__(
        self,
        username: str,
        web_rid: str,
        *,
        browser: BrowserBootstrap | None = None,
    ) -> None:
        self.username = username
        self.web_rid = web_rid
        self.browser = browser
        self.stop_event = asyncio.Event()
        self.events: deque[dict[str, Any]] = deque(maxlen=500)
        self.condition = asyncio.Condition()
        self.sequence = 0
        self.task: asyncio.Task[None] | None = None

    async def publish(self, payload: dict[str, Any]) -> None:
        async with self.condition:
            self.sequence += 1
            event = {
                "sequence": self.sequence,
                "room_id": self.web_rid,
                "timestamp": utc_now(),
                **payload,
            }
            self.events.append(event)
            self.condition.notify_all()

    async def start(self) -> None:
        await self.publish(
            {"type": "status", "status": "starting", "message": "正在连接直播间"}
        )
        self.task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        target: dict[str, Any] | None = None
        try:
            logger.info(
                "Starting Douyin live capture username=%s room_id=%s",
                self.username,
                self.web_rid,
            )
            if self.browser is None:
                raise RuntimeError("共享抖音浏览器尚未启动")
            await self.publish(
                {
                    "type": "status",
                    "status": "starting",
                    "message": "浏览器已启动，正在打开直播间",
                }
            )
            target = await self.browser.new_target()

            async def on_connected() -> None:
                await self.publish(
                    {"type": "status", "status": "running", "message": "抓取中"}
                )

            async def on_event(raw: dict[str, Any]) -> None:
                user = raw.get("user") or {}
                if raw["type"] == "chat":
                    await self.publish(
                        {
                            "type": "chat",
                            "nickname": user.get("nickname") or "直播间用户",
                            "content": raw.get("content") or "",
                            "msg_id": raw.get("msg_id"),
                        }
                    )
                elif raw["type"] == "gift":
                    gift = raw.get("gift") or {}
                    await self.publish(
                        {
                            "type": "gift",
                            "nickname": user.get("nickname") or "直播间用户",
                            "gift_name": gift.get("name") or "礼物",
                            "gift_count": gift.get("count") or 1,
                            "diamond_count": gift.get("diamond_count"),
                            "msg_id": raw.get("msg_id"),
                        }
                    )

            async def on_status(message: str) -> None:
                await self.publish(
                    {"type": "status", "status": "starting", "message": message}
                )

            await monitor_room(
                target,
                self.web_rid,
                self.stop_event,
                on_event,
                on_connected,
                on_status,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "Douyin live capture failed username=%s room_id=%s",
                self.username,
                self.web_rid,
            )
            await self.publish(
                {"type": "status", "status": "error", "message": str(exc)}
            )
        finally:
            if self.browser and target:
                await self.browser.close_target(target.get("id"))
            if not any(
                event.get("type") == "status" and event.get("status") == "error"
                for event in list(self.events)[-1:]
            ):
                await self.publish(
                    {"type": "status", "status": "stopped", "message": "已停止抓取"}
                )

    async def stop(self) -> None:
        self.stop_event.set()
        if not self.task:
            return
        try:
            await asyncio.wait_for(self.task, timeout=8)
        except TimeoutError:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task

    async def stream(self, after_sequence: int) -> AsyncIterator[dict[str, Any]]:
        cursor = after_sequence
        while True:
            heartbeat: dict[str, Any] | None = None
            async with self.condition:
                pending = [
                    event for event in self.events if event["sequence"] > cursor
                ]
                if not pending:
                    try:
                        await asyncio.wait_for(self.condition.wait(), timeout=15)
                    except TimeoutError:
                        heartbeat = {
                            "type": "heartbeat",
                            "sequence": self.sequence,
                            "room_id": self.web_rid,
                            "timestamp": utc_now(),
                        }
                    else:
                        pending = [
                            event for event in self.events if event["sequence"] > cursor
                        ]
            if heartbeat:
                yield heartbeat
                continue
            for event in pending:
                cursor = event["sequence"]
                yield event


class DouyinLiveManager:
    def __init__(self, profile_root: Path | None = None) -> None:
        self.profile_root = (
            profile_root
            if profile_root is not None
            else Path(tempfile.gettempdir()) / "live_streaming_agent-douyin-profiles"
        ).resolve()
        self.profile_root.mkdir(parents=True, exist_ok=True)
        self.shared_profile_path = self.profile_root / "shared"
        self.sessions: dict[str, DouyinCaptureSession] = {}
        self.browser: BrowserBootstrap | None = None
        self.login_target: dict[str, Any] | None = None
        self.authenticated = False
        self.lock = asyncio.Lock()

    async def _ensure_browser_locked(self) -> BrowserBootstrap:
        if self.browser and self.browser.process and self.browser.process.poll() is None:
            return self.browser
        if self.browser:
            await self.browser.close()
        browser = BrowserBootstrap(
            find_browser(),
            profile_path=self.shared_profile_path,
        )
        await browser.start()
        self.browser = browser
        self.authenticated = await browser.has_douyin_session()
        return browser

    async def _close_login_target_locked(self) -> None:
        if self.browser and self.login_target:
            await self.browser.close_target(self.login_target.get("id"))
        self.login_target = None

    async def start_login(self) -> dict[str, Any]:
        async with self.lock:
            browser = await self._ensure_browser_locked()
            self.authenticated = await browser.has_douyin_session()
            if self.authenticated:
                return {
                    "status": "ready",
                    "message": "抖音账号已登录，所有用户将共用此登录态",
                    "qr_image": None,
                }
            await self._close_login_target_locked()
            self.login_target = await browser.open_url(DOUYIN_HOME_URL)
            await asyncio.sleep(2)
            await self._trigger_login_dialog_locked()
            return await self._login_status_locked()

    async def _trigger_login_dialog_locked(self) -> None:
        if not self.login_target:
            return
        debugger_url = self.login_target.get("webSocketDebuggerUrl")
        if not debugger_url:
            return
        script = r"""
        (() => {
          const button = [...document.querySelectorAll("button")].find(
            element => (element.innerText || "").trim() === "登录"
          );
          const reactKey = button && Object.keys(button).find(
            key => key.startsWith("__reactProps")
          );
          const onClick = reactKey && button[reactKey] && button[reactKey].onClick;
          if (typeof onClick === "function") {
            onClick();
            return {triggered: true, method: "react"};
          }
          if (button) {
            return {triggered: false, reason: "not_hydrated"};
          }
          return {triggered: false, reason: "button_missing"};
        })()
        """
        trigger: dict[str, Any] = {}
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            async with CdpConnection(debugger_url) as cdp:
                result = await cdp.command(
                    "Runtime.evaluate",
                    {"expression": script, "returnByValue": True},
                )
                trigger = result.get("result", {}).get("value") or {}
            if trigger.get("triggered"):
                break
            await asyncio.sleep(0.5)
        logger.info(
            "Douyin login trigger result=%s",
            trigger,
        )
        await asyncio.sleep(1)

    async def _login_status_locked(self) -> dict[str, Any]:
        browser = await self._ensure_browser_locked()
        self.authenticated = await browser.has_douyin_session()
        if self.authenticated:
            await self._close_login_target_locked()
            return {
                "status": "ready",
                "message": "登录成功，所有用户将共用此抖音账号",
                "qr_image": None,
            }
        if not self.login_target:
            return {
                "status": "idle",
                "message": "尚未开始抖音扫码登录",
                "qr_image": None,
            }
        debugger_url = self.login_target.get("webSocketDebuggerUrl")
        if not debugger_url:
            return {
                "status": "error",
                "message": "无法读取抖音登录页面",
                "qr_image": None,
            }
        async with CdpConnection(debugger_url) as cdp:
            qr_result = await cdp.command(
                "Runtime.evaluate",
                {
                    "expression": r"""
                    (() => {
                      const candidates = [...document.querySelectorAll("img")]
                        .map(element => {
                          const rect = element.getBoundingClientRect();
                          return {
                            src: String(element.src || ""),
                            width: rect.width,
                            height: rect.height,
                          };
                        })
                        .filter(item =>
                          item.src.startsWith("data:image/") &&
                          item.width >= 120 &&
                          item.height >= 120 &&
                          Math.abs(item.width - item.height) <= 12
                        )
                        .sort((left, right) =>
                          right.width * right.height - left.width * left.height
                        );
                      return candidates[0]?.src || null;
                    })()
                    """,
                    "returnByValue": True,
                },
            )
        qr_image = qr_result.get("result", {}).get("value")
        return {
            "status": "waiting_scan",
            "message": "请使用抖音 App 扫描二维码并确认登录",
            "qr_image": qr_image,
        }

    async def login_status(self) -> dict[str, Any]:
        async with self.lock:
            return await self._login_status_locked()

    async def finish_login(self) -> dict[str, Any]:
        async with self.lock:
            status = await self._login_status_locked()
            if status["status"] != "ready":
                await self._close_login_target_locked()
            return status

    async def start(self, username: str, room_id: str) -> DouyinCaptureSession:
        web_rid = extract_web_rid(room_id)
        async with self.lock:
            browser = await self._ensure_browser_locked()
            self.authenticated = await browser.has_douyin_session()
            if not self.authenticated:
                raise RuntimeError("请先在前端完成抖音扫码登录")
            await self._close_login_target_locked()
            previous = self.sessions.get(username)
            if previous:
                await previous.stop()
            session = DouyinCaptureSession(
                username,
                web_rid,
                browser=browser,
            )
            self.sessions[username] = session
            await session.start()
            return session

    async def stop(self, username: str) -> DouyinCaptureSession | None:
        async with self.lock:
            session = self.sessions.pop(username, None)
            if session:
                await session.stop()
            return session

    async def release(self, username: str) -> DouyinCaptureSession | None:
        async with self.lock:
            session = self.sessions.pop(username, None)
            if session:
                await session.stop()
            return session

    def get(self, username: str) -> DouyinCaptureSession | None:
        return self.sessions.get(username)

    async def stop_all(self) -> None:
        async with self.lock:
            sessions = list(self.sessions.values())
            browser = self.browser
            self.sessions.clear()
            self.browser = None
            self.login_target = None
        await asyncio.gather(*(session.stop() for session in sessions))
        if browser:
            await browser.close()
