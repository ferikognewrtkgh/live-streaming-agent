from __future__ import annotations

import argparse
import sys
import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable


PATH_HINTS = (
    "link",
    "linked",
    "linkmic",
    "link_mic",
    "cohost",
    "co_host",
    "guest",
    "rival",
    "opponent",
    "pk",
    "battle",
    "connect",
    "connection",
    "mic",
    "\u8fde\u7ebf",
    "\u8fde\u9ea6",
    "\u5609\u5bbe",
    "\u5bf9\u65b9",
    "\u5bf9\u6218",
)
NAME_KEYS = (
    "Nickname",
    "NickName",
    "nickname",
    "nick_name",
    "Name",
    "name",
    "DisplayName",
    "display_name",
    "DisplayId",
    "DisplayID",
    "display_id",
    "ShortId",
    "short_id",
    "UniqueId",
    "unique_id",
    "ScreenName",
    "screen_name",
)
ID_KEYS = (
    "SecUid",
    "sec_uid",
    "UserId",
    "UserID",
    "user_id",
    "Id",
    "ID",
    "uid",
    "RoomId",
    "room_id",
    "WebcastRoomId",
)
REJECT_NAMES = {
    "",
    "unknown",
    "viewer",
    "\u4e3b\u64ad",
    "\u8fde\u7ebf\u4e3b\u64ad",
    "\u6296\u97f3",
    "\u6296\u97f3\u76f4\u64ad",
    "\u6296\u97f3\u76f4\u64ad\u4f34\u4fa3",
    "\u76f4\u64ad\u4f34\u4fa3",
    "\u76f4\u64ad\u95f4",
    "\u8fde\u9ea6\u4e2d",
    "\u8fde\u7ebf\u4e2d",
    "pk\u8fde\u7ebf",
    "\u8fde\u7ebfpk",
    "pk\u4e2d",
    "\u4e0e",
    "\u548c",
    "\u9000\u51fapk",
    "\u6bd4\u62fc\u65b9\u5f0f",
    "\u5e38\u89c4pk",
    "\u8fdb\u884c\u4e2d",
    "\u66f4\u591a\u73a9\u6cd5",
    "\u7acb\u5373\u5339\u914d",
    "\u968f\u673a\u5339\u914d",
    "\u8bf4\u70b9\u4ec0\u4e48",
    "\u53d1\u9001",
    "\u5bfc\u64ad",
    "\u573a\u666f",
    "\u573a\u666f\u4e00",
    "\u573a\u666f\u4e8c",
    "\u573a\u666f\u4e09",
    "\u5e38\u89c4\u6a21\u5f0f",
    "\u4e92\u52a8\u73a9\u6cd5",
    "\u89c2\u4f17\u8fde\u7ebf",
    "ai\u5609\u5bbe",
    "\u798f\u888b",
    "\u793c\u7269\u83dc\u5355",
    "\u5ba0\u7c89\u7ea2\u5305",
    "\u5ba0\u7c89",
    "\u5fc3\u613f",
    "\u793c\u7269\u6295\u7968",
    "\u6e38\u620f\u80fd\u529b",
    "\u7559\u8a00\u4e0a\u5899",
    "\u7559\u8a00",
    "\u4eba\u6c14\u4efb\u52a1",
    "\u793c\u7269\u5c55\u9986",
    "\u8d77\u6d41\u6311\u6218",
    "\u76f4\u64ad\u5de5\u5177",
    "\u76f4\u64ad\u8bbe\u7f6e",
    "\u4e2d\u63a7\u53f0",
    "\u7d20\u6750\u5e93",
    "\u6e38\u620f\u73a9\u6cd5",
    "\u4e92\u52a8\u5de5\u5177",
    "\u865a\u62df\u5f62\u8c61",
    "\u7eff\u5e55\u76f4\u64ad",
    "ai\u7ecf\u7eaa\u4eba",
    "\u6296\u97f3\u5c0f\u52a9\u624b",
    "\u6211\u65b9\u8d21\u732e\u699c",
    "\u8d21\u732e\u699c",
    "pk\u8d21\u732e\u699c",
    "\u5728\u7ebf\u89c2\u4f17\u699c",
    "\u672c\u573a\u89c2\u4f17\u699c",
    "\u518d\u6765\u4e00\u5c40",
    "\u7ed9ta\u70b9\u70b9",
    "\u6444\u50cf\u5934\u5e03\u5c40",
    "\u8fde\u7ebf\u8bbe\u7f6e",
    "\u672a\u5206\u7c7b",
    "\u4eba\u6c14\u699c",
    "\u5c0f\u8377\u699c",
    "pk\u7ed3\u675f",
    "\u5e73\u5c40",
    "\u5173\u64ad",
    "\u4e3b\u64ad\u4e2d\u5fc3",
    "\u663e\u793a\u5668",
    "\u6dfb\u52a0\u7d20\u6750",
    "obs64.exe",
    "dream maker live console",
    "chrome legacy window",
}
REJECT_NAME_SUBSTRINGS = {
    "\u8981\u83b7\u53d6\u7f3a\u5931\u7684\u56fe\u7247\u8bf4\u660e",
    "\u6b22\u8fce\u6765\u5230\u76f4\u64ad\u95f4",
    "\u5e73\u53f0\u4e25\u7981",
    "\u8bf7\u6587\u660epk",
    "pk\u8fde\u7ebf",
    "\u8fde\u7ebf",
    "\u9000\u51fapk",
    "\u6bd4\u62fc\u65b9\u5f0f",
    "\u5e38\u89c4pk",
    "\u8fdb\u884c\u4e2d",
    "\u66f4\u591a\u73a9\u6cd5",
    "\u7acb\u5373\u5339\u914d",
    "\u968f\u673a\u5339\u914d",
    "\u8bf4\u70b9\u4ec0\u4e48",
    "\u8d21\u732e\u699c",
    "\u793c\u7269\u83dc\u5355",
    "\u5ba0\u7c89",
    "\u798f\u888b",
    "\u6e38\u620f\u80fd\u529b",
    "\u7559\u8a00",
    "\u7ed9ta\u70b9",
    "websocket",
    "chrome legacy",
}
TEXT_LABEL_RE = re.compile(
    r"(?P<label>\u5bf9\u65b9|\u5609\u5bbe|\u4e3b\u64ad|\u7528\u6237|\u6635\u79f0|"
    r"\u6296\u97f3\u53f7|ID|id)[:：\s]+(?P<value>[^\n\r|｜]{2,40})"
)
PK_RELATION_NAME_RE = re.compile(
    r"(?:^|[\s\r\n])(?:\u4e0e|\u548c)\s*"
    r"(?P<name>[\w\u4e00-\u9fff\u00b7\u2022._-]{1,24})\s*"
    r"(?:PK\u4e2d|pk\u4e2d|PK|pk|\u8fde\u7ebf\u4e2d|\u8fde\u9ea6\u4e2d)"
)


@dataclass
class LinkAnchorCandidate:
    nickname: str | None = None
    display_id: str | None = None
    sec_uid: str | None = None
    room_id: str | None = None
    source: str = "unknown"
    confidence: float = 0.0
    path: str | None = None
    raw_text: str | None = None

    @property
    def usable(self) -> bool:
        return bool(self.nickname or self.display_id or self.sec_uid)


@dataclass
class ProbeResult:
    found: bool
    candidate: LinkAnchorCandidate | None
    sources: list[dict[str, Any]]
    errors: list[str]
    elapsed_seconds: float


def normalize_candidate_name(value: Any) -> str | None:
    if value is None:
        return None
    name = str(value).strip()
    if not name:
        return None
    name = re.sub(r"[\r\n\t]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip(" @:：|-_/\\")
    name = re.sub(r"^(\u5bf9\u65b9|\u5609\u5bbe|\u4e3b\u64ad|\u7528\u6237|\u6635\u79f0)[:：\s]+", "", name)
    name = re.sub(r"(\u7684)?\u76f4\u64ad\u95f4$", "", name).strip()
    name = re.sub(r"(\u6b63\u5728)?\u76f4\u64ad(\u4e2d)?$", "", name).strip()
    name = name.strip(" @:：|-_/\\")
    if len(name) > 32:
        return None
    if len(name) < 2 and not re.fullmatch(r"[\u4e00-\u9fff]", name):
        return None
    lowered = name.lower()
    if lowered in REJECT_NAMES:
        return None
    if any(part.lower() in lowered for part in REJECT_NAME_SUBSTRINGS):
        return None
    if "://" in lowered or lowered.startswith("ws"):
        return None
    if re.fullmatch(r"\d{5,}", name):
        return None
    return name


def normalize_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().strip("@:：| ")
    if not text:
        return None
    if len(text) > 128:
        return None
    if "://" in text or re.search(r"[\s\u4e00-\u9fff]", text):
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.-]{2,128}", text):
        return None
    return text


def _path_has_hint(path: tuple[str, ...]) -> bool:
    joined = ".".join(path).lower()
    return any(hint in joined for hint in PATH_HINTS)


def _first_key(data: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return None


def iter_structured_candidates(
    value: Any,
    *,
    source: str,
    path: tuple[str, ...] = (),
    depth: int = 0,
    hinted: bool = False,
) -> Iterable[LinkAnchorCandidate]:
    if depth > 9:
        return

    current_hinted = hinted or _path_has_hint(path)
    if isinstance(value, dict):
        if current_hinted:
            nickname = normalize_candidate_name(_first_key(value, NAME_KEYS))
            display_id = normalize_id(
                _first_key(value, ("DisplayId", "DisplayID", "display_id", "UniqueId", "unique_id", "ShortId", "short_id"))
            )
            sec_uid = normalize_id(_first_key(value, ("SecUid", "sec_uid")))
            user_id = normalize_id(_first_key(value, ("UserId", "UserID", "user_id", "uid")))
            room_id = normalize_id(_first_key(value, ("RoomId", "room_id", "WebcastRoomId")))
            if nickname or display_id or sec_uid or user_id:
                yield LinkAnchorCandidate(
                    nickname=nickname,
                    display_id=display_id or user_id,
                    sec_uid=sec_uid,
                    room_id=room_id,
                    source=source,
                    confidence=0.88 if sec_uid or display_id else 0.8,
                    path=".".join(path),
                )

            for key in ("User", "user", "Anchor", "anchor", "Guest", "guest", "Owner", "owner"):
                nested = value.get(key)
                if isinstance(nested, dict):
                    nested_path = (*path, key)
                    yield from iter_structured_candidates(
                        nested,
                        source=source,
                        path=nested_path,
                        depth=depth + 1,
                        hinted=True,
                    )

        for key, nested in value.items():
            yield from iter_structured_candidates(
                nested,
                source=source,
                path=(*path, str(key)),
                depth=depth + 1,
                hinted=current_hinted,
            )
        return

    if isinstance(value, list):
        for index, item in enumerate(value[:30]):
            yield from iter_structured_candidates(
                item,
                source=source,
                path=(*path, str(index)),
                depth=depth + 1,
                hinted=current_hinted,
            )


def candidates_from_text(text: str, *, source: str) -> list[LinkAnchorCandidate]:
    if not text:
        return []
    candidates: list[LinkAnchorCandidate] = []
    seen: set[str] = set()

    for match in PK_RELATION_NAME_RE.finditer(text):
        raw = match.group("name").strip()
        nickname = normalize_candidate_name(raw)
        if nickname and nickname not in seen:
            seen.add(nickname)
            candidates.append(
                LinkAnchorCandidate(
                    nickname=nickname,
                    source=source,
                    confidence=0.74,
                    raw_text=match.group(0).strip(),
                )
            )

    for match in TEXT_LABEL_RE.finditer(text):
        label = match.group("label").strip().lower()
        raw = match.group("value").strip()
        if label == "id" or "\u6296\u97f3\u53f7" in label:
            display_id = normalize_id(raw)
            if display_id and display_id not in seen:
                seen.add(display_id)
                candidates.append(
                    LinkAnchorCandidate(
                        display_id=display_id,
                        source=source,
                        confidence=0.58,
                        raw_text=raw,
                    )
                )
            continue
        nickname = normalize_candidate_name(raw)
        if nickname and nickname not in seen:
            seen.add(nickname)
            candidates.append(
                LinkAnchorCandidate(
                    nickname=nickname,
                    source=source,
                    confidence=0.62,
                    raw_text=raw,
                )
            )

    lines = [
        line.strip()
        for line in re.split(r"[\r\n]+", text)
        if line and line.strip()
    ]
    for line in lines[:120]:
        for part in re.split(r"\s{2,}|[|｜]", line):
            nickname = normalize_candidate_name(part)
            if nickname and nickname not in seen:
                lowered = nickname.lower()
                if any(reject.lower() == lowered for reject in REJECT_NAMES):
                    continue
                seen.add(nickname)
                candidates.append(
                    LinkAnchorCandidate(
                        nickname=nickname,
                        source=source,
                        confidence=0.5,
                        raw_text=part,
                    )
                )

    return candidates


def best_candidate(candidates: Iterable[LinkAnchorCandidate]) -> LinkAnchorCandidate | None:
    usable = [candidate for candidate in candidates if candidate.usable]
    if not usable:
        return None
    return sorted(
        usable,
        key=lambda item: (
            item.confidence,
            bool(item.sec_uid),
            bool(item.display_id),
            bool(item.nickname),
        ),
        reverse=True,
    )[0]


def _win_enumerate_windows() -> list[tuple[int, str]]:
    if sys.platform != "win32":
        return []
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsIconic.restype = wintypes.BOOL

    results: list[tuple[int, str]] = []
    enum_proc = ctypes.WINFUNCTYPE(
        wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
    )

    def callback(hwnd: Any, _lparam: Any) -> bool:
        try:
            if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            title = (buffer.value or "").strip()
            if title:
                results.append((int(hwnd), title))
        except Exception:
            pass
        return True

    try:
        user32.EnumWindows(enum_proc(callback), 0)
    except Exception:
        return []
    return results


def _win_find_window_by_title(
    title: str | None,
    preferred_hwnd: int | None = None,
) -> tuple[int | None, str | None]:
    if not title and not preferred_hwnd:
        return None, None
    windows = _win_enumerate_windows()
    if preferred_hwnd:
        for hwnd, window_title in windows:
            if hwnd != preferred_hwnd:
                continue
            if not title or window_title == title or title in window_title or window_title in title:
                return hwnd, window_title
    for hwnd, window_title in windows:
        if window_title == title:
            return hwnd, window_title
    for hwnd, window_title in windows:
        if title in window_title or window_title in title:
            return hwnd, window_title
    lowered = title.lower()
    for hwnd, window_title in windows:
        if lowered in window_title.lower():
            return hwnd, window_title
    return None, None


def probe_live_companion_ui(
    window_title: str | None,
    window_hwnd: int | None = None,
) -> tuple[list[LinkAnchorCandidate], dict[str, Any]]:
    if not window_title and not window_hwnd:
        return [], {"source": "live_companion_uia", "status": "skipped", "reason": "missing window title"}

    hwnd, actual_title = _win_find_window_by_title(window_title, preferred_hwnd=window_hwnd)
    if not hwnd:
        return [], {
            "source": "live_companion_uia",
            "status": "unavailable",
            "reason": f"window not found: title={window_title} hwnd={window_hwnd}",
        }

    texts: list[str] = []
    errors: list[str] = []
    try:
        import uiautomation as auto  # type: ignore
    except Exception as exc:
        errors.append(f"uiautomation unavailable: {exc}")
    else:
        try:
            root = auto.ControlFromHandle(hwnd)
            values: list[str] = []
            seen: set[str] = set()

            def collect(control: Any, depth: int = 0) -> None:
                if depth > 5 or len(values) >= 160:
                    return
                try:
                    name = str(getattr(control, "Name", "") or "").strip()
                except Exception:
                    name = ""
                if name and name not in seen:
                    seen.add(name)
                    values.append(name)
                try:
                    children = control.GetChildren()
                except Exception:
                    children = []
                for child in children[:40]:
                    collect(child, depth + 1)

            collect(root)
            texts.append("\n".join(values))
        except Exception as exc:
            errors.append(f"uiautomation failed: {exc}")

    try:
        from pywinauto import Desktop  # type: ignore
    except Exception as exc:
        errors.append(f"pywinauto unavailable: {exc}")
    else:
        try:
            window = Desktop(backend="uia").window(handle=hwnd)
            values: list[str] = []
            seen: set[str] = set()
            for control in window.descendants()[:180]:
                for value in control.texts():
                    text = str(value or "").strip()
                    if text and text not in seen:
                        seen.add(text)
                        values.append(text)
            texts.append("\n".join(values))
        except Exception as exc:
            errors.append(f"pywinauto failed: {exc}")

    combined = "\n".join(text for text in texts if text)
    candidates = candidates_from_text(combined, source="live_companion_uia")
    status = "ok" if combined else "unavailable"
    return candidates, {
        "source": "live_companion_uia",
        "status": status,
        "window_title": actual_title,
        "window_hwnd": hwnd,
        "text_chars": len(combined),
        "text_sample": combined[:1200],
        "text_full": combined[:20000],
        "candidate_count": len(candidates),
        "errors": errors,
    }


def probe_web_page(
    *,
    url: str | None,
    cdp_url: str | None,
    duration_seconds: float,
    headless: bool,
) -> tuple[list[LinkAnchorCandidate], dict[str, Any]]:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as exc:
        return [], {
            "source": "web",
            "status": "unavailable",
            "reason": f"playwright unavailable: {exc}",
        }

    candidates: list[LinkAnchorCandidate] = []
    errors: list[str] = []
    with sync_playwright() as playwright:
        browser = None
        context = None
        try:
            if cdp_url:
                browser = playwright.chromium.connect_over_cdp(cdp_url)
                context = browser.contexts[0] if browser.contexts else browser.new_context()
            else:
                browser = playwright.chromium.launch(headless=headless)
                context = browser.new_context()

            page = context.pages[0] if context.pages else context.new_page()

            def inspect_payload(payload: Any, source: str) -> None:
                for candidate in iter_structured_candidates(payload, source=source):
                    candidates.append(candidate)

            def handle_response(response: Any) -> None:
                try:
                    content_type = response.headers.get("content-type", "")
                    if "json" not in content_type.lower():
                        return
                    inspect_payload(response.json(), "web_network")
                except Exception:
                    return

            def handle_websocket(ws: Any) -> None:
                def handle_frame(frame: Any) -> None:
                    try:
                        payload = getattr(frame, "payload", frame)
                        if isinstance(payload, bytes):
                            payload = payload.decode("utf-8", errors="ignore")
                        if isinstance(payload, str):
                            payload = json.loads(payload)
                        inspect_payload(payload, "web_websocket")
                    except Exception:
                        return

                ws.on("framereceived", handle_frame)

            page.on("response", handle_response)
            page.on("websocket", handle_websocket)
            if url:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)

            deadline = time.time() + max(0.5, duration_seconds)
            while time.time() < deadline:
                try:
                    body_text = page.evaluate(
                        "() => document.body ? document.body.innerText.slice(0, 50000) : ''"
                    )
                    candidates.extend(candidates_from_text(body_text, source="web_dom"))
                except Exception:
                    pass
                if best_candidate(candidates):
                    break
                page.wait_for_timeout(500)
        except Exception as exc:
            errors.append(str(exc))
        finally:
            if not cdp_url and browser:
                browser.close()

    return candidates, {
        "source": "web",
        "status": "ok" if candidates else "empty",
        "candidate_count": len(candidates),
        "errors": errors,
    }


def run_probe(
    *,
    url: str | None = None,
    cdp_url: str | None = None,
    live_companion_window: str | None = None,
    live_companion_hwnd: int | None = None,
    duration_seconds: float = 8.0,
    headless: bool = False,
) -> ProbeResult:
    started_at = time.time()
    all_candidates: list[LinkAnchorCandidate] = []
    sources: list[dict[str, Any]] = []
    errors: list[str] = []

    if url or cdp_url:
        candidates, source_status = probe_web_page(
            url=url,
            cdp_url=cdp_url,
            duration_seconds=duration_seconds,
            headless=headless,
        )
        all_candidates.extend(candidates)
        sources.append(source_status)

    if live_companion_window or live_companion_hwnd:
        candidates, source_status = probe_live_companion_ui(
            live_companion_window,
            window_hwnd=live_companion_hwnd,
        )
        all_candidates.extend(candidates)
        sources.append(source_status)

    if not sources:
        errors.append(
            "no probe source configured; pass --url/--cdp-url, "
            "--live-companion-window, or --live-companion-hwnd"
        )

    candidate = best_candidate(all_candidates)
    return ProbeResult(
        found=bool(candidate),
        candidate=candidate,
        sources=sources,
        errors=errors,
        elapsed_seconds=round(time.time() - started_at, 3),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe Douyin web/live companion for the co-host anchor name/id."
    )
    parser.add_argument("--url", help="Douyin live web URL to open with Playwright.")
    parser.add_argument(
        "--cdp-url",
        help="Existing Chromium/Edge remote debugging endpoint, e.g. http://127.0.0.1:9222.",
    )
    parser.add_argument(
        "--live-companion-window",
        help="Part or full title of the Douyin Live Companion window.",
    )
    parser.add_argument(
        "--live-companion-hwnd",
        type=int,
        help="Exact Windows hwnd for the Douyin Live Companion window.",
    )
    parser.add_argument(
        "--list-windows",
        action="store_true",
        help="List visible top-level Windows window titles and exit.",
    )
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if args.list_windows:
        print(
            json.dumps(
                [
                    {"hwnd": hwnd, "title": title}
                    for hwnd, title in _win_enumerate_windows()
                ],
                ensure_ascii=False,
                indent=2 if args.pretty else None,
            )
        )
        return 0
    result = run_probe(
        url=args.url,
        cdp_url=args.cdp_url,
        live_companion_window=args.live_companion_window,
        live_companion_hwnd=args.live_companion_hwnd,
        duration_seconds=args.duration,
        headless=args.headless,
    )
    print(
        json.dumps(
            asdict(result),
            ensure_ascii=False,
            indent=2 if args.pretty else None,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
