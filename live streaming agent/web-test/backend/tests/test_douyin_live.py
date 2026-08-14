import asyncio

import pytest
from backend.app.douyin_live import (
    CdpConnection,
    DouyinCaptureSession,
    DouyinLiveManager,
    cleanup_browser_profile,
    decode_message,
    extract_web_rid,
)


def protobuf_varint(value: int) -> bytes:
    result = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        result.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(result)


def protobuf_bytes(field_number: int, value: bytes) -> bytes:
    return (
        protobuf_varint((field_number << 3) | 2)
        + protobuf_varint(len(value))
        + value
    )


def protobuf_uint(field_number: int, value: int) -> bytes:
    return protobuf_varint(field_number << 3) + protobuf_varint(value)


def test_extract_web_rid_accepts_room_number_and_url() -> None:
    assert extract_web_rid(" 187786036155 ") == "187786036155"
    assert (
        extract_web_rid("https://live.douyin.com/187786036155?foo=bar")
        == "187786036155"
    )


def test_extract_web_rid_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="房间号"):
        extract_web_rid("not-a-room")


def test_decode_chat_message() -> None:
    user = protobuf_bytes(3, "测试用户".encode())
    payload = protobuf_bytes(2, user) + protobuf_bytes(3, "你好 Live Streaming Agent".encode())

    event = decode_message("WebcastChatMessage", payload)

    assert event["type"] == "chat"
    assert event["user"]["nickname"] == "测试用户"
    assert event["content"] == "你好 Live Streaming Agent"


def test_decode_gift_message() -> None:
    user = protobuf_bytes(3, "送礼用户".encode())
    gift = (
        protobuf_uint(5, 1001)
        + protobuf_uint(12, 10)
        + protobuf_bytes(16, "小心心".encode())
    )
    payload = (
        protobuf_uint(5, 3)
        + protobuf_bytes(7, user)
        + protobuf_bytes(15, gift)
    )

    event = decode_message("WebcastGiftMessage", payload)

    assert event["type"] == "gift"
    assert event["user"]["nickname"] == "送礼用户"
    assert event["gift"] == {
        "id": 1001,
        "name": "小心心",
        "diamond_count": 10,
        "count": 3,
        "repeat_end": None,
    }


@pytest.mark.asyncio
async def test_session_stream_replays_events_after_sequence() -> None:
    session = DouyinCaptureSession("tester", "123456")
    await session.publish({"type": "chat", "nickname": "甲", "content": "第一条"})
    await session.publish({"type": "chat", "nickname": "乙", "content": "第二条"})

    stream = session.stream(after_sequence=1)
    event = await asyncio.wait_for(anext(stream), timeout=1)
    await stream.aclose()

    assert event["sequence"] == 2
    assert event["content"] == "第二条"


@pytest.mark.asyncio
async def test_manager_stop_removes_session(tmp_path) -> None:
    manager = DouyinLiveManager(tmp_path)
    session = DouyinCaptureSession("tester", "123456")
    manager.sessions["tester"] = session

    stopped = await manager.stop("tester")

    assert stopped is session
    assert manager.get("tester") is None
    assert session.stop_event.is_set()


def test_manager_uses_single_shared_profile_path(tmp_path) -> None:
    manager = DouyinLiveManager(tmp_path)

    assert manager.shared_profile_path == tmp_path.resolve() / "shared"


def test_cleanup_browser_profile_keeps_authentication_data(tmp_path) -> None:
    profile = tmp_path / "shared"
    cache = profile / "Default" / "Cache"
    cookies = profile / "Default" / "Network" / "Cookies"
    local_state = profile / "Local State"
    local_storage = profile / "Default" / "Local Storage" / "leveldb" / "data"
    for path in (cache / "cache.bin", cookies, local_state, local_storage):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"auth-or-cache")

    removed_paths, removed_bytes = cleanup_browser_profile(profile)

    assert removed_paths == 1
    assert removed_bytes == len(b"auth-or-cache")
    assert not cache.exists()
    assert cookies.exists()
    assert local_state.exists()
    assert local_storage.exists()


@pytest.mark.asyncio
async def test_cdp_command_timeout_removes_pending_request() -> None:
    class FakeWebSocket:
        async def send(self, _message: str) -> None:
            return None

    cdp = CdpConnection("ws://127.0.0.1/devtools/page/test")
    cdp.websocket = FakeWebSocket()

    with pytest.raises(TimeoutError, match="Page.navigate"):
        await cdp.command("Page.navigate", timeout=0.001)

    assert cdp.pending == {}
