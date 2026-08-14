"""Extract Douyin link-mic anchor candidates from raw protobuf payloads.

Douyin Webcast payloads are protobuf messages, but the app does not ship with
Douyin's private .proto files. This module uses a conservative wire-format
reader and a small set of observed field heuristics from WebcastLinkMessage.
"""

from __future__ import annotations

import base64
import binascii
import re
from typing import Any


MAX_PROTO_DEPTH = 8
MAX_PROTO_FIELDS = 800
HOST_ROLE_TEXT = "\u4e3b\u6301"
LINK_METHOD_HINTS = (
    "link",
    "linkmic",
    "battle",
    "pk",
)
REJECT_NAMES = {
    "",
    "\u4e3b\u6301",
    "\u4e3b\u64ad",
    "\u8fde\u7ebf\u4e3b\u64ad",
    "\u6211\u65b9\u8d21\u732e\u699c",
    "pk\u8fde\u7ebf",
    "webcastlinkmessage",
    "webcastlinkmicmethod",
    "webcastlinkmicbattlemethod",
}


class ProtoDecodeError(ValueError):
    pass


def extract_link_anchor_candidates_from_payload_base64(
    payload_base64: Any,
    method: str | None = None,
) -> list[dict[str, Any]]:
    """Return likely anchor records from a Douyin raw Webcast protobuf payload."""

    if not _is_link_method(method):
        return []

    payload = _decode_base64(payload_base64)
    if not payload:
        return []

    try:
        fields = _decode_message(payload, 0)
    except ProtoDecodeError:
        return []

    candidates = list(_walk_candidate_messages(fields))
    return _dedupe_candidates(candidates)


def _is_link_method(method: str | None) -> bool:
    if not method:
        return True
    lowered = str(method).lower()
    return any(hint in lowered for hint in LINK_METHOD_HINTS)


def _decode_base64(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if not isinstance(value, str) or not value.strip():
        return b""
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return b""


def _decode_message(data: bytes, depth: int) -> list[dict[str, Any]]:
    if depth > MAX_PROTO_DEPTH:
        return []

    fields: list[dict[str, Any]] = []
    pos = 0
    length = len(data)
    while pos < length:
        if len(fields) >= MAX_PROTO_FIELDS:
            break

        key, pos = _read_varint(data, pos)
        field_no = key >> 3
        wire_type = key & 0x07
        if field_no <= 0:
            raise ProtoDecodeError("invalid field number")

        item: dict[str, Any] = {
            "field": field_no,
            "wire": wire_type,
        }
        if wire_type == 0:
            value, pos = _read_varint(data, pos)
            item["value"] = value
        elif wire_type == 1:
            end = pos + 8
            if end > length:
                raise ProtoDecodeError("fixed64 out of range")
            item["value"] = int.from_bytes(data[pos:end], "little")
            pos = end
        elif wire_type == 2:
            item_length, pos = _read_varint(data, pos)
            end = pos + item_length
            if end > length:
                raise ProtoDecodeError("length-delimited out of range")
            raw = data[pos:end]
            pos = end
            item["raw"] = raw
            text = _decode_printable_text(raw)
            if text is not None:
                item["text"] = text
            elif raw:
                nested = _try_decode_nested(raw, depth + 1)
                if nested:
                    item["nested"] = nested
        elif wire_type == 5:
            end = pos + 4
            if end > length:
                raise ProtoDecodeError("fixed32 out of range")
            item["value"] = int.from_bytes(data[pos:end], "little")
            pos = end
        else:
            raise ProtoDecodeError("unsupported wire type")

        fields.append(item)

    if pos != length:
        raise ProtoDecodeError("trailing bytes")
    return fields


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while pos < len(data):
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
        if shift >= 70:
            break
    raise ProtoDecodeError("invalid varint")


def _decode_printable_text(raw: bytes) -> str | None:
    if not raw:
        return ""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not text:
        return ""

    printable = 0
    for char in text:
        if char in "\r\n\t" or not char.isprintable():
            continue
        printable += 1
    if printable / max(len(text), 1) < 0.9:
        return None
    if any(ord(char) < 32 and char not in "\r\n\t" for char in text):
        return None
    return text.strip()


def _try_decode_nested(raw: bytes, depth: int) -> list[dict[str, Any]]:
    if depth > MAX_PROTO_DEPTH or len(raw) > 20_000:
        return []
    try:
        fields = _decode_message(raw, depth)
    except ProtoDecodeError:
        return []
    if not fields:
        return []
    return fields


def _walk_candidate_messages(
    fields: list[dict[str, Any]],
    path: tuple[str, ...] = (),
    inherited_host_role: bool = False,
):
    host_role = inherited_host_role or _has_text_shallow(fields, HOST_ROLE_TEXT, 3)
    candidate = _candidate_from_direct_fields(fields, path, host_role)
    if candidate:
        yield candidate

    for field in fields:
        nested = field.get("nested")
        if isinstance(nested, list) and nested:
            next_path = (*path, str(field.get("field")))
            yield from _walk_candidate_messages(nested, next_path, host_role)


def _candidate_from_direct_fields(
    fields: list[dict[str, Any]],
    path: tuple[str, ...],
    is_host: bool,
) -> dict[str, Any] | None:
    name = _normalize_name(_first_text(fields, 3))
    if not name:
        return None

    user_id = _first_numericish(fields, 1) or _first_numericish(fields, 1028)
    sec_uid = _first_text(fields, 46)
    room_id = _nested_text(fields, (33, 2))
    short_id = _first_numericish(fields, 2)

    if not user_id and not sec_uid and not room_id:
        return None

    confidence = 0.94
    if is_host:
        confidence = 0.48

    return {
        "name": name,
        "id": user_id,
        "short_id": short_id,
        "room_id": room_id,
        "sec_uid": sec_uid,
        "is_host": is_host,
        "source": "barrage_raw_protobuf",
        "path": ".".join(path) or "<root>",
        "confidence": confidence,
    }


def _first_text(fields: list[dict[str, Any]], field_no: int) -> str | None:
    for field in fields:
        if field.get("field") != field_no:
            continue
        text = field.get("text")
        if isinstance(text, str) and text:
            return text
        value = field.get("value")
        if value not in (None, ""):
            return str(value)
    return None


def _first_numericish(fields: list[dict[str, Any]], field_no: int) -> str | None:
    text = _first_text(fields, field_no)
    if text and re.fullmatch(r"\d{5,}", text):
        return text
    for field in fields:
        if field.get("field") != field_no:
            continue
        value = field.get("value")
        if isinstance(value, int) and value >= 10000:
            return str(value)
    return None


def _nested_text(fields: list[dict[str, Any]], field_path: tuple[int, ...]) -> str | None:
    if not field_path:
        return None
    current_no = field_path[0]
    rest = field_path[1:]
    for field in fields:
        if field.get("field") != current_no:
            continue
        if not rest:
            text = field.get("text")
            if isinstance(text, str) and text:
                return text
            value = field.get("value")
            if value not in (None, ""):
                return str(value)
            continue
        nested = field.get("nested")
        if isinstance(nested, list):
            value = _nested_text(nested, rest)
            if value:
                return value
    return None


def _has_text_shallow(
    fields: list[dict[str, Any]],
    target: str,
    max_edges: int,
) -> bool:
    if max_edges <= 0:
        return False
    for field in fields:
        text = field.get("text")
        if isinstance(text, str) and target in text:
            return True
        nested = field.get("nested")
        if isinstance(nested, list) and _has_text_shallow(
            nested,
            target,
            max_edges - 1,
        ):
            return True
    return False


def _normalize_name(value: Any) -> str | None:
    if value is None:
        return None
    name = str(value).strip()
    if not name:
        return None
    name = re.sub(r"[\r\n\t]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip(" @:\u951b\u6b80-_/\\")
    name = re.sub(r"(\u7684)?\u76f4\u64ad\u95f4$", "", name).strip()
    name = re.sub(r"(\u6b63\u5728)?\u76f4\u64ad(\u4e2d)?$", "", name).strip()
    if not name:
        return None

    lowered = name.lower()
    if lowered in REJECT_NAMES:
        return None
    if "://" in lowered or lowered.startswith("ws"):
        return None
    if lowered.startswith(("webcast", "sslocal")):
        return None
    if len(name) < 2 and not re.fullmatch(r"[\u4e00-\u9fff]", name):
        return None
    if len(name) > 40:
        return None
    if re.fullmatch(r"\d{5,}", name):
        return None
    if re.fullmatch(r"\d+\s*\u7c89\u4e1d", name):
        return None
    if not re.search(r"[\u4e00-\u9fffA-Za-z]", name):
        return None
    return name


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in sorted(
        candidates,
        key=lambda item: (
            bool(item.get("is_host")),
            -float(item.get("confidence") or 0.0),
            str(item.get("name") or ""),
        ),
    ):
        key = (
            str(candidate.get("id") or ""),
            str(candidate.get("room_id") or ""),
            str(candidate.get("name") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result
