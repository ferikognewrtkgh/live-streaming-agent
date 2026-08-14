from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

TURN_TRACE_LOG_ROOT = Path("logs") / "turn_trace"
_WRITE_LOCK = threading.Lock()
_MAX_STRING_LENGTH = 500
_MAX_LIST_ITEMS = 20


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) <= _MAX_STRING_LENGTH:
            return value
        return {
            "preview": value[:_MAX_STRING_LENGTH],
            "omitted_chars": len(value) - _MAX_STRING_LENGTH,
        }

    if isinstance(value, bytes):
        return {"bytes": len(value)}

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        safe_items = [_safe_json_value(item) for item in items[:_MAX_LIST_ITEMS]]
        if len(items) > _MAX_LIST_ITEMS:
            safe_items.append({"omitted_items": len(items) - _MAX_LIST_ITEMS})
        return safe_items

    return str(value)


def record_turn_event(
    turn_id: str | None,
    module: str,
    event: str,
    **fields: Any,
) -> None:
    """Append one turn trace event to logs/turn_trace/YYYY-MM-DD.jsonl."""
    if not turn_id:
        return

    try:
        now = datetime.now().astimezone()
        record = {
            "timestamp": now.isoformat(timespec="milliseconds"),
            "turn_id": turn_id,
            "module": module,
            "event": event,
            **{key: _safe_json_value(value) for key, value in fields.items()},
        }
        log_path = TURN_TRACE_LOG_ROOT / f"{now:%Y-%m-%d}.jsonl"
        with _WRITE_LOCK:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("Failed to write turn trace event: {!r}", exc)
