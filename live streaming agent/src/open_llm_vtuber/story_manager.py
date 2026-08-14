import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any

from loguru import logger

from prompts import prompt_loader

from .chat_history_manager import (
    CHAT_HISTORY_ROOT,
    WORKING_DIR_NAME,
    get_metadata,
    update_metadate,
)

STORY_FILE_NAME = "story.txt"
STORY_WINDOW_SIZE = 3
STORY_MATCH_THRESHOLD = 0.5
STORY_PROGRESS_KEY = "story_progress_index"
STORY_SIGNATURE_KEY = "story_signature"


def _conf_base_dir(conf_uid: str) -> Path:
    return Path(CHAT_HISTORY_ROOT) / conf_uid


def _working_dir(conf_uid: str) -> Path:
    return _conf_base_dir(conf_uid) / WORKING_DIR_NAME


def _working_story_path(conf_uid: str) -> Path:
    return _working_dir(conf_uid) / STORY_FILE_NAME


def ensure_working_story_file(conf_uid: str) -> Path | None:
    """Return working/story.txt when the current working session provides one."""
    story_path = _working_story_path(conf_uid)
    if story_path.exists():
        return story_path
    return None


def _read_story_text(conf_uid: str) -> str:
    story_path = ensure_working_story_file(conf_uid)
    if not story_path:
        return ""

    try:
        return prompt_loader._load_file_content(str(story_path))
    except Exception:
        logger.exception("Failed to load story file: {}", story_path)
        return ""


def _story_signature(story_text: str) -> str:
    return hashlib.sha256(story_text.encode("utf-8")).hexdigest()


def _split_story_line(line: str) -> tuple[str | None, str]:
    stripped = line.strip()
    match = re.match(r"^\{\{\s*(user|human|char|ai|assistant)\s*\}\}\s*[:：]\s*(.*)$", stripped, re.I)
    if not match:
        return None, stripped

    role = match.group(1).lower()
    if role in {"user", "human"}:
        return "user", match.group(2).strip()
    return "ai", match.group(2).strip()


def parse_story_entries(story_text: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    current: dict[str, str] = {}
    last_role: str | None = None

    for raw_line in story_text.splitlines():
        if not raw_line.strip():
            continue

        role, content = _split_story_line(raw_line)
        if role == "user":
            if current.get("user") and current.get("ai"):
                entries.append(current)
            current = {"user": content, "ai": ""}
            last_role = "user"
            continue

        if role == "ai":
            if not current.get("user"):
                continue
            current["ai"] = content
            last_role = "ai"
            entries.append(current)
            current = {}
            continue

        if last_role and current:
            current[last_role] = f"{current.get(last_role, '')}\n{content}".strip()

    if current.get("user") and current.get("ai"):
        entries.append(current)

    return [
        {
            "index": index,
            "user": entry.get("user", ""),
            "ai": entry.get("ai", ""),
        }
        for index, entry in enumerate(entries)
    ]


def _load_story_entries(conf_uid: str) -> tuple[list[dict[str, Any]], str]:
    story_text = _read_story_text(conf_uid)
    if not story_text:
        return [], ""
    return parse_story_entries(story_text), _story_signature(story_text)


def _current_story_progress(
    conf_uid: str,
    history_uid: str,
    signature: str,
) -> int:
    metadata = get_metadata(conf_uid, history_uid)
    if metadata.get(STORY_SIGNATURE_KEY) == signature:
        progress = metadata.get(STORY_PROGRESS_KEY, 0)
        if isinstance(progress, int):
            return max(0, progress)

    progress = 0
    update_metadate(
        conf_uid,
        history_uid,
        {
            STORY_SIGNATURE_KEY: signature,
            STORY_PROGRESS_KEY: progress,
        },
    )
    return progress


def _clamp_progress(progress: int, total: int) -> int:
    if total <= 0:
        return 0
    return min(max(0, progress), total)


def get_story_state_payload(
    conf_uid: str,
    history_uid: str,
    *,
    start_index: int | None = None,
) -> dict[str, Any]:
    entries, signature = _load_story_entries(conf_uid)
    if not entries:
        return {
            "has_story": False,
            "progress_index": 0,
            "total": 0,
            "items": [],
        }

    if start_index is None:
        start_index = _current_story_progress(conf_uid, history_uid, signature)
    start_index = _clamp_progress(start_index, len(entries))

    return {
        "has_story": True,
        "progress_index": start_index,
        "total": len(entries),
        "items": entries[start_index : start_index + STORY_WINDOW_SIZE],
    }


def _normalize_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in normalized if ch.isalnum())


def _lcs_length(a: str, b: str) -> int:
    if not a or not b:
        return 0

    previous = [0] * (len(b) + 1)
    for char_a in a:
        current = [0]
        for index_b, char_b in enumerate(b, start=1):
            if char_a == char_b:
                current.append(previous[index_b - 1] + 1)
            else:
                current.append(max(previous[index_b], current[-1]))
        previous = current
    return previous[-1]


def _candidate_indices(story_candidates: Any) -> list[int]:
    if not isinstance(story_candidates, list):
        return []

    indices = []
    for candidate in story_candidates:
        if not isinstance(candidate, dict):
            continue
        try:
            indices.append(int(candidate.get("index")))
        except (TypeError, ValueError):
            continue
    return indices


def _build_story_guidance(entry: dict[str, Any]) -> str:
    return (
        "[剧情剧本指令]\n"
        "用户刚刚的话命中了下面这条剧本。请按照“AI剧本台词”的核心意思回答，"
        "可以自然口语化并保持角色语气，但不要提到剧本、匹配或系统指令。\n"
        f"剧本用户台词：{entry.get('user', '')}\n"
        f"AI剧本台词：{entry.get('ai', '')}"
    )


def match_story_and_advance(
    *,
    conf_uid: str,
    history_uid: str,
    translated_user_text: str,
    story_candidates: Any,
) -> dict[str, Any] | None:
    entries, signature = _load_story_entries(conf_uid)
    if not entries:
        return None

    candidate_indices = _candidate_indices(story_candidates)
    if not candidate_indices:
        progress = _current_story_progress(conf_uid, history_uid, signature)
        candidate_indices = [
            entry["index"]
            for entry in entries[progress : progress + STORY_WINDOW_SIZE]
        ]

    normalized_user_text = _normalize_for_match(translated_user_text)
    best_match: dict[str, Any] | None = None
    best_score = 0.0

    for index in candidate_indices:
        if index < 0 or index >= len(entries):
            continue
        entry = entries[index]
        normalized_story_user = _normalize_for_match(str(entry.get("user") or ""))
        if not normalized_story_user:
            continue
        score = _lcs_length(normalized_story_user, normalized_user_text) / len(
            normalized_story_user
        )
        if score > best_score:
            best_score = score
            best_match = entry

    if not best_match or best_score < STORY_MATCH_THRESHOLD:
        return None

    next_index = _clamp_progress(int(best_match["index"]) + 1, len(entries))
    update_metadate(
        conf_uid,
        history_uid,
        {
            STORY_SIGNATURE_KEY: signature,
            STORY_PROGRESS_KEY: next_index,
        },
    )

    story_state = get_story_state_payload(
        conf_uid,
        history_uid,
        start_index=next_index,
    )
    story_state["active_entry"] = best_match
    story_state["match_score"] = round(best_score, 4)

    logger.info(
        "Matched story entry {} with score {:.2f}; progress advanced to {}",
        best_match["index"],
        best_score,
        next_index,
    )

    return {
        "entry": best_match,
        "score": best_score,
        "next_index": next_index,
        "story_state": story_state,
        "guidance": _build_story_guidance(best_match),
    }
