import json
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, List, Optional, TypedDict

from loguru import logger

CHAT_HISTORY_ROOT = os.path.join("logs", "chat_history")
WORKING_DIR_NAME = "working"
ARCHIVE_DIR_NAME = "archive"
SUMMARIES_DIR_NAME = "summaries"
METADATA_ROLE = "metadata"
LEGACY_SUMMARY_DIR_NAME = "basic_memory_summaries"
STORY_FILE_NAME = "story.txt"


class HistoryMessage(TypedDict):
    role: Literal["human", "ai", "system"]
    timestamp: str
    content: str
    # Optional display information for the message
    name: Optional[str]


def _is_safe_filename(filename: str) -> bool:
    """Validate filename for safety and allowed characters."""
    if not filename or len(filename) > 255:
        return False
    if any(char in filename for char in '/\\:*?"<>|'):
        return False

    pattern = re.compile(r"^[\w\- .\u00A0-\uFFFF]+$")
    return bool(pattern.match(filename))


def _sanitize_path_component(component: str) -> str:
    """Sanitize and validate a path component."""
    sanitized = os.path.basename(component.strip())

    if not _is_safe_filename(sanitized):
        raise ValueError(f"Invalid characters in path component: {component}")

    return sanitized


def _get_conf_dir(safe_conf_uid: str) -> str:
    return os.path.normpath(os.path.join(CHAT_HISTORY_ROOT, safe_conf_uid))


def _ensure_conf_dir(conf_uid: str) -> str:
    """Ensure the directory for a specific conf exists and return its path."""
    if not conf_uid:
        raise ValueError("conf_uid cannot be empty")

    safe_conf_uid = _sanitize_path_component(conf_uid)
    base_dir = _get_conf_dir(safe_conf_uid)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _get_subdir(conf_uid: str, subdir_name: str) -> Path:
    base_dir = Path(_ensure_conf_dir(conf_uid))
    return base_dir / subdir_name


def _unique_path(directory: Path, filename: str) -> Path:
    target = directory / filename
    if not target.exists():
        return target

    stem = target.stem
    suffix = target.suffix
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for index in range(1, 1000):
        candidate = directory / f"{stem}_archived_{timestamp}_{index}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Unable to build unique archive path for {target}")


def _move_history_file(source: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = _unique_path(target_dir, source.name)
    os.replace(source, target)
    return target


def _move_file_to_dir(source: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = _unique_path(target_dir, source.name)
    os.replace(source, target)
    return target


def _copy_file_to_dir(source: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = _unique_path(target_dir, source.name)
    shutil.copy2(source, target)
    return target


def _iter_history_files(directory: Path, *, recursive: bool = False) -> list[Path]:
    if not directory.exists():
        return []
    if recursive:
        return [
            path
            for path in directory.rglob("*.json")
            if path.is_file()
        ]
    return [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix == ".json"
    ]


def _unique_archive_bundle_dir(
    archive_dir: Path,
    preferred_name: str | None = None,
) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    base_name = preferred_name or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    candidate = archive_dir / base_name
    if not candidate.exists():
        return candidate

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for index in range(1, 1000):
        candidate = archive_dir / f"{base_name}_archived_{timestamp}_{index}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Unable to build unique archive directory for {archive_dir}")


def _history_sort_key(path: Path) -> tuple[float, str]:
    try:
        return (path.stat().st_mtime, path.name)
    except OSError:
        return (0.0, path.name)


def _migrate_legacy_histories(
    base_dir: Path,
    working_dir: Path,
    archive_dir: Path,
) -> None:
    """Move old root-level history files into the new working/archive layout."""
    legacy_files = [
        path
        for path in base_dir.iterdir()
        if path.is_file() and path.suffix == ".json"
    ]
    if not legacy_files:
        return

    working_files = _iter_history_files(working_dir)
    newest_legacy = max(legacy_files, key=_history_sort_key)
    for legacy_file in legacy_files:
        target_dir = archive_dir
        if legacy_file == newest_legacy and not working_files:
            target_dir = working_dir
        try:
            moved_path = _move_history_file(legacy_file, target_dir)
            logger.info(f"Migrated legacy history {legacy_file} to {moved_path}")
        except Exception:
            logger.exception(f"Failed to migrate legacy history file: {legacy_file}")


def _summary_day_from_legacy_file(summary_file: Path) -> str:
    match = re.search(r"_(\d{8})T", summary_file.name)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            pass
    return datetime.fromtimestamp(summary_file.stat().st_mtime).strftime("%Y-%m-%d")


def _migrate_legacy_summaries(summaries_dir: Path) -> None:
    legacy_dir = Path(CHAT_HISTORY_ROOT) / LEGACY_SUMMARY_DIR_NAME
    if not legacy_dir.exists() or not legacy_dir.is_dir():
        return

    for summary_file in legacy_dir.iterdir():
        if not summary_file.is_file() or summary_file.suffix != ".json":
            continue
        day_dir = summaries_dir / _summary_day_from_legacy_file(summary_file)
        try:
            moved_path = _move_history_file(summary_file, day_dir)
            logger.info(f"Migrated legacy summary {summary_file} to {moved_path}")
        except Exception:
            logger.exception(f"Failed to migrate legacy summary file: {summary_file}")

    try:
        legacy_dir.rmdir()
    except OSError:
        pass


def _ensure_conf_dirs(conf_uid: str) -> dict[str, Path]:
    base_dir = Path(_ensure_conf_dir(conf_uid))
    working_dir = base_dir / WORKING_DIR_NAME
    archive_dir = base_dir / ARCHIVE_DIR_NAME
    summaries_dir = base_dir / SUMMARIES_DIR_NAME
    working_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_histories(base_dir, working_dir, archive_dir)
    _migrate_legacy_summaries(summaries_dir)
    return {
        "base": base_dir,
        "working": working_dir,
        "archive": archive_dir,
        "summaries": summaries_dir,
    }


def _assert_inside(base_dir: Path, full_path: Path) -> None:
    base_abs = base_dir.resolve()
    full_abs = full_path.resolve()
    if (
        os.path.normcase(os.path.commonpath([str(base_abs), str(full_abs)]))
        != os.path.normcase(str(base_abs))
    ):
        raise ValueError("Invalid path: Path traversal detected")


def _get_safe_history_path(
    conf_uid: str,
    history_uid: str,
    location: Literal["working", "archive"] = "working",
) -> Path:
    """Get sanitized path for a history file in the new layout."""
    safe_history_uid = _sanitize_path_component(history_uid)
    dirs = _ensure_conf_dirs(conf_uid)
    target_dir = dirs[location]
    full_path = target_dir / f"{safe_history_uid}.json"
    _assert_inside(target_dir, full_path)
    return full_path


def _find_history_path(conf_uid: str, history_uid: str) -> Path | None:
    if not conf_uid or not history_uid:
        return None
    safe_history_uid = _sanitize_path_component(history_uid)
    dirs = _ensure_conf_dirs(conf_uid)
    for location in ("working", "archive"):
        candidate = dirs[location] / f"{safe_history_uid}.json"
        _assert_inside(dirs[location], candidate)
        if candidate.exists():
            return candidate

    for candidate in _iter_history_files(dirs["archive"], recursive=True):
        if candidate.stem == safe_history_uid:
            _assert_inside(dirs["archive"], candidate)
            return candidate

    legacy_candidate = dirs["base"] / f"{safe_history_uid}.json"
    _assert_inside(dirs["base"], legacy_candidate)
    if legacy_candidate.exists():
        return legacy_candidate
    return None


def _read_history_data(filepath: Path) -> list[dict[str, Any]]:
    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    logger.warning(f"History file is not a list: {filepath}")
    return []


def _write_history_data(filepath: Path, history_data: list[dict[str, Any]]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(history_data, f, ensure_ascii=False, indent=2)


def _initial_history_data(history_uid: str) -> list[dict[str, Any]]:
    return [
        {
            "role": METADATA_ROLE,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "history_uid": history_uid,
            "summary_files": [],
            "active_summary_file": None,
        }
    ]


def _ensure_history_metadata(
    history_data: list[dict[str, Any]],
    history_uid: str,
) -> list[dict[str, Any]]:
    if history_data and history_data[0].get("role") == METADATA_ROLE:
        history_data[0].setdefault("history_uid", history_uid)
        history_data[0].setdefault("summary_files", [])
        history_data[0].setdefault("active_summary_file", None)
        return history_data

    return _initial_history_data(history_uid) + history_data


def _keep_latest_working_history(conf_uid: str) -> Path | None:
    dirs = _ensure_conf_dirs(conf_uid)
    working_files = sorted(
        _iter_history_files(dirs["working"]),
        key=_history_sort_key,
        reverse=True,
    )
    if not working_files:
        return None

    latest_file = working_files[0]
    for stale_file in working_files[1:]:
        try:
            moved_path = _move_history_file(stale_file, dirs["archive"])
            logger.info(f"Archived stale working history {stale_file} to {moved_path}")
        except Exception:
            logger.exception(f"Failed to archive stale working history: {stale_file}")
    return latest_file


def get_or_create_working_history(conf_uid: str) -> str:
    """Return the current working history, creating one if the folder is empty."""
    if not conf_uid:
        logger.warning("No conf_uid provided")
        return ""

    latest_file = _keep_latest_working_history(conf_uid)
    if latest_file:
        return latest_file.stem
    return create_new_history(conf_uid, archive_existing=False)


def create_new_history(conf_uid: str, archive_existing: bool = True) -> str:
    """Create a new working history file with a unique ID."""
    if not conf_uid:
        logger.warning("No conf_uid provided")
        return ""

    if archive_existing:
        archive_working_history(conf_uid)

    history_uid = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{uuid.uuid4().hex}"

    try:
        filepath = _get_safe_history_path(conf_uid, history_uid, "working")
        _write_history_data(filepath, _initial_history_data(history_uid))
    except Exception as e:
        logger.error(f"Failed to create new history file: {e}")
        return ""

    logger.debug(f"Created new working history file: {filepath}")
    return history_uid


def archive_working_history(conf_uid: str) -> list[str]:
    """Archive working history files and copy story.txt into one archive bundle."""
    if not conf_uid:
        return []

    dirs = _ensure_conf_dirs(conf_uid)
    working_files = _iter_history_files(dirs["working"])
    story_file = dirs["working"] / STORY_FILE_NAME
    story_files = [story_file] if story_file.exists() and story_file.is_file() else []
    if not working_files and not story_files:
        return []

    preferred_name = None
    if working_files:
        newest_history = max(working_files, key=_history_sort_key)
        preferred_name = newest_history.stem
    elif story_files:
        preferred_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_story")

    archive_bundle_dir = _unique_archive_bundle_dir(dirs["archive"], preferred_name)
    archived_paths = []
    for working_file in working_files:
        try:
            archived_path = _move_file_to_dir(working_file, archive_bundle_dir)
            archived_paths.append(str(archived_path))
            logger.info(f"Archived working history {working_file} to {archived_path}")
        except Exception:
            logger.exception(f"Failed to archive working file: {working_file}")
    for story_file in story_files:
        try:
            archived_path = _copy_file_to_dir(story_file, archive_bundle_dir)
            archived_paths.append(str(archived_path))
            logger.info(f"Copied working story {story_file} to {archived_path}")
        except Exception:
            logger.exception(f"Failed to copy working story file: {story_file}")
    return archived_paths


def archive_all_working_histories() -> list[str]:
    """Archive current working history files for every character folder."""
    root = Path(CHAT_HISTORY_ROOT)
    if not root.exists():
        return []

    archived_paths = []
    for conf_dir in root.iterdir():
        if not conf_dir.is_dir() or conf_dir.name == LEGACY_SUMMARY_DIR_NAME:
            continue
        archived_paths.extend(archive_working_history(conf_dir.name))
    return archived_paths


def store_message(
    conf_uid: str,
    history_uid: str,
    role: Literal["human", "ai", "system"],
    content: str,
    name: str | None = None,
    avatar: str | None = None,
):
    """Store a message in a specific history file.

    The avatar argument is accepted for compatibility with older call sites, but
    avatars are no longer persisted in chat history.
    """
    if not conf_uid or not history_uid:
        if not conf_uid:
            logger.warning("Missing conf_uid")
        if not history_uid:
            logger.warning("Missing history_uid")
        return

    filepath = _find_history_path(conf_uid, history_uid) or _get_safe_history_path(
        conf_uid,
        history_uid,
        "working",
    )
    logger.debug(f"Storing {role} message to {filepath}")

    history_data = []
    if filepath.exists():
        try:
            history_data = _read_history_data(filepath)
        except Exception:
            logger.exception(f"Failed to load history file: {filepath}")

    history_data = _ensure_history_metadata(history_data, history_uid)

    new_item = {
        "role": role,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "content": content,
    }

    if name is not None:
        new_item["name"] = name

    history_data.append(new_item)
    _write_history_data(filepath, history_data)
    logger.debug(f"Successfully stored {role} message")


def get_metadata(conf_uid: str, history_uid: str) -> dict:
    """Get metadata from history file."""
    if not conf_uid or not history_uid:
        return {}

    filepath = _find_history_path(conf_uid, history_uid)
    if not filepath:
        return {}

    try:
        history_data = _read_history_data(filepath)
        if history_data and history_data[0].get("role") == METADATA_ROLE:
            return history_data[0]
    except Exception as e:
        logger.error(f"Failed to get metadata: {e}")
    return {}


def update_metadate(conf_uid: str, history_uid: str, metadata: dict) -> bool:
    """Set metadata in history file.

    Updates existing metadata with new fields, preserving existing ones.
    If no metadata exists, creates new metadata entry.
    """
    if not conf_uid or not history_uid:
        return False

    filepath = _find_history_path(conf_uid, history_uid)
    if not filepath:
        return False

    try:
        history_data = _ensure_history_metadata(
            _read_history_data(filepath),
            history_uid,
        )
        history_data[0].update(metadata)
        _write_history_data(filepath, history_data)

        logger.debug(f"Updated metadata for history {history_uid}")
        return True
    except Exception as e:
        logger.error(f"Failed to set metadata: {e}")
    return False


def get_history(conf_uid: str, history_uid: str) -> List[HistoryMessage]:
    """Read chat history for the given conf_uid and history_uid."""
    if not conf_uid or not history_uid:
        if not conf_uid:
            logger.warning("Missing conf_uid")
        if not history_uid:
            logger.warning("Missing history_uid")
        return []

    filepath = _find_history_path(conf_uid, history_uid)

    if not filepath:
        logger.warning(f"History file not found: {conf_uid}/{history_uid}")
        return []

    try:
        history_data = _read_history_data(filepath)
        return [msg for msg in history_data if msg.get("role") != METADATA_ROLE]
    except Exception:
        logger.exception(f"Failed to read history file: {filepath}")
        return []


def delete_history(conf_uid: str, history_uid: str) -> bool:
    """Delete a specific history file."""
    if not conf_uid or not history_uid:
        logger.warning("Missing conf_uid or history_uid")
        return False

    filepath = _find_history_path(conf_uid, history_uid)
    try:
        if filepath and filepath.exists():
            os.remove(filepath)
            logger.debug(f"Successfully deleted history file: {filepath}")
            return True
    except Exception as e:
        logger.error(f"Failed to delete history file: {e}")
    return False


def _history_info_from_file(filepath: Path, location: str) -> dict | None:
    history_uid = filepath.stem
    try:
        messages = _read_history_data(filepath)
        actual_messages = [
            msg for msg in messages if msg.get("role") != METADATA_ROLE
        ]
        latest_message = actual_messages[-1] if actual_messages else None
        if location == ARCHIVE_DIR_NAME and latest_message is None:
            return None
        timestamp = latest_message.get("timestamp") if latest_message else None
        return {
            "uid": history_uid,
            "latest_message": latest_message,
            "timestamp": timestamp or datetime.fromtimestamp(
                filepath.stat().st_mtime,
            ).isoformat(timespec="seconds"),
            "location": location,
        }
    except Exception as e:
        logger.error(f"Error reading history file {filepath}: {e}")
        return None


def get_history_list(conf_uid: str) -> List[dict]:
    """Get list of working and archived histories with their latest messages."""
    if not conf_uid:
        return []

    histories = []
    try:
        dirs = _ensure_conf_dirs(conf_uid)
        for filepath in _iter_history_files(dirs[WORKING_DIR_NAME]):
            history_info = _history_info_from_file(filepath, WORKING_DIR_NAME)
            if history_info:
                histories.append(history_info)

        for filepath in _iter_history_files(dirs[ARCHIVE_DIR_NAME], recursive=True):
            history_info = _history_info_from_file(filepath, ARCHIVE_DIR_NAME)
            if history_info:
                histories.append(history_info)

        histories.sort(
            key=lambda x: x["timestamp"] if x["timestamp"] else "",
            reverse=True,
        )
        return histories

    except Exception as e:
        logger.error(f"Error listing histories: {e}")
        return []


def get_summary_storage_dir(conf_uid: str, created_at: datetime | None = None) -> Path:
    """Return the per-character, per-day summary folder."""
    created_at = created_at or datetime.now()
    dirs = _ensure_conf_dirs(conf_uid)
    summary_dir = dirs["summaries"] / created_at.strftime("%Y-%m-%d")
    summary_dir.mkdir(parents=True, exist_ok=True)
    return summary_dir


def _summary_file_sort_key(path: Path) -> tuple[float, str]:
    try:
        return (path.stat().st_mtime, path.name)
    except OSError:
        return (0.0, path.name)


def list_summary_files(conf_uid: str) -> list[Path]:
    """List all per-character summary files."""
    dirs = _ensure_conf_dirs(conf_uid)
    summaries_dir = dirs["summaries"]
    if not summaries_dir.exists():
        return []
    return sorted(
        [
            path
            for path in summaries_dir.rglob("*.json")
            if path.is_file()
        ],
        key=_summary_file_sort_key,
    )


def _summary_relative_path(conf_uid: str, summary_file_path: str | Path) -> str:
    dirs = _ensure_conf_dirs(conf_uid)
    path = Path(summary_file_path)
    try:
        return path.resolve().relative_to(dirs["base"].resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_summary_file(conf_uid: str, summary_file_name: str) -> Path | None:
    dirs = _ensure_conf_dirs(conf_uid)
    candidate = dirs["base"] / summary_file_name
    try:
        _assert_inside(dirs["base"], candidate)
    except ValueError:
        return None
    if candidate.exists():
        return candidate
    return None


def record_summary_file(
    conf_uid: str,
    history_uid: str,
    summary_file_path: str | Path,
    *,
    active: bool = True,
) -> bool:
    """Record summary file usage in a history file's metadata."""
    relative_path = _summary_relative_path(conf_uid, summary_file_path)
    metadata = get_metadata(conf_uid, history_uid)
    summary_files = metadata.get("summary_files", [])
    if not isinstance(summary_files, list):
        summary_files = []
    if relative_path not in summary_files:
        summary_files.append(relative_path)

    update = {
        "summary_files": summary_files,
        "last_summary_file": relative_path,
    }
    if active:
        update["active_summary_file"] = relative_path
    return update_metadate(conf_uid, history_uid, update)


def get_latest_summary_payload(
    conf_uid: str,
    history_uid: str,
) -> dict[str, Any] | None:
    """Load the latest summary payload explicitly recorded by this history."""
    if not get_history(conf_uid, history_uid):
        return None

    metadata = get_metadata(conf_uid, history_uid)
    recorded_files = metadata.get("summary_files", [])
    if not isinstance(recorded_files, list):
        recorded_files = []

    candidate_files = []
    active_summary_file = metadata.get("active_summary_file")
    if isinstance(active_summary_file, str) and active_summary_file:
        resolved_active = _resolve_summary_file(conf_uid, active_summary_file)
        if resolved_active:
            candidate_files.append(resolved_active)

    for summary_file_name in recorded_files:
        if not isinstance(summary_file_name, str):
            continue
        resolved_file = _resolve_summary_file(conf_uid, summary_file_name)
        if resolved_file and resolved_file not in candidate_files:
            candidate_files.append(resolved_file)

    if not candidate_files:
        return None

    latest_file = max(candidate_files, key=_summary_file_sort_key)
    try:
        with latest_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        logger.exception(f"Failed to load summary file: {latest_file}")
        return None

    if not isinstance(payload, dict):
        logger.warning(f"Summary file is not a JSON object: {latest_file}")
        return None

    recorded_summary_files = [
        _summary_relative_path(conf_uid, path)
        for path in candidate_files
    ]
    return {
        "path": str(latest_file),
        "relative_path": _summary_relative_path(conf_uid, latest_file),
        "summary_files": recorded_summary_files,
        "payload": payload,
    }


def modify_latest_message(
    conf_uid: str,
    history_uid: str,
    role: Literal["human", "ai", "system"],
    new_content: str,
) -> bool:
    """Modify the latest message in a specific history file if its role matches."""
    if not conf_uid or not history_uid:
        logger.warning("Missing conf_uid or history_uid")
        return False

    filepath = _find_history_path(conf_uid, history_uid)
    if not filepath:
        logger.warning(f"History file not found: {conf_uid}/{history_uid}")
        return False

    try:
        history_data = _read_history_data(filepath)

        if not history_data:
            logger.warning("History is empty")
            return False

        latest_message = history_data[-1]
        if latest_message.get("role") != role:
            logger.warning(
                f"Latest message role ({latest_message.get('role')}) doesn't match "
                f"requested role ({role})"
            )
            return False

        latest_message["content"] = new_content
        _write_history_data(filepath, history_data)

        logger.debug(f"Successfully modified latest {role} message")
        return True

    except Exception as e:
        logger.error(f"Failed to modify latest message: {e}")
        return False


def rename_history_file(
    conf_uid: str,
    old_history_uid: str,
    new_history_uid: str,
) -> bool:
    """Rename a history file with a new history_uid."""
    if not conf_uid or not old_history_uid or not new_history_uid:
        logger.warning("Missing required parameters for rename")
        return False

    old_filepath = _find_history_path(conf_uid, old_history_uid)
    if not old_filepath:
        return False

    new_filepath = old_filepath.with_name(
        f"{_sanitize_path_component(new_history_uid)}.json"
    )

    try:
        os.rename(old_filepath, new_filepath)
        logger.info(
            f"Renamed history file from {old_history_uid} to {new_history_uid}"
        )
        return True
    except Exception as e:
        logger.error(f"Failed to rename history file: {e}")
    return False
