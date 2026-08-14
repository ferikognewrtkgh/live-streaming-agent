import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = BACKEND_DIR / "logs"
DEFAULT_LOG_MAX_BYTES = 20 * 1024 * 1024
MODEL_CALL_LOGGER_NAME = "live_streaming_agent.model_calls"
CHAT_LATENCY_LOGGER_NAME = "live_streaming_agent.chat_latency"


class DateSizeRotatingFileHandler(logging.Handler):
    terminator = "\n"

    def __init__(
        self,
        log_root: Path,
        filename_prefix: str,
        extension: str,
        *,
        max_bytes: int = DEFAULT_LOG_MAX_BYTES,
    ) -> None:
        super().__init__()
        self.log_root = log_root
        self.filename_prefix = filename_prefix
        self.extension = extension.lstrip(".")
        self.max_bytes = max_bytes
        self._stream: TextIO | None = None
        self._file_size = 0
        self._date_key = ""

    def _open_new_file(self, now: datetime) -> None:
        if self._stream is not None:
            self._stream.flush()
            self._stream.close()

        self._date_key = now.strftime("%Y-%m-%d")
        date_dir = self.log_root / self._date_key
        date_dir.mkdir(parents=True, exist_ok=True)
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
        path = (
            date_dir
            / f"{self.filename_prefix}_{timestamp}.{self.extension}"
        )
        collision_number = 1
        while path.exists():
            path = (
                date_dir
                / (
                    f"{self.filename_prefix}_{timestamp}_"
                    f"{collision_number:02d}.{self.extension}"
                )
            )
            collision_number += 1
        self._stream = path.open("a", encoding="utf-8")
        self._file_size = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            rendered = f"{self.format(record)}{self.terminator}"
            rendered_size = len(rendered.encode("utf-8"))
            now = datetime.now().astimezone()
            date_key = now.strftime("%Y-%m-%d")
            should_rotate = (
                self._stream is None
                or date_key != self._date_key
                or (
                    self._file_size > 0
                    and self._file_size + rendered_size > self.max_bytes
                )
            )
            if should_rotate:
                self._open_new_file(now)
            if self._stream is None:
                raise RuntimeError("log file stream was not initialized")
            self._stream.write(rendered)
            self._stream.flush()
            self._file_size += rendered_size
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.flush()
            self._stream.close()
            self._stream = None
        super().close()


def _rotating_handler(
    log_root: Path,
    filename_prefix: str,
    extension: str,
    formatter: logging.Formatter,
) -> DateSizeRotatingFileHandler:
    handler = DateSizeRotatingFileHandler(
        log_root,
        filename_prefix,
        extension,
    )
    handler.setFormatter(formatter)
    return handler


def configure_logging(log_dir: Path | None = None) -> None:
    """Keep detailed logs in files while showing API access logs in the terminal."""
    target_dir = log_dir or DEFAULT_LOG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    standard_formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.close()
    root_logger.handlers.clear()
    root_logger.addHandler(
        _rotating_handler(
            target_dir,
            "backend",
            "log",
            standard_formatter,
        )
    )
    root_logger.setLevel(logging.INFO)

    terminal_handler = logging.StreamHandler()
    terminal_handler.setFormatter(standard_formatter)

    for logger_name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(logger_name)
        for handler in uvicorn_logger.handlers:
            handler.close()
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
        uvicorn_logger.setLevel(logging.INFO)

    for logger_name in ("uvicorn.access", "httpx"):
        api_logger = logging.getLogger(logger_name)
        for handler in api_logger.handlers:
            handler.close()
        api_logger.handlers.clear()
        api_logger.addHandler(terminal_handler)
        api_logger.propagate = True
        api_logger.setLevel(logging.INFO)

    model_logger = logging.getLogger(MODEL_CALL_LOGGER_NAME)
    for handler in model_logger.handlers:
        handler.close()
    model_logger.handlers.clear()
    model_logger.addHandler(
        _rotating_handler(
            target_dir,
            "model_calls",
            "jsonl",
            logging.Formatter("%(message)s"),
        )
    )
    model_logger.setLevel(logging.INFO)
    model_logger.propagate = False

    latency_logger = logging.getLogger(CHAT_LATENCY_LOGGER_NAME)
    for handler in latency_logger.handlers:
        handler.close()
    latency_logger.handlers.clear()
    latency_logger.addHandler(
        _rotating_handler(
            target_dir,
            "chat_latency",
            "jsonl",
            logging.Formatter("%(message)s"),
        )
    )
    latency_logger.setLevel(logging.INFO)
    latency_logger.propagate = False

    logging.captureWarnings(True)


def log_model_event(event: dict[str, Any]) -> None:
    logging.getLogger(MODEL_CALL_LOGGER_NAME).info(
        json.dumps(event, ensure_ascii=False, default=str)
    )


def log_chat_latency_event(event: dict[str, Any]) -> None:
    logging.getLogger(CHAT_LATENCY_LOGGER_NAME).info(
        json.dumps(event, ensure_ascii=False, default=str)
    )
