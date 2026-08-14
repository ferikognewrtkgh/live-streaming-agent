import json
import logging
import sys
from datetime import datetime

from backend.app.logging_setup import (
    CHAT_LATENCY_LOGGER_NAME,
    DEFAULT_LOG_DIR,
    MODEL_CALL_LOGGER_NAME,
    DateSizeRotatingFileHandler,
    configure_logging,
    log_chat_latency_event,
    log_model_event,
)


def test_logging_keeps_model_context_off_console(tmp_path) -> None:
    try:
        configure_logging(tmp_path)
        log_model_event(
            {
                "event": "model_call_started",
                "model": "test-model",
                "request": {"messages": [{"role": "user", "content": "你好"}]},
            }
        )
        log_chat_latency_event(
            {
                "event": "frontend_first_paint",
                "trace_id": "trace-12345678",
                "click_to_first_paint_ms": 812.34,
            }
        )

        root_handlers = logging.getLogger().handlers
        model_handlers = logging.getLogger(MODEL_CALL_LOGGER_NAME).handlers
        latency_handlers = logging.getLogger(CHAT_LATENCY_LOGGER_NAME).handlers
        all_handlers = [*root_handlers, *model_handlers, *latency_handlers]
        assert all(
            getattr(handler, "stream", None) not in {sys.stdout, sys.stderr}
            for handler in all_handlers
        )

        date_dir = tmp_path / datetime.now().astimezone().strftime("%Y-%m-%d")
        model_log = next(date_dir.glob("model_calls_*.jsonl"))
        records = [
            json.loads(line)
            for line in model_log.read_text(encoding="utf-8").splitlines()
        ]
        assert records[-1]["model"] == "test-model"
        assert records[-1]["request"]["messages"][0]["content"] == "你好"
        latency_log = next(date_dir.glob("chat_latency_*.jsonl"))
        latency_records = [
            json.loads(line)
            for line in latency_log.read_text(encoding="utf-8").splitlines()
        ]
        assert latency_records[-1]["trace_id"] == "trace-12345678"
        assert latency_records[-1]["click_to_first_paint_ms"] == 812.34
    finally:
        configure_logging(DEFAULT_LOG_DIR)


def test_log_files_rotate_by_size_with_timestamped_names(tmp_path) -> None:
    logger = logging.getLogger("test.date_size_rotation")
    handler = DateSizeRotatingFileHandler(
        tmp_path,
        "backend",
        "log",
        max_bytes=20,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    try:
        logger.info("第一条较长的日志")
        logger.info("第二条较长的日志")
    finally:
        handler.close()
        logger.handlers.clear()

    date_dir = tmp_path / datetime.now().astimezone().strftime("%Y-%m-%d")
    files = sorted(date_dir.glob("backend_*.log"))
    assert len(files) == 2
    assert all(file.stem.startswith("backend_20") for file in files)


def test_api_access_loggers_have_terminal_handler(tmp_path) -> None:
    try:
        configure_logging(tmp_path)
        for logger_name in ("uvicorn.access", "httpx"):
            handlers = logging.getLogger(logger_name).handlers
            assert any(
                getattr(handler, "stream", None) in {sys.stdout, sys.stderr}
                for handler in handlers
            )
    finally:
        configure_logging(DEFAULT_LOG_DIR)
