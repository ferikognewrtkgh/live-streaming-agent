from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

from loguru import logger


BACKEND_PERFORMANCE_METRICS = (
    "asr_seconds",
    "knowledge_seconds",
    "web_search_seconds",
    "llm_first_token_seconds",
    "llm_first_sentence_seconds",
    "llm_total_seconds",
    "tts_first_audio_seconds",
    "tts_total_seconds",
)
FRONTEND_PERFORMANCE_METRICS = (
    "user_speech_seconds",
    "speech_end_to_audio_start_seconds",
    "ai_playback_seconds",
    "speech_end_to_playback_complete_seconds",
)
ALL_PERFORMANCE_METRICS = (
    *BACKEND_PERFORMANCE_METRICS,
    *FRONTEND_PERFORMANCE_METRICS,
)
VOICE_INPUT_SOURCES = {"mic", "link_microphone"}


class PerformanceElasticsearchStore:
    """Incrementally upsert one performance document per conversation turn."""

    def __init__(self, client: Any, index_name: str) -> None:
        self.client = client
        self.index_name = index_name

    def ensure_index(self) -> None:
        metric_properties = {
            metric: {"type": "double"} for metric in ALL_PERFORMANCE_METRICS
        }
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.put_mapping(
                index=self.index_name,
                properties=metric_properties,
            )
            return
        self.client.indices.create(
            index=self.index_name,
            mappings={
                "properties": {
                    "turn_id": {"type": "keyword"},
                    "client_uid": {"type": "keyword"},
                    "input_source": {"type": "keyword"},
                    "backend_complete": {"type": "boolean"},
                    "playback_completed": {"type": "boolean"},
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"},
                    **metric_properties,
                }
            },
        )
        logger.info("Created Elasticsearch performance index: {}", self.index_name)

    def upsert(
        self,
        turn_id: str,
        metrics: dict[str, float],
        *,
        client_uid: str,
        input_source: str,
        backend_complete: bool | None,
        playback_completed: bool | None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        document: dict[str, Any] = {
            "turn_id": turn_id,
            "client_uid": client_uid,
            "input_source": input_source,
            "updated_at": now,
            **metrics,
        }
        if backend_complete is not None:
            document["backend_complete"] = bool(backend_complete)
        if playback_completed is not None:
            document["playback_completed"] = bool(playback_completed)
        self.client.update(
            index=self.index_name,
            id=turn_id,
            doc=document,
            upsert={"created_at": now, **document},
            refresh=False,
        )


_PERFORMANCE_STORE: PerformanceElasticsearchStore | None = None
_PERFORMANCE_STORE_LOCK = threading.Lock()


class TurnPerformanceRegistry:
    """Thread-safe in-memory timings for active and recently completed turns."""

    def __init__(self, max_turns: int = 256) -> None:
        self._max_turns = max(16, int(max_turns))
        self._records: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def ensure_turn(self, turn_id: str | None, **details: Any) -> None:
        if not turn_id:
            return
        with self._lock:
            record = self._record_locked(turn_id)
            for key, value in details.items():
                if value is not None:
                    record[key] = value

    def start_phase(self, turn_id: str | None, phase: str) -> None:
        if not turn_id:
            return
        now = time.perf_counter()
        with self._lock:
            record = self._record_locked(turn_id)
            record["phase_started"].setdefault(phase, now)

    def set_metric(
        self,
        turn_id: str | None,
        metric: str,
        seconds: float,
        *,
        overwrite: bool = False,
    ) -> None:
        if not turn_id:
            return
        value = max(0.0, float(seconds))
        with self._lock:
            metrics = self._record_locked(turn_id)["metrics"]
            if overwrite or metric not in metrics:
                metrics[metric] = value

    def mark_elapsed(
        self,
        turn_id: str | None,
        metric: str,
        phase: str,
        *,
        overwrite: bool = False,
    ) -> None:
        if not turn_id:
            return
        now = time.perf_counter()
        with self._lock:
            record = self._record_locked(turn_id)
            started_at = record["phase_started"].get(phase)
            if started_at is None:
                return
            metrics = record["metrics"]
            if overwrite or metric not in metrics:
                metrics[metric] = max(0.0, now - started_at)

    def snapshot(self, turn_id: str | None) -> dict[str, Any]:
        if not turn_id:
            return {}
        with self._lock:
            record = self._records.get(turn_id)
            if not record:
                return {}
            return {
                "turn_id": turn_id,
                "input_source": record.get("input_source"),
                "metrics": {
                    key: float(value)
                    for key, value in record["metrics"].items()
                    if key in BACKEND_PERFORMANCE_METRICS
                },
            }

    def _record_locked(self, turn_id: str) -> dict[str, Any]:
        record = self._records.get(turn_id)
        if record is None:
            record = {
                "metrics": {},
                "phase_started": {},
            }
            self._records[turn_id] = record
            while len(self._records) > self._max_turns:
                self._records.popitem(last=False)
        else:
            self._records.move_to_end(turn_id)
        return record


_REGISTRY = TurnPerformanceRegistry()


async def configure_performance_storage(
    config: Any,
    knowledge_runtime: Any | None = None,
) -> None:
    """Initialize performance persistence without making it a startup requirement."""
    global _PERFORMANCE_STORE

    if not bool(getattr(config, "performance_storage_enabled", False)):
        with _PERFORMANCE_STORE_LOCK:
            _PERFORMANCE_STORE = None
        logger.info("Elasticsearch performance storage is disabled.")
        return

    try:
        client = getattr(knowledge_runtime, "client", None)
        if client is None:
            from .knowledge_elasticsearch import create_es_connection

            client = await asyncio.to_thread(
                create_es_connection,
                es_url=getattr(config, "es_url", "auto"),
                api_key=getattr(config, "api_key", ""),
                username=getattr(config, "username", ""),
                password=getattr(config, "password", ""),
                request_timeout=float(getattr(config, "request_timeout", 30.0)),
                verify_certs=bool(getattr(config, "verify_certs", False)),
            )
        store = PerformanceElasticsearchStore(
            client,
            str(
                getattr(
                    config,
                    "performance_index",
                    "vtuber_performance_metrics",
                )
            ),
        )
        await asyncio.to_thread(store.ensure_index)
    except Exception:
        with _PERFORMANCE_STORE_LOCK:
            _PERFORMANCE_STORE = None
        logger.exception("Failed to initialize Elasticsearch performance storage.")
        return

    with _PERFORMANCE_STORE_LOCK:
        _PERFORMANCE_STORE = store
    logger.info(
        "Elasticsearch performance storage initialized: index={}",
        store.index_name,
    )


def _performance_store() -> PerformanceElasticsearchStore | None:
    with _PERFORMANCE_STORE_LOCK:
        return _PERFORMANCE_STORE


def _validated_metrics(metrics: Any) -> dict[str, float]:
    if not isinstance(metrics, dict):
        return {}
    result: dict[str, float] = {}
    for metric in ALL_PERFORMANCE_METRICS:
        if metric not in metrics:
            continue
        try:
            value = float(metrics[metric])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value >= 0:
            result[metric] = value
    return result


async def persist_performance_metrics(
    turn_id: str | None,
    metrics: Any,
    *,
    client_uid: str,
    input_source: str | None,
    backend_complete: bool | None = None,
    playback_completed: bool | None = None,
) -> None:
    store = _performance_store()
    source = str(input_source or "").strip()
    if store is None or not turn_id or source not in VOICE_INPUT_SOURCES:
        return
    validated_metrics = _validated_metrics(metrics)
    try:
        await asyncio.to_thread(
            store.upsert,
            str(turn_id),
            validated_metrics,
            client_uid=str(client_uid or "default"),
            input_source=source,
            backend_complete=backend_complete,
            playback_completed=playback_completed,
        )
        logger.debug(
            "Persisted performance metrics: turn_id={} index={} metrics={} "
            "backend_complete={} playback_completed={}",
            turn_id,
            store.index_name,
            sorted(validated_metrics),
            backend_complete,
            playback_completed,
        )
    except Exception:
        logger.exception(
            "Failed to persist performance metrics: turn_id={} index={}",
            turn_id,
            store.index_name,
        )


def ensure_performance_turn(turn_id: str | None, **details: Any) -> None:
    _REGISTRY.ensure_turn(turn_id, **details)


def start_performance_phase(turn_id: str | None, phase: str) -> None:
    _REGISTRY.start_phase(turn_id, phase)


def set_performance_metric(
    turn_id: str | None,
    metric: str,
    seconds: float,
    *,
    overwrite: bool = False,
) -> None:
    _REGISTRY.set_metric(turn_id, metric, seconds, overwrite=overwrite)


def mark_performance_elapsed(
    turn_id: str | None,
    metric: str,
    phase: str,
    *,
    overwrite: bool = False,
) -> None:
    _REGISTRY.mark_elapsed(
        turn_id,
        metric,
        phase,
        overwrite=overwrite,
    )


def build_performance_payload(turn_id: str | None) -> dict[str, Any] | None:
    snapshot = _REGISTRY.snapshot(turn_id)
    if not snapshot:
        return None
    return {
        "type": "performance-metrics",
        **snapshot,
    }


async def send_performance_stage(
    websocket_send: Any,
    turn_id: str | None,
    stage: str,
) -> None:
    if not turn_id or not stage:
        return
    try:
        await websocket_send(
            json.dumps(
                {
                    "type": "performance-stage",
                    "turn_id": turn_id,
                    "stage": stage,
                }
            )
        )
    except Exception as exc:
        logger.debug(
            "Failed to send performance stage: turn_id={} stage={} error={}",
            turn_id,
            stage,
            exc,
        )
