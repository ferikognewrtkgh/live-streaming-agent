import asyncio
import json
import random
import re
import shutil
import time
import uuid
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from ..agent.output_types import Actions, DisplayText
from ..emotion_tags import normalize_tts_emotion_tag
from ..live2d_model import Live2dModel
from ..resource_paths import SLEEP_VOICE_ROOT
from ..tts.tts_interface import TTSInterface, TTSStreamChunk
from ..utils.stream_audio import prepare_audio_payload, prepare_audio_payload_from_bytes
from ..utils.turn_trace import record_turn_event
from ..performance_metrics import (
    mark_performance_elapsed,
    send_performance_stage,
    start_performance_phase,
)
from .types import WebSocketSend

TTS_LOG_ROOT = Path("logs") / "tts"
TTS_LOG_COUNTER = count()
SLEEP_VOICE_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
DEFAULT_SLEEP_VOICE_TEXT = "\u6211\u6709\u70b9\u56f0\u4e86\uff0c\u6211\u53bb\u7761\u4e00\u4f1a"
_SLEEP_VOICE_QUEUE: list[Path] = []
_SLEEP_VOICE_LAST: Path | None = None
_SLEEP_VOICE_SIGNATURE: tuple[Path, ...] = ()


class TTSTaskManager:
    """Manage TTS tasks and ordered websocket audio delivery."""

    def __init__(
        self,
        turn_id: str | None = None,
        audio_not_before_monotonic: float | None = None,
    ) -> None:
        self.task_list: List[asyncio.Task] = []
        self._payload_queue: asyncio.Queue[tuple[int, Optional[Dict], bool]] = (
            asyncio.Queue()
        )
        self.turn_id = turn_id
        self._sender_task: Optional[asyncio.Task] = None
        self._sequence_counter = 0
        self._next_sequence_to_send = 0
        self._has_queued_spoken_tts = False
        self._tts_failure_fallback_attempted = False
        self._sleep_after_failure_sequences: set[int] = set()
        self._sleep_entered_after_tts_failure = False
        self._tts_request_lock = asyncio.Lock()
        self._audio_not_before_monotonic = audio_not_before_monotonic
        self._audio_start_delay_applied = False
        self._performance_stages_sent: set[str] = set()
        record_turn_event(
            self.turn_id,
            "tts_manager",
            "initialized",
            audio_not_before_monotonic=audio_not_before_monotonic,
        )

    @property
    def sleep_entered_after_tts_failure(self) -> bool:
        return self._sleep_entered_after_tts_failure

    async def send_performance_stage_once(
        self,
        websocket_send: WebSocketSend,
        stage: str,
    ) -> None:
        if stage in self._performance_stages_sent:
            return
        self._performance_stages_sent.add(stage)
        await send_performance_stage(websocket_send, self.turn_id, stage)

    async def speak(
        self,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        live2d_model: Live2dModel,
        tts_engine: TTSInterface,
        websocket_send: WebSocketSend,
    ) -> None:
        """Queue a TTS task while preserving sentence order."""
        if self._tts_failure_fallback_attempted:
            logger.debug(
                "Skipping TTS text because a TTS failure fallback was already triggered."
            )
            record_turn_event(
                self.turn_id,
                "tts_manager",
                "tts_skipped_after_failure",
                tts_text_len=len(tts_text),
                tts_text_preview=tts_text[:120],
            )
            return

        if len(re.sub(r'[\s.,!?"\']+', "", tts_text)) == 0:
            logger.debug("Empty TTS text, sending silent display payload")
            sequence_number = self._next_sequence()
            record_turn_event(
                self.turn_id,
                "tts_manager",
                "silent_tts_queued",
                sequence=sequence_number,
                display_text_len=len(display_text.text),
            )
            self._ensure_sender_task(websocket_send)
            await self._send_silent_payload(display_text, actions, sequence_number)
            return

        logger.debug("Queuing TTS task for: {!r} (by {})", tts_text, display_text.name)
        sequence_number = self._next_sequence()
        is_first = not self._has_queued_spoken_tts
        self._has_queued_spoken_tts = True
        record_turn_event(
            self.turn_id,
            "tts_manager",
            "tts_queued",
            sequence=sequence_number,
            is_first=is_first,
            tts_text_len=len(tts_text),
            tts_text_preview=tts_text[:120],
            display_text_len=len(display_text.text),
        )
        self._ensure_sender_task(websocket_send)

        task = asyncio.create_task(
            self._process_tts(
                tts_text=tts_text,
                display_text=display_text,
                actions=actions,
                live2d_model=live2d_model,
                tts_engine=tts_engine,
                sequence_number=sequence_number,
                is_first=is_first,
                websocket_send=websocket_send,
            )
        )
        task.add_done_callback(self._consume_task_result)
        self.task_list.append(task)

    def _next_sequence(self) -> int:
        sequence_number = self._sequence_counter
        self._sequence_counter += 1
        return sequence_number

    def _ensure_sender_task(self, websocket_send: WebSocketSend) -> None:
        if not self._sender_task or self._sender_task.done():
            self._sender_task = asyncio.create_task(
                self._process_payload_queue(websocket_send)
            )
            record_turn_event(
                self.turn_id,
                "tts_manager",
                "payload_sender_started",
            )

    async def _process_payload_queue(self, websocket_send: WebSocketSend) -> None:
        """Send queued audio payloads in sequence order."""
        buffered_payloads: Dict[int, list[Dict[str, Any]]] = {}
        finished_sequences: set[int] = set()

        while True:
            try:
                sequence_number, payload, is_sequence_done = (
                    await self._payload_queue.get()
                )
                try:
                    record_turn_event(
                        self.turn_id,
                        "tts_manager",
                        "payload_queue_item_received",
                        sequence=sequence_number,
                        has_payload=payload is not None,
                        is_sequence_done=is_sequence_done,
                    )
                    await self._handle_payload_queue_item(
                        websocket_send=websocket_send,
                        buffered_payloads=buffered_payloads,
                        finished_sequences=finished_sequences,
                        sequence_number=sequence_number,
                        payload=payload,
                        is_sequence_done=is_sequence_done,
                    )
                finally:
                    self._payload_queue.task_done()
            except asyncio.CancelledError:
                break

    async def _handle_payload_queue_item(
        self,
        websocket_send: WebSocketSend,
        buffered_payloads: Dict[int, list[Dict[str, Any]]],
        finished_sequences: set[int],
        sequence_number: int,
        payload: Optional[Dict],
        is_sequence_done: bool,
    ) -> None:
        if sequence_number == self._next_sequence_to_send:
            if payload:
                await self._wait_for_first_audio_slot(payload, sequence_number)
                await websocket_send(json.dumps(payload))
                record_turn_event(
                    self.turn_id,
                    "tts_manager",
                    "payload_sent",
                    sequence=sequence_number,
                    payload_type=payload.get("type"),
                    has_audio=bool(payload.get("audio")),
                    display_text=payload.get("display_text"),
                    actions=payload.get("actions"),
                    forwarded=payload.get("forwarded"),
                )
            if is_sequence_done:
                self._next_sequence_to_send += 1
                record_turn_event(
                    self.turn_id,
                    "tts_manager",
                    "sequence_completed",
                    sequence=sequence_number,
                )
                await self._maybe_enter_sleep_after_tts_failure(sequence_number)
                await self._flush_ready_sequences(
                    websocket_send, buffered_payloads, finished_sequences
                )
            return

        if payload:
            buffered_payloads.setdefault(sequence_number, []).append(payload)
            record_turn_event(
                self.turn_id,
                "tts_manager",
                "payload_buffered",
                sequence=sequence_number,
                next_sequence=self._next_sequence_to_send,
            )
        if is_sequence_done:
            finished_sequences.add(sequence_number)

    async def _flush_ready_sequences(
        self,
        websocket_send: WebSocketSend,
        buffered_payloads: Dict[int, list[Dict[str, Any]]],
        finished_sequences: set[int],
    ) -> None:
        while self._next_sequence_to_send in finished_sequences:
            sequence_number = self._next_sequence_to_send
            for payload in buffered_payloads.pop(sequence_number, []):
                await self._wait_for_first_audio_slot(payload, sequence_number)
                await websocket_send(json.dumps(payload))
                record_turn_event(
                    self.turn_id,
                    "tts_manager",
                    "buffered_payload_sent",
                    sequence=sequence_number,
                    payload_type=payload.get("type"),
                    has_audio=bool(payload.get("audio")),
                    display_text=payload.get("display_text"),
                    actions=payload.get("actions"),
                )
            finished_sequences.remove(sequence_number)
            self._next_sequence_to_send += 1
            record_turn_event(
                self.turn_id,
                "tts_manager",
                "buffered_sequence_completed",
                sequence=sequence_number,
            )
            await self._maybe_enter_sleep_after_tts_failure(sequence_number)

    async def _wait_for_first_audio_slot(
        self,
        payload: Dict[str, Any],
        sequence_number: int,
    ) -> None:
        if self._audio_start_delay_applied:
            return
        if not self._audio_not_before_monotonic:
            return
        if payload.get("type") != "audio" or not payload.get("audio"):
            return

        self._audio_start_delay_applied = True
        remaining = self._audio_not_before_monotonic - time.monotonic()
        if remaining <= 0:
            record_turn_event(
                self.turn_id,
                "tts_manager",
                "first_audio_slot_already_ready",
                sequence=sequence_number,
            )
            return

        logger.info(
            "Delaying first audio payload for barrage speech interval: {:.3f}s",
            remaining,
        )
        record_turn_event(
            self.turn_id,
            "tts_manager",
            "first_audio_delay_wait_started",
            sequence=sequence_number,
            wait_seconds=remaining,
        )
        await asyncio.sleep(remaining)
        record_turn_event(
            self.turn_id,
            "tts_manager",
            "first_audio_delay_wait_completed",
            sequence=sequence_number,
        )

    def _payload_log_view(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return a log-safe payload view without large audio arrays."""
        log_payload = dict(payload)
        if isinstance(log_payload.get("audio"), str):
            log_payload["audio"] = f"<base64 omitted, chars={len(log_payload['audio'])}>"
        if isinstance(log_payload.get("volumes"), list):
            log_payload["volumes"] = (
                f"<volumes omitted, count={len(log_payload['volumes'])}>"
            )
        return log_payload

    def _save_tts_segment_log(
        self,
        *,
        text: str,
        display_text: Optional[DisplayText],
        actions: Optional[Actions],
        metadata: Dict[str, Any],
        audio_format: str = "wav",
        audio_bytes: Optional[bytes] = None,
        audio_path: Optional[str] = None,
    ) -> Optional[Path]:
        """Save one returned TTS audio segment and its text metadata for debugging."""
        if not audio_bytes and not audio_path:
            return None

        try:
            now = datetime.now()
            day_dir = TTS_LOG_ROOT / now.strftime("%Y-%m-%d")
            day_dir.mkdir(parents=True, exist_ok=True)

            timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
            extension = self._resolve_audio_extension(audio_format, audio_path)
            stem = timestamp
            audio_log_path = day_dir / f"{stem}.{extension}"
            text_log_path = day_dir / f"{stem}.txt"
            while audio_log_path.exists() or text_log_path.exists():
                stem = f"{timestamp}_{next(TTS_LOG_COUNTER):04d}"
                audio_log_path = day_dir / f"{stem}.{extension}"
                text_log_path = day_dir / f"{stem}.txt"
            index_log_path = day_dir / "segments.txt"
            audio_text_index_path = day_dir / "audio_text.jsonl"

            if audio_bytes:
                audio_log_path.write_bytes(audio_bytes)
            elif audio_path:
                shutil.copyfile(audio_path, audio_log_path)
            audio_text_record = {
                "audio_file": audio_log_path.name,
                "text": text,
            }
            log_record = {
                "timestamp": timestamp,
                **audio_text_record,
                "display_text": self._display_text_log_data(display_text),
                "actions": self._actions_log_data(actions),
                "metadata": metadata,
            }
            text_log_path.write_text(
                json.dumps(log_record, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            with index_log_path.open("a", encoding="utf-8") as index_log:
                index_log.write(
                    json.dumps(log_record, ensure_ascii=False, default=str) + "\n"
                )
            with audio_text_index_path.open("a", encoding="utf-8") as audio_text_log:
                audio_text_log.write(
                    json.dumps(audio_text_record, ensure_ascii=False, default=str)
                    + "\n"
                )
            return audio_log_path
        except Exception as e:
            logger.warning("Failed to save TTS segment log: {!r}", e)
            return None

    def _resolve_audio_extension(
        self,
        audio_format: str,
        audio_path: Optional[str],
    ) -> str:
        if audio_path:
            suffix = Path(audio_path).suffix.lower().lstrip(".")
            if suffix:
                return suffix

        extension = re.sub(r"[^a-zA-Z0-9]+", "", audio_format).lower()
        return extension or "wav"

    def _display_text_log_data(
        self,
        display_text: Optional[DisplayText],
    ) -> Optional[Dict[str, Any]]:
        if not display_text:
            return None
        return display_text.to_dict()

    def _actions_log_data(self, actions: Optional[Actions]) -> Optional[Dict[str, Any]]:
        if not actions:
            return None
        actions_data = actions.to_dict()
        return actions_data or {"emotions": []}

    async def _send_silent_payload(
        self,
        display_text: DisplayText,
        actions: Optional[Actions],
        sequence_number: int,
    ) -> None:
        audio_payload = prepare_audio_payload(
            audio_path=None,
            display_text=display_text,
            actions=actions,
            turn_id=self.turn_id,
        )
        await self._payload_queue.put((sequence_number, audio_payload, True))

    async def _process_tts(
        self,
        tts_text: str,
        display_text: DisplayText,
        actions: Optional[Actions],
        live2d_model: Live2dModel,
        tts_engine: TTSInterface,
        sequence_number: int,
        is_first: bool,
        websocket_send: WebSocketSend,
    ) -> None:
        audio_file_path = None
        payload_sent = False
        sequence_done = False
        try:
            if self._tts_request_lock.locked():
                record_turn_event(
                    self.turn_id,
                    "tts_manager",
                    "tts_request_wait_started",
                    sequence=sequence_number,
                )

            async with self._tts_request_lock:
                if is_first:
                    start_performance_phase(self.turn_id, "tts")
                    await self.send_performance_stage_once(
                        websocket_send,
                        "tts-start",
                    )
                mood = self._resolve_mood(actions, live2d_model, tts_engine)
                record_turn_event(
                    self.turn_id,
                    "tts_manager",
                    "tts_processing_started",
                    sequence=sequence_number,
                    is_first=is_first,
                    tts_text_len=len(tts_text),
                    mood=mood,
                    supports_streaming=tts_engine.supports_streaming_audio(),
                )
                if tts_engine.supports_streaming_audio():
                    async for chunk in tts_engine.async_generate_audio_stream(
                        text=tts_text,
                        file_name_no_ext=self._make_cache_file_stem(),
                        mood=mood,
                        is_first=is_first,
                    ):
                        payload = self._prepare_stream_chunk_payload(
                            chunk=chunk,
                            display_text=display_text,
                            actions=actions,
                        )
                        if payload:
                            if payload.get("audio"):
                                mark_performance_elapsed(
                                    self.turn_id,
                                    "tts_first_audio_seconds",
                                    "tts",
                                )
                                await self.send_performance_stage_once(
                                    websocket_send,
                                    "tts-first-audio",
                                )
                            record_turn_event(
                                self.turn_id,
                                "tts_manager",
                                "stream_chunk_payload_prepared",
                                sequence=sequence_number,
                                request_id=chunk.metadata.get("request_id"),
                                chunk_index=chunk.metadata.get("index"),
                                is_final_chunk=chunk.metadata.get("is_final_chunk"),
                                chunk_text_len=len(chunk.text or ""),
                                audio_bytes=len(chunk.audio_bytes or b""),
                                has_audio_path=bool(chunk.audio_path),
                            )
                            logger.debug(
                                "received TTS payload: {}",
                                self._payload_log_view(payload),
                            )
                            await self._payload_queue.put(
                                (sequence_number, payload, False)
                            )
                            payload_sent = True
                        if chunk.audio_path:
                            tts_engine.remove_file(chunk.audio_path)

                    if not payload_sent:
                        raise RuntimeError("Streaming TTS returned no audio payload.")

                    await self._payload_queue.put((sequence_number, None, True))
                    sequence_done = True
                    record_turn_event(
                        self.turn_id,
                        "tts_manager",
                        "streaming_tts_sequence_done",
                        sequence=sequence_number,
                        payload_sent=payload_sent,
                    )
                else:
                    audio_file_path = await self._generate_audio(tts_engine, tts_text)
                    if not audio_file_path:
                        raise RuntimeError("TTS returned no audio file.")
                    self._save_tts_segment_log(
                        text=tts_text,
                        display_text=display_text,
                        actions=actions,
                        metadata={"source": "non_streaming_tts"},
                        audio_path=audio_file_path,
                    )
                    payload = prepare_audio_payload(
                        audio_path=audio_file_path,
                        display_text=display_text,
                        actions=actions,
                        turn_id=self.turn_id,
                        trim_long_tts_tail=True,
                        tts_text=tts_text,
                    )
                    if payload.get("audio"):
                        mark_performance_elapsed(
                            self.turn_id,
                            "tts_first_audio_seconds",
                            "tts",
                        )
                        await self.send_performance_stage_once(
                            websocket_send,
                            "tts-first-audio",
                        )
                    logger.debug(
                        "received TTS payload: {}", self._payload_log_view(payload)
                    )
                    await self._payload_queue.put((sequence_number, payload, True))
                    payload_sent = True
                    sequence_done = True
                    record_turn_event(
                        self.turn_id,
                        "tts_manager",
                        "non_streaming_tts_payload_queued",
                        sequence=sequence_number,
                        audio_file_path=audio_file_path,
                    )
                record_turn_event(
                    self.turn_id,
                    "tts_manager",
                    "tts_request_completed",
                    sequence=sequence_number,
                    payload_sent=payload_sent,
                    sequence_done=sequence_done,
                )

        except Exception as e:
            logger.error("Error preparing audio payload: {!r}", e)
            record_turn_event(
                self.turn_id,
                "tts_manager",
                "tts_processing_error",
                sequence=sequence_number,
                error=str(e),
                payload_sent=payload_sent,
            )
            if not sequence_done:
                payload = self._prepare_tts_failure_sleep_payload(
                    original_tts_text=tts_text,
                    original_display_text=display_text,
                    error=e,
                )
                if payload:
                    if payload.get("audio"):
                        mark_performance_elapsed(
                            self.turn_id,
                            "tts_first_audio_seconds",
                            "tts",
                        )
                        await self.send_performance_stage_once(
                            websocket_send,
                            "tts-first-audio",
                        )
                    payload_sent = True
                self._sleep_after_failure_sequences.add(sequence_number)
                await self._payload_queue.put((sequence_number, payload, True))
                sequence_done = True

        except asyncio.CancelledError:
            record_turn_event(
                self.turn_id,
                "tts_manager",
                "tts_processing_cancelled",
                sequence=sequence_number,
                payload_sent=payload_sent,
                sequence_done=sequence_done,
            )
            raise

        finally:
            if audio_file_path:
                tts_engine.remove_file(audio_file_path)
                logger.debug("Audio cache file cleaned.")
            record_turn_event(
                self.turn_id,
                "tts_manager",
                "tts_processing_finished",
                sequence=sequence_number,
                payload_sent=payload_sent,
                sequence_done=sequence_done,
            )

    def _resolve_mood(
        self,
        actions: Optional[Actions],
        live2d_model: Live2dModel,
        tts_engine: TTSInterface,
    ) -> str | None:
        """Resolve the semantic emotion tag for TTS."""
        raw_default_emotion = getattr(tts_engine, "default_emotion", None)
        default_emotion = normalize_tts_emotion_tag(raw_default_emotion) or "default"
        emotions = actions.emotions if actions and actions.emotions else []
        if emotions:
            return normalize_tts_emotion_tag(emotions[0]) or default_emotion

        expressions = actions.expressions if actions and actions.expressions else []
        if not expressions:
            return default_emotion

        expression = expressions[0]
        if isinstance(expression, str):
            mood = normalize_tts_emotion_tag(expression)
            if mood:
                return mood

        for emotion, mapped_expression in live2d_model.emo_map.items():
            if mapped_expression == expression or str(mapped_expression) == str(
                expression
            ):
                return normalize_tts_emotion_tag(emotion) or default_emotion

        return default_emotion

    def _prepare_stream_chunk_payload(
        self,
        chunk: TTSStreamChunk,
        display_text: DisplayText,
        actions: Optional[Actions],
    ) -> Optional[Dict]:
        """Convert one streaming TTS chunk into the existing websocket audio payload."""
        chunk_display_text = None
        if chunk.text:
            chunk_display_text = DisplayText(
                text=chunk.text,
                name=display_text.name,
                avatar=display_text.avatar,
            )

        if chunk.audio_bytes:
            audio_format = str(chunk.metadata.get("audio_format") or "wav")
            self._save_tts_segment_log(
                text=chunk.text or display_text.text,
                display_text=chunk_display_text or display_text,
                actions=actions,
                metadata=chunk.metadata,
                audio_format=audio_format,
                audio_bytes=chunk.audio_bytes,
            )
            return prepare_audio_payload_from_bytes(
                audio_bytes=chunk.audio_bytes,
                # 跟下面 audio_path 分支保持一致: chunk.text 为空时回落到外层
                # display_text. 否则流式 TTS (GPT-SoVITS) 只在首帧回 text,
                # 后续 chunk 的 display_text=None, 前端就拿不到任何文字.
                display_text=chunk_display_text or display_text,
                actions=actions,
                audio_format=audio_format,
                turn_id=self.turn_id,
                trim_long_tts_tail=True,
                tts_text=chunk.text or display_text.text,
            )
        if chunk.audio_path:
            self._save_tts_segment_log(
                text=chunk.text or display_text.text,
                display_text=chunk_display_text or display_text,
                actions=actions,
                metadata=chunk.metadata,
                audio_path=chunk.audio_path,
            )
            return prepare_audio_payload(
                audio_path=chunk.audio_path,
                display_text=chunk_display_text or display_text,
                actions=actions,
                turn_id=self.turn_id,
                trim_long_tts_tail=True,
                tts_text=chunk.text or display_text.text,
            )
        return None

    async def _generate_audio(self, tts_engine: TTSInterface, text: str) -> str:
        """Generate audio file from text."""
        logger.debug("Generating audio for {!r}...", text)
        return await tts_engine.async_generate_audio(
            text=text,
            file_name_no_ext=self._make_cache_file_stem(),
        )

    def _make_cache_file_stem(self) -> str:
        return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

    def _consume_task_result(self, task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("Unhandled TTS task error: {!r}", exc)

    async def wait_for_delivery(self) -> None:
        """Wait until all queued websocket audio payloads have been sent."""
        record_turn_event(
            self.turn_id,
            "tts_manager",
            "delivery_wait_started",
        )
        await self._payload_queue.join()
        record_turn_event(
            self.turn_id,
            "tts_manager",
            "delivery_wait_completed",
        )

    def clear(self) -> None:
        """Clear all pending tasks and reset state."""
        record_turn_event(
            self.turn_id,
            "tts_manager",
            "clear_called",
            task_count=len(self.task_list),
        )
        cancelled_count = 0
        for task in self.task_list:
            if not task.done():
                task.cancel()
                cancelled_count += 1
        record_turn_event(
            self.turn_id,
            "tts_manager",
            "tts_tasks_cancel_requested",
            cancelled_count=cancelled_count,
        )
        self.task_list.clear()
        if self._sender_task:
            self._sender_task.cancel()
        self._sequence_counter = 0
        self._next_sequence_to_send = 0
        self._has_queued_spoken_tts = False
        self._tts_failure_fallback_attempted = False
        self._sleep_after_failure_sequences.clear()
        self._sleep_entered_after_tts_failure = False
        self._payload_queue = asyncio.Queue()

    def _select_sleep_voice(self) -> tuple[Path, str] | None:
        global _SLEEP_VOICE_LAST, _SLEEP_VOICE_QUEUE, _SLEEP_VOICE_SIGNATURE

        if not SLEEP_VOICE_ROOT.exists():
            logger.error(
                "TTS failure sleep voice directory does not exist: {}",
                SLEEP_VOICE_ROOT,
            )
            return None

        captions: dict[str, str] = {}
        for caption_path in sorted(SLEEP_VOICE_ROOT.glob("*.txt")):
            try:
                for line in caption_path.read_text(encoding="utf-8-sig").splitlines():
                    filename, separator, text = line.partition("|")
                    if separator and filename.strip() and text.strip():
                        captions[filename.strip()] = text.strip()
            except Exception as e:
                logger.warning(
                    "Failed to read TTS failure sleep voice captions from {}: {!r}",
                    caption_path,
                    e,
                )

        audio_paths = sorted(
            path
            for path in SLEEP_VOICE_ROOT.iterdir()
            if path.is_file() and path.suffix.lower() in SLEEP_VOICE_AUDIO_EXTENSIONS
        )
        if not audio_paths:
            logger.error(
                "No TTS failure sleep voice audio found in {}", SLEEP_VOICE_ROOT
            )
            return None

        signature = tuple(audio_paths)
        if signature != _SLEEP_VOICE_SIGNATURE:
            _SLEEP_VOICE_SIGNATURE = signature
            _SLEEP_VOICE_QUEUE = []
            _SLEEP_VOICE_LAST = None

        if not _SLEEP_VOICE_QUEUE:
            _SLEEP_VOICE_QUEUE = list(audio_paths)
            random.shuffle(_SLEEP_VOICE_QUEUE)
            if (
                len(_SLEEP_VOICE_QUEUE) > 1
                and _SLEEP_VOICE_LAST is not None
                and _SLEEP_VOICE_QUEUE[0] == _SLEEP_VOICE_LAST
            ):
                _SLEEP_VOICE_QUEUE.append(_SLEEP_VOICE_QUEUE.pop(0))

        audio_path = _SLEEP_VOICE_QUEUE.pop(0)
        _SLEEP_VOICE_LAST = audio_path
        caption = captions.get(audio_path.name) or captions.get(audio_path.stem)
        return audio_path, caption or DEFAULT_SLEEP_VOICE_TEXT

    def _prepare_tts_failure_sleep_payload(
        self,
        *,
        original_tts_text: str,
        original_display_text: DisplayText,
        error: Exception,
    ) -> Optional[Dict]:
        if self._tts_failure_fallback_attempted:
            return None

        self._tts_failure_fallback_attempted = True
        selection = self._select_sleep_voice()
        if not selection:
            record_turn_event(
                self.turn_id,
                "tts_manager",
                "tts_failure_sleep_voice_missing",
                error=str(error),
                original_tts_text_len=len(original_tts_text),
            )
            return None

        audio_path, caption = selection
        display_text = DisplayText(
            text=caption,
            name=original_display_text.name,
            avatar=original_display_text.avatar,
        )
        actions = Actions()
        try:
            self._save_tts_segment_log(
                text=caption,
                display_text=display_text,
                actions=actions,
                metadata={
                    "source": "tts_failure_sleep_voice",
                    "sleep_voice_file": audio_path.name,
                    "original_tts_text": original_tts_text,
                    "error": str(error),
                },
                audio_path=str(audio_path),
            )
            payload = prepare_audio_payload(
                audio_path=str(audio_path),
                display_text=display_text,
                actions=actions,
                turn_id=self.turn_id,
            )
            payload["source"] = "tts_failure_sleep_voice"
            payload["sleep_voice_file"] = audio_path.name
        except Exception as fallback_error:
            logger.error(
                "Failed to prepare TTS failure sleep voice payload from {}: {!r}",
                audio_path,
                fallback_error,
            )
            record_turn_event(
                self.turn_id,
                "tts_manager",
                "tts_failure_sleep_payload_error",
                audio_path=str(audio_path),
                error=str(fallback_error),
            )
            return None

        logger.debug(
            "received TTS failure sleep payload: {}", self._payload_log_view(payload)
        )
        record_turn_event(
            self.turn_id,
            "tts_manager",
            "tts_failure_sleep_payload_prepared",
            audio_path=str(audio_path),
            caption=caption,
            error=str(error),
        )
        return payload

    async def _maybe_enter_sleep_after_tts_failure(self, sequence_number: int) -> None:
        if sequence_number not in self._sleep_after_failure_sequences:
            return

        self._sleep_after_failure_sequences.discard(sequence_number)
        if self._sleep_entered_after_tts_failure:
            return

        self._sleep_entered_after_tts_failure = True
        try:
            from ..vtuber_state_machine import get_vtuber_state_machine

            sm = get_vtuber_state_machine()
            if sm is None:
                record_turn_event(
                    self.turn_id,
                    "tts_manager",
                    "tts_failure_sleep_skipped",
                    reason="state_machine_missing",
                    sequence=sequence_number,
                )
                return

            state = await sm.enter_sleep(
                reason="tts-failure",
                interrupt_current=False,
            )
            record_turn_event(
                self.turn_id,
                "tts_manager",
                "tts_failure_sleep_entered",
                sequence=sequence_number,
                vtuber_state=state,
            )
        except Exception as e:
            logger.warning("Failed to enter sleep after TTS failure: {!r}", e)
            record_turn_event(
                self.turn_id,
                "tts_manager",
                "tts_failure_sleep_error",
                sequence=sequence_number,
                error=str(e),
            )
