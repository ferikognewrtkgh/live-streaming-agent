import asyncio
import re
import queue
import threading
from dataclasses import dataclass, field
from typing import Optional, Union, Any, List, Dict, AsyncIterator
import numpy as np
import json
import base64
import binascii
from loguru import logger

from ..agent.events import (
    LLM_FIRST_TOKEN_EVENT_TYPE,
    LLM_FINAL_ERROR_FALLBACK_EVENT_TYPE,
    WEB_SEARCH_START_EVENT_TYPE,
    WEB_SEARCH_TIMING_EVENT_TYPE,
)
from ..message_handler import message_handler
from .types import WebSocketSend, BroadcastContext
from .tts_manager import TTSTaskManager
from ..agent.output_types import (
    Actions,
    DisplayText,
    SentenceOutput,
    AudioOutput,
)
from ..agent.input_types import BatchInput, TextData, ImageData, TextSource, ImageSource
from ..asr.asr_interface import ASRInterface
from ..blocked_words_loader import sanitize_blocked_words_text_with_matches
from ..live2d_model import Live2dModel
from ..painting import extract_paint_commands, get_paint_manager
from ..tts.tts_interface import TTSInterface
from ..utils.sentence_divider import paint_command_spans
from ..utils.prompt_trigger_registry import get_prompt_trigger_registry
from ..utils.stream_audio import prepare_audio_payload
from ..utils.turn_trace import record_turn_event
from ..performance_metrics import (
    build_performance_payload,
    mark_performance_elapsed,
    persist_performance_metrics,
    set_performance_metric,
    start_performance_phase,
)

LLM_DELAY_TRIGGER_NAME = "delay"
LLM_DELAY_TRIGGER_SECONDS = 10
MAX_VISION_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_VISION_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _merge_blocked_word_matches(*match_groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in match_groups:
        for word in group:
            key = word.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(word)
    return merged


def format_human_input_for_llm(input_text: str, human_name: str | None) -> str:
    text = str(input_text or "").strip()
    name = str(human_name or "").strip()
    if not text or not name:
        return text

    knowledge_marker = "[知识库检索结果]"
    user_question_marker = "[用户当前提问]"
    if text.startswith(knowledge_marker) and user_question_marker in text:
        knowledge_text, question_text = text.rsplit(user_question_marker, 1)
        question_text = question_text.strip()
        if question_text:
            formatted_question = format_human_input_for_llm(question_text, name)
            return (
                f"{knowledge_text.rstrip()}\n\n"
                f"{user_question_marker}\n{formatted_question}"
            )

    old_prefix = f"{name}：\u201c"
    prefix = f"{name}说：\u201c"
    if text.endswith("\u201d") and (
        text.startswith(old_prefix) or text.startswith(prefix)
    ):
        return text
    return f"{prefix}{text}\u201d"


# Convert class methods to standalone functions
def create_batch_input(
    input_text: str,
    images: Optional[List[Dict[str, Any]]],
    from_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    include_human_name_prefix: bool = True,
) -> BatchInput:
    """Create batch input for agent processing"""
    validated_images: List[ImageData] = []
    for image in images or []:
        if not isinstance(image, dict):
            logger.warning("Skipping non-dictionary image payload.")
            continue

        source = image.get("source")
        data = image.get("data")
        mime_type = str(image.get("mime_type") or "").lower()
        if source not in {item.value for item in ImageSource}:
            logger.warning("Skipping image with unsupported source: {}", source)
            continue
        if mime_type not in ALLOWED_VISION_MIME_TYPES:
            logger.warning("Skipping image with unsupported MIME type: {}", mime_type)
            continue

        expected_prefix = f"data:{mime_type};base64,"
        if not isinstance(data, str) or not data.startswith(expected_prefix):
            logger.warning("Skipping image with invalid data URL.")
            continue

        try:
            decoded_size = len(
                base64.b64decode(data[len(expected_prefix) :], validate=True)
            )
        except (binascii.Error, ValueError, TypeError):
            logger.warning("Skipping image with invalid base64 content.")
            continue

        if decoded_size > MAX_VISION_IMAGE_BYTES:
            logger.warning(
                "Skipping oversized image payload: {} bytes (limit {}).",
                decoded_size,
                MAX_VISION_IMAGE_BYTES,
            )
            continue

        validated_images.append(
            ImageData(
                source=ImageSource(source),
                data=data,
                mime_type=mime_type,
            )
        )

    llm_user_text = (
        format_human_input_for_llm(input_text, from_name)
        if include_human_name_prefix
        else str(input_text or "").strip()
    )
    metadata = dict(metadata) if isinstance(metadata, dict) else None
    llm_input_text = llm_user_text
    if metadata and metadata.get("story_guidance"):
        llm_input_text = f"{llm_user_text}\n\n{metadata['story_guidance']}"

    if metadata and metadata.get("game_vision_context"):
        llm_input_text = (
            f"{llm_input_text}\n\n"
            f"[\u6e38\u620f\u753b\u9762\u8bc6\u522b]\n{metadata['game_vision_context']}"
        )

    if metadata and metadata.get("game_vision_reply_mode") == "vision_model":
        llm_input_text = (
            f"{llm_input_text}\n\n"
            "[\u8bf7\u7ed3\u5408\u968f\u9644\u7684\u6e38\u620f\u753b\u9762\u622a\u56fe"
            "\u76f4\u63a5\u56de\u7b54\u4e3b\u64ad\u7684\u95ee\u9898\u6216\u53d1\u8a00]"
        )

    if metadata and metadata.get("vision_context_reused"):
        llm_input_text = (
            f"{llm_input_text}\n\n"
            "[请继续参考此前上传的图片；后端已将同一张图片随本轮再次附带，"
            "请结合当前问题直接回复。]"
        )

    if metadata and llm_input_text != llm_user_text:
        metadata = {
            **metadata,
            "memory_input_text": llm_user_text,
        }

    return BatchInput(
        texts=[
            TextData(source=TextSource.INPUT, content=llm_input_text, from_name=from_name)
        ],
        images=validated_images or None,
        metadata=metadata,
    )


async def process_agent_output(
    output: Union[AudioOutput, SentenceOutput],
    character_config: Any,
    live2d_model: Live2dModel,
    tts_engine: TTSInterface,
    websocket_send: WebSocketSend,
    tts_manager: TTSTaskManager,
    translate_engine: Optional[Any] = None,
    turn_id: str | None = None,
) -> str:
    """Process agent output with character information and optional translation"""
    record_turn_event(
        turn_id,
        "conversation_utils",
        "process_agent_output_entered",
        output_type=type(output).__name__,
    )
    output.display_text.name = character_config.character_name
    output.display_text.avatar = character_config.avatar

    full_response = ""
    try:
        if isinstance(output, SentenceOutput):
            full_response = await handle_sentence_output(
                output,
                live2d_model,
                tts_engine,
                websocket_send,
                tts_manager,
                translate_engine,
                turn_id=turn_id,
            )
        elif isinstance(output, AudioOutput):
            full_response = await handle_audio_output(
                output,
                websocket_send,
                turn_id=turn_id,
            )
        else:
            logger.warning(f"Unknown output type: {type(output)}")
    except Exception as e:
        logger.error(f"Error processing agent output: {e}")
        record_turn_event(
            turn_id,
            "conversation_utils",
            "process_agent_output_error",
            output_type=type(output).__name__,
            error=str(e),
        )
        await websocket_send(
            json.dumps(
                {"type": "error", "message": f"Error processing response: {str(e)}"}
            )
        )

    record_turn_event(
        turn_id,
        "conversation_utils",
        "process_agent_output_completed",
        output_type=type(output).__name__,
        response_len=len(full_response),
    )
    return full_response


async def handle_sentence_output(
    output: SentenceOutput,
    live2d_model: Live2dModel,
    tts_engine: TTSInterface,
    websocket_send: WebSocketSend,
    tts_manager: TTSTaskManager,
    translate_engine: Optional[Any] = None,
    turn_id: str | None = None,
) -> str:
    """Handle sentence output type with optional translation support"""
    full_response = ""
    async for display_text, tts_text, actions in output:
        logger.debug(f"🏃 Processing output: '''{tts_text}'''...")

        memory_text = display_text.text
        paint_extraction = extract_paint_commands(display_text.text)
        if paint_extraction.prompts:
            display_text.text = paint_extraction.text
            tts_text = extract_paint_commands(tts_text).text
            paint_manager = get_paint_manager()
            for prompt in paint_extraction.prompts:
                started = await paint_manager.request_paint(
                    prompt=prompt,
                    websocket_send=websocket_send,
                    turn_id=turn_id,
                )
                record_turn_event(
                    turn_id,
                    "conversation_utils",
                    "paint_command_detected",
                    prompt_len=len(prompt),
                    prompt_preview=prompt[:120],
                    started=started,
                    busy=paint_manager.busy,
                )

        if translate_engine:
            if len(re.sub(r'[\s.,!?，。！？\'"』」）】\s]+', "", tts_text)):
                tts_text = translate_engine.translate(tts_text)
            logger.info(f"🏃 Text after translation: '''{tts_text}'''...")
        else:
            logger.debug("🚫 No translation engine available. Skipping translation.")

        sanitized_display_text, display_blocked_words = (
            sanitize_blocked_words_text_with_matches(display_text.text)
        )
        sanitized_tts_text, tts_blocked_words = (
            sanitize_blocked_words_text_with_matches(tts_text)
        )
        sanitized_memory_text, memory_blocked_words = (
            sanitize_blocked_words_text_with_matches(
                memory_text,
                ignored_spans=paint_command_spans(memory_text),
            )
        )
        if (
            sanitized_display_text != display_text.text
            or sanitized_tts_text != tts_text
            or sanitized_memory_text != memory_text
        ):
            blocked_words = _merge_blocked_word_matches(
                display_blocked_words,
                tts_blocked_words,
                memory_blocked_words,
            )
            logger.info(
                "Blocked word sanitized in AI sentence output: "
                f"words={blocked_words!r}"
            )
            record_turn_event(
                turn_id,
                "conversation_utils",
                "blocked_words_sanitized",
                output_type="SentenceOutput",
                blocked_words=blocked_words,
                display_text_blocked_words=display_blocked_words,
                tts_text_blocked_words=tts_blocked_words,
                memory_text_blocked_words=memory_blocked_words,
                display_text_changed=sanitized_display_text != display_text.text,
                tts_text_changed=sanitized_tts_text != tts_text,
                memory_text_changed=sanitized_memory_text != memory_text,
            )
        display_text.text = sanitized_display_text
        tts_text = sanitized_tts_text
        memory_text = sanitized_memory_text

        if len(re.sub(r'[\s.,!?，。！？\'"』」）】\s]+', "", display_text.text)) == 0 and len(
            re.sub(r'[\s.,!?，。！？\'"』」）】\s]+', "", tts_text)
        ) == 0:
            if memory_text:
                full_response += memory_text
            record_turn_event(
                turn_id,
                "conversation_utils",
                "empty_sentence_skipped_after_paint_command",
                memory_text_len=len(memory_text),
                memory_text_preview=memory_text[:120],
            )
            continue

        full_response += memory_text
        record_turn_event(
            turn_id,
            "conversation_utils",
            "sentence_output_ready_for_tts",
            display_text_len=len(display_text.text),
            tts_text_len=len(tts_text),
            memory_text_len=len(memory_text),
            display_text_preview=display_text.text[:120],
            tts_text_preview=tts_text[:120],
            memory_text_preview=memory_text[:120],
            has_actions=bool(actions and actions.to_dict()),
        )
        await tts_manager.speak(
            tts_text=tts_text,
            display_text=display_text,
            actions=actions,
            live2d_model=live2d_model,
            tts_engine=tts_engine,
            websocket_send=websocket_send,
        )
    return full_response


async def handle_audio_output(
    output: AudioOutput,
    websocket_send: WebSocketSend,
    turn_id: str | None = None,
) -> str:
    """Process and send AudioOutput directly to the client"""
    full_response = ""
    async for audio_path, display_text, transcript, actions in output:
        sanitized_transcript, transcript_blocked_words = (
            sanitize_blocked_words_text_with_matches(transcript)
        )
        sanitized_display_text, display_blocked_words = (
            sanitize_blocked_words_text_with_matches(display_text.text)
        )
        if (
            sanitized_transcript != transcript
            or sanitized_display_text != display_text.text
        ):
            blocked_words = _merge_blocked_word_matches(
                transcript_blocked_words,
                display_blocked_words,
            )
            logger.info(
                "Blocked word sanitized in AI audio output: "
                f"words={blocked_words!r}"
            )
            record_turn_event(
                turn_id,
                "conversation_utils",
                "blocked_words_sanitized",
                output_type="AudioOutput",
                blocked_words=blocked_words,
                transcript_blocked_words=transcript_blocked_words,
                display_text_blocked_words=display_blocked_words,
                transcript_changed=sanitized_transcript != transcript,
                display_text_changed=sanitized_display_text != display_text.text,
            )
        transcript = sanitized_transcript
        display_text.text = sanitized_display_text
        full_response += transcript
        record_turn_event(
            turn_id,
            "conversation_utils",
            "audio_output_ready",
            transcript_len=len(transcript),
            has_audio_path=bool(audio_path),
        )
        audio_payload = prepare_audio_payload(
            audio_path=audio_path,
            display_text=display_text,
            actions=actions.to_dict() if actions else None,
            turn_id=turn_id,
        )
        await websocket_send(json.dumps(audio_payload))
        record_turn_event(
            turn_id,
            "conversation_utils",
            "audio_output_sent",
            transcript_len=len(transcript),
        )
    return full_response


def is_llm_first_token_event(output_item: Any) -> bool:
    return (
        isinstance(output_item, dict)
        and output_item.get("type") == LLM_FIRST_TOKEN_EVENT_TYPE
    )


def is_llm_final_error_fallback_event(output_item: Any) -> bool:
    return (
        isinstance(output_item, dict)
        and output_item.get("type") == LLM_FINAL_ERROR_FALLBACK_EVENT_TYPE
    )


def is_web_search_start_event(output_item: Any) -> bool:
    return (
        isinstance(output_item, dict)
        and output_item.get("type") == WEB_SEARCH_START_EVENT_TYPE
    )


def is_web_search_timing_event(output_item: Any) -> bool:
    return (
        isinstance(output_item, dict)
        and output_item.get("type") == WEB_SEARCH_TIMING_EVENT_TYPE
    )


async def speak_trigger_prompt(
    *,
    trigger_name: str,
    character_config: Any,
    live2d_model: Live2dModel,
    tts_engine: TTSInterface,
    websocket_send: WebSocketSend,
    tts_manager: TTSTaskManager,
    turn_id: str | None = None,
) -> bool:
    prompt = get_prompt_trigger_registry().get_next(trigger_name)
    if not prompt:
        record_turn_event(
            turn_id,
            "conversation_utils",
            "trigger_prompt_missing",
            trigger_name=trigger_name,
        )
        return False

    display_text = DisplayText(
        text=prompt.text,
        name=character_config.character_name,
        avatar=character_config.avatar,
    )
    emotion = str(prompt.expression or "").strip()
    actions = Actions(emotions=[emotion]) if emotion else Actions()
    record_turn_event(
        turn_id,
        "conversation_utils",
        "trigger_prompt_tts_queued",
        trigger_name=trigger_name,
        text_len=len(prompt.text),
        text_preview=prompt.text[:120],
        expression=prompt.expression,
    )
    await tts_manager.speak(
        tts_text=prompt.text,
        display_text=display_text,
        actions=actions,
        live2d_model=live2d_model,
        tts_engine=tts_engine,
        websocket_send=websocket_send,
    )
    return True


async def speak_trigger_prompt_turn(
    *,
    trigger_name: str,
    context: Any,
    websocket_send: WebSocketSend,
    client_uid: str,
    turn_id: str | None = None,
) -> bool:
    """Speak one trigger prompt as a complete websocket conversation turn."""
    tts_manager = TTSTaskManager(turn_id=turn_id)
    spoken = False
    try:
        await send_conversation_start_signals(websocket_send, turn_id=turn_id)
        record_turn_event(
            turn_id,
            "conversation_utils",
            "trigger_prompt_turn_started",
            client_uid=client_uid,
            trigger_name=trigger_name,
        )
        spoken = await speak_trigger_prompt(
            trigger_name=trigger_name,
            character_config=context.character_config,
            live2d_model=context.live2d_model,
            tts_engine=context.tts_engine,
            websocket_send=websocket_send,
            tts_manager=tts_manager,
            turn_id=turn_id,
        )
        await finalize_conversation_turn(
            tts_manager=tts_manager,
            websocket_send=websocket_send,
            client_uid=client_uid,
            turn_id=turn_id,
        )
        record_turn_event(
            turn_id,
            "conversation_utils",
            "trigger_prompt_turn_completed",
            client_uid=client_uid,
            trigger_name=trigger_name,
            spoken=spoken,
        )
        return spoken
    finally:
        cleanup_conversation(tts_manager, session_emoji=trigger_name)


async def speak_delay_trigger_if_llm_is_slow(
    *,
    first_token_event: asyncio.Event,
    character_config: Any,
    live2d_model: Live2dModel,
    tts_engine: TTSInterface,
    websocket_send: WebSocketSend,
    tts_manager: TTSTaskManager,
    turn_id: str | None = None,
    delay_seconds: float = LLM_DELAY_TRIGGER_SECONDS,
) -> None:
    try:
        await asyncio.wait_for(first_token_event.wait(), timeout=delay_seconds)
        record_turn_event(
            turn_id,
            "conversation_utils",
            "llm_delay_trigger_cancelled",
            delay_seconds=delay_seconds,
        )
        return
    except asyncio.TimeoutError:
        if first_token_event.is_set():
            return

    record_turn_event(
        turn_id,
        "conversation_utils",
        "llm_delay_trigger_fired",
        delay_seconds=delay_seconds,
    )
    try:
        await speak_trigger_prompt(
            trigger_name=LLM_DELAY_TRIGGER_NAME,
            character_config=character_config,
            live2d_model=live2d_model,
            tts_engine=tts_engine,
            websocket_send=websocket_send,
            tts_manager=tts_manager,
            turn_id=turn_id,
        )
    except Exception as exc:
        logger.exception("Failed to queue LLM delay trigger prompt.")
        record_turn_event(
            turn_id,
            "conversation_utils",
            "llm_delay_trigger_error",
            delay_seconds=delay_seconds,
            error=str(exc),
        )


async def send_conversation_start_signals(
    websocket_send: WebSocketSend,
    turn_id: str | None = None,
) -> None:
    """Send initial conversation signals"""
    start_payload = {
        "type": "control",
        "text": "conversation-chain-start",
    }
    thinking_payload = {"type": "full-text", "text": "Thinking..."}
    if turn_id:
        start_payload["turn_id"] = turn_id
        thinking_payload["turn_id"] = turn_id
    await websocket_send(json.dumps(start_payload))
    record_turn_event(
        turn_id,
        "conversation_utils",
        "conversation_start_signal_sent",
        payload_type=start_payload["type"],
        control_text=start_payload["text"],
    )
    await websocket_send(json.dumps(thinking_payload))
    record_turn_event(
        turn_id,
        "conversation_utils",
        "thinking_signal_sent",
        payload_type=thinking_payload["type"],
    )


async def process_user_input(
    user_input: Union[str, np.ndarray],
    asr_engine: ASRInterface,
    websocket_send: WebSocketSend,
    turn_id: str | None = None,
) -> str:
    """Process user input, converting audio to text if needed"""
    if isinstance(user_input, np.ndarray):
        logger.info("Transcribing audio input...")
        start_performance_phase(turn_id, "asr")
        record_turn_event(
            turn_id,
            "conversation_utils",
            "asr_started",
            audio_samples=len(user_input),
        )
        input_text = await asr_engine.async_transcribe_np(user_input)
        mark_performance_elapsed(turn_id, "asr_seconds", "asr")
        record_turn_event(
            turn_id,
            "conversation_utils",
            "asr_completed",
            audio_samples=len(user_input),
            input_text_len=len(input_text),
            input_text_preview=input_text[:120],
        )
        payload = {"type": "user-input-transcription", "text": input_text}
        if turn_id:
            payload["turn_id"] = turn_id
        await websocket_send(json.dumps(payload))
        record_turn_event(
            turn_id,
            "conversation_utils",
            "asr_transcription_sent",
            input_text_len=len(input_text),
        )
        return input_text
    record_turn_event(
        turn_id,
        "conversation_utils",
        "text_input_used",
        input_text_len=len(user_input),
        input_text_preview=user_input[:120],
    )
    return user_input


async def finalize_conversation_turn(
    tts_manager: TTSTaskManager,
    websocket_send: WebSocketSend,
    client_uid: str,
    broadcast_ctx: Optional[BroadcastContext] = None,
    turn_id: str | None = None,
) -> None:
    """Finalize a conversation turn"""
    record_turn_event(
        turn_id,
        "conversation_utils",
        "finalize_entered",
        client_uid=client_uid,
        tts_task_count=len(tts_manager.task_list),
    )
    performance_payload_sent = False
    if tts_manager.task_list:
        record_turn_event(
            turn_id,
            "conversation_utils",
            "tts_tasks_wait_started",
            client_uid=client_uid,
            tts_task_count=len(tts_manager.task_list),
        )
        await asyncio.gather(*tts_manager.task_list)
        mark_performance_elapsed(turn_id, "tts_total_seconds", "tts")
        await tts_manager.send_performance_stage_once(
            websocket_send,
            "tts-complete",
        )
        record_turn_event(
            turn_id,
            "conversation_utils",
            "tts_tasks_wait_completed",
            client_uid=client_uid,
        )
        await tts_manager.wait_for_delivery()
        record_turn_event(
            turn_id,
            "conversation_utils",
            "tts_delivery_wait_completed",
            client_uid=client_uid,
        )
        performance_payload = build_performance_payload(turn_id)
        if performance_payload:
            await persist_performance_metrics(
                turn_id,
                performance_payload.get("metrics"),
                client_uid=client_uid,
                input_source=performance_payload.get("input_source"),
                backend_complete=True,
            )
            await websocket_send(json.dumps(performance_payload))
            performance_payload_sent = True
            record_turn_event(
                turn_id,
                "conversation_utils",
                "performance_metrics_sent",
                client_uid=client_uid,
                metrics=performance_payload.get("metrics"),
            )

        synth_complete_payload = {"type": "backend-synth-complete"}
        if turn_id:
            synth_complete_payload["turn_id"] = turn_id
        response_waiter = message_handler.prepare_response_wait(
            client_uid,
            "frontend-playback-complete",
            request_id=turn_id,
            timeout=30.0,
        )
        await websocket_send(json.dumps(synth_complete_payload))
        record_turn_event(
            turn_id,
            "conversation_utils",
            "backend_synth_complete_sent",
            client_uid=client_uid,
        )

        response = await response_waiter

        if not response:
            logger.warning(f"No playback completion response from {client_uid}")
            record_turn_event(
                turn_id,
                "conversation_utils",
                "frontend_playback_complete_missing",
                client_uid=client_uid,
            )
            return
        record_turn_event(
            turn_id,
            "conversation_utils",
            "frontend_playback_complete_received",
            client_uid=client_uid,
        )

    if not performance_payload_sent:
        performance_payload = build_performance_payload(turn_id)
        if performance_payload:
            await persist_performance_metrics(
                turn_id,
                performance_payload.get("metrics"),
                client_uid=client_uid,
                input_source=performance_payload.get("input_source"),
                backend_complete=True,
            )
            await websocket_send(json.dumps(performance_payload))
            record_turn_event(
                turn_id,
                "conversation_utils",
                "performance_metrics_sent",
                client_uid=client_uid,
                metrics=performance_payload.get("metrics"),
            )

    force_new_payload = {"type": "force-new-message"}
    if turn_id:
        force_new_payload["turn_id"] = turn_id
    await websocket_send(json.dumps(force_new_payload))
    record_turn_event(
        turn_id,
        "conversation_utils",
        "force_new_message_sent",
        client_uid=client_uid,
    )

    if broadcast_ctx and broadcast_ctx.broadcast_func:
        await broadcast_ctx.broadcast_func(
            broadcast_ctx.group_members,
            force_new_payload,
            broadcast_ctx.current_client_uid,
        )

    await send_conversation_end_signal(websocket_send, broadcast_ctx, turn_id=turn_id)
    record_turn_event(
        turn_id,
        "conversation_utils",
        "finalize_completed",
        client_uid=client_uid,
    )


async def send_conversation_end_signal(
    websocket_send: WebSocketSend,
    broadcast_ctx: Optional[BroadcastContext],
    turn_id: str | None = None,
    session_emoji: str = "😊",
) -> None:
    """Send conversation chain end signal"""
    chain_end_msg = {
        "type": "control",
        "text": "conversation-chain-end",
    }
    if turn_id:
        chain_end_msg["turn_id"] = turn_id

    await websocket_send(json.dumps(chain_end_msg))
    record_turn_event(
        turn_id,
        "conversation_utils",
        "conversation_end_signal_sent",
        payload_type=chain_end_msg["type"],
        control_text=chain_end_msg["text"],
    )

    if broadcast_ctx and broadcast_ctx.broadcast_func and broadcast_ctx.group_members:
        await broadcast_ctx.broadcast_func(
            broadcast_ctx.group_members,
            chain_end_msg,
        )
        record_turn_event(
            turn_id,
            "conversation_utils",
            "conversation_end_signal_broadcast",
            group_members=list(broadcast_ctx.group_members),
        )

    logger.info(f"😎👍✅ Conversation Chain {session_emoji} completed!")


def cleanup_conversation(tts_manager: TTSTaskManager, session_emoji: str) -> None:
    """Clean up conversation resources"""
    tts_manager.clear()
    logger.debug(f"🧹 Clearing up conversation {session_emoji}.")


EMOJI_LIST = [
    "🐶",
    "🐱",
    "🐭",
    "🐹",
    "🐰",
    "🦊",
    "🐻",
    "🐼",
    "🐨",
    "🐯",
    "🦁",
    "🐮",
    "🐷",
    "🐸",
    "🐵",
    "🐔",
    "🐧",
    "🐦",
    "🐤",
    "🐣",
    "🐥",
    "🦆",
    "🦅",
    "🦉",
    "🦇",
    "🐺",
    "🐗",
    "🐴",
    "🦄",
    "🐝",
    "🌵",
    "🎄",
    "🌲",
    "🌳",
    "🌴",
    "🌱",
    "🌿",
    "☘️",
    "🍀",
    "🍂",
    "🍁",
    "🍄",
    "🌾",
    "💐",
    "🌹",
    "🌸",
    "🌛",
    "🌍",
    "⭐️",
    "🔥",
    "🌈",
    "🌩",
    "⛄️",
    "🎃",
    "🎄",
    "🎉",
    "🎏",
    "🎗",
    "🀄️",
    "🎭",
    "🎨",
    "🧵",
    "🪡",
    "🧶",
    "🥽",
    "🥼",
    "🦺",
    "👔",
    "👕",
    "👜",
    "👑",
]


_AGENT_STREAM_WORKER_ATTR = "_threaded_agent_output_worker"
_STREAM_SENTINEL = object()


@dataclass
class _StreamError:
    exception: BaseException


@dataclass
class _AgentStreamJob:
    batch_input: BatchInput
    output_loop: asyncio.AbstractEventLoop
    output_queue: asyncio.Queue
    turn_id: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    completed_event: threading.Event = field(default_factory=threading.Event)
    task: Optional[asyncio.Task] = None


class _AgentOutputThreadWorker:
    """Runs agent.chat() on a dedicated event loop thread."""

    def __init__(self, agent: Any):
        self._agent = agent
        self._jobs: queue.Queue[Optional[_AgentStreamJob]] = queue.Queue()
        self._loop_ready = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"agent-output-stream-{id(agent):x}",
            daemon=True,
        )
        self._thread.start()
        if not self._loop_ready.wait(timeout=5):
            raise RuntimeError("Timed out while starting agent output stream thread.")

    def submit(
        self,
        batch_input: BatchInput,
        output_loop: asyncio.AbstractEventLoop,
        output_queue: asyncio.Queue,
        turn_id: str | None = None,
    ) -> _AgentStreamJob:
        job = _AgentStreamJob(
            batch_input=batch_input,
            output_loop=output_loop,
            output_queue=output_queue,
            turn_id=turn_id,
        )
        self._jobs.put(job)
        record_turn_event(
            turn_id,
            "conversation_utils.agent_thread",
            "job_submitted",
            worker_thread=self._thread.name,
        )
        return job

    def cancel(self, job: _AgentStreamJob) -> None:
        job.cancel_event.set()
        record_turn_event(
            job.turn_id,
            "conversation_utils.agent_thread",
            "job_cancel_requested",
            worker_thread=self._thread.name,
        )
        loop = self._loop
        task = job.task
        if loop and task and not task.done() and not loop.is_closed():
            loop.call_soon_threadsafe(task.cancel)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._loop_ready.set()
        logger.debug(
            "Started threaded agent output stream loop: {}",
            threading.current_thread().name,
        )

        try:
            while True:
                job = self._jobs.get()
                if job is None:
                    break

                task = loop.create_task(self._consume_job(job))
                job.task = task
                record_turn_event(
                    job.turn_id,
                    "conversation_utils.agent_thread",
                    "job_started",
                    worker_thread=threading.current_thread().name,
                )
                try:
                    loop.run_until_complete(task)
                except asyncio.CancelledError:
                    logger.debug("Threaded agent output stream job cancelled.")
                except BaseException as exc:
                    logger.exception("Threaded agent output stream job failed.")
                    self._post(job, _StreamError(exc))
                finally:
                    mark_performance_elapsed(
                        job.turn_id,
                        "llm_total_seconds",
                        "llm",
                    )
                    job.completed_event.set()
                    self._post(job, _STREAM_SENTINEL)
                    job.task = None
                    record_turn_event(
                        job.turn_id,
                        "conversation_utils.agent_thread",
                        "job_finished",
                        worker_thread=threading.current_thread().name,
                        cancelled=job.cancel_event.is_set(),
                    )
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _consume_job(self, job: _AgentStreamJob) -> None:
        if job.cancel_event.is_set():
            return

        logger.debug(
            "Consuming agent output stream in thread: {}",
            threading.current_thread().name,
        )
        record_turn_event(
            job.turn_id,
            "conversation_utils.agent_thread",
            "agent_chat_entered",
            worker_thread=threading.current_thread().name,
        )
        start_performance_phase(job.turn_id, "llm")
        async for output_item in self._agent.chat(job.batch_input):
            if job.cancel_event.is_set():
                break
            record_turn_event(
                job.turn_id,
                "conversation_utils.agent_thread",
                "agent_chat_output_posted",
                worker_thread=threading.current_thread().name,
                output_type=type(output_item).__name__,
                message_type=output_item.get("type")
                if isinstance(output_item, dict)
                else None,
            )
            if is_llm_first_token_event(output_item):
                mark_performance_elapsed(
                    job.turn_id,
                    "llm_first_token_seconds",
                    "llm",
                )
            elif is_web_search_timing_event(output_item):
                set_performance_metric(
                    job.turn_id,
                    "web_search_seconds",
                    output_item.get("seconds", 0.0),
                    overwrite=True,
                )
            elif isinstance(output_item, SentenceOutput):
                mark_performance_elapsed(
                    job.turn_id,
                    "llm_first_sentence_seconds",
                    "llm",
                )
            self._post(job, output_item)

    @staticmethod
    def _post(job: _AgentStreamJob, item: Any) -> None:
        if job.output_loop.is_closed():
            return
        try:
            job.output_loop.call_soon_threadsafe(job.output_queue.put_nowait, item)
        except RuntimeError:
            pass


def _get_agent_output_worker(agent: Any) -> _AgentOutputThreadWorker:
    worker = getattr(agent, _AGENT_STREAM_WORKER_ATTR, None)
    if worker is None:
        worker = _AgentOutputThreadWorker(agent)
        setattr(agent, _AGENT_STREAM_WORKER_ATTR, worker)
    return worker


async def stream_agent_output_in_thread(
    agent: Any,
    batch_input: BatchInput,
    turn_id: str | None = None,
) -> AsyncIterator[Any]:
    """Yield agent.chat() output from a dedicated thread-backed event loop."""

    output_loop = asyncio.get_running_loop()
    output_queue: asyncio.Queue = asyncio.Queue()
    worker = _get_agent_output_worker(agent)
    job = worker.submit(batch_input, output_loop, output_queue, turn_id=turn_id)

    try:
        while True:
            item = await output_queue.get()
            if item is _STREAM_SENTINEL:
                break
            if isinstance(item, _StreamError):
                record_turn_event(
                    turn_id,
                    "conversation_utils",
                    "agent_stream_error_received",
                    error=str(item.exception),
                )
                raise item.exception
            record_turn_event(
                turn_id,
                "conversation_utils",
                "agent_stream_item_yielded",
                output_type=type(item).__name__,
                message_type=item.get("type") if isinstance(item, dict) else None,
            )
            yield item
    finally:
        if not job.completed_event.is_set():
            worker.cancel(job)
        record_turn_event(
            turn_id,
            "conversation_utils",
            "agent_stream_completed",
            cancelled=not job.completed_event.is_set(),
        )
