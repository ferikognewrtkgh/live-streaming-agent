from typing import Union, List, Dict, Any, Optional
import asyncio
import json
import uuid
from contextlib import suppress
from loguru import logger
import numpy as np

from .conversation_utils import (
    create_batch_input,
    process_agent_output,
    send_conversation_start_signals,
    process_user_input,
    finalize_conversation_turn,
    cleanup_conversation,
    is_llm_final_error_fallback_event,
    is_llm_first_token_event,
    is_web_search_start_event,
    is_web_search_timing_event,
    speak_delay_trigger_if_llm_is_slow,
    stream_agent_output_in_thread,
    EMOJI_LIST,
)
from .types import WebSocketSend
from .tts_manager import TTSTaskManager
from ..chat_history_manager import store_message, create_new_history
from ..knowledge_elasticsearch import maybe_enhance_input_with_knowledge
from ..service_context import ServiceContext
from ..utils.turn_trace import record_turn_event
from ..performance_metrics import ensure_performance_turn, send_performance_stage

# Import necessary types from agent outputs
from ..agent.output_types import SentenceOutput, AudioOutput


async def process_single_conversation(
    context: ServiceContext,
    websocket_send: WebSocketSend,
    client_uid: str,
    user_input: Union[str, np.ndarray],
    images: Optional[List[Dict[str, Any]]] = None,
    session_emoji: str = np.random.choice(EMOJI_LIST),
    metadata: Optional[Dict[str, Any]] = None,
    turn_id: str | None = None,
) -> str:
    """Process a single-user conversation turn

    Args:
        context: Service context containing all configurations and engines
        websocket_send: WebSocket send function
        client_uid: Client unique identifier
        user_input: Text or audio input from user
        images: Optional list of image data
        session_emoji: Emoji identifier for the conversation
        metadata: Optional metadata for special processing flags

    Returns:
        str: Complete response text
    """
    turn_id = turn_id or uuid.uuid4().hex
    input_source = str((metadata or {}).get("input_source") or "").strip()
    if not input_source:
        input_source = "audio" if isinstance(user_input, np.ndarray) else "text"
    ensure_performance_turn(turn_id, input_source=input_source)
    record_turn_event(
        turn_id,
        "single_conversation",
        "entered",
        client_uid=client_uid,
        input_kind="audio" if isinstance(user_input, np.ndarray) else "text",
        audio_samples=len(user_input) if isinstance(user_input, np.ndarray) else None,
        text_len=len(user_input) if isinstance(user_input, str) else None,
        images_count=len(images or []),
        metadata=metadata or {},
    )

    audio_not_before_monotonic = None
    if metadata and metadata.get("audio_not_before_monotonic") is not None:
        try:
            audio_not_before_monotonic = float(metadata["audio_not_before_monotonic"])
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring invalid audio_not_before_monotonic metadata: {}",
                metadata.get("audio_not_before_monotonic"),
            )

    # Create TTSTaskManager for this conversation
    tts_manager = TTSTaskManager(
        turn_id=turn_id,
        audio_not_before_monotonic=audio_not_before_monotonic,
    )
    full_response = ""  # Initialize full_response here
    should_sleep_after_llm_failure = False

    try:
        # Send initial signals
        await send_conversation_start_signals(websocket_send, turn_id=turn_id)
        record_turn_event(
            turn_id,
            "single_conversation",
            "start_signals_sent",
            client_uid=client_uid,
        )
        logger.info(f"New Conversation Chain {session_emoji} started!")
        human_name = str(
            (metadata or {}).get("human_name") or context.character_config.human_name
        )

        # Process user input
        input_text = await process_user_input(
            user_input, context.asr_engine, websocket_send, turn_id=turn_id
        )
        record_turn_event(
            turn_id,
            "single_conversation",
            "user_input_processed",
            client_uid=client_uid,
            input_text_len=len(input_text),
            input_text_preview=input_text[:120],
        )

        # Store user message (check if we should skip storing to history)
        skip_history = bool(metadata and metadata.get("skip_history", False))
        history_user_input_already_stored = bool(
            metadata and metadata.get("history_user_input_already_stored")
        )
        if not skip_history:
            input_text, metadata = await maybe_enhance_input_with_knowledge(
                context=context,
                input_text=input_text,
                metadata=metadata,
                turn_id=turn_id,
                client_uid=client_uid,
            )

        if (
            metadata
            and metadata.get("reload_memory_from_history_before_turn")
            and context.history_uid
            and not skip_history
        ):
            try:
                context.agent_engine.set_memory_from_history(
                    conf_uid=context.character_config.conf_uid,
                    history_uid=context.history_uid,
                )
                record_turn_event(
                    turn_id,
                    "single_conversation",
                    "agent_memory_reloaded_before_turn",
                    client_uid=client_uid,
                    history_uid=context.history_uid,
                    reason=metadata.get("reload_memory_reason"),
                )
                logger.info(
                    "Reloaded agent memory from history before turn: history_uid={} reason={}",
                    context.history_uid,
                    metadata.get("reload_memory_reason"),
                )
                if history_user_input_already_stored:
                    remove_current_user = getattr(
                        context.agent_engine,
                        "remove_recent_user_input_from_memory",
                        None,
                    )
                    if callable(remove_current_user):
                        removed_current = bool(
                            remove_current_user(input_text, human_name)
                        )
                        record_turn_event(
                            turn_id,
                            "single_conversation",
                            "prestored_current_user_removed_after_memory_reload",
                            client_uid=client_uid,
                            removed=removed_current,
                            input_text_len=len(input_text),
                        )
            except Exception as exc:
                logger.exception("Failed to reload agent memory before turn.")
                record_turn_event(
                    turn_id,
                    "single_conversation",
                    "agent_memory_reload_before_turn_failed",
                    client_uid=client_uid,
                    history_uid=context.history_uid,
                    error=str(exc),
                )

        # Create batch input
        batch_input = create_batch_input(
            input_text=input_text,
            images=images,
            from_name=human_name,
            metadata=metadata,
        )
        logger.info(
            "LLM input prepared: text_len={}, images_count={}, metadata_keys={}",
            len(input_text),
            len(batch_input.images or []),
            sorted((metadata or {}).keys()),
        )

        # Auto-create history if not yet set (e.g. barrage mode where
        # conversations are initiated server-side, not by the frontend)
        if not context.history_uid and not skip_history:
            auto_uid = create_new_history(context.character_config.conf_uid)
            if auto_uid:
                context.history_uid = auto_uid
                logger.info(
                    f"Auto-created chat history for "
                    f"{context.character_config.conf_uid}: {auto_uid}"
                )
                record_turn_event(
                    turn_id,
                    "single_conversation",
                    "history_auto_created",
                    client_uid=client_uid,
                    history_uid=auto_uid,
                )

        if context.history_uid and not skip_history and not history_user_input_already_stored:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="human",
                content=input_text,
                name=human_name,
            )
            record_turn_event(
                turn_id,
                "single_conversation",
                "human_message_stored",
                client_uid=client_uid,
                history_uid=context.history_uid,
                input_text_len=len(input_text),
            )
        elif history_user_input_already_stored:
            record_turn_event(
                turn_id,
                "single_conversation",
                "human_message_store_skipped_already_stored",
                client_uid=client_uid,
                history_uid=context.history_uid,
                input_text_len=len(input_text),
            )

        if skip_history:
            logger.debug("Skipping storing user input to history (proactive speak)")

        logger.info(f"User input: {input_text}")
        if images:
            logger.info(f"With {len(images)} images")

        first_token_event = asyncio.Event()
        delay_trigger_task = asyncio.create_task(
            speak_delay_trigger_if_llm_is_slow(
                first_token_event=first_token_event,
                character_config=context.character_config,
                live2d_model=context.live2d_model,
                tts_engine=context.tts_engine,
                websocket_send=websocket_send,
                tts_manager=tts_manager,
                turn_id=turn_id,
            )
        )
        llm_first_token_stage_sent = False
        llm_first_sentence_stage_sent = False
        llm_stream_failed = False
        await send_performance_stage(websocket_send, turn_id, "llm-start")
        try:
            # agent.chat includes LLM streaming and sentence segmentation; run it
            # off the main loop so TTS/WebSocket I/O stays responsive.
            async for output_item in stream_agent_output_in_thread(
                context.agent_engine,
                batch_input,
                turn_id=turn_id,
            ):
                record_turn_event(
                    turn_id,
                    "single_conversation",
                    "agent_stream_item_received",
                    client_uid=client_uid,
                    output_type=type(output_item).__name__,
                    message_type=output_item.get("type")
                    if isinstance(output_item, dict)
                    else None,
                )
                if is_llm_first_token_event(output_item):
                    if not llm_first_token_stage_sent:
                        llm_first_token_stage_sent = True
                        await send_performance_stage(
                            websocket_send,
                            turn_id,
                            "llm-first-token",
                        )
                    first_token_event.set()
                    record_turn_event(
                        turn_id,
                        "single_conversation",
                        "llm_first_token_event_received",
                        client_uid=client_uid,
                    )
                    continue

                if is_web_search_start_event(output_item):
                    await send_performance_stage(
                        websocket_send,
                        turn_id,
                        "web-search-start",
                    )
                    continue

                if is_web_search_timing_event(output_item):
                    await send_performance_stage(
                        websocket_send,
                        turn_id,
                        "web-search-complete",
                    )
                    continue

                if is_llm_final_error_fallback_event(output_item):
                    should_sleep_after_llm_failure = True
                    record_turn_event(
                        turn_id,
                        "single_conversation",
                        "llm_final_error_fallback_detected",
                        client_uid=client_uid,
                    )
                    continue

                if (
                    isinstance(output_item, dict)
                    and output_item.get("type") == "tool_call_status"
                ):
                    # Handle tool status event: send WebSocket message
                    output_item["name"] = context.character_config.character_name
                    logger.debug(f"Sending tool status update: {output_item}")

                    await websocket_send(json.dumps(output_item))

                elif isinstance(output_item, (SentenceOutput, AudioOutput)):
                    if not first_token_event.is_set():
                        if not llm_first_token_stage_sent:
                            llm_first_token_stage_sent = True
                            await send_performance_stage(
                                websocket_send,
                                turn_id,
                                "llm-first-token",
                            )
                        first_token_event.set()
                        record_turn_event(
                            turn_id,
                            "single_conversation",
                            "llm_first_token_event_inferred",
                            client_uid=client_uid,
                            output_type=type(output_item).__name__,
                        )

                    if (
                        isinstance(output_item, SentenceOutput)
                        and not llm_first_sentence_stage_sent
                    ):
                        llm_first_sentence_stage_sent = True
                        await send_performance_stage(
                            websocket_send,
                            turn_id,
                            "llm-first-sentence",
                        )

                    # Handle SentenceOutput or AudioOutput
                    response_part = await process_agent_output(
                        output=output_item,
                        character_config=context.character_config,
                        live2d_model=context.live2d_model,
                        tts_engine=context.tts_engine,
                        websocket_send=websocket_send,  # Pass websocket_send for audio/tts messages
                        tts_manager=tts_manager,
                        translate_engine=context.translate_engine,
                        turn_id=turn_id,
                    )
                    # Ensure response_part is treated as a string before concatenation
                    response_part_str = (
                        str(response_part) if response_part is not None else ""
                    )
                    full_response += response_part_str  # Accumulate text response
                    record_turn_event(
                        turn_id,
                        "single_conversation",
                        "agent_output_processed",
                        client_uid=client_uid,
                        response_part_len=len(response_part_str),
                        full_response_len=len(full_response),
                    )
                else:
                    logger.warning(
                        f"Received unexpected item type from agent chat stream: {type(output_item)}"
                    )
                    logger.debug(f"Unexpected item content: {output_item}")

        except Exception as e:
            llm_stream_failed = True
            logger.exception(
                f"Error processing agent response stream: {e}"
            )  # Log with stack trace
            await websocket_send(
                json.dumps(
                    {
                        "type": "error",
                        "message": f"Error processing agent response: {str(e)}",
                    }
                )
            )
            # full_response will contain partial response before error
        finally:
            first_token_event.set()
            if not delay_trigger_task.done():
                delay_trigger_task.cancel()
                with suppress(asyncio.CancelledError):
                    await delay_trigger_task
        await send_performance_stage(
            websocket_send,
            turn_id,
            "llm-failed" if llm_stream_failed else "llm-complete",
        )
        # --- End processing agent response ---

        # Wait for any pending TTS tasks

        await finalize_conversation_turn(
            tts_manager=tts_manager,
            websocket_send=websocket_send,
            client_uid=client_uid,
            turn_id=turn_id,
        )
        record_turn_event(
            turn_id,
            "single_conversation",
            "finalize_completed",
            client_uid=client_uid,
            full_response_len=len(full_response),
            interrupted=False,
        )

        if context.history_uid and full_response:  # Check full_response before storing
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="ai",
                content=full_response,
                name=context.character_config.character_name,
                avatar=context.character_config.avatar,
            )
            logger.info(f"AI response: {full_response}")
            record_turn_event(
                turn_id,
                "single_conversation",
                "ai_message_stored",
                client_uid=client_uid,
                history_uid=context.history_uid,
                full_response_len=len(full_response),
            )

        if should_sleep_after_llm_failure and not (
            tts_manager.sleep_entered_after_tts_failure
        ):
            from ..vtuber_state_machine import get_vtuber_state_machine

            sm = get_vtuber_state_machine()
            if sm is not None:
                vtuber_state = await sm.enter_sleep(
                    reason="llm-final-error-fallback",
                    interrupt_current=False,
                )
                record_turn_event(
                    turn_id,
                    "single_conversation",
                    "vtuber_sleep_after_llm_failure",
                    client_uid=client_uid,
                    vtuber_state=vtuber_state,
                )
            else:
                record_turn_event(
                    turn_id,
                    "single_conversation",
                    "vtuber_sleep_after_llm_failure_skipped",
                    client_uid=client_uid,
                    reason="state_machine_missing",
                )
        elif should_sleep_after_llm_failure:
            record_turn_event(
                turn_id,
                "single_conversation",
                "vtuber_sleep_after_llm_failure_skipped",
                client_uid=client_uid,
                reason="tts_failure_already_entered_sleep",
            )

        record_turn_event(
            turn_id,
            "single_conversation",
            "completed",
            client_uid=client_uid,
            full_response_len=len(full_response),
            interrupted=False,
        )
        return full_response  # Return accumulated full_response

    except asyncio.CancelledError:
        record_turn_event(
            turn_id,
            "single_conversation",
            "cancelled",
            client_uid=client_uid,
            full_response_len=len(full_response),
            interrupted=True,
        )
        logger.info(f"🤡👍 Conversation {session_emoji} cancelled because interrupted.")
        raise
    except Exception as e:
        logger.error(f"Error in conversation chain: {e}")
        record_turn_event(
            turn_id,
            "single_conversation",
            "error",
            client_uid=client_uid,
            error=str(e),
            full_response_len=len(full_response),
        )
        await websocket_send(
            json.dumps({"type": "error", "message": f"Conversation error: {str(e)}"})
        )
        raise
    finally:
        cleanup_conversation(tts_manager, session_emoji)
        record_turn_event(
            turn_id,
            "single_conversation",
            "cleanup_completed",
            client_uid=client_uid,
        )
