from typing import Any, Dict, List, Optional, Union
import asyncio
import json
import uuid
from contextlib import suppress
from loguru import logger
from fastapi import WebSocket
import numpy as np

from ..agent.output_types import AudioOutput, SentenceOutput

from .conversation_utils import (
    create_batch_input,
    process_agent_output,
    process_user_input,
    finalize_conversation_turn,
    cleanup_conversation,
    is_llm_first_token_event,
    speak_delay_trigger_if_llm_is_slow,
    stream_agent_output_in_thread,
    EMOJI_LIST,
)
from .types import (
    BroadcastFunc,
    GroupConversationState,
    BroadcastContext,
    WebSocketSend,
)
from ..service_context import ServiceContext
from ..chat_history_manager import store_message
from ..knowledge_elasticsearch import maybe_enhance_input_with_knowledge
from ..utils.turn_trace import record_turn_event
from .tts_manager import TTSTaskManager


async def process_group_conversation(
    client_contexts: Dict[str, ServiceContext],
    client_connections: Dict[str, WebSocket],
    broadcast_func: BroadcastFunc,
    group_members: List[str],
    initiator_client_uid: str,
    user_input: Union[str, np.ndarray],
    images: Optional[List[Dict[str, Any]]] = None,
    session_emoji: str = np.random.choice(EMOJI_LIST),
    metadata: Optional[Dict[str, Any]] = None,
    turn_id: str | None = None,
) -> None:
    """Process group conversation

    Args:
        client_contexts: Dictionary of client contexts
        client_connections: Dictionary of client WebSocket connections
        broadcast_func: Function to broadcast messages to group
        group_members: List of group member UIDs
        initiator_client_uid: UID of conversation initiator
        user_input: Text or audio input from user
        images: Optional list of image data
        session_emoji: Emoji identifier for the conversation
        metadata: Optional metadata for special processing flags
    """
    turn_id = turn_id or uuid.uuid4().hex
    record_turn_event(
        turn_id,
        "group_conversation",
        "entered",
        initiator_client_uid=initiator_client_uid,
        group_members=list(group_members),
        input_kind="audio" if isinstance(user_input, np.ndarray) else "text",
        audio_samples=len(user_input) if isinstance(user_input, np.ndarray) else None,
        text_len=len(user_input) if isinstance(user_input, str) else None,
        images_count=len(images or []),
        metadata=metadata or {},
    )

    # Create TTSTaskManager for each member
    tts_managers = {uid: TTSTaskManager(turn_id=turn_id) for uid in group_members}

    try:
        logger.info(f"Group Conversation Chain {session_emoji} started!")
        record_turn_event(
            turn_id,
            "group_conversation",
            "started",
            initiator_client_uid=initiator_client_uid,
            group_members=list(group_members),
        )

        # Initialize state with group_id
        state = GroupConversationState(
            group_id=f"group_{initiator_client_uid}",  # Use same format as chat_group
            session_emoji=session_emoji,
            group_queue=list(group_members),
            memory_index={
                uid: 0 for uid in group_members
            },  # Initialize memory index for each member
        )

        # Initialize group conversation context for each AI
        init_group_conversation_contexts(client_contexts)

        # Get human name from initiator context
        initiator_context = client_contexts.get(initiator_client_uid)
        human_name = (
            initiator_context.character_config.human_name
            if initiator_context
            else "Human"
        )
        human_name = str((metadata or {}).get("human_name") or human_name)

        # Process initial input
        input_text = await process_group_input(
            user_input=user_input,
            initiator_context=initiator_context,
            initiator_ws_send=client_connections[initiator_client_uid].send_text,
            broadcast_func=broadcast_func,
            group_members=group_members,
            initiator_client_uid=initiator_client_uid,
            turn_id=turn_id,
        )
        record_turn_event(
            turn_id,
            "group_conversation",
            "group_input_processed",
            initiator_client_uid=initiator_client_uid,
            input_text_len=len(input_text),
            input_text_preview=input_text[:120],
        )

        # Check if we should skip storing this input to history
        skip_history = metadata and metadata.get("skip_history", False)
        if not skip_history and initiator_context:
            input_text, metadata = await maybe_enhance_input_with_knowledge(
                context=initiator_context,
                input_text=input_text,
                metadata=metadata,
                turn_id=turn_id,
                client_uid=initiator_client_uid,
            )

        if not skip_history:
            for member_uid in group_members:
                member_context = client_contexts[member_uid]
                store_message(
                    conf_uid=member_context.character_config.conf_uid,
                    history_uid=member_context.history_uid,
                    role="human",
                    content=input_text,
                    name=human_name,
                )
            record_turn_event(
                turn_id,
                "group_conversation",
                "human_message_stored_for_group",
                group_members=list(group_members),
                input_text_len=len(input_text),
            )
        else:
            logger.debug("Skipping storing proactive speak input to group history")

        state.conversation_history = [f"{human_name}: {input_text}"]

        is_first_responder = False
        # Main conversation loop
        while state.group_queue:
            try:
                current_member_uid = state.group_queue.pop(0)

                # Only pass metadata to the first responder
                current_metadata = None
                if is_first_responder:
                    current_metadata = metadata
                    is_first_responder = False

                await handle_group_member_turn(
                    current_member_uid=current_member_uid,
                    state=state,
                    client_contexts=client_contexts,
                    client_connections=client_connections,
                    broadcast_func=broadcast_func,
                    group_members=group_members,
                    images=images,
                    tts_manager=tts_managers[current_member_uid],
                    metadata=current_metadata,
                    turn_id=turn_id,
                )
            except Exception as e:
                logger.error(f"Error in group member turn: {e}")
                await handle_member_error(
                    broadcast_func, group_members, f"Error in conversation: {str(e)}"
                )

    except asyncio.CancelledError:
        record_turn_event(
            turn_id,
            "group_conversation",
            "cancelled",
            initiator_client_uid=initiator_client_uid,
            interrupted=True,
        )
        logger.info(
            f"🤡👍 Group Conversation {session_emoji} cancelled because interrupted."
        )
        raise
    except Exception as e:
        logger.error(f"Error in group conversation chain: {e}")
        record_turn_event(
            turn_id,
            "group_conversation",
            "error",
            initiator_client_uid=initiator_client_uid,
            error=str(e),
        )
        await handle_member_error(
            broadcast_func, group_members, f"Fatal error in conversation: {str(e)}"
        )
        raise
    finally:
        # Cleanup all TTS managers
        for tts_manager in tts_managers.values():
            cleanup_conversation(tts_manager, session_emoji)
        # Clean up
        GroupConversationState.remove_state(state.group_id)
        record_turn_event(
            turn_id,
            "group_conversation",
            "cleanup_completed",
            initiator_client_uid=initiator_client_uid,
        )


def init_group_conversation_state(
    group_members: List[str], session_emoji: str
) -> GroupConversationState:
    """Initialize group conversation state"""
    return GroupConversationState(
        conversation_history=[],
        memory_index={uid: 0 for uid in group_members},
        group_queue=list(group_members),
        session_emoji=session_emoji,
    )


def init_group_conversation_contexts(
    client_contexts: Dict[str, ServiceContext],
) -> None:
    """Initialize group conversation context for each AI participant"""
    ai_names = [ctx.character_config.character_name for ctx in client_contexts.values()]

    for context in client_contexts.values():
        agent = context.agent_engine
        if hasattr(agent, "start_group_conversation"):
            agent.start_group_conversation(
                human_name="Human",
                ai_participants=[
                    name
                    for name in ai_names
                    if name != context.character_config.character_name
                ],
            )
            logger.debug(
                f"Initialized group conversation context for "
                f"{context.character_config.character_name}"
            )


async def process_group_input(
    user_input: Union[str, np.ndarray],
    initiator_context: ServiceContext,
    initiator_ws_send: WebSocketSend,
    broadcast_func: BroadcastFunc,
    group_members: List[str],
    initiator_client_uid: str,
    turn_id: str | None = None,
) -> str:
    """Process and broadcast user input to group"""
    record_turn_event(
        turn_id,
        "group_conversation",
        "process_group_input_entered",
        initiator_client_uid=initiator_client_uid,
    )
    input_text = await process_user_input(
        user_input, initiator_context.asr_engine, initiator_ws_send, turn_id=turn_id
    )
    await broadcast_transcription(
        broadcast_func, group_members, input_text, initiator_client_uid, turn_id=turn_id
    )
    return input_text


async def broadcast_transcription(
    broadcast_func: BroadcastFunc,
    group_members: List[str],
    text: str,
    exclude_uid: str,
    turn_id: str | None = None,
) -> None:
    """Broadcast transcription to group members"""
    payload = {
        "type": "user-input-transcription",
        "text": text,
    }
    if turn_id:
        payload["turn_id"] = turn_id
    await broadcast_func(
        group_members,
        payload,
        exclude_uid,
    )
    record_turn_event(
        turn_id,
        "group_conversation",
        "transcription_broadcast",
        group_members=list(group_members),
        exclude_uid=exclude_uid,
        text_len=len(text),
    )


async def handle_group_member_turn(
    current_member_uid: str,
    state: GroupConversationState,
    client_contexts: Dict[str, ServiceContext],
    client_connections: Dict[str, WebSocket],
    broadcast_func: BroadcastFunc,
    group_members: List[str],
    images: Optional[List[Dict[str, Any]]],
    tts_manager: TTSTaskManager,
    metadata: Optional[Dict[str, Any]] = None,
    turn_id: str | None = None,
) -> None:
    """Handle a single group member's conversation turn"""
    # Update current speaker before processing
    state.current_speaker_uid = current_member_uid
    record_turn_event(
        turn_id,
        "group_conversation",
        "member_turn_entered",
        current_member_uid=current_member_uid,
        group_id=state.group_id,
        queue_len=len(state.group_queue),
    )

    await broadcast_thinking_state(broadcast_func, group_members, turn_id=turn_id)

    context = client_contexts[current_member_uid]
    current_ws_send = client_connections[current_member_uid].send_text

    new_messages = state.conversation_history[state.memory_index[current_member_uid] :]
    new_context = "\n".join(new_messages) if new_messages else ""

    batch_input = create_batch_input(
        input_text=new_context,
        images=images,
        from_name="Human",
        metadata=metadata,
        include_human_name_prefix=False,
    )

    logger.info(
        f"AI {context.character_config.character_name} "
        f"(client {current_member_uid}) receiving context:\n{new_context}"
    )

    full_response = await process_member_response(
        context=context,
        batch_input=batch_input,
        current_ws_send=current_ws_send,
        tts_manager=tts_manager,
        broadcast_func=broadcast_func,
        group_members=group_members,
        turn_id=turn_id,
    )
    record_turn_event(
        turn_id,
        "group_conversation",
        "member_response_completed",
        current_member_uid=current_member_uid,
        response_len=len(full_response),
    )

    broadcast_ctx = BroadcastContext(
        broadcast_func=broadcast_func,
        group_members=group_members,
        current_client_uid=current_member_uid,
    )

    await finalize_conversation_turn(
        tts_manager=tts_manager,
        websocket_send=current_ws_send,
        client_uid=current_member_uid,
        broadcast_ctx=broadcast_ctx,
        turn_id=turn_id,
    )
    record_turn_event(
        turn_id,
        "group_conversation",
        "member_finalize_completed",
        current_member_uid=current_member_uid,
    )

    if full_response:
        ai_message = f"{context.character_config.character_name}: {full_response}"
        state.conversation_history.append(ai_message)
        logger.info(f"Appended complete response: {ai_message}")

        for member_uid in group_members:
            member_context = client_contexts[member_uid]
            store_message(
                conf_uid=member_context.character_config.conf_uid,
                history_uid=member_context.history_uid,
                role="ai",
                content=full_response,
                name=context.character_config.character_name,
                avatar=context.character_config.avatar,
            )
        else:
            logger.debug("Skipping storing AI response to history (proactive speak)")
        record_turn_event(
            turn_id,
            "group_conversation",
            "ai_message_stored_for_group",
            speaker_uid=current_member_uid,
            group_members=list(group_members),
            response_len=len(full_response),
        )

    state.memory_index[current_member_uid] = len(state.conversation_history)
    state.group_queue.append(current_member_uid)

    # Clear speaker after turn completes
    state.current_speaker_uid = None
    record_turn_event(
        turn_id,
        "group_conversation",
        "member_turn_completed",
        current_member_uid=current_member_uid,
        group_id=state.group_id,
    )


async def broadcast_thinking_state(
    broadcast_func: BroadcastFunc,
    group_members: List[str],
    turn_id: str | None = None,
) -> None:
    """Broadcast thinking state to group"""
    start_payload = {"type": "control", "text": "conversation-chain-start"}
    thinking_payload = {"type": "full-text", "text": "Thinking..."}
    if turn_id:
        start_payload["turn_id"] = turn_id
        thinking_payload["turn_id"] = turn_id
    await broadcast_func(
        group_members,
        start_payload,
    )
    record_turn_event(
        turn_id,
        "group_conversation",
        "group_start_signal_broadcast",
        group_members=list(group_members),
    )
    await broadcast_func(
        group_members,
        thinking_payload,
    )
    record_turn_event(
        turn_id,
        "group_conversation",
        "group_thinking_signal_broadcast",
        group_members=list(group_members),
    )


async def handle_member_error(
    broadcast_func: BroadcastFunc,
    group_members: List[str],
    error_message: str,
) -> None:
    """Handle and broadcast member error"""
    await broadcast_func(
        group_members,
        {
            "type": "error",
            "message": error_message,
        },
    )


async def process_member_response(
    context: ServiceContext,
    batch_input: Any,
    current_ws_send: WebSocketSend,
    tts_manager: TTSTaskManager,
    broadcast_func: Optional[BroadcastFunc] = None,
    group_members: Optional[List[str]] = None,
    turn_id: str | None = None,
) -> str:
    """Process group member's response, handling text/audio and tool status events."""
    full_response = ""
    record_turn_event(
        turn_id,
        "group_conversation",
        "process_member_response_entered",
        character_name=context.character_config.character_name,
    )

    first_token_event = asyncio.Event()
    delay_trigger_task = asyncio.create_task(
        speak_delay_trigger_if_llm_is_slow(
            first_token_event=first_token_event,
            character_config=context.character_config,
            live2d_model=context.live2d_model,
            tts_engine=context.tts_engine,
            websocket_send=current_ws_send,
            tts_manager=tts_manager,
            turn_id=turn_id,
        )
    )
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
                "group_conversation",
                "member_agent_stream_item_received",
                character_name=context.character_config.character_name,
                output_type=type(output_item).__name__,
                message_type=output_item.get("type")
                if isinstance(output_item, dict)
                else None,
            )
            if is_llm_first_token_event(output_item):
                first_token_event.set()
                record_turn_event(
                    turn_id,
                    "group_conversation",
                    "member_llm_first_token_event_received",
                    character_name=context.character_config.character_name,
                )
                continue

            if (
                isinstance(output_item, dict)
                and output_item.get("type") == "tool_call_status"
            ):
                if broadcast_func and group_members:
                    logger.debug(f"Broadcasting tool status update: {output_item}")
                    output_item["name"] = context.character_config.character_name
                    await broadcast_func(group_members, output_item)
                else:
                    logger.warning(
                        "Cannot broadcast tool status: broadcast_func or group_members missing."
                    )
            elif isinstance(output_item, (SentenceOutput, AudioOutput)):
                if not first_token_event.is_set():
                    first_token_event.set()
                    record_turn_event(
                        turn_id,
                        "group_conversation",
                        "member_llm_first_token_event_inferred",
                        character_name=context.character_config.character_name,
                        output_type=type(output_item).__name__,
                    )

                # Handle SentenceOutput or AudioOutput: Send to current user, broadcast audio later if needed
                response_part = await process_agent_output(
                    output=output_item,
                    character_config=context.character_config,
                    live2d_model=context.live2d_model,
                    tts_engine=context.tts_engine,
                    websocket_send=current_ws_send,  # Send TTS/display text directly to speaker's client
                    tts_manager=tts_manager,
                    translate_engine=context.translate_engine,
                    turn_id=turn_id,
                )
                full_response += response_part  # Accumulate text response
                record_turn_event(
                    turn_id,
                    "group_conversation",
                    "member_agent_output_processed",
                    character_name=context.character_config.character_name,
                    response_part_len=len(response_part),
                    full_response_len=len(full_response),
                )
            else:
                logger.warning(
                    f"Received unexpected item type from agent chat stream: {type(output_item)}"
                )

    except Exception as e:
        logger.exception(f"Error processing group member response stream: {e}")
        record_turn_event(
            turn_id,
            "group_conversation",
            "process_member_response_error",
            character_name=context.character_config.character_name,
            error=str(e),
            full_response_len=len(full_response),
        )
        await current_ws_send(
            json.dumps(
                {"type": "error", "message": f"Error processing response: {str(e)}"}
            )
        )
    finally:
        first_token_event.set()
        if not delay_trigger_task.done():
            delay_trigger_task.cancel()
            with suppress(asyncio.CancelledError):
                await delay_trigger_task

    record_turn_event(
        turn_id,
        "group_conversation",
        "process_member_response_completed",
        character_name=context.character_config.character_name,
        full_response_len=len(full_response),
    )
    return full_response
