import asyncio
import json
import uuid
from typing import Dict, Optional, Callable

import numpy as np
from fastapi import WebSocket
from loguru import logger

from ..chat_group import ChatGroupManager
from ..chat_history_manager import create_new_history, store_message
from ..service_context import ServiceContext
from ..knowledge_elasticsearch import maybe_enhance_input_with_knowledge
from ..performance_metrics import (
    mark_performance_elapsed,
    send_performance_stage,
    start_performance_phase,
)
from .group_conversation import process_group_conversation
from .single_conversation import process_single_conversation
from .conversation_utils import EMOJI_LIST
from .types import GroupConversationState
from ..utils.turn_trace import record_turn_event
from prompts import prompt_loader


async def handle_conversation_trigger(
    msg_type: str,
    data: dict,
    client_uid: str,
    context: ServiceContext,
    websocket: WebSocket,
    client_contexts: Dict[str, ServiceContext],
    client_connections: Dict[str, WebSocket],
    chat_group_manager: ChatGroupManager,
    received_data_buffers: Dict[str, np.ndarray],
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    broadcast_to_group: Callable,
    turn_id: str | None = None,
) -> None:
    """Handle triggers that start a conversation"""
    # ---- VTuber 模式检查门卫 ----
    from ..vtuber_state_machine import get_vtuber_state_machine, VTuberMode

    sm = get_vtuber_state_machine()
    if sm is not None:
        if sm.mode == VTuberMode.IDLE:
            logger.info(
                f"[state] IDLE mode, ignoring conversation trigger: {msg_type}"
            )
            return
    incoming_metadata = data.get("metadata")
    metadata = dict(incoming_metadata) if isinstance(incoming_metadata, dict) else None
    websocket_send = context.send_text or websocket.send_text
    turn_id = turn_id or uuid.uuid4().hex
    record_turn_event(
        turn_id,
        "conversation_handler",
        "trigger_entered",
        client_uid=client_uid,
        message_type=msg_type,
        has_images=bool(data.get("images")),
        text_len=len(data.get("text") or ""),
    )

    if msg_type == "ai-speak-signal":
        try:
            # Get proactive speak prompt from config
            prompt_name = "proactive_speak_prompt"
            prompt_file = context.system_config.tool_prompts.get(prompt_name)
            if prompt_file:
                user_input = prompt_loader.load_util(prompt_file)
            else:
                logger.warning("Proactive speak prompt not configured, using default")
                user_input = "Please say something."
        except Exception as e:
            logger.error(f"Error loading proactive speak prompt: {e}")
            user_input = "Please say something."

        # Add metadata to indicate this is a proactive speak request
        # that should be skipped in both memory and history
        metadata = {
            "proactive_speak": True,
            "skip_memory": True,  # Skip storing in AI's internal memory
            "skip_history": True,  # Skip storing in local conversation history
        }
        record_turn_event(
            turn_id,
            "conversation_handler",
            "proactive_speak_prompt_loaded",
            client_uid=client_uid,
            prompt_len=len(user_input),
        )

        await websocket_send(
            json.dumps(
                {
                    "type": "full-text",
                    "text": "AI wants to speak something...",
                }
            )
        )
    elif msg_type == "text-input":
        user_input = data.get("text", "")
    elif msg_type == "mic-audio-end" and isinstance(data.get("text"), str):
        user_input = data.get("text", "")
    else:  # mic-audio-end
        user_input = received_data_buffers[client_uid]
        received_data_buffers[client_uid] = np.array([])

    if data.get("story_guidance"):
        metadata = {
            **(metadata or {}),
            "story_guidance": data["story_guidance"],
            "story_match": data.get("story_match"),
        }

    images = data.get("images")
    session_emoji = np.random.choice(EMOJI_LIST)

    group = chat_group_manager.get_client_group(client_uid)
    if group and len(group.members) > 1:
        # Use group_id as task key for group conversations
        task_key = group.group_id
        if (
            task_key not in current_conversation_tasks
            or current_conversation_tasks[task_key].done()
        ):
            logger.info(f"Starting new group conversation for {task_key}")
            record_turn_event(
                turn_id,
                "conversation_handler",
                "group_task_created",
                client_uid=client_uid,
                task_key=task_key,
                group_members=list(group.members),
                input_kind="audio" if isinstance(user_input, np.ndarray) else "text",
            )

            current_conversation_tasks[task_key] = asyncio.create_task(
                process_group_conversation(
                    client_contexts=client_contexts,
                    client_connections=client_connections,
                    broadcast_func=broadcast_to_group,
                    group_members=group.members,
                    initiator_client_uid=client_uid,
                    user_input=user_input,
                    images=images,
                    session_emoji=session_emoji,
                    metadata=metadata,
                    turn_id=turn_id,
                )
            )
    else:
        # Use client_uid as task key for individual conversations
        skip_history = bool(metadata and metadata.get("skip_history", False))
        if (
            msg_type in {"mic-audio-end", "text-input"}
            and isinstance(user_input, str)
            and user_input.strip()
            and not skip_history
        ):
            knowledge_config = getattr(context.config, "knowledge_config", None)
            knowledge_enabled = bool(
                knowledge_config and getattr(knowledge_config, "enabled", False)
            )
            if knowledge_enabled:
                start_performance_phase(turn_id, "knowledge")
                await send_performance_stage(
                    websocket_send,
                    turn_id,
                    "knowledge-start",
                )
            try:
                user_input, metadata = await maybe_enhance_input_with_knowledge(
                    context=context,
                    input_text=user_input,
                    metadata=metadata,
                    turn_id=turn_id,
                    client_uid=client_uid,
                )
            finally:
                if knowledge_enabled:
                    mark_performance_elapsed(
                        turn_id,
                        "knowledge_seconds",
                        "knowledge",
                    )
                    await send_performance_stage(
                        websocket_send,
                        turn_id,
                        "knowledge-complete",
                    )
            if not context.history_uid:
                auto_uid = create_new_history(context.character_config.conf_uid)
                if auto_uid:
                    context.history_uid = auto_uid
                    logger.info(
                        "Auto-created chat history before task creation for "
                        "{}: {}",
                        context.character_config.conf_uid,
                        auto_uid,
                    )
                    record_turn_event(
                        turn_id,
                        "conversation_handler",
                        "history_auto_created_before_task",
                        client_uid=client_uid,
                        history_uid=auto_uid,
                    )

            if context.history_uid:
                human_name = str(
                    (metadata or {}).get("human_name")
                    or context.character_config.human_name
                )
                store_message(
                    conf_uid=context.character_config.conf_uid,
                    history_uid=context.history_uid,
                    role="human",
                    content=user_input,
                    name=human_name,
                )
                metadata = {
                    **(metadata or {}),
                    "history_user_input_already_stored": True,
                }
                record_turn_event(
                    turn_id,
                    "conversation_handler",
                    "human_message_stored_before_task",
                    client_uid=client_uid,
                    history_uid=context.history_uid,
                    input_text_len=len(user_input),
                    human_name=human_name,
                )

        existing_task = current_conversation_tasks.get(client_uid)
        if existing_task and not existing_task.done():
            existing_task.cancel()
            metadata = {
                **(metadata or {}),
                "reload_memory_from_history_before_turn": True,
                "reload_memory_reason": "new-user-input-interrupted-running-turn",
            }
            logger.info(
                "New individual conversation trigger cancelled previous running task for {}",
                client_uid,
            )
            record_turn_event(
                turn_id,
                "conversation_handler",
                "previous_individual_task_cancelled_by_new_trigger",
                client_uid=client_uid,
                interrupted=True,
            )

        record_turn_event(
            turn_id,
            "conversation_handler",
            "individual_task_created",
            client_uid=client_uid,
            task_key=client_uid,
            input_kind="audio" if isinstance(user_input, np.ndarray) else "text",
        )
        current_conversation_tasks[client_uid] = asyncio.create_task(
            process_single_conversation(
                context=context,
                websocket_send=websocket_send,
                client_uid=client_uid,
                user_input=user_input,
                images=images,
                session_emoji=session_emoji,
                metadata=metadata,
                turn_id=turn_id,
            )
        )


async def handle_individual_interrupt(
    client_uid: str,
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    context: ServiceContext,
    heard_response: str,
    turn_id: str | None = None,
):
    record_turn_event(
        turn_id,
        "conversation_handler",
        "individual_interrupt_entered",
        client_uid=client_uid,
        heard_response_len=len(heard_response),
        interrupted=True,
    )
    if client_uid in current_conversation_tasks:
        task = current_conversation_tasks[client_uid]
        if task and not task.done():
            task.cancel()
            record_turn_event(
                turn_id,
                "conversation_handler",
                "individual_task_cancelled",
                client_uid=client_uid,
                interrupted=True,
            )
            logger.info("🛑 Conversation task was successfully interrupted")

        try:
            context.agent_engine.handle_interrupt(heard_response)
            record_turn_event(
                turn_id,
                "conversation_handler",
                "agent_interrupt_handled",
                client_uid=client_uid,
                interrupted=True,
            )
        except Exception as e:
            logger.error(f"Error handling interrupt: {e}")
            record_turn_event(
                turn_id,
                "conversation_handler",
                "agent_interrupt_error",
                client_uid=client_uid,
                error=str(e),
                interrupted=True,
            )

        if context.history_uid:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="ai",
                content=heard_response,
                name=context.character_config.character_name,
                avatar=context.character_config.avatar,
            )
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="system",
                content="[Interrupted by user]",
            )
            record_turn_event(
                turn_id,
                "conversation_handler",
                "interrupt_stored_to_history",
                client_uid=client_uid,
                history_uid=context.history_uid,
                heard_response_len=len(heard_response),
                interrupted=True,
            )


async def handle_group_interrupt(
    group_id: str,
    heard_response: str,
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    chat_group_manager: ChatGroupManager,
    client_contexts: Dict[str, ServiceContext],
    broadcast_to_group: Callable,
    turn_id: str | None = None,
) -> None:
    """Handles interruption for a group conversation"""
    task = current_conversation_tasks.get(group_id)
    if not task or task.done():
        return
    record_turn_event(
        turn_id,
        "conversation_handler",
        "group_interrupt_entered",
        group_id=group_id,
        heard_response_len=len(heard_response),
        interrupted=True,
    )

    # Get state and speaker info before cancellation
    state = GroupConversationState.get_state(group_id)
    current_speaker_uid = state.current_speaker_uid if state else None

    # Get context from current speaker
    context = None
    group = chat_group_manager.get_group_by_id(group_id)
    if current_speaker_uid:
        context = client_contexts.get(current_speaker_uid)
        logger.info(f"Found current speaker context for {current_speaker_uid}")
    if not context and group and group.members:
        logger.warning(f"No context found for group {group_id}, using first member")
        context = client_contexts.get(next(iter(group.members)))

    # Now cancel the task
    task.cancel()
    record_turn_event(
        turn_id,
        "conversation_handler",
        "group_task_cancel_requested",
        group_id=group_id,
        interrupted=True,
    )
    try:
        await task
    except asyncio.CancelledError:
        logger.info(f"🛑 Group conversation {group_id} cancelled successfully.")

    record_turn_event(
        turn_id,
        "conversation_handler",
        "group_task_cancelled",
        group_id=group_id,
        interrupted=True,
    )
    current_conversation_tasks.pop(group_id, None)
    GroupConversationState.remove_state(group_id)  # Clean up state after we've used it

    # Store messages with speaker info
    if context and group:
        for member_uid in group.members:
            if member_uid in client_contexts:
                try:
                    member_ctx = client_contexts[member_uid]
                    member_ctx.agent_engine.handle_interrupt(heard_response)
                    store_message(
                        conf_uid=member_ctx.character_config.conf_uid,
                        history_uid=member_ctx.history_uid,
                        role="ai",
                        content=heard_response,
                        name=context.character_config.character_name,
                        avatar=context.character_config.avatar,
                    )
                    store_message(
                        conf_uid=member_ctx.character_config.conf_uid,
                        history_uid=member_ctx.history_uid,
                        role="system",
                        content="[Interrupted by user]",
                    )
                    record_turn_event(
                        turn_id,
                        "conversation_handler",
                        "group_interrupt_stored_to_history",
                        group_id=group_id,
                        member_uid=member_uid,
                        history_uid=member_ctx.history_uid,
                        heard_response_len=len(heard_response),
                        interrupted=True,
                    )
                except Exception as e:
                    logger.error(f"Error handling interrupt for {member_uid}: {e}")

    await broadcast_to_group(
        list(group.members),
        {
            "type": "interrupt-signal",
            "text": "conversation-interrupted",
            **({"turn_id": turn_id} if turn_id else {}),
        },
    )
