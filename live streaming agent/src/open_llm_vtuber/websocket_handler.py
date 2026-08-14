from typing import Any, Awaitable, Dict, List, Optional, Callable, TypedDict
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import json
import re
import time
import uuid
from enum import Enum
import numpy as np
from loguru import logger

from .service_context import ServiceContext
from .chat_group import (
    ChatGroupManager,
    handle_group_operation,
    handle_client_disconnect,
    broadcast_to_group,
)
from .message_handler import message_handler
from .utils.stream_audio import prepare_audio_payload
from .utils.turn_trace import record_turn_event
from .chat_history_manager import (
    create_new_history,
    get_history,
    delete_history,
    get_history_list,
)
from .story_manager import get_story_state_payload, match_story_and_advance
from .config_manager.utils import scan_config_alts_directory, scan_bg_directory
from .conversations.conversation_handler import (
    handle_conversation_trigger,
    handle_group_interrupt,
    handle_individual_interrupt,
)
from .conversations.conversation_utils import create_batch_input, speak_trigger_prompt_turn
from .painting import get_paint_manager
from .performance_metrics import (
    ensure_performance_turn,
    persist_performance_metrics,
    set_performance_metric,
)
from .project_model_config import ProjectModelConfigManager

STORY_COLD_SILENCE_TRIGGER_NAME = "cold_silence"
FIRST_WAKE_TRIGGER_NAME = "first_wake"
SECOND_WAKE_TRIGGER_NAME = "second_wake"
BARRAGE_SLEEP_TRIGGER_NAME = "error"
STORY_COLD_SILENCE_BARRAGE_THRESHOLD = 2
WAKE_ANIMATION_TIMEOUT_SECONDS = 20.0
GAME_VISION_USER_PROMPT = (
    "\u8bf7\u8bc6\u522b\u8fd9\u5f20\u6e38\u620f\u5c4f\u5e55\u622a\u56fe\uff0c"
    "\u4e3a\u4e3b\u64ad\u7b49\u4f1a\u513f\u56de\u7b54\u95ee\u9898\u63d0\u4f9b\u53ef\u7528\u7684\u753b\u9762\u4e0a\u4e0b\u6587\u3002"
)
LINK_NAME_VISION_USER_PROMPT = (
    "\u8bf7\u4ece\u8fd9\u5f20\u6296\u97f3\u76f4\u64ad\u4f34\u4fa3\u6216\u6296\u97f3\u76f4\u64ad\u95f4\u622a\u56fe\u4e2d\uff0c"
    "\u8bc6\u522b\u6b63\u5728\u8fde\u7ebf/\u8fde\u9ea6/PK \u7684\u5bf9\u65b9\u4e3b\u64ad\u6635\u79f0\u548c\u53ef\u80fd\u7684\u6296\u97f3\u53f7\u3002"
    "\u622a\u56fe\u901a\u5e38\u662f\u8fde\u7ebf\u753b\u9762\u53f3\u4e0b\u89d2\u88c1\u526a\u533a\uff0c"
    "\u4e3b\u64ad\u540d\u5e38\u5728\u5934\u50cf\u9644\u8fd1\uff0c\u53ef\u80fd\u53ea\u67091\u4e2a\u4e2d\u6587\u5b57\u3002"
    "\u4e0d\u8981\u8bc6\u522b\u672c\u5730\u4e3b\u64ad\u6216\u666e\u901a\u89c2\u4f17\u3002"
    "\u4e0d\u8981\u628a\u6211\u65b9\u8d21\u732e\u699c\u3001PK\u8d21\u732e\u699c\u3001\u518d\u6765\u4e00\u5c40\u3001"
    "\u7ed9TA\u70b9\u70b9\u7b49\u529f\u80fd\u6309\u94ae\u5f53\u6210\u4e3b\u64ad\u540d\u3002"
    "\u53ea\u8fd4\u56de JSON\uff0c\u683c\u5f0f\uff1a"
    "{\"nickname\":\"\",\"display_id\":\"\",\"sec_uid\":\"\",\"confidence\":0.0}."
    "\u5982\u679c\u770b\u4e0d\u5230\u660e\u786e\u5bf9\u65b9\u4e3b\u64ad\u540d\uff0c"
    "\u8fd4\u56de {\"nickname\":\"\",\"display_id\":\"\",\"sec_uid\":\"\",\"confidence\":0.0}\u3002"
)
VISION_CONTEXT_MODE_ONE_SHOT = "vision_one_shot"
VISION_CONTEXT_MODE_PERSISTENT = "vision_persistent"
DIRECTOR_METRIC_TO_BARRAGE_VARIABLE = {
    "wealth_level": "wealth",
    "fan_badge_level": "fan_badge",
    "session_diamonds": "diamond_rank",
}


class MessageType(Enum):
    """Enum for WebSocket message types"""

    GROUP = ["add-client-to-group", "remove-client-from-group"]
    HISTORY = [
        "fetch-history-list",
        "fetch-and-set-history",
        "create-new-history",
        "delete-history",
    ]
    CONVERSATION = ["mic-audio-end", "text-input", "ai-speak-signal"]
    CONFIG = ["fetch-configs", "switch-config"]
    CONTROL = ["interrupt-signal", "audio-play-start", "console-message"]
    DATA = ["mic-audio-data"]

def truncate_data(data: Any, max_len: int = 30) -> Any:
    """
    递归处理字典/列表，截断超长字符串
    """
    if isinstance(data, dict):
        return {k: truncate_data(v, max_len) for k, v in data.items()}
    elif isinstance(data, list):
        return [truncate_data(item, max_len) for item in data[:10]]
    elif isinstance(data, str):
        if len(data) > max_len:
            return data[:max_len] + f"...(len:{len(data)})"
        return data
    return data


def websocket_send_log_view(message: Any, max_len: int = 80) -> Any:
    """Return a compact, audio-safe view of an outgoing WebSocket message."""
    if isinstance(message, str):
        try:
            message = json.loads(message)
        except json.JSONDecodeError:
            return truncate_data(message, max_len)

    if isinstance(message, dict):
        result = {}
        for key, value in message.items():
            if key == "audio":
                if isinstance(value, str):
                    result[key] = f"<base64 omitted, chars={len(value)}>"
                elif isinstance(value, list):
                    result[key] = f"<audio omitted, samples={len(value)}>"
                else:
                    result[key] = "<audio omitted>"
            elif key == "volumes":
                result[key] = (
                    f"<volumes omitted, count={len(value)}>"
                    if isinstance(value, list)
                    else "<volumes omitted>"
                )
            else:
                result[key] = websocket_send_log_view(value, max_len)
        return result

    if isinstance(message, list):
        return [websocket_send_log_view(item, max_len) for item in message[:10]]

    if isinstance(message, str):
        return truncate_data(message, max_len)

    return message


class WSMessage(TypedDict, total=False):
    """Type definition for WebSocket messages"""

    type: str
    action: Optional[str]
    text: Optional[str]
    audio: Optional[List[float]]
    images: Optional[List[Dict[str, Any]]]
    metadata: Optional[Dict[str, Any]]
    history_uid: Optional[str]
    file: Optional[str]
    display_text: Optional[dict]
    turn_id: Optional[str]
    request_id: Optional[str]


class ClientConnectionGroup:
    """Multiple physical WebSocket connections attached to one client session."""

    def __init__(self, client_uid: str) -> None:
        self.client_uid = client_uid
        self._connections: list[WebSocket] = []

    @property
    def count(self) -> int:
        return len(self._connections)

    def add(self, websocket: WebSocket) -> None:
        if websocket not in self._connections:
            self._connections.append(websocket)

    def remove(self, websocket: WebSocket) -> None:
        if websocket in self._connections:
            self._connections.remove(websocket)

    def is_empty(self) -> bool:
        return not self._connections

    async def send_text(
        self,
        message: str,
        *,
        connection_filter: Callable[[WebSocket], bool] | None = None,
    ) -> None:
        stale_connections: list[WebSocket] = []
        logger.info(
            "sending websocket message to {}: {}",
            self.client_uid,
            websocket_send_log_view(message),
        )
        for websocket in list(self._connections):
            if connection_filter and not connection_filter(websocket):
                continue
            try:
                await websocket.send_text(message)
            except Exception as exc:
                logger.warning(
                    "Failed to send WebSocket message to {} connection: {}",
                    self.client_uid,
                    exc,
                )
                stale_connections.append(websocket)

        for websocket in stale_connections:
            self.remove(websocket)

    async def send_json(self, message: dict[str, Any]) -> None:
        await self.send_text(json.dumps(message, ensure_ascii=False))


class WebSocketHandler:
    """Handles WebSocket connections and message routing"""

    def __init__(self, default_context_cache: ServiceContext):
        """Initialize the WebSocket handler with default context"""
        self.client_connections: Dict[str, ClientConnectionGroup] = {}
        self.display_client_modes: dict[int, str] = {}
        self.client_contexts: Dict[str, ServiceContext] = {}
        self.chat_group_manager = ChatGroupManager()
        self.current_conversation_tasks: Dict[str, Optional[asyncio.Task]] = {}
        self.current_turn_ids: Dict[str, str] = {}
        self.default_context_cache = default_context_cache
        self.received_data_buffers: Dict[str, np.ndarray] = {}
        self.mic_asr_tasks: Dict[str, list[asyncio.Task[str]]] = {}
        self.mic_asr_locks: Dict[str, asyncio.Lock] = {}
        self.mic_asr_elapsed_seconds: Dict[str, float] = {}
        self.game_vision_captures: Dict[str, dict[str, Any]] = {}
        self.vision_image_contexts: Dict[str, dict[str, Any]] = {}
        self.clients_needing_memory_reload: set[str] = set()
        self.proactive_idle_seconds: float = 100
        self._proactive_last_activity_at: float = time.monotonic()
        self._proactive_timer_task: Optional[asyncio.Task] = None
        self._consecutive_cold_silence_triggers = 0
        self.display_live2d_open = True
        self.display_microphone_enabled = True
        self.display_link_microphone_enabled = False
        self.display_link_microphone_faulted = False
        self.display_link_microphone_pending = False
        self.display_link_microphone_confirmed = False
        self.display_link_human_name = "\u8fde\u7ebf\u4e3b\u64ad"
        self.display_gift_thanks_enabled = False
        self.display_live_streaming_agent_subtitle_enabled = False
        self.display_barrage_subtitle_enabled = False
        self.display_game_vision_enabled = False
        self.display_paint_enabled = False
        self.display_wake_animation_pending = False
        self.pending_wake_triggers: Dict[str, dict[str, Any]] = {}
        self.wake_animation_timeout_tasks: Dict[str, asyncio.Task] = {}
        self.project_model_manager: ProjectModelConfigManager | None = None
        self._project_model_config_lock = asyncio.Lock()

        # Message handlers mapping
        self._message_handlers = self._init_message_handlers()

    def _init_message_handlers(self) -> Dict[str, Callable]:
        """Initialize message type to handler mapping"""
        return {
            "add-client-to-group": self._handle_group_operation,
            "remove-client-from-group": self._handle_group_operation,
            "request-group-info": self._handle_group_info,
            "fetch-history-list": self._handle_history_list_request,
            "fetch-and-set-history": self._handle_fetch_history,
            "create-new-history": self._handle_create_history,
            "delete-history": self._handle_delete_history,
            "interrupt-signal": self._handle_interrupt,
            "console-message": self._handle_console_message,
            "mic-audio-data": self._handle_audio_data,
            "mic-audio-segment-end": self._handle_mic_audio_segment_end,
            "mic-audio-end": self._handle_conversation_trigger,
            "game-vision-capture": self._handle_game_vision_capture,
            "link-name-vision-capture": self._handle_link_name_vision_capture,
            "raw-audio-data": self._handle_raw_audio_data,
            "text-input": self._handle_conversation_trigger,
            "ai-speak-signal": self._handle_conversation_trigger,
            "fetch-configs": self._handle_fetch_configs,
            "switch-config": self._handle_config_switch,
            "fetch-backgrounds": self._handle_fetch_backgrounds,
            "audio-play-start": self._handle_audio_play_start,
            "frontend-playback-complete": self._handle_frontend_playback_complete,
            "performance-monitor-sync": self._handle_performance_monitor_sync,
            "project-config-request": self._handle_project_config_request,
            "project-config-update": self._handle_project_config_update,
            "project-config-test": self._handle_project_config_test,
            "request-init-config": self._handle_init_config_request,
            "heartbeat": self._handle_heartbeat,
        }

    async def handle_new_connection(
        self, websocket: WebSocket, client_uid: str
    ) -> None:
        """
        Handle new WebSocket connection setup

        Args:
            websocket: The WebSocket connection
            client_uid: Unique identifier for the client

        Raises:
            Exception: If initialization fails
        """
        try:
            if client_uid in self.client_contexts:
                session_service_context = self.client_contexts[client_uid]
            else:
                session_service_context = await self._init_service_context(
                    self._make_client_sender(client_uid), client_uid
                )

            await self._store_client_data(
                websocket, client_uid, session_service_context
            )

            await self._send_initial_messages(
                websocket, client_uid, session_service_context
            )

            connection_group = self.client_connections[client_uid]
            logger.info(
                "Connection established for client {} ({} active connection{})",
                client_uid,
                connection_group.count,
                "" if connection_group.count == 1 else "s",
            )
            self._mark_proactive_activity("client-connected")

        except Exception as e:
            logger.error(
                f"Failed to initialize connection for client {client_uid}: {e}"
            )
            await self._cleanup_failed_connection(client_uid, websocket)
            raise

    async def _store_client_data(
        self,
        websocket: WebSocket,
        client_uid: str,
        session_service_context: ServiceContext,
    ):
        """Store client data and initialize group status"""
        if client_uid not in self.client_connections:
            self.client_connections[client_uid] = ClientConnectionGroup(client_uid)
        self.client_connections[client_uid].add(websocket)

        self.client_contexts.setdefault(client_uid, session_service_context)
        self.received_data_buffers.setdefault(client_uid, np.array([]))
        self.chat_group_manager.client_group_map.setdefault(client_uid, "")
        await self.send_group_update(websocket, client_uid)

    def _make_client_sender(self, client_uid: str) -> Callable[[str], Awaitable[None]]:
        async def send_text(message: str) -> None:
            await self._send_text_to_client(client_uid, message)

        return send_text

    async def _send_text_to_client(self, client_uid: str, message: str) -> None:
        connection_group = self.client_connections.get(client_uid)
        if not connection_group or connection_group.is_empty():
            logger.warning("No active WebSocket connections for client {}", client_uid)
            return

        await connection_group.send_text(message)

    async def _send_text_to_display_mode(
        self,
        client_uid: str,
        message: str,
        display_mode: str,
    ) -> None:
        connection_group = self.client_connections.get(client_uid)
        if not connection_group or connection_group.is_empty():
            return
        await connection_group.send_text(
            message,
            connection_filter=lambda connection: self.display_client_modes.get(
                id(connection)
            )
            == display_mode,
        )

    async def _request_performance_monitor_snapshot(self, client_uid: str) -> None:
        await self._send_text_to_display_mode(
            client_uid,
            json.dumps({"type": "performance-monitor-request"}),
            "streamer",
        )

    def _get_message_turn_id(self, data: dict[str, Any]) -> str | None:
        return data.get("turn_id") or data.get("request_id")

    def _new_turn_id(self) -> str:
        return uuid.uuid4().hex

    def _current_vtuber_state_payload(self) -> dict[str, Any] | None:
        from .vtuber_state_machine import VTuberMode, get_vtuber_state_machine

        sm = get_vtuber_state_machine()
        if sm is None:
            return None

        return {
            "mode": sm.mode.value,
            "sub_mode": (
                sm.idle_sub_mode.value if sm.mode == VTuberMode.IDLE else None
            ),
            "interaction_mode": sm.interaction_mode.value,
            "sleeping": sm.sleeping,
            "punished": sm.punished,
            "wake_animation_pending": self.display_wake_animation_pending,
        }

    def _story_state_for_context(self, context: ServiceContext) -> dict[str, Any]:
        if not context.history_uid:
            return {
                "has_story": False,
                "progress_index": 0,
                "total": 0,
                "items": [],
            }
        return get_story_state_payload(
            context.character_config.conf_uid,
            context.history_uid,
        )

    def _model_and_conf_payload(
        self,
        context: ServiceContext,
        client_uid: str,
    ) -> dict[str, Any]:
        return {
            "type": "set-model-and-conf",
            "model_info": context.live2d_model.model_info,
            "conf_name": context.character_config.conf_name,
            "conf_uid": context.character_config.conf_uid,
            "client_uid": client_uid,
            "vtuber_state": self._current_vtuber_state_payload(),
            "proactive_idle_seconds": self.proactive_idle_seconds,
            "story_state": self._story_state_for_context(context),
            "display_state": self._display_state_payload(),
        }

    def _display_state_payload(self) -> dict[str, Any]:
        return {
            "live2d_open": self.display_live2d_open,
            "microphone_enabled": self.display_microphone_enabled,
            "link_microphone_enabled": self.display_link_microphone_enabled,
            "link_microphone_faulted": self.display_link_microphone_faulted,
            "link_microphone_pending": self.display_link_microphone_pending,
            "link_microphone_confirmed": self.display_link_microphone_confirmed,
            "link_human_name": self.display_link_human_name,
            "gift_thanks_enabled": self.display_gift_thanks_enabled,
            "live_streaming_agent_subtitle_enabled": self.display_live_streaming_agent_subtitle_enabled,
            "barrage_subtitle_enabled": self.display_barrage_subtitle_enabled,
            "game_vision_enabled": self.display_game_vision_enabled,
            "paint_enabled": self.display_paint_enabled,
        }

    def _mode_change_payload(self, result: dict[str, Any]) -> dict[str, Any]:
        detail = {
            **result,
            "wake_animation_pending": self.display_wake_animation_pending,
        }
        return {
            "type": "mode-changed",
            "mode": result["new_mode"],
            "sub_mode": result.get("sub_mode"),
            "interaction_mode": result.get("interaction_mode"),
            "sleeping": result.get("sleeping", False),
            "punished": result.get("punished", False),
            "wake_animation_pending": self.display_wake_animation_pending,
            "detail": detail,
        }

    async def _broadcast_current_vtuber_state(self, reason: str) -> None:
        state = self._current_vtuber_state_payload()
        if not state:
            return

        message = json.dumps(
            {
                "type": "mode-changed",
                "mode": state["mode"],
                "sub_mode": state.get("sub_mode"),
                "interaction_mode": state.get("interaction_mode"),
                "sleeping": state.get("sleeping", False),
                "punished": state.get("punished", False),
                "wake_animation_pending": self.display_wake_animation_pending,
                "detail": {
                    **state,
                    "new_mode": state["mode"],
                    "wake_animation_pending": self.display_wake_animation_pending,
                    "reason": reason,
                },
            },
            ensure_ascii=False,
        )
        for uid in list(self.client_connections.keys()):
            await self._send_text_to_client(uid, message)

    async def _broadcast_display_control(self, control_text: str) -> None:
        message = json.dumps(
            {
                "type": "control",
                "text": control_text,
                "display_state": self._display_state_payload(),
            },
            ensure_ascii=False,
        )
        for uid in list(self.client_connections.keys()):
            await self._send_text_to_client(uid, message)

    @staticmethod
    def _normalize_display_client_mode(mode: Any) -> str | None:
        value = str(mode or "").strip().lower()
        if value in {"streamer", "anchor", "host", "live", "\u4e3b\u64ad"}:
            return "streamer"
        if value in {
            "director",
            "producer",
            "control",
            "controller",
            "\u7f16\u5bfc",
            "\u5bfc\u64ad",
        }:
            return "director"
        return None

    def _has_streamer_display_client(self) -> bool:
        return any(mode == "streamer" for mode in self.display_client_modes.values())

    async def _forget_display_client_mode(self, websocket: WebSocket | None) -> None:
        if websocket is None:
            return
        mode = self.display_client_modes.pop(id(websocket), None)
        if mode != "streamer":
            return
        if not self.display_link_microphone_enabled:
            return
        if self._has_streamer_display_client():
            return
        self.display_link_microphone_faulted = True
        self.display_link_microphone_pending = False
        self.display_link_microphone_confirmed = False
        await self._broadcast_display_control("link-mic-fault")
        logger.warning(
            "[console] link microphone marked faulted because streamer display disconnected"
        )

    def _match_story_for_turn(
        self,
        *,
        context: ServiceContext,
        user_text: str,
        story_candidates: Any,
    ) -> dict[str, Any] | None:
        if not context.history_uid:
            return None
        return match_story_and_advance(
            conf_uid=context.character_config.conf_uid,
            history_uid=context.history_uid,
            translated_user_text=user_text,
            story_candidates=story_candidates,
        )

    def _is_vtuber_active_for_proactive_speak(self) -> bool:
        from .vtuber_state_machine import VTuberMode, get_vtuber_state_machine

        sm = get_vtuber_state_machine()
        return sm is not None and sm.mode != VTuberMode.IDLE

    def _is_vtuber_accepting_input(self) -> bool:
        from .vtuber_state_machine import VTuberMode, get_vtuber_state_machine

        sm = get_vtuber_state_machine()
        return sm is None or sm.mode != VTuberMode.IDLE

    def _is_story_interaction_mode(self) -> bool:
        from .vtuber_state_machine import VTuberMode, get_vtuber_state_machine

        sm = get_vtuber_state_machine()
        return sm is not None and sm.interaction_mode == VTuberMode.CO_HOST

    def _is_barrage_interaction_mode(self) -> bool:
        from .vtuber_state_machine import VTuberMode, get_vtuber_state_machine

        sm = get_vtuber_state_machine()
        return sm is not None and sm.interaction_mode == VTuberMode.BARRAGE

    def _context_has_story(self, context: ServiceContext) -> bool:
        return bool(self._story_state_for_context(context).get("has_story"))

    def _is_current_working_history_empty(self, context: ServiceContext) -> bool:
        if not context.history_uid:
            return True
        return not get_history(
            context.character_config.conf_uid,
            context.history_uid,
        )

    def _wake_trigger_name_for_context(self, context: ServiceContext) -> str:
        if self._is_current_working_history_empty(context):
            return FIRST_WAKE_TRIGGER_NAME
        return SECOND_WAKE_TRIGGER_NAME

    def _reset_cold_silence_streak(self, reason: str) -> None:
        if self._consecutive_cold_silence_triggers:
            logger.debug(
                "Resetting cold silence streak by {}: previous_count={}",
                reason,
                self._consecutive_cold_silence_triggers,
            )
        self._consecutive_cold_silence_triggers = 0

    def _start_trigger_prompt_task(
        self,
        *,
        client_uid: str,
        trigger_name: str,
        turn_id: str,
        reset_proactive_on_done: bool = True,
    ) -> asyncio.Task | None:
        context = self.client_contexts[client_uid]
        websocket_send = context.send_text
        if websocket_send is None:
            connection_group = self.client_connections.get(client_uid)
            if not connection_group:
                logger.warning(
                    "Cannot start trigger prompt task; no connection for {}",
                    client_uid,
                )
                return None
            websocket_send = connection_group.send_text

        task_key = self._conversation_task_key(client_uid)
        task = asyncio.create_task(
            speak_trigger_prompt_turn(
                trigger_name=trigger_name,
                context=context,
                websocket_send=websocket_send,
                client_uid=client_uid,
                turn_id=turn_id,
            )
        )
        self.current_conversation_tasks[task_key] = task
        record_turn_event(
            turn_id,
            "websocket_handler",
            "trigger_prompt_task_created",
            client_uid=client_uid,
            task_key=task_key,
            trigger_name=trigger_name,
        )
        if reset_proactive_on_done:
            self._attach_proactive_task_callback(client_uid)
        return task

    async def _set_wake_animation_pending(self, active: bool, reason: str) -> None:
        changed = self.display_wake_animation_pending != active
        self.display_wake_animation_pending = active
        if changed:
            logger.info(
                "Display wake animation state changed: active={} reason={}",
                active,
                reason,
            )
            await self._broadcast_current_vtuber_state(reason)

    def _queue_pending_wake_trigger(
        self,
        *,
        client_uid: str,
        trigger_name: str,
        turn_id: str,
    ) -> None:
        old_task = self.wake_animation_timeout_tasks.pop(client_uid, None)
        if old_task and not old_task.done():
            old_task.cancel()

        self.pending_wake_triggers[client_uid] = {
            "trigger_name": trigger_name,
            "turn_id": turn_id,
        }
        self.wake_animation_timeout_tasks[client_uid] = asyncio.create_task(
            self._wake_animation_timeout(client_uid, turn_id)
        )
        record_turn_event(
            turn_id,
            "websocket_handler",
            "wake_trigger_waiting_for_animation",
            client_uid=client_uid,
            trigger_name=trigger_name,
            timeout_seconds=WAKE_ANIMATION_TIMEOUT_SECONDS,
        )

    async def _wake_animation_timeout(self, client_uid: str, turn_id: str) -> None:
        try:
            await asyncio.sleep(WAKE_ANIMATION_TIMEOUT_SECONDS)
            pending = self.pending_wake_triggers.get(client_uid)
            if not pending:
                if self.display_wake_animation_pending:
                    logger.warning(
                        "Wake animation completion timed out for {} turn_id={}; clearing wake animation state",
                        client_uid,
                        turn_id,
                    )
                    await self._set_wake_animation_pending(
                        False,
                        "wake-animation-timeout",
                    )
                return

            if pending.get("turn_id") != turn_id:
                return

            logger.warning(
                "Wake animation start timed out for {} turn_id={}; starting pending trigger",
                client_uid,
                turn_id,
            )
            record_turn_event(
                turn_id,
                "websocket_handler",
                "wake_animation_start_timeout",
                client_uid=client_uid,
                timeout_seconds=WAKE_ANIMATION_TIMEOUT_SECONDS,
            )
            await self._set_wake_animation_pending(False, "wake-animation-timeout")
            await self._start_pending_wake_trigger(
                client_uid,
                reason="wake-animation-timeout",
            )
        except asyncio.CancelledError:
            return

    async def _start_pending_wake_trigger(
        self,
        client_uid: str,
        reason: str,
        *,
        keep_timeout: bool = False,
    ) -> None:
        pending_client_uid = client_uid
        pending = self.pending_wake_triggers.pop(client_uid, None)
        if not pending:
            for queued_uid, queued_pending in list(
                self.pending_wake_triggers.items()
            ):
                pending_client_uid = queued_uid
                pending = queued_pending
                self.pending_wake_triggers.pop(queued_uid, None)
                break

        if not keep_timeout:
            timeout_task = self.wake_animation_timeout_tasks.pop(
                pending_client_uid,
                None,
            )
            current = asyncio.current_task()
            if (
                timeout_task
                and timeout_task is not current
                and not timeout_task.done()
            ):
                timeout_task.cancel()
        if not pending:
            return

        from .vtuber_state_machine import get_vtuber_state_machine

        sm = get_vtuber_state_machine()
        if sm is not None and (sm.sleeping or sm.punished):
            record_turn_event(
                pending.get("turn_id"),
                "websocket_handler",
                "wake_trigger_cancelled_before_start",
                client_uid=pending_client_uid,
                reason=reason,
                sleeping=sm.sleeping,
                punished=sm.punished,
            )
            return

        turn_id = str(pending["turn_id"])
        trigger_name = str(pending["trigger_name"])
        if pending_client_uid not in self.client_contexts:
            logger.warning(
                "Cannot start pending wake trigger; client {} is disconnected",
                pending_client_uid,
            )
            record_turn_event(
                turn_id,
                "websocket_handler",
                "wake_trigger_cancelled_before_start",
                client_uid=pending_client_uid,
                trigger_name=trigger_name,
                reason="client-disconnected",
            )
            return

        record_turn_event(
            turn_id,
            "websocket_handler",
            "wake_trigger_animation_ready",
            client_uid=pending_client_uid,
            trigger_name=trigger_name,
            reason=reason,
        )
        self._start_trigger_prompt_task(
            client_uid=pending_client_uid,
            trigger_name=trigger_name,
            turn_id=turn_id,
        )

    async def cancel_pending_wake_animation(self, reason: str) -> None:
        for task in list(self.wake_animation_timeout_tasks.values()):
            if task and not task.done():
                task.cancel()
        self.wake_animation_timeout_tasks.clear()
        self.pending_wake_triggers.clear()
        await self._set_wake_animation_pending(False, reason)

    def _is_wake_transition(
        self,
        action: str,
        result: dict[str, Any],
    ) -> bool:
        return (
            action == "sleep"
            and result.get("old_mode") == "idle"
            and result.get("new_mode") != "idle"
            and not result.get("sleeping", True)
        )

    def _mic_source_from_data(self, data: dict[str, Any] | None) -> str:
        if not isinstance(data, dict):
            return "local"
        source = str(data.get("mic_source") or data.get("audio_source") or "local")
        source = source.strip().lower()
        return source or "local"

    def _mic_state_key(self, client_uid: str, mic_source: str | None = None) -> str:
        source = (mic_source or "local").strip().lower() or "local"
        return client_uid if source == "local" else f"{client_uid}:{source}"

    def _game_vision_state_key(
        self,
        client_uid: str,
        mic_source: str | None = None,
    ) -> str:
        return f"{self._mic_state_key(client_uid, mic_source)}:game_vision"

    def _clear_game_vision_state(self, client_uid: str, reason: str) -> None:
        prefix = f"{client_uid}:"
        keys = {
            key
            for key in self.game_vision_captures.keys()
            if key.startswith(prefix)
        }
        for key in keys:
            self.game_vision_captures.pop(key, None)
        if keys:
            logger.info(
                "Cleared game vision state for {} because {}: keys={}",
                client_uid,
                reason,
                sorted(keys),
            )

    async def _send_game_vision_state(
        self,
        client_uid: str,
        state: str,
        *,
        request_id: str | None = None,
        provider: str | None = None,
        message: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "type": "game-vision-state",
            "state": state,
        }
        if request_id:
            payload["request_id"] = request_id
        if provider:
            payload["provider"] = provider
        if message:
            payload["message"] = message
        await self._send_text_to_client(
            client_uid,
            json.dumps(payload, ensure_ascii=False),
        )

    def _resolved_game_vision_provider(
        self,
        context: ServiceContext,
        requested_provider: str | None,
    ) -> str:
        agent = context.agent_engine
        vision_llms = getattr(agent, "_vision_llms", {}) or {}
        if requested_provider and requested_provider in vision_llms:
            return requested_provider

        default_provider = getattr(agent, "_default_vision_llm_provider", None)
        if default_provider and default_provider in vision_llms:
            return str(default_provider)

        if vision_llms:
            return str(next(iter(vision_llms)))

        return requested_provider or "main_llm"

    def _clear_visual_image_context(self, client_uid: str, reason: str) -> None:
        context = self.vision_image_contexts.pop(client_uid, None)
        if context:
            logger.info(
                "Cleared visual image context for {} because {}: provider={} image_name={}",
                client_uid,
                reason,
                context.get("provider"),
                context.get("image_name"),
            )

    def _validated_visual_image_payloads(
        self,
        images: Any,
        *,
        provider: str | None = None,
    ) -> list[dict[str, Any]]:
        try:
            batch_input = create_batch_input(
                input_text="",
                images=images,
                from_name="visual_image",
                metadata=(
                    {"vision_model_provider": provider}
                    if provider
                    else None
                ),
                include_human_name_prefix=False,
            )
        except Exception as exc:
            logger.warning("Visual image context validation failed: {}", exc)
            return []

        return [
            {
                "source": image.source.value,
                "data": image.data,
                "mime_type": image.mime_type,
            }
            for image in batch_input.images or []
        ]

    def _prepare_visual_image_context_for_reply(
        self,
        client_uid: str,
        turn_id: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        if data.get("type") not in {"mic-audio-end", "text-input"}:
            return data

        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        images = list(data.get("images") or [])
        mode = str(metadata.get("vision_context_mode") or "").strip()
        is_manual_visual_upload = bool(metadata.get("visual_image_attached")) and bool(
            images
        )

        if is_manual_visual_upload:
            if mode == VISION_CONTEXT_MODE_ONE_SHOT:
                self._clear_visual_image_context(client_uid, "new one-shot visual image")
                return data

            if mode != VISION_CONTEXT_MODE_PERSISTENT:
                return data

            provider = str(metadata.get("vision_model_provider") or "").strip() or None
            context = self.client_contexts.get(client_uid)
            if context:
                provider = self._resolved_game_vision_provider(context, provider)

            stored_images = self._validated_visual_image_payloads(
                images,
                provider=provider,
            )
            if not stored_images:
                logger.warning(
                    "Persistent visual image context was requested but no valid images were supplied: client={}",
                    client_uid,
                )
                return data

            image_name = str(metadata.get("vision_image_name") or "").strip() or None
            self.vision_image_contexts[client_uid] = {
                "provider": provider,
                "images": stored_images,
                "image_name": image_name,
                "created_at": time.monotonic(),
            }

            merged_metadata = {
                **metadata,
                "vision_context_mode": VISION_CONTEXT_MODE_PERSISTENT,
            }
            if provider:
                merged_metadata["vision_model_provider"] = provider

            next_data = dict(data)
            next_data["metadata"] = merged_metadata
            record_turn_event(
                turn_id,
                "websocket_handler",
                "visual_image_context_stored",
                client_uid=client_uid,
                provider=provider,
                image_name=image_name,
                image_count=len(stored_images),
            )
            logger.info(
                "Stored persistent visual image context: client={} provider={} image_name={} images={}",
                client_uid,
                provider,
                image_name,
                len(stored_images),
            )
            return next_data

        if images or bool(metadata.get("skip_visual_context")):
            return data

        context = self.vision_image_contexts.get(client_uid)
        if not context:
            return data

        stored_images = list(context.get("images") or [])
        if not stored_images:
            self._clear_visual_image_context(client_uid, "empty stored visual context")
            return data

        provider = str(context.get("provider") or "").strip() or None
        image_name = str(context.get("image_name") or "").strip() or None
        merged_metadata = {
            **metadata,
            "vision_context_mode": VISION_CONTEXT_MODE_PERSISTENT,
            "vision_context_reused": True,
            "vision_image_name": image_name,
        }
        if provider:
            merged_metadata["vision_model_provider"] = provider

        next_data = dict(data)
        next_data["images"] = stored_images
        next_data["metadata"] = merged_metadata
        record_turn_event(
            turn_id,
            "websocket_handler",
            "visual_image_context_reused",
            client_uid=client_uid,
            provider=provider,
            image_name=image_name,
            image_count=len(stored_images),
        )
        logger.info(
            "Reused persistent visual image context for turn: client={} provider={} image_name={} images={}",
            client_uid,
            provider,
            image_name,
            len(stored_images),
        )
        return next_data

    async def _handle_game_vision_capture(
        self,
        websocket: WebSocket,
        client_uid: str,
        data: WSMessage,
    ) -> None:
        mic_source = self._mic_source_from_data(data)
        state_key = self._game_vision_state_key(client_uid, mic_source)
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        request_id = str(
            data.get("request_id")
            or metadata.get("game_vision_request_id")
            or self._new_turn_id()
        )
        provider = str(metadata.get("vision_model_provider") or "").strip() or None
        context = self.client_contexts.get(client_uid)
        if context:
            provider = self._resolved_game_vision_provider(context, provider)

        self.game_vision_captures.pop(state_key, None)

        try:
            batch_input = create_batch_input(
                input_text=GAME_VISION_USER_PROMPT,
                images=data.get("images"),
                from_name="game_screen",
                metadata={"vision_model_provider": provider},
                include_human_name_prefix=False,
            )
            if not batch_input.images:
                raise RuntimeError("No valid game screenshot image was received.")

            stored_images = [
                {
                    "source": image.source.value,
                    "data": image.data,
                    "mime_type": image.mime_type,
                }
                for image in batch_input.images
            ]
        except Exception as exc:
            await self._send_game_vision_state(
                client_uid,
                "error",
                request_id=request_id,
                provider=provider,
                message=f"\u6e38\u620f\u8bc6\u56fe\u5931\u8d25\uff1a{exc}",
            )
            logger.warning(
                "Game vision screenshot rejected: client={} source={} request_id={} error={}",
                client_uid,
                mic_source,
                request_id,
                exc,
            )
            return

        self.game_vision_captures[state_key] = {
            "request_id": request_id,
            "provider": provider,
            "images": stored_images,
            "created_at": time.monotonic(),
        }
        await self._send_game_vision_state(
            client_uid,
            "started",
            request_id=request_id,
            provider=provider,
        )
        logger.info(
            "Stored game vision screenshot for final visual reply: client={} source={} request_id={} provider={} images={}",
            client_uid,
            mic_source,
            request_id,
            provider,
            len(stored_images),
        )

    @staticmethod
    def _normalize_link_name_vision_value(value: Any) -> str | None:
        text = str(value or "").strip()
        if not text:
            return None
        text = re.sub(r"[\r\n\t]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip(" @:：|-_/\\")
        if not text or len(text) > 64:
            return None
        if len(text) < 2 and not re.fullmatch(r"[\u4e00-\u9fff]", text):
            return None
        lowered = text.lower()
        if lowered in {
            "unknown",
            "none",
            "null",
            "\u65e0",
            "\u672a\u77e5",
            "\u770b\u4e0d\u5230",
            "\u8fde\u7ebf\u4e3b\u64ad",
            "\u4e3b\u64ad",
            "pk\u8fde\u7ebf",
            "\u8fde\u7ebfpk",
            "\u4e0e",
            "\u548c",
            "\u9000\u51fapk",
            "\u6bd4\u62fc\u65b9\u5f0f",
            "\u5e38\u89c4pk",
            "\u8fdb\u884c\u4e2d",
            "\u66f4\u591a\u73a9\u6cd5",
            "\u7acb\u5373\u5339\u914d",
            "\u968f\u673a\u5339\u914d",
            "\u8bf4\u70b9\u4ec0\u4e48",
            "\u53d1\u9001",
            "\u6211\u65b9\u8d21\u732e\u699c",
            "\u8d21\u732e\u699c",
            "pk\u8d21\u732e\u699c",
            "\u798f\u888b",
            "\u793c\u7269\u83dc\u5355",
            "\u5ba0\u7c89",
            "\u5ba0\u7c89\u7ea2\u5305",
            "\u5728\u7ebf\u89c2\u4f17\u699c",
            "\u672c\u573a\u89c2\u4f17\u699c",
            "\u518d\u6765\u4e00\u5c40",
            "\u7ed9ta\u70b9\u70b9",
            "\u6444\u50cf\u5934\u5e03\u5c40",
            "\u8fde\u7ebf\u8bbe\u7f6e",
            "\u5c0f\u8377\u699c",
            "pk\u7ed3\u675f",
            "\u5e73\u5c40",
            "\u5173\u64ad",
            "\u4e3b\u64ad\u4e2d\u5fc3",
            "\u663e\u793a\u5668",
            "\u6dfb\u52a0\u7d20\u6750",
            "obs64.exe",
            "chrome legacy window",
        }:
            return None
        if any(
            part in lowered
            for part in (
                "pk\u8fde\u7ebf",
                "\u8fde\u7ebf",
                "\u9000\u51fapk",
                "\u6bd4\u62fc\u65b9\u5f0f",
                "\u5e38\u89c4pk",
                "\u8fdb\u884c\u4e2d",
                "\u66f4\u591a\u73a9\u6cd5",
                "\u7acb\u5373\u5339\u914d",
                "\u968f\u673a\u5339\u914d",
                "\u8bf4\u70b9\u4ec0\u4e48",
                "\u8d21\u732e\u699c",
                "\u793c\u7269\u83dc\u5355",
                "\u5ba0\u7c89",
                "\u798f\u888b",
                "\u6e38\u620f\u80fd\u529b",
                "\u7559\u8a00",
                "\u7ed9ta\u70b9",
                "websocket",
                "chrome legacy",
            )
        ):
            return None
        return text

    def _parse_link_name_vision_response(self, text: str) -> dict[str, Any]:
        raw_text = str(text or "").strip()
        if not raw_text:
            return {}

        json_text = raw_text
        match = re.search(r"\{.*\}", raw_text, flags=re.S)
        if match:
            json_text = match.group(0)

        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError:
            payload = {}

        if isinstance(payload, dict):
            nickname = self._normalize_link_name_vision_value(
                payload.get("nickname")
                or payload.get("name")
                or payload.get("\u6635\u79f0")
            )
            display_id = self._normalize_link_name_vision_value(
                payload.get("display_id")
                or payload.get("douyin_id")
                or payload.get("\u6296\u97f3\u53f7")
            )
            sec_uid = self._normalize_link_name_vision_value(
                payload.get("sec_uid")
            )
            try:
                confidence = float(payload.get("confidence") or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            return {
                "nickname": nickname,
                "display_id": display_id,
                "sec_uid": sec_uid,
                "confidence": max(0.0, min(confidence, 1.0)),
                "raw_response": raw_text,
            }

        fallback = self._normalize_link_name_vision_value(raw_text)
        if fallback:
            return {
                "nickname": fallback,
                "confidence": 0.45,
                "raw_response": raw_text,
            }
        return {"raw_response": raw_text}

    async def _handle_link_name_vision_capture(
        self,
        websocket: WebSocket,
        client_uid: str,
        data: WSMessage,
    ) -> None:
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        request_id = str(data.get("request_id") or self._new_turn_id())
        context = self.client_contexts.get(client_uid)
        provider = str(metadata.get("vision_model_provider") or "").strip() or None
        if not context:
            await self._send_text_to_client(
                client_uid,
                json.dumps(
                    {
                        "type": "link-microphone-name-detect",
                        "request_id": request_id,
                        "found": False,
                        "source": "vision_model",
                        "reason": "client context is not ready",
                    },
                    ensure_ascii=False,
                ),
            )
            return

        provider = self._resolved_game_vision_provider(context, provider)
        agent = context.agent_engine
        vision_llms = getattr(agent, "_vision_llms", {}) or {}
        active_llm = vision_llms.get(provider)
        if active_llm is None and vision_llms:
            provider, active_llm = next(iter(vision_llms.items()))

        if active_llm is None:
            await self._send_text_to_client(
                client_uid,
                json.dumps(
                    {
                        "type": "link-microphone-name-detect",
                        "request_id": request_id,
                        "found": False,
                        "source": "vision_model",
                        "reason": "no visual LLM is configured",
                    },
                    ensure_ascii=False,
                ),
            )
            logger.warning("Link name vision skipped: no visual LLM configured")
            return

        try:
            batch_input = create_batch_input(
                input_text=LINK_NAME_VISION_USER_PROMPT,
                images=data.get("images"),
                from_name="link_anchor_screenshot",
                metadata={"vision_model_provider": provider},
                include_human_name_prefix=False,
            )
            if not batch_input.images:
                raise RuntimeError("No valid screenshot image was received.")

            image = batch_input.images[0]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": LINK_NAME_VISION_USER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image.data,
                                "detail": "high",
                            },
                        },
                    ],
                }
            ]
            chunks: list[str] = []
            async for chunk in active_llm.chat_completion(
                messages,
                system=(
                    "\u4f60\u662f\u4e00\u4e2a\u53ea\u8f93\u51fa JSON \u7684\u89c6\u89c9\u8bc6\u522b\u5668\u3002"
                    "\u4e0d\u8981\u89e3\u91ca\uff0c\u4e0d\u8981\u8f93\u51fa markdown\u3002"
                ),
                call_source="link_name_vision",
            ):
                if isinstance(chunk, str):
                    chunks.append(chunk)

            response_text = "".join(chunks).strip()
            parsed = self._parse_link_name_vision_response(response_text)
            candidate = (
                parsed.get("nickname")
                or parsed.get("display_id")
                or parsed.get("sec_uid")
            )
            confidence = parsed.get("confidence")
            found = bool(candidate)
            await self._send_text_to_client(
                client_uid,
                json.dumps(
                    {
                        "type": "link-microphone-name-detect",
                        "request_id": request_id,
                        "found": found,
                        "candidate": candidate,
                        "nickname": parsed.get("nickname"),
                        "display_id": parsed.get("display_id"),
                        "sec_uid": parsed.get("sec_uid"),
                        "source": "vision_model",
                        "confidence": confidence,
                        "provider": provider,
                        "reason": None if found else "vision model returned no clear name",
                        "raw_response": parsed.get("raw_response"),
                    },
                    ensure_ascii=False,
                ),
            )
            logger.info(
                "Link name vision result: client={} request_id={} provider={} found={} candidate={} response={}",
                client_uid,
                request_id,
                provider,
                found,
                candidate,
                truncate_data(response_text, 120),
            )
        except Exception as exc:
            await self._send_text_to_client(
                client_uid,
                json.dumps(
                    {
                        "type": "link-microphone-name-detect",
                        "request_id": request_id,
                        "found": False,
                        "source": "vision_model",
                        "provider": provider,
                        "reason": str(exc),
                    },
                    ensure_ascii=False,
                ),
            )
            logger.warning(
                "Link name vision failed: client={} request_id={} provider={} error={}",
                client_uid,
                request_id,
                provider,
                exc,
            )

    def _mic_state_keys_for_client(
        self,
        client_uid: str,
        mic_source: str | None = None,
    ) -> list[str]:
        if mic_source is not None:
            return [self._mic_state_key(client_uid, mic_source)]

        prefix = f"{client_uid}:"
        keys = {client_uid}
        keys.update(
            key
            for key in self.received_data_buffers.keys()
            if key.startswith(prefix)
        )
        keys.update(key for key in self.mic_asr_tasks.keys() if key.startswith(prefix))
        keys.update(key for key in self.mic_asr_locks.keys() if key.startswith(prefix))
        keys.update(
            key for key in self.mic_asr_elapsed_seconds.keys() if key.startswith(prefix)
        )
        return list(keys)

    def _clear_received_audio_buffer(
        self,
        client_uid: str,
        reason: str,
        mic_source: str | None = None,
    ) -> None:
        keys = self._mic_state_keys_for_client(client_uid, mic_source)
        for key in keys:
            existing = self.received_data_buffers.get(key)
            if existing is not None and len(existing):
                logger.info(
                    "Clearing buffered microphone audio for {} because {}: samples={}",
                    key,
                    reason,
                    len(existing),
                )
            self.received_data_buffers[key] = np.array([], dtype=np.float32)

    def _clear_mic_asr_state(
        self,
        client_uid: str,
        reason: str,
        mic_source: str | None = None,
    ) -> None:
        keys = self._mic_state_keys_for_client(client_uid, mic_source)
        self._clear_received_audio_buffer(client_uid, reason, mic_source)
        for key in keys:
            tasks = self.mic_asr_tasks.pop(key, [])
            if tasks:
                logger.info(
                    "Clearing pending microphone ASR tasks for {} because {}: tasks={}",
                    key,
                    reason,
                    len(tasks),
                )
            for task in tasks:
                if not task.done():
                    task.cancel()
            self.mic_asr_locks.pop(key, None)
            self.mic_asr_elapsed_seconds.pop(key, None)

    def _mic_asr_lock_for(self, state_key: str) -> asyncio.Lock:
        lock = self.mic_asr_locks.get(state_key)
        if lock is None:
            lock = asyncio.Lock()
            self.mic_asr_locks[state_key] = lock
        return lock

    def _append_mic_audio_buffer(
        self,
        client_uid: str,
        mic_source: str,
        audio: np.ndarray,
    ) -> None:
        state_key = self._mic_state_key(client_uid, mic_source)
        self.received_data_buffers[state_key] = np.append(
            self.received_data_buffers.get(
                state_key,
                np.array([], dtype=np.float32),
            ),
            audio,
        )

    def _link_human_name_from_data(self, data: dict[str, Any]) -> str:
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            name = str(metadata.get("human_name") or "").strip()
            if name:
                return name
        name = str(data.get("link_human_name") or "").strip()
        return name or self.display_link_human_name

    def _metadata_with_mic_source(
        self,
        data: dict[str, Any],
        mic_source: str,
    ) -> dict[str, Any] | None:
        metadata = data.get("metadata")
        merged = dict(metadata) if isinstance(metadata, dict) else {}
        merged["mic_source"] = mic_source
        if mic_source == "link":
            merged["input_source"] = "link_microphone"
            merged["human_name"] = self._link_human_name_from_data(data)
        elif mic_source == "local":
            merged.setdefault("input_source", "mic")
        return merged or None

    async def _route_mic_conversation_to_orchestrator(
        self,
        client_uid: str,
        data: dict[str, Any],
        turn_id: str,
        mic_source: str,
    ) -> bool:
        if data.get("type") != "mic-audio-end":
            return False

        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        if metadata.get("orchestrator_dispatched"):
            return False

        try:
            from .orchestrator import (
                MsgPriority,
                MsgSource,
                OrchestratorMessage,
                get_orchestrator,
            )
        except Exception as exc:
            logger.warning("Failed to import Orchestrator for mic routing: {}", exc)
            return False

        orch = get_orchestrator()
        if not orch:
            return False

        text = str(data.get("text") or "").strip()
        if not text:
            return False

        source = MsgSource.PARTNER if mic_source == "link" else MsgSource.MIC
        priority = MsgPriority.PARTNER if mic_source == "link" else MsgPriority.MIC
        human_name = str(metadata.get("human_name") or "").strip()
        nickname = human_name if mic_source == "link" else "本地麦克风"
        user_id = f"link:{human_name}" if mic_source == "link" and human_name else client_uid
        payload = dict(data)
        payload["metadata"] = dict(metadata)

        await orch.put_message(
            OrchestratorMessage(
                priority=priority,
                source=source,
                text=text,
                user_id=user_id,
                nickname=nickname,
                extra={
                    "client_uid": client_uid,
                    "turn_id": turn_id,
                    "conversation_data": payload,
                    "mic_source": mic_source,
                    "metadata": dict(metadata),
                },
            )
        )
        record_turn_event(
            turn_id,
            "websocket_handler",
            "mic_conversation_routed_to_orchestrator",
            client_uid=client_uid,
            mic_source=mic_source,
            orchestrator_source=source.value,
            orchestrator_priority=int(priority),
            text_len=len(text),
        )
        logger.info(
            "Routed mic conversation to Orchestrator: client={} source={} priority={} text={}",
            client_uid,
            source.value,
            int(priority),
            text[:80],
        )
        return True

    async def _dispatch_prepared_conversation(
        self,
        websocket: Any,
        client_uid: str,
        data: dict[str, Any],
        turn_id: str | None,
    ) -> None:
        context = self.client_contexts.get(client_uid)
        if context is None:
            logger.warning(
                "Dropping prepared conversation for missing client context: {}",
                client_uid,
            )
            return

        turn_id = turn_id or self._new_turn_id()
        self._remember_turn_id_for_client_group(client_uid, turn_id)
        await handle_conversation_trigger(
            msg_type=data.get("type", ""),
            data=data,
            client_uid=client_uid,
            context=context,
            websocket=websocket,
            client_contexts=self.client_contexts,
            client_connections=self.client_connections,
            chat_group_manager=self.chat_group_manager,
            received_data_buffers=self.received_data_buffers,
            current_conversation_tasks=self.current_conversation_tasks,
            broadcast_to_group=self.broadcast_to_group,
            turn_id=turn_id,
        )
        self._attach_proactive_task_callback(client_uid)

    async def _attach_game_vision_images_for_reply(
        self,
        client_uid: str,
        turn_id: str,
        data: dict[str, Any],
        mic_source: str,
    ) -> dict[str, Any]:
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        request_id = str(metadata.get("game_vision_request_id") or "").strip()
        state_key = self._game_vision_state_key(client_uid, mic_source)
        capture = self.game_vision_captures.pop(state_key, None)
        if not capture:
            return data

        capture_request_id = str(capture.get("request_id") or "").strip()
        if request_id and capture_request_id and request_id != capture_request_id:
            logger.info(
                "Ignoring stale game vision capture for final reply: client={} expected={} got={}",
                client_uid,
                request_id,
                capture_request_id,
            )
            return data

        images = list(data.get("images") or [])
        images.extend(capture.get("images") or [])
        if not images:
            return data

        provider = capture.get("provider")
        merged_metadata = {
            **metadata,
            "game_vision_request_id": capture_request_id or request_id,
            "game_vision_provider": provider,
            "game_vision_reply_mode": "vision_model",
        }
        if provider:
            merged_metadata["vision_model_provider"] = provider

        next_data = dict(data)
        next_data["images"] = images
        next_data["metadata"] = merged_metadata

        await self._send_game_vision_state(
            client_uid,
            "completed",
            request_id=capture_request_id or request_id or None,
            provider=str(provider) if provider else None,
            message="\u6e38\u620f\u622a\u56fe\u5df2\u4ea4\u7ed9\u89c6\u89c9\u6a21\u578b\u751f\u6210\u56de\u590d",
        )
        record_turn_event(
            turn_id,
            "websocket_handler",
            "game_vision_images_attached_for_visual_reply",
            client_uid=client_uid,
            mic_source=mic_source,
            request_id=capture_request_id,
            provider=provider,
            image_count=len(images),
        )
        logger.info(
            "Attached game screenshot to final visual reply: client={} source={} request_id={} provider={} images={}",
            client_uid,
            mic_source,
            capture_request_id,
            provider,
            len(images),
        )
        return next_data

    def _mark_memory_reload_after_interrupt(
        self,
        client_uid: str,
        turn_id: str | None,
        reason: str,
    ) -> None:
        self.clients_needing_memory_reload.add(client_uid)
        record_turn_event(
            turn_id,
            "websocket_handler",
            "memory_reload_marked_after_interrupt",
            client_uid=client_uid,
            reason=reason,
        )

    def _apply_pending_memory_reload_metadata(
        self,
        client_uid: str,
        turn_id: str,
        data: WSMessage,
    ) -> WSMessage:
        if client_uid not in self.clients_needing_memory_reload:
            return data

        self.clients_needing_memory_reload.discard(client_uid)
        updated_data = dict(data)
        metadata = updated_data.get("metadata")
        merged_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        merged_metadata["reload_memory_from_history_before_turn"] = True
        merged_metadata["reload_memory_reason"] = "previous-turn-interrupted"
        updated_data["metadata"] = merged_metadata
        record_turn_event(
            turn_id,
            "websocket_handler",
            "pending_memory_reload_metadata_applied",
            client_uid=client_uid,
        )
        return updated_data


    def _combine_mic_transcripts(self, parts: list[str]) -> str:
        combined = ""
        for part in parts:
            text = part.strip()
            if not text:
                continue
            if (
                combined
                and combined[-1].isascii()
                and combined[-1].isalnum()
                and text[0].isascii()
                and text[0].isalnum()
            ):
                combined += " "
            combined += text
        return combined

    async def _transcribe_mic_audio_segment(
        self,
        client_uid: str,
        state_key: str,
        mic_source: str,
        audio: np.ndarray,
        segment_index: int,
    ) -> str:
        context = self.client_contexts[client_uid]
        logger.info(
            "Starting microphone ASR segment: client={} source={} key={} segment={} samples={}",
            client_uid,
            mic_source,
            state_key,
            segment_index,
            len(audio),
        )
        try:
            async with self._mic_asr_lock_for(state_key):
                asr_started_at = time.perf_counter()
                try:
                    text = await context.asr_engine.async_transcribe_np(audio)
                finally:
                    elapsed = time.perf_counter() - asr_started_at
                    self.mic_asr_elapsed_seconds[state_key] = (
                        self.mic_asr_elapsed_seconds.get(state_key, 0.0) + elapsed
                    )
        except asyncio.CancelledError:
            logger.info(
                "Microphone ASR segment cancelled: client={} source={} segment={}",
                client_uid,
                mic_source,
                segment_index,
            )
            raise
        except Exception:
            logger.exception(
                "Microphone ASR segment failed: client={} source={} segment={}",
                client_uid,
                mic_source,
                segment_index,
            )
            return ""

        logger.info(
            "Microphone ASR segment completed: client={} source={} segment={} text_len={} text={}",
            client_uid,
            mic_source,
            segment_index,
            len(text),
            text,
        )
        return text

    def _start_mic_asr_segment_from_buffer(
        self,
        client_uid: str,
        reason: str,
        mic_source: str = "local",
    ) -> bool:
        state_key = self._mic_state_key(client_uid, mic_source)
        audio = self.received_data_buffers.get(state_key)
        if audio is None or len(audio) == 0:
            logger.debug(
                "No buffered microphone audio to start ASR segment for {} ({})",
                state_key,
                reason,
            )
            self.received_data_buffers[state_key] = np.array([], dtype=np.float32)
            return False

        audio = audio.astype(np.float32, copy=True)
        self.received_data_buffers[state_key] = np.array([], dtype=np.float32)
        tasks = self.mic_asr_tasks.setdefault(state_key, [])
        segment_index = len(tasks)
        task = asyncio.create_task(
            self._transcribe_mic_audio_segment(
                client_uid,
                state_key,
                mic_source,
                audio,
                segment_index,
            )
        )
        tasks.append(task)
        logger.info(
            "Queued microphone ASR segment: client={} source={} key={} segment={} samples={} reason={}",
            client_uid,
            mic_source,
            state_key,
            segment_index,
            len(audio),
            reason,
        )
        return True

    async def _finalize_mic_asr_text(
        self,
        client_uid: str,
        turn_id: str,
        mic_source: str = "local",
    ) -> str:
        state_key = self._mic_state_key(client_uid, mic_source)
        self._start_mic_asr_segment_from_buffer(
            client_uid,
            "final-confirm",
            mic_source=mic_source,
        )
        tasks = self.mic_asr_tasks.pop(state_key, [])
        if not tasks:
            self.mic_asr_elapsed_seconds.pop(state_key, None)
            logger.info("No microphone ASR segments to finalize for {}", state_key)
            return ""

        logger.info(
            "Waiting for microphone ASR segments: client={} source={} key={} segments={}",
            client_uid,
            mic_source,
            state_key,
            len(tasks),
        )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        parts: list[str] = []
        for index, result in enumerate(results):
            if isinstance(result, asyncio.CancelledError):
                logger.info(
                    "Microphone ASR segment skipped after cancellation: client={} source={} segment={}",
                    client_uid,
                    mic_source,
                    index,
                )
                continue
            if isinstance(result, Exception):
                logger.warning(
                    "Microphone ASR segment returned error: client={} source={} segment={} error={}",
                    client_uid,
                    mic_source,
                    index,
                    result,
                )
                continue
            parts.append(result)

        text = self._combine_mic_transcripts(parts)
        asr_seconds = self.mic_asr_elapsed_seconds.pop(state_key, 0.0)
        input_source = "link_microphone" if mic_source == "link" else "mic"
        ensure_performance_turn(turn_id, input_source=input_source)
        set_performance_metric(
            turn_id,
            "asr_seconds",
            asr_seconds,
            overwrite=True,
        )
        logger.info(
            "Final microphone ASR text: client={} source={} turn_id={} segments={} text_len={} text={}",
            client_uid,
            mic_source,
            turn_id,
            len(parts),
            len(text),
            text,
        )
        return text

    def _has_connected_clients(self) -> bool:
        return any(
            not connection_group.is_empty()
            for connection_group in self.client_connections.values()
        )

    def _has_running_conversation(self) -> bool:
        return any(
            task is not None and not task.done()
            for task in self.current_conversation_tasks.values()
        )

    def _get_proactive_client_uid(self) -> str | None:
        default_group = self.client_connections.get("default")
        if (
            default_group
            and not default_group.is_empty()
            and "default" in self.client_contexts
        ):
            return "default"

        for client_uid, connection_group in self.client_connections.items():
            if connection_group.is_empty() or client_uid not in self.client_contexts:
                continue
            return client_uid

        return None

    def _conversation_task_key(self, client_uid: str) -> str:
        group = self.chat_group_manager.get_client_group(client_uid)
        if group and len(group.members) > 1:
            return group.group_id
        return client_uid

    def _cancel_proactive_idle_timer(self) -> None:
        current_task = asyncio.current_task()
        if (
            self._proactive_timer_task
            and not self._proactive_timer_task.done()
            and self._proactive_timer_task is not current_task
        ):
            self._proactive_timer_task.cancel()
        self._proactive_timer_task = None

    def _schedule_proactive_idle_timer(self) -> None:
        if (
            not self._is_vtuber_active_for_proactive_speak()
            or not self._has_connected_clients()
        ):
            self._cancel_proactive_idle_timer()
            return

        current_task = asyncio.current_task()
        if self._proactive_timer_task and not self._proactive_timer_task.done():
            if self._proactive_timer_task is current_task:
                self._proactive_timer_task = None
            else:
                self._proactive_timer_task.cancel()

        scheduled_at = self._proactive_last_activity_at
        self._proactive_timer_task = asyncio.create_task(
            self._run_proactive_idle_timer(scheduled_at)
        )

    def _mark_proactive_activity(self, reason: str) -> None:
        self._proactive_last_activity_at = time.monotonic()
        if reason not in {"message:mic-audio-data", "message:raw-audio-data"}:
            logger.debug(
                "Proactive idle timer reset by {} ({}s)",
                reason,
                self.proactive_idle_seconds,
            )
        self._schedule_proactive_idle_timer()

    def _attach_proactive_task_callback(self, client_uid: str) -> None:
        task_key = self._conversation_task_key(client_uid)
        task = self.current_conversation_tasks.get(task_key)
        if not task or task.done():
            return

        def _on_done(_task: asyncio.Task) -> None:
            self._mark_proactive_activity("conversation-task-done")

        task.add_done_callback(_on_done)

    async def _run_proactive_idle_timer(self, scheduled_at: float) -> None:
        try:
            await asyncio.sleep(max(0.1, self.proactive_idle_seconds))
            if scheduled_at != self._proactive_last_activity_at:
                return
            if (
                not self._is_vtuber_active_for_proactive_speak()
                or not self._has_connected_clients()
            ):
                return
            if self._has_running_conversation():
                self._mark_proactive_activity("conversation-still-running")
                return

            idle_time = time.monotonic() - self._proactive_last_activity_at
            await self._trigger_proactive_speak(idle_time)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error("Proactive idle timer failed: {}", exc)

    async def _trigger_proactive_speak(self, idle_time: float) -> None:
        client_uid = self._get_proactive_client_uid()
        if not client_uid:
            return

        connection_group = self.client_connections.get(client_uid)
        if not connection_group or connection_group.is_empty():
            return

        if self.display_game_vision_enabled:
            logger.info(
                "Skipping backend proactive text cold speak after {:.2f}s idle "
                "because game vision cold capture is handled by the display",
                idle_time,
            )
            self._mark_proactive_activity("game-vision-cold-handled-by-display")
            return

        self._proactive_last_activity_at = time.monotonic()
        if self._is_story_interaction_mode():
            turn_id = self._new_turn_id()
            self._consecutive_cold_silence_triggers += 1
            logger.info(
                "Triggering story cold silence prompt for {} after {:.2f}s idle "
                "(streak={})",
                client_uid,
                idle_time,
                self._consecutive_cold_silence_triggers,
            )
            record_turn_event(
                turn_id,
                "websocket_handler",
                "story_cold_silence_triggered",
                client_uid=client_uid,
                idle_time=idle_time,
                streak=self._consecutive_cold_silence_triggers,
            )
            self._remember_turn_id_for_client_group(client_uid, turn_id)
            self._start_trigger_prompt_task(
                client_uid=client_uid,
                trigger_name=STORY_COLD_SILENCE_TRIGGER_NAME,
                turn_id=turn_id,
            )
            if (
                self._consecutive_cold_silence_triggers
                >= STORY_COLD_SILENCE_BARRAGE_THRESHOLD
            ):
                if self.display_paint_enabled:
                    logger.info(
                        "Skipping auto switch to barrage after cold silence because paint mode is enabled."
                    )
                    self._mark_proactive_activity("paint-mode-skip-barrage-switch")
                    return
                await self._switch_to_barrage_after_cold_silence(turn_id, client_uid)
            return

        if self._is_barrage_interaction_mode():
            turn_id = self._new_turn_id()
            logger.info(
                "Speaking error prompt before sleep because barrage mode stayed idle "
                "for {:.2f}s",
                idle_time,
            )
            record_turn_event(
                turn_id,
                "websocket_handler",
                "barrage_cold_silence_error_prompt_triggered",
                client_uid=client_uid,
                idle_time=idle_time,
            )
            self._remember_turn_id_for_client_group(client_uid, turn_id)
            trigger_task = self._start_trigger_prompt_task(
                client_uid=client_uid,
                trigger_name=BARRAGE_SLEEP_TRIGGER_NAME,
                turn_id=turn_id,
                reset_proactive_on_done=False,
            )
            if trigger_task:
                asyncio.create_task(
                    self._enter_sleep_after_barrage_prompt(
                        trigger_task,
                        turn_id,
                        client_uid,
                    )
                )
                record_turn_event(
                    turn_id,
                    "websocket_handler",
                    "barrage_cold_silence_sleep_task_created",
                    client_uid=client_uid,
                )
            else:
                await self._enter_sleep_after_barrage_cold_silence(turn_id, client_uid)
            return

        self._reset_cold_silence_streak("proactive-llm-speak")
        logger.info(
            "Triggering proactive speak for {} after {:.2f}s idle",
            client_uid,
            idle_time,
        )
        await self._handle_conversation_trigger(
            connection_group,
            client_uid,
            {
                "type": "ai-speak-signal",
                "idle_time": idle_time,
                "source": "proactive_idle_timer",
            },
        )

    async def _enter_sleep_after_barrage_prompt(
        self,
        trigger_task: asyncio.Task,
        turn_id: str,
        client_uid: str,
    ) -> None:
        try:
            await trigger_task
            record_turn_event(
                turn_id,
                "websocket_handler",
                "barrage_cold_silence_error_prompt_completed",
                client_uid=client_uid,
            )
        except asyncio.CancelledError:
            record_turn_event(
                turn_id,
                "websocket_handler",
                "barrage_cold_silence_error_prompt_cancelled",
                client_uid=client_uid,
            )
            return
        except Exception as exc:
            logger.warning(
                "Barrage cold silence error prompt failed before sleep: {}",
                exc,
            )
            record_turn_event(
                turn_id,
                "websocket_handler",
                "barrage_cold_silence_error_prompt_failed",
                client_uid=client_uid,
                error=str(exc),
            )

        await self._enter_sleep_after_barrage_cold_silence(turn_id, client_uid)

    async def _enter_sleep_after_barrage_cold_silence(
        self,
        turn_id: str,
        client_uid: str,
    ) -> None:
        from .vtuber_state_machine import VTuberMode, get_vtuber_state_machine

        sm = get_vtuber_state_machine()
        if sm is None:
            return

        already_sleeping = sm.mode == VTuberMode.IDLE and sm.sleeping
        if already_sleeping:
            result = self._current_vtuber_state_payload() or {
                "new_mode": sm.mode.value,
                "sub_mode": sm.idle_sub_mode.value,
                "interaction_mode": sm.interaction_mode.value,
                "sleeping": sm.sleeping,
                "punished": sm.punished,
            }
            sleep_event = "barrage_cold_silence_sleep_already_entered"
        else:
            result = await sm.enter_sleep(
                reason="barrage-cold-silence",
                interrupt_current=False,
            )
            sleep_event = "barrage_cold_silence_sleep_entered"

        for uid in list(self.received_data_buffers.keys()):
            self._clear_mic_asr_state(uid, "barrage-cold-silence")
        self._cancel_proactive_idle_timer()
        self._reset_cold_silence_streak("barrage-cold-silence")
        record_turn_event(
            turn_id,
            "websocket_handler",
            sleep_event,
            client_uid=client_uid,
            vtuber_state=result,
        )

    async def _switch_to_barrage_after_cold_silence(
        self,
        turn_id: str,
        client_uid: str,
    ) -> None:
        from .vtuber_state_machine import get_vtuber_state_machine

        sm = get_vtuber_state_machine()
        if sm is None:
            return

        result = await sm.handle_switch("barrage")
        self._consecutive_cold_silence_triggers = 0
        record_turn_event(
            turn_id,
            "websocket_handler",
            "story_cold_silence_switched_to_barrage",
            client_uid=client_uid,
            vtuber_state=result,
        )
        await self._broadcast_mode_change_result(result)

    async def _broadcast_mode_change_result(self, result: dict[str, Any]) -> None:
        message = json.dumps(self._mode_change_payload(result), ensure_ascii=False)
        for uid in list(self.client_connections.keys()):
            await self._send_text_to_client(uid, message)

    def _remember_turn_id_for_client_group(
        self,
        client_uid: str,
        turn_id: str,
    ) -> None:
        group = self.chat_group_manager.get_client_group(client_uid)
        if group and len(group.members) > 1:
            for member_uid in group.members:
                self.current_turn_ids[member_uid] = turn_id
            record_turn_event(
                turn_id,
                "websocket_handler",
                "turn_remembered_for_group",
                client_uid=client_uid,
                group_id=group.group_id,
                group_members=list(group.members),
            )
            return

        self.current_turn_ids[client_uid] = turn_id
        record_turn_event(
            turn_id,
            "websocket_handler",
            "turn_remembered_for_client",
            client_uid=client_uid,
        )

    async def _send_initial_messages(
        self,
        websocket: WebSocket,
        client_uid: str,
        session_service_context: ServiceContext,
    ):
        """Send initial connection messages to the client"""
        await websocket.send_text(
            json.dumps({"type": "full-text", "text": "Connection established"})
        )

        await websocket.send_text(
            json.dumps(
                self._model_and_conf_payload(
                    session_service_context,
                    client_uid,
                ),
                ensure_ascii=False,
            )
        )
        if self.project_model_manager is not None:
            await websocket.send_text(
                json.dumps(
                    self.project_model_manager.public_state(),
                    ensure_ascii=False,
                )
            )

        if session_service_context.history_uid:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "working-history-loaded",
                        "history_uid": session_service_context.history_uid,
                    }
                )
            )
            messages = [
                msg
                for msg in get_history(
                    session_service_context.character_config.conf_uid,
                    session_service_context.history_uid,
                )
                if msg["role"] != "system"
            ]
            await websocket.send_text(
                json.dumps({"type": "history-data", "messages": messages})
            )

        # Send initial group status
        await self.send_group_update(websocket, client_uid)

        await websocket.send_text(
            json.dumps(
                {
                    "type": "control",
                    "text": "open-live2d"
                    if self.display_live2d_open
                    else "close-live2d",
                    "display_state": self._display_state_payload(),
                },
                ensure_ascii=False,
            )
        )
        await websocket.send_text(
            json.dumps(
                {
                    "type": "control",
                    "text": "start-mic"
                    if self.display_microphone_enabled
                    else "stop-mic",
                    "display_state": self._display_state_payload(),
                },
                ensure_ascii=False,
            )
        )
        await websocket.send_text(
            json.dumps(
                {
                    "type": "control",
                    "text": "start-link-mic"
                    if self.display_link_microphone_enabled
                    else "stop-link-mic",
                    "display_state": self._display_state_payload(),
                },
                ensure_ascii=False,
            )
        )

    async def _init_service_context(
        self, send_text: Callable, client_uid: str
    ) -> ServiceContext:
        """Initialize service context for a new session by cloning the default context"""
        session_service_context = ServiceContext()
        await session_service_context.load_cache(
            config=self.default_context_cache.config.model_copy(deep=True),
            system_config=self.default_context_cache.system_config.model_copy(
                deep=True
            ),
            character_config=self.default_context_cache.character_config.model_copy(
                deep=True
            ),
            live2d_model=self.default_context_cache.live2d_model,
            asr_engine=self.default_context_cache.asr_engine,
            tts_engine=self.default_context_cache.tts_engine,
            vad_engine=self.default_context_cache.vad_engine,
            agent_engine=self.default_context_cache.agent_engine,
            translate_engine=self.default_context_cache.translate_engine,
            mcp_server_registery=self.default_context_cache.mcp_server_registery,
            tool_adapter=self.default_context_cache.tool_adapter,
            knowledge_runtime=self.default_context_cache.knowledge_runtime,
            send_text=send_text,
            client_uid=client_uid,
        )
        return session_service_context

    async def handle_websocket_communication(
        self, websocket: WebSocket, client_uid: str
    ) -> None:
        """
        Handle ongoing WebSocket communication

        Args:
            websocket: The WebSocket connection
            client_uid: Unique identifier for the client
        """
        try:
            while True:
                try:
                    data = await websocket.receive_json()
                    message_handler.handle_message(client_uid, data)
                    await self._route_message(websocket, client_uid, data)
                except WebSocketDisconnect:
                    raise
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": str(e)})
                    )
                    continue

        except WebSocketDisconnect:
            logger.info(f"Client {client_uid} disconnected")
            raise
        except Exception as e:
            logger.error(f"Fatal error in WebSocket communication: {e}")
            raise

    async def _route_message(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """
        Route incoming message to appropriate handler

        Args:
            websocket: The WebSocket connection
            client_uid: Client identifier
            data: Message data
        """

        msg_type = data.get("type")
        turn_id = self._get_message_turn_id(data)
        quiet_message_types = {
            "heartbeat",
            "performance-monitor-sync",
            "project-config-request",
            "project-config-update",
            "project-config-test",
        }
        if msg_type not in quiet_message_types:
            turn_id = turn_id or self.current_turn_ids.get(client_uid)
            record_turn_event(
                turn_id,
                "websocket_handler",
                "message_received",
                client_uid=client_uid,
                message_type=msg_type,
                has_audio=bool(data.get("audio")),
                has_text=bool(data.get("text")),
            )
        if msg_type not in (
            "heartbeat",
            "frontend-playback-complete",
            "audio-play-start",
            "performance-monitor-sync",
        ):
            logger.info(
                f"received message from {websocket} Client {client_uid}: "
                f"{truncate_data(data)}"
            )
        if not msg_type:
            logger.warning("Message received without type")
            return
        if msg_type not in quiet_message_types:
            self._mark_proactive_activity(f"message:{msg_type}")

        handler = self._message_handlers.get(msg_type)
        if handler:
            await handler(websocket, client_uid, data)
        else:
            logger.warning(f"Unknown message type: {msg_type}")

    async def _handle_frontend_playback_complete(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Acknowledge playback-complete messages already handled by message_handler."""
        record_turn_event(
            self._get_message_turn_id(data),
            "websocket_handler",
            "frontend_playback_complete_received",
            client_uid=client_uid,
        )
        pass

    async def _handle_performance_monitor_sync(
        self,
        websocket: WebSocket,
        client_uid: str,
        data: WSMessage,
    ) -> None:
        """Synchronize performance data between streamer and director displays."""
        display_mode = self.display_client_modes.get(id(websocket))
        payload = dict(data)
        payload["type"] = "performance-monitor-sync"
        if payload.get("kind") == "reset":
            if display_mode not in {"streamer", "director"}:
                logger.warning(
                    "Ignoring performance monitor reset from unknown display: {}",
                    client_uid,
                )
                return
            await self._send_text_to_client(
                client_uid,
                json.dumps(payload, ensure_ascii=False),
            )
            return
        if display_mode != "streamer":
            logger.warning(
                "Ignoring performance monitor sync from non-streamer display: {}",
                client_uid,
            )
            return
        await self._send_text_to_display_mode(
            client_uid,
            json.dumps(payload, ensure_ascii=False),
            "director",
        )
        if payload.get("kind") == "turn":
            await persist_performance_metrics(
                self._get_message_turn_id(payload),
                payload.get("metrics"),
                client_uid=client_uid,
                input_source=payload.get("source"),
                playback_completed=bool(payload.get("completed")),
            )

    async def _handle_project_config_request(
        self,
        websocket: WebSocket,
        client_uid: str,
        data: WSMessage,
    ) -> None:
        if self.project_model_manager is None:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "project-config-error",
                        "message": "项目模型配置尚未初始化",
                    },
                    ensure_ascii=False,
                )
            )
            return
        await websocket.send_text(
            json.dumps(
                self.project_model_manager.public_state(),
                ensure_ascii=False,
            )
        )

    async def _handle_project_config_update(
        self,
        websocket: WebSocket,
        client_uid: str,
        data: WSMessage,
    ) -> None:
        if self.project_model_manager is None:
            await self._handle_project_config_request(websocket, client_uid, data)
            return
        try:
            async with self._project_model_config_lock:
                runtime = self.project_model_manager.save(data)
                await self.apply_project_model_config(runtime)
        except Exception as exc:
            logger.exception("Failed to update project model config.")
            await websocket.send_text(
                json.dumps(
                    {"type": "project-config-error", "message": str(exc)},
                    ensure_ascii=False,
                )
            )
            return
        await self._broadcast_project_config_state()
        logger.info(
            "Project model config applied: provider={} model={} temperature={} "
            "web_search={}",
            runtime.provider,
            runtime.model,
            runtime.temperature,
            runtime.web_search_enabled,
        )

    async def _handle_project_config_test(
        self,
        websocket: WebSocket,
        client_uid: str,
        data: WSMessage,
    ) -> None:
        if self.project_model_manager is None:
            await self._handle_project_config_request(websocket, client_uid, data)
            return
        try:
            result = await self.project_model_manager.test_connection(data)
        except Exception as exc:
            result = {"ok": False, "message": str(exc)}
        await websocket.send_text(
            json.dumps(
                {"type": "project-config-test-result", **result},
                ensure_ascii=False,
            )
        )

    async def apply_project_model_config(self, runtime: Any | None = None) -> None:
        if self.project_model_manager is None:
            return
        llm = self.project_model_manager.build_llm(runtime)
        contexts = [self.default_context_cache, *self.client_contexts.values()]
        updated_agents: set[int] = set()
        for context in contexts:
            agent = getattr(context, "agent_engine", None)
            if agent is None or id(agent) in updated_agents:
                continue
            set_llm = getattr(agent, "_set_llm", None)
            if not callable(set_llm):
                logger.warning(
                    "Project model hot switch is unsupported for agent type: {}",
                    type(agent).__name__,
                )
                continue
            set_llm(llm)
            updated_agents.add(id(agent))

    async def _broadcast_project_config_state(self) -> None:
        if self.project_model_manager is None:
            return
        message = json.dumps(
            self.project_model_manager.public_state(),
            ensure_ascii=False,
        )
        for uid in list(self.client_connections):
            await self._send_text_to_client(uid, message)

    async def _handle_group_operation(
        self, websocket: WebSocket, client_uid: str, data: dict
    ) -> None:
        """Handle group-related operations"""
        operation = data.get("type")
        target_uid = data.get(
            "invitee_uid" if operation == "add-client-to-group" else "target_uid"
        )

        await handle_group_operation(
            operation=operation,
            client_uid=client_uid,
            target_uid=target_uid,
            chat_group_manager=self.chat_group_manager,
            client_connections=self.client_connections,
            send_group_update=self.send_group_update,
        )

    async def handle_disconnect(
        self, client_uid: str, websocket: WebSocket | None = None
    ) -> None:
        """Handle client disconnection"""
        connection_group = self.client_connections.get(client_uid)
        if websocket and connection_group:
            connection_group.remove(websocket)
        await self._forget_display_client_mode(websocket)
        if websocket and connection_group:
            if not connection_group.is_empty():
                logger.info(
                    "WebSocket connection for client {} disconnected ({} still active)",
                    client_uid,
                    connection_group.count,
                )
                return

        group = self.chat_group_manager.get_client_group(client_uid)
        if group:
            await handle_group_interrupt(
                group_id=group.group_id,
                heard_response="",
                current_conversation_tasks=self.current_conversation_tasks,
                chat_group_manager=self.chat_group_manager,
                client_contexts=self.client_contexts,
                broadcast_to_group=self.broadcast_to_group,
                turn_id=self.current_turn_ids.get(client_uid),
            )

        await handle_client_disconnect(
            client_uid=client_uid,
            chat_group_manager=self.chat_group_manager,
            client_connections=self.client_connections,
            send_group_update=self.send_group_update,
        )

        # Clean up other client data
        self._clear_mic_asr_state(client_uid, "client-disconnect")
        self._clear_game_vision_state(client_uid, "client-disconnect")
        self._clear_visual_image_context(client_uid, "client-disconnect")
        self.mic_asr_locks.pop(client_uid, None)
        self.client_connections.pop(client_uid, None)
        context = self.client_contexts.pop(client_uid, None)
        self.received_data_buffers.pop(client_uid, None)
        self.clients_needing_memory_reload.discard(client_uid)
        if client_uid in self.current_conversation_tasks:
            task = self.current_conversation_tasks[client_uid]
            if task and not task.done():
                task.cancel()
            self.current_conversation_tasks.pop(client_uid, None)
        self.current_turn_ids.pop(client_uid, None)
        if not self._has_connected_clients():
            self._cancel_proactive_idle_timer()

        # Call context close to clean up resources (e.g., MCPClient)
        if context:
            await context.close()

        logger.info(f"Client {client_uid} disconnected")
        message_handler.cleanup_client(client_uid)

    async def _cleanup_failed_connection(
        self, client_uid: str, websocket: WebSocket | None = None
    ) -> None:
        """Clean up failed connection data"""
        connection_group = self.client_connections.get(client_uid)
        if websocket and connection_group:
            connection_group.remove(websocket)
        await self._forget_display_client_mode(websocket)
        if websocket and connection_group:
            if not connection_group.is_empty():
                return

        self._clear_mic_asr_state(client_uid, "failed-connection")
        self._clear_game_vision_state(client_uid, "failed-connection")
        self._clear_visual_image_context(client_uid, "failed-connection")
        self.mic_asr_locks.pop(client_uid, None)
        self.client_connections.pop(client_uid, None)
        self.client_contexts.pop(client_uid, None)
        self.received_data_buffers.pop(client_uid, None)
        self.clients_needing_memory_reload.discard(client_uid)
        self.chat_group_manager.client_group_map.pop(client_uid, None)

        if client_uid in self.current_conversation_tasks:
            task = self.current_conversation_tasks[client_uid]
            if task and not task.done():
                task.cancel()
            self.current_conversation_tasks.pop(client_uid, None)
        self.current_turn_ids.pop(client_uid, None)
        if not self._has_connected_clients():
            self._cancel_proactive_idle_timer()

        message_handler.cleanup_client(client_uid)

    async def broadcast_to_group(
        self, group_members: list[str], message: dict, exclude_uid: str = None
    ) -> None:
        """Broadcasts a message to group members"""
        await broadcast_to_group(
            group_members=group_members,
            message=message,
            client_connections=self.client_connections,
            exclude_uid=exclude_uid,
        )

    async def send_group_update(self, websocket: Any, client_uid: str):
        """Sends group information to a client"""
        group = self.chat_group_manager.get_client_group(client_uid)
        if group:
            current_members = self.chat_group_manager.get_group_members(client_uid)
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "group-update",
                        "members": current_members,
                        "is_owner": group.owner_uid == client_uid,
                    }
                )
            )
        else:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "group-update",
                        "members": [],
                        "is_owner": False,
                    }
                )
            )

    async def _handle_interrupt(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle conversation interruption"""
        heard_response = data.get("text", "")
        turn_id = self._get_message_turn_id(data) or self.current_turn_ids.get(
            client_uid
        )
        record_turn_event(
            turn_id,
            "websocket_handler",
            "interrupt_signal_received",
            client_uid=client_uid,
            heard_response_len=len(heard_response),
            interrupted=True,
        )
        context = self.client_contexts[client_uid]
        group = self.chat_group_manager.get_client_group(client_uid)

        if group and len(group.members) > 1:
            await handle_group_interrupt(
                group_id=group.group_id,
                heard_response=heard_response,
                current_conversation_tasks=self.current_conversation_tasks,
                chat_group_manager=self.chat_group_manager,
                client_contexts=self.client_contexts,
                broadcast_to_group=self.broadcast_to_group,
                turn_id=turn_id,
            )
        else:
            control_payload = {"type": "control", "text": "interrupt"}
            if turn_id:
                control_payload["turn_id"] = turn_id
            await self._send_text_to_client(
                client_uid,
                json.dumps(control_payload),
            )
            await handle_individual_interrupt(
                client_uid=client_uid,
                current_conversation_tasks=self.current_conversation_tasks,
                context=context,
                heard_response=heard_response,
                turn_id=turn_id,
            )
            self._mark_memory_reload_after_interrupt(
                client_uid,
                turn_id,
                "interrupt-signal",
            )

    @staticmethod
    def _coerce_director_metric_int(value: Any, default: int = 0) -> int:
        if value in (None, ""):
            return default
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    def _director_metrics_to_barrage_variables(
        self,
        metrics: Any,
    ) -> list[dict[str, Any]]:
        if not isinstance(metrics, list):
            logger.warning(
                "Ignoring director-metrics payload because metrics is not a list: {}",
                type(metrics),
            )
            return []

        variables: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        allowed_names = set(DIRECTOR_METRIC_TO_BARRAGE_VARIABLE.values())
        for fallback_priority, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                continue

            key = str(metric.get("key") or metric.get("name") or "")
            name = DIRECTOR_METRIC_TO_BARRAGE_VARIABLE.get(key)
            if name is None and key in allowed_names:
                name = key
            if not name or name in seen_names:
                continue

            seen_names.add(name)
            variables.append(
                {
                    "name": name,
                    "enabled": bool(metric.get("enabled")),
                    "threshold": self._coerce_director_metric_int(
                        metric.get("value", metric.get("raw_value")),
                    ),
                    "priority": self._coerce_director_metric_int(
                        metric.get("order"),
                        fallback_priority,
                    ),
                }
            )

        return variables

    async def _handle_console_message(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle messages from the PyQt live console."""
        action = data.get("action")
        logger.info("Received console-message from {}: {}", client_uid, action)

        if action == "display-client-mode":
            mode = self._normalize_display_client_mode(data.get("mode"))
            if not mode:
                logger.warning(
                    "[console] ignoring unknown display client mode: {}",
                    data.get("mode"),
                )
                return
            self.display_client_modes[id(websocket)] = mode
            logger.info(
                "[console] registered display client mode: client_uid={} mode={} streamer_online={}",
                client_uid,
                mode,
                self._has_streamer_display_client(),
            )
            await self._request_performance_monitor_snapshot(client_uid)
            return

        if action in {"wake-animation-start", "wake-animation-complete"}:
            await self._set_wake_animation_pending(
                action == "wake-animation-start",
                f"{action}:{client_uid}",
            )
            if action == "wake-animation-start":
                await self._start_pending_wake_trigger(
                    client_uid,
                    reason="wake-animation-start",
                    keep_timeout=True,
                )
            elif action == "wake-animation-complete":
                timeout_task = self.wake_animation_timeout_tasks.pop(
                    client_uid,
                    None,
                )
                if timeout_task and not timeout_task.done():
                    timeout_task.cancel()
            return

        if action in {"live2d-toggle", "live2d-open", "live2d-close"}:
            if action == "live2d-toggle":
                self.display_live2d_open = not self.display_live2d_open
            else:
                self.display_live2d_open = action == "live2d-open"
            await self._broadcast_display_control(
                "open-live2d" if self.display_live2d_open else "close-live2d"
            )
            return

        if action in {"microphone-toggle", "microphone-start", "microphone-stop"}:
            if action == "microphone-toggle":
                self.display_microphone_enabled = not self.display_microphone_enabled
            else:
                self.display_microphone_enabled = action == "microphone-start"
            await self._broadcast_display_control(
                "start-mic" if self.display_microphone_enabled else "stop-mic"
            )
            return

        if action in {
            "link-microphone-toggle",
            "link-microphone-start",
            "link-microphone-stop",
        }:
            if action == "link-microphone-toggle":
                self.display_link_microphone_enabled = (
                    not self.display_link_microphone_enabled
                )
            else:
                self.display_link_microphone_enabled = (
                    action == "link-microphone-start"
                )
            self.display_link_microphone_faulted = False
            self.display_link_microphone_pending = self.display_link_microphone_enabled
            self.display_link_microphone_confirmed = False
            if (
                self.display_link_microphone_enabled
                and not self._has_streamer_display_client()
            ):
                self.display_link_microphone_faulted = True
                self.display_link_microphone_pending = False
                await self._broadcast_display_control("link-mic-fault")
                logger.warning(
                    "[console] link microphone requested but no streamer display is online"
                )
                return
            await self._broadcast_display_control(
                "start-link-mic"
                if self.display_link_microphone_enabled
                else "stop-link-mic"
            )
            return

        if action == "link-microphone-fault":
            self.display_link_microphone_faulted = bool(data.get("faulted"))
            self.display_link_microphone_pending = False
            self.display_link_microphone_confirmed = (
                self.display_link_microphone_enabled
                and not self.display_link_microphone_faulted
            )
            await self._broadcast_display_control("link-mic-fault")
            logger.info(
                "[console] link microphone faulted={} reason={}",
                self.display_link_microphone_faulted,
                data.get("reason") or "",
            )
            return

        if action == "link-microphone-name":
            name = str(data.get("name") or "").strip()
            self.display_link_human_name = name or "\u8fde\u7ebf\u4e3b\u64ad"
            await self._broadcast_display_control("link-mic-name")
            logger.info(
                "[console] link microphone human name set to {}",
                self.display_link_human_name,
            )
            return

        if action == "link-microphone-detect-name":
            from .barrage_adapter import _adapter_instance

            result: dict[str, Any] = {
                "found": False,
                "reason": "barrage adapter is not running",
            }
            if _adapter_instance:
                result = _adapter_instance.get_link_anchor_candidate()

            await self._send_text_to_client(
                client_uid,
                json.dumps(
                    {
                        "type": "link-microphone-name-detect",
                        "request_id": data.get("request_id"),
                        "candidate": result.get("name"),
                        "source": result.get("source") or "barrage_structured",
                        "confidence": result.get("confidence"),
                        "path": result.get("path"),
                        "found": bool(result.get("found")),
                        "reason": result.get("reason"),
                        "age_seconds": result.get("age_seconds"),
                        "id": result.get("id"),
                        "short_id": result.get("short_id"),
                        "room_id": result.get("room_id"),
                        "sec_uid": result.get("sec_uid"),
                        "is_host": result.get("is_host"),
                    },
                    ensure_ascii=False,
                ),
            )
            logger.info("[console] link microphone name detect result={}", result)
            return

        if action == "gift-thanks":
            self.display_gift_thanks_enabled = bool(data.get("enabled"))
            from .barrage_adapter import _adapter_instance

            adapter_result = None
            if _adapter_instance:
                adapter_result = _adapter_instance.set_gift_thanks_enabled(
                    self.display_gift_thanks_enabled
                )
            await self._broadcast_display_control(
                "gift-thanks-on"
                if self.display_gift_thanks_enabled
                else "gift-thanks-off"
            )
            logger.info(
                "[console] gift-thanks enabled={} adapter_result={}",
                self.display_gift_thanks_enabled,
                adapter_result,
            )
            return

        if action == "game-vision-mode":
            self.display_game_vision_enabled = bool(data.get("enabled"))
            await self._broadcast_display_control(
                "game-vision-on"
                if self.display_game_vision_enabled
                else "game-vision-off"
            )
            logger.info(
                "[console] game-vision enabled={} cold_idle_seconds={}",
                self.display_game_vision_enabled,
                data.get("cold_idle_seconds"),
            )
            self._mark_proactive_activity("game-vision-mode-updated")
            return

        if action == "paint-mode":
            self.display_paint_enabled = bool(data.get("enabled"))
            get_paint_manager().set_enabled(self.display_paint_enabled)
            await self._broadcast_display_control(
                "paint-on" if self.display_paint_enabled else "paint-off"
            )
            logger.info("[console] paint enabled={}", self.display_paint_enabled)
            return

        if action == "clear-vision-context":
            self._clear_visual_image_context(client_uid, "console-clear")
            return

        if action in {
            "live-streaming-agent-subtitle-toggle",
            "live-streaming-agent-subtitle-start",
            "live-streaming-agent-subtitle-stop",
        }:
            if action == "live-streaming-agent-subtitle-toggle":
                self.display_live_streaming_agent_subtitle_enabled = (
                    not self.display_live_streaming_agent_subtitle_enabled
                )
            else:
                self.display_live_streaming_agent_subtitle_enabled = action == "live-streaming-agent-subtitle-start"
            await self._broadcast_display_control(
                "open-live-streaming-agent-subtitle"
                if self.display_live_streaming_agent_subtitle_enabled
                else "close-live-streaming-agent-subtitle"
            )
            logger.info(
                "[console] live_streaming_agent-subtitle enabled={}",
                self.display_live_streaming_agent_subtitle_enabled,
            )
            return

        if action in {
            "barrage-subtitle-toggle",
            "barrage-subtitle-start",
            "barrage-subtitle-stop",
        }:
            if action == "barrage-subtitle-toggle":
                self.display_barrage_subtitle_enabled = (
                    not self.display_barrage_subtitle_enabled
                )
            else:
                self.display_barrage_subtitle_enabled = (
                    action == "barrage-subtitle-start"
                )
            await self._broadcast_display_control(
                "open-barrage-subtitle"
                if self.display_barrage_subtitle_enabled
                else "close-barrage-subtitle"
            )
            logger.info(
                "[console] barrage-subtitle enabled={}",
                self.display_barrage_subtitle_enabled,
            )
            return

        if action == "reply-probability":
            from .barrage_adapter import _adapter_instance

            if not _adapter_instance:
                await self._send_text_to_client(
                    client_uid,
                    json.dumps(
                        {
                            "type": "reply-probability",
                            "ok": False,
                            "error": "adapter not running",
                        },
                        ensure_ascii=False,
                    ),
                )
                return

            result = _adapter_instance.set_high_fan_percent(
                data.get("value", 10)
            )
            await self._send_text_to_client(
                client_uid,
                json.dumps(
                    {
                        "type": "reply-probability",
                        "ok": True,
                        "data": result,
                    },
                    ensure_ascii=False,
                ),
            )
            logger.info(
                "[console] reply-probability -> high_fan_percent={} cutoff={}",
                result["percent"],
                result["level_cutoff"],
            )
            return

        if action == "cold-time":
            try:
                self.proactive_idle_seconds = max(1.0, float(data.get("value", 5)))
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid cold-time value from {}: {}",
                    client_uid,
                    data.get("value"),
                )
                return
            logger.info(
                "Updated proactive cold time to {:.2f}s by {}",
                self.proactive_idle_seconds,
                client_uid,
            )
            self._mark_proactive_activity("cold-time-updated")
            return

        # ---- VTuber 模式切换开关 ----
        if action == "mode-switch":
            mode = data.get("mode")
            if mode == "barrage":
                action = "barrage"
            elif mode in {"focus", "co_host"}:
                action = "co_host"

        if action in {"sleep", "co_host", "barrage", "punish"}:
            from .vtuber_state_machine import get_vtuber_state_machine

            sm = get_vtuber_state_machine()
            if sm is None:
                logger.warning("[console] VTuber state machine not initialized")
                return
            result = await sm.handle_switch(action)
            wake_transition = self._is_wake_transition(action, result)
            if result["new_mode"] == "idle":
                await self.cancel_pending_wake_animation(f"mode-switch:{action}")
                await self._broadcast_mode_change_result(result)
                for uid in list(self.received_data_buffers.keys()):
                    self._clear_mic_asr_state(uid, f"mode-switch:{action}")
                self._cancel_proactive_idle_timer()
            else:
                if wake_transition and self.display_live2d_open:
                    self.display_wake_animation_pending = True
                    logger.info(
                        "Display wake animation state changed: active=True reason={}",
                        f"wake-transition:{client_uid}",
                    )
                await self._broadcast_mode_change_result(result)
                self._mark_proactive_activity(f"mode-switch:{action}")
                if wake_transition:
                    context = self.client_contexts[client_uid]
                    turn_id = self._new_turn_id()
                    trigger_name = self._wake_trigger_name_for_context(context)
                    record_turn_event(
                        turn_id,
                        "websocket_handler",
                        "wake_trigger_prompt_selected",
                        client_uid=client_uid,
                        trigger_name=trigger_name,
                        history_uid=context.history_uid,
                    )
                    self._remember_turn_id_for_client_group(client_uid, turn_id)
                    if self.display_live2d_open:
                        self._queue_pending_wake_trigger(
                            client_uid=client_uid,
                            trigger_name=trigger_name,
                            turn_id=turn_id,
                        )
                    else:
                        self._start_trigger_prompt_task(
                            client_uid=client_uid,
                            trigger_name=trigger_name,
                            turn_id=turn_id,
                        )
            return

        if action == "barrage-status":
            from .barrage_adapter import _adapter_instance
            if _adapter_instance:
                status = _adapter_instance.get_queue_status()
                await self._send_text_to_client(
                    client_uid,
                    json.dumps(
                        {"type": "barrage-status", "data": status},
                        ensure_ascii=False,
                    ),
                )
                logger.info(
                    f"[console] barrage-status: {json.dumps(status, ensure_ascii=False)}"
                )
            else:
                await self._send_text_to_client(
                    client_uid,
                    json.dumps(
                        {"type": "barrage-status", "data": None, "error": "adapter not running"},
                        ensure_ascii=False,
                    ),
                )
            return

        if action == "barrage-reconnect":
            from .barrage_adapter import _adapter_instance
            if _adapter_instance:
                _adapter_instance.force_reconnect()
                await self._send_text_to_client(
                    client_uid,
                    json.dumps(
                        {"type": "barrage-reconnect", "ok": True,
                         "message": "已触发重连"},
                        ensure_ascii=False,
                    ),
                )
                logger.info("[console] barrage-reconnect triggered")
            else:
                await self._send_text_to_client(
                    client_uid,
                    json.dumps(
                        {"type": "barrage-reconnect", "ok": False,
                         "error": "adapter not running"},
                        ensure_ascii=False,
                    ),
                )
            return

        if action == "barrage-info":
            from .barrage_adapter import _adapter_instance
            if _adapter_instance:
                info = _adapter_instance.get_connection_info()
                await self._send_text_to_client(
                    client_uid,
                    json.dumps(
                        {"type": "barrage-info", "data": info},
                        ensure_ascii=False,
                    ),
                )
                logger.info(
                    f"[console] barrage-info: "
                    f"{json.dumps(info, ensure_ascii=False)}"
                )
            else:
                await self._send_text_to_client(
                    client_uid,
                    json.dumps(
                        {"type": "barrage-info", "data": None,
                         "error": "adapter not running"},
                        ensure_ascii=False,
                    ),
                )
            return

        # ---- 自定义筛选/排序 (阶段4) ----
        if action == "director-metrics":
            from .barrage_adapter import _adapter_instance

            variables = self._director_metrics_to_barrage_variables(
                data.get("metrics"),
            )
            if _adapter_instance:
                snapshot = _adapter_instance.update_custom_config(variables)
                await self._send_text_to_client(
                    client_uid,
                    json.dumps(
                        {
                            "type": "director-metrics",
                            "ok": True,
                            "variables": variables,
                            "data": snapshot,
                        },
                        ensure_ascii=False,
                    ),
                )
                logger.info(
                    "[console] director-metrics -> variables={} snapshot={}",
                    json.dumps(variables, ensure_ascii=False),
                    json.dumps(snapshot, ensure_ascii=False),
                )
            else:
                await self._send_text_to_client(
                    client_uid,
                    json.dumps(
                        {
                            "type": "director-metrics",
                            "ok": False,
                            "variables": variables,
                            "error": "adapter not running",
                        },
                        ensure_ascii=False,
                    ),
                )
            return

        if action == "barrage-custom-config":
            from .barrage_adapter import _adapter_instance
            if _adapter_instance:
                variables = data.get("variables") or []
                snapshot = _adapter_instance.update_custom_config(variables)
                await self._send_text_to_client(
                    client_uid,
                    json.dumps(
                        {"type": "barrage-custom-config",
                         "ok": True, "data": snapshot},
                        ensure_ascii=False,
                    ),
                )
                logger.info(
                    f"[console] barrage-custom-config -> "
                    f"{json.dumps(snapshot, ensure_ascii=False)}"
                )
            else:
                await self._send_text_to_client(
                    client_uid,
                    json.dumps(
                        {"type": "barrage-custom-config",
                         "ok": False, "error": "adapter not running"},
                        ensure_ascii=False,
                    ),
                )
            return

        if action == "barrage-custom-status":
            from .barrage_adapter import _adapter_instance
            if _adapter_instance:
                snapshot = _adapter_instance._snapshot_custom_config()
                await self._send_text_to_client(
                    client_uid,
                    json.dumps(
                        {"type": "barrage-custom-status", "data": snapshot},
                        ensure_ascii=False,
                    ),
                )
            else:
                await self._send_text_to_client(
                    client_uid,
                    json.dumps(
                        {"type": "barrage-custom-status",
                         "data": None, "error": "adapter not running"},
                        ensure_ascii=False,
                    ),
                )
            return

        if action == "barrage-metrics":
            from .barrage_adapter import _adapter_instance
            if _adapter_instance:
                metrics = _adapter_instance.get_metrics()
                await self._send_text_to_client(
                    client_uid,
                    json.dumps(
                        {"type": "barrage-metrics", "data": metrics},
                        ensure_ascii=False,
                    ),
                )
            else:
                await self._send_text_to_client(
                    client_uid,
                    json.dumps(
                        {"type": "barrage-metrics",
                         "data": None, "error": "adapter not running"},
                        ensure_ascii=False,
                    ),
                )
            return

        if action == "barrage-runtime-config":
            from .barrage_adapter import _adapter_instance
            if not _adapter_instance:
                await self._send_text_to_client(
                    client_uid,
                    json.dumps(
                        {"type": "barrage-runtime-config",
                         "ok": False, "error": "adapter not running"},
                        ensure_ascii=False,
                    ),
                )
                return
            patch = data.get("patch")
            if patch is None:
                # 读: 返回所有可热更新字段当前值
                cfg = _adapter_instance.get_runtime_config()
                await self._send_text_to_client(
                    client_uid,
                    json.dumps(
                        {"type": "barrage-runtime-config", "data": cfg},
                        ensure_ascii=False,
                    ),
                )
            else:
                result = _adapter_instance.update_runtime_config(patch)
                await self._send_text_to_client(
                    client_uid,
                    json.dumps(
                        {"type": "barrage-runtime-config",
                         "ok": True, "data": result},
                        ensure_ascii=False,
                    ),
                )
                logger.info(
                    f"[console] barrage-runtime-config: "
                    f"{json.dumps(result, ensure_ascii=False)}"
                )
            return

        if action == "barrage-diamond-list":
            from .barrage_adapter import _adapter_instance
            if _adapter_instance:
                top_n = int(data.get("top_n") or 0)
                ranking = _adapter_instance.get_diamond_ranking(top_n=top_n)
                await self._send_text_to_client(
                    client_uid,
                    json.dumps(
                        {"type": "barrage-diamond-list", "data": ranking},
                        ensure_ascii=False,
                    ),
                )
                logger.info(
                    f"[console] barrage-diamond-list (top_n={top_n}): "
                    f"users={ranking['total_users']} "
                    f"total_diamonds={ranking['total_diamonds']}"
                )
            else:
                await self._send_text_to_client(
                    client_uid,
                    json.dumps(
                        {"type": "barrage-diamond-list",
                         "data": None, "error": "adapter not running"},
                        ensure_ascii=False,
                    ),
                )
            return

        if action == "barrage-reset-diamond":
            from .barrage_adapter import _adapter_instance
            if _adapter_instance:
                cleared = _adapter_instance.reset_session_diamond_totals()
                await self._send_text_to_client(
                    client_uid,
                    json.dumps(
                        {"type": "barrage-reset-diamond",
                         "ok": True, "cleared_users": cleared},
                        ensure_ascii=False,
                    ),
                )
                logger.info(
                    f"[console] barrage-reset-diamond: 清空 {cleared} 个用户"
                )
            else:
                await self._send_text_to_client(
                    client_uid,
                    json.dumps(
                        {"type": "barrage-reset-diamond",
                         "ok": False, "error": "adapter not running"},
                        ensure_ascii=False,
                    ),
                )
            return

        if action not in {"voice-cutoff", "voice-cut", "interrupt"}:
            return

        heard_response = data.get("text", "")
        turn_id = self._get_message_turn_id(data) or self.current_turn_ids.get(
            client_uid
        )
        record_turn_event(
            turn_id,
            "websocket_handler",
            "console_interrupt_received",
            client_uid=client_uid,
            action=action,
            heard_response_len=len(heard_response),
            interrupted=True,
        )
        context = self.client_contexts[client_uid]
        group = self.chat_group_manager.get_client_group(client_uid)

        if group and len(group.members) > 1:
            await handle_group_interrupt(
                group_id=group.group_id,
                heard_response=heard_response,
                current_conversation_tasks=self.current_conversation_tasks,
                chat_group_manager=self.chat_group_manager,
                client_contexts=self.client_contexts,
                broadcast_to_group=self.broadcast_to_group,
                turn_id=turn_id,
            )
        else:
            interrupt_payload = {"type": "interrupt-signal", "text": heard_response}
            if turn_id:
                interrupt_payload["turn_id"] = turn_id
            await self._send_text_to_client(
                client_uid,
                json.dumps(interrupt_payload, ensure_ascii=False),
            )
            await handle_individual_interrupt(
                client_uid=client_uid,
                current_conversation_tasks=self.current_conversation_tasks,
                context=context,
                heard_response=heard_response,
                turn_id=turn_id,
            )
            self._mark_memory_reload_after_interrupt(
                client_uid,
                turn_id,
                action,
            )

    async def _handle_history_list_request(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request for chat history list"""
        context = self.client_contexts[client_uid]
        histories = get_history_list(context.character_config.conf_uid)
        await websocket.send_text(
            json.dumps({"type": "history-list", "histories": histories})
        )

    async def _handle_fetch_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle fetching and setting specific chat history"""
        history_uid = data.get("history_uid")
        if not history_uid:
            return

        context = self.client_contexts[client_uid]
        # Update history_uid in service context
        self._clear_visual_image_context(client_uid, "history-switch")
        context.history_uid = history_uid
        context.agent_engine.set_memory_from_history(
            conf_uid=context.character_config.conf_uid,
            history_uid=history_uid,
        )

        messages = [
            msg
            for msg in get_history(
                context.character_config.conf_uid,
                history_uid,
            )
            if msg["role"] != "system"
        ]
        await websocket.send_text(
            json.dumps({"type": "history-data", "messages": messages})
        )
        await websocket.send_text(
            json.dumps(
                {
                    "type": "story-state",
                    "story_state": self._story_state_for_context(context),
                },
                ensure_ascii=False,
            )
        )

    async def _handle_create_history(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle creation of new chat history"""
        context = self.client_contexts[client_uid]
        history_uid = create_new_history(context.character_config.conf_uid)
        if history_uid:
            self._clear_visual_image_context(client_uid, "new-history")
            context.history_uid = history_uid
            context.agent_engine.set_memory_from_history(
                conf_uid=context.character_config.conf_uid,
                history_uid=history_uid,
            )
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "new-history-created",
                        "history_uid": history_uid,
                    }
                )
            )
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "story-state",
                        "story_state": self._story_state_for_context(context),
                    },
                    ensure_ascii=False,
                )
            )

    async def _handle_delete_history(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle deletion of chat history"""
        history_uid = data.get("history_uid")
        if not history_uid:
            return

        context = self.client_contexts[client_uid]
        success = delete_history(
            context.character_config.conf_uid,
            history_uid,
        )
        await websocket.send_text(
            json.dumps(
                {
                    "type": "history-deleted",
                    "success": success,
                    "history_uid": history_uid,
                }
            )
        )
        if history_uid == context.history_uid:
            context.history_uid = None

    async def _handle_audio_data(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle incoming audio data"""
        mic_source = self._mic_source_from_data(data)
        if not self._is_vtuber_accepting_input():
            self._clear_mic_asr_state(
                client_uid,
                "vtuber-input-disabled",
                mic_source=mic_source,
            )
            logger.debug(
                "Dropping mic-audio-data from {} source={} while input is disabled",
                client_uid,
                mic_source,
            )
            return

        audio_data = data.get("audio", [])
        if audio_data:
            self._append_mic_audio_buffer(
                client_uid,
                mic_source,
                np.array(audio_data, dtype=np.float32),
            )

    async def _handle_mic_audio_segment_end(
        self,
        websocket: WebSocket,
        client_uid: str,
        data: WSMessage,
    ) -> None:
        """Start ASR for the current microphone segment without starting LLM yet."""
        mic_source = self._mic_source_from_data(data)
        if not self._is_vtuber_accepting_input():
            self._clear_mic_asr_state(
                client_uid,
                "vtuber-input-disabled",
                mic_source=mic_source,
            )
            logger.debug(
                "Dropping mic-audio-segment-end from {} source={} while input is disabled",
                client_uid,
                mic_source,
            )
            return

        started = self._start_mic_asr_segment_from_buffer(
            client_uid,
            "mic-audio-segment-end",
            mic_source=mic_source,
        )
        logger.info(
            "Microphone audio segment end handled: client={} source={} started_asr={}",
            client_uid,
            mic_source,
            started,
        )

    async def _handle_raw_audio_data(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle incoming raw audio data for VAD processing"""
        if not self._is_vtuber_accepting_input():
            self._clear_mic_asr_state(client_uid, "vtuber-input-disabled")
            logger.debug("Dropping raw-audio-data from {} while input is disabled", client_uid)
            return

        logger.warning('detected audio data {}'.format(client_uid))
        context = self.client_contexts[client_uid]
        chunk = data.get("audio", [])
        if chunk:
            for audio_bytes in context.vad_engine.detect_speech(chunk):
                if audio_bytes == b"<|PAUSE|>":
                    control_payload = {"type": "control", "text": "interrupt"}
                    turn_id = self.current_turn_ids.get(client_uid)
                    record_turn_event(
                        turn_id,
                        "websocket_handler",
                        "vad_pause_detected",
                        client_uid=client_uid,
                        interrupted=True,
                    )
                    if turn_id:
                        control_payload["turn_id"] = turn_id
                    await self._send_text_to_client(
                        client_uid,
                        json.dumps(control_payload)
                    )
                elif audio_bytes == b"<|RESUME|>":
                    record_turn_event(
                        self.current_turn_ids.get(client_uid),
                        "websocket_handler",
                        "vad_resume_detected",
                        client_uid=client_uid,
                    )
                    pass
                elif len(audio_bytes) > 1024:
                    # Detected audio activity (voice)
                    self.received_data_buffers[client_uid] = np.append(
                        self.received_data_buffers[client_uid],
                        np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32),
                    )
                    await websocket.send_text(
                        json.dumps({"type": "control", "text": "mic-audio-end"})
                    )
                    record_turn_event(
                        self.current_turn_ids.get(client_uid),
                        "websocket_handler",
                        "vad_audio_end_prompted",
                        client_uid=client_uid,
                        audio_bytes=len(audio_bytes),
                    )

    async def _handle_conversation_trigger(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle triggers that start a conversation"""
        mic_source = (
            self._mic_source_from_data(data)
            if data.get("type") == "mic-audio-end"
            else "local"
        )
        if self.display_wake_animation_pending:
            if data.get("type") == "mic-audio-end":
                self._clear_mic_asr_state(
                    client_uid,
                    "mic-audio-end while wake animation pending",
                    mic_source=mic_source,
                )
            logger.info(
                "Dropping conversation trigger from {} while wake animation is pending: {}",
                client_uid,
                data.get("type"),
            )
            record_turn_event(
                self.current_turn_ids.get(client_uid),
                "websocket_handler",
                "conversation_trigger_dropped_wake_animation_pending",
                client_uid=client_uid,
                message_type=data.get("type"),
                has_audio=bool(data.get("audio")),
                has_text=bool(data.get("text")),
            )
            return

        if not self._is_vtuber_accepting_input():
            if data.get("type") == "mic-audio-end":
                self._clear_mic_asr_state(
                    client_uid,
                    "mic-audio-end while input disabled",
                    mic_source=mic_source,
                )
            logger.info(
                "Dropping conversation trigger from {} while input is disabled: {}",
                client_uid,
                data.get("type"),
            )
            return

        turn_id = self._new_turn_id()
        record_turn_event(
            turn_id,
            "websocket_handler",
            "turn_created",
            client_uid=client_uid,
            trigger_type=data.get("type"),
            has_images=bool(data.get("images")),
            text_len=len(data.get("text") or ""),
            audio_samples=len(data.get("audio") or []),
        )
        if data.get("type") in {"mic-audio-end", "text-input"}:
            self._reset_cold_silence_streak(f"user-trigger:{data.get('type')}")

        if data.get("type") == "mic-audio-end":
            input_text = await self._finalize_mic_asr_text(
                client_uid,
                turn_id,
                mic_source=mic_source,
            )
            if not input_text:
                logger.info(
                    "Dropping mic-audio-end from {} source={} because ASR produced no text",
                    client_uid,
                    mic_source,
                )
                record_turn_event(
                    turn_id,
                    "websocket_handler",
                    "mic_audio_confirm_dropped_empty_asr",
                    client_uid=client_uid,
                )
                return

            transcription_payload = {
                "type": "user-input-transcription",
                "text": input_text,
                "turn_id": turn_id,
            }
            performance_id = str(data.get("performance_id") or "").strip()
            if performance_id:
                transcription_payload["performance_id"] = performance_id
            await self._send_text_to_client(
                client_uid,
                json.dumps(transcription_payload, ensure_ascii=False),
            )
            data = dict(data)
            data["text"] = input_text
            data["pre_transcribed_audio"] = True
            data["metadata"] = self._metadata_with_mic_source(data, mic_source)
            record_turn_event(
                turn_id,
                "websocket_handler",
                "mic_audio_confirm_transcript_ready",
                client_uid=client_uid,
                mic_source=mic_source,
                input_text_len=len(input_text),
                input_text_preview=input_text[:120],
            )
            data = await self._attach_game_vision_images_for_reply(
                client_uid,
                turn_id,
                data,
                mic_source,
            )

        if data.get("type") in {"mic-audio-end", "text-input"}:
            data = self._prepare_visual_image_context_for_reply(
                client_uid,
                turn_id,
                data,
            )

        context = self.client_contexts[client_uid]
        metadata_for_story = (
            data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        )
        if (
            data.get("type") in {"mic-audio-end", "text-input"}
            and self._is_story_interaction_mode()
            and self._context_has_story(context)
            and not bool(metadata_for_story.get("skip_story_match"))
        ):
            story_match = self._match_story_for_turn(
                context=context,
                user_text=str(data.get("text") or ""),
                story_candidates=data.get("story_candidates"),
            )
            if story_match:
                await self._send_text_to_client(
                    client_uid,
                    json.dumps(
                        {
                            "type": "story-state",
                            "story_state": story_match["story_state"],
                            "turn_id": turn_id,
                        },
                        ensure_ascii=False,
                    ),
                )
                data["story_guidance"] = story_match["guidance"]
                data["story_match"] = {
                    "entry": story_match["entry"],
                    "score": story_match["score"],
                    "next_index": story_match["next_index"],
                }
                record_turn_event(
                    turn_id,
                    "websocket_handler",
                    "story_match_applied",
                    client_uid=client_uid,
                    story_index=story_match["entry"].get("index"),
                    match_score=story_match["score"],
                )
            else:
                record_turn_event(
                    turn_id,
                    "websocket_handler",
                    "story_match_missing_llm_fallback",
                    client_uid=client_uid,
                    text_len=len(data.get("text") or ""),
                )
                logger.info(
                    "Story mode input from {} did not match current story window; "
                    "falling back to normal LLM response.",
                    client_uid,
                )

        if data.get("type") in {"mic-audio-end", "text-input"}:
            data = self._apply_pending_memory_reload_metadata(
                client_uid,
                turn_id,
                data,
            )

        if await self._route_mic_conversation_to_orchestrator(
            client_uid,
            data,
            turn_id,
            mic_source,
        ):
            return

        await self._dispatch_prepared_conversation(
            websocket,
            client_uid,
            data,
            turn_id,
        )

    async def _handle_fetch_configs(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle fetching available configurations"""
        context = self.client_contexts[client_uid]
        config_files = scan_config_alts_directory(context.system_config.config_alts_dir)
        await websocket.send_text(
            json.dumps({"type": "config-files", "configs": config_files})
        )

    async def _handle_config_switch(
        self, websocket: WebSocket, client_uid: str, data: dict
    ):
        """Handle switching to a different configuration"""
        config_file_name = data.get("file")
        if config_file_name:
            context = self.client_contexts[client_uid]
            self._clear_visual_image_context(client_uid, "config-switch")
            await context.handle_config_switch(
                self.client_connections.get(client_uid, websocket),
                config_file_name,
            )
            await self.apply_project_model_config()

    async def _handle_fetch_backgrounds(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle fetching available background images"""
        bg_files = scan_bg_directory()
        await websocket.send_text(
            json.dumps({"type": "background-files", "files": bg_files})
        )

    async def _handle_audio_play_start(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """
        Handle audio playback start notification
        """
        return
        group_members = self.chat_group_manager.get_group_members(client_uid)
        if len(group_members) > 1:
            display_text = data.get("display_text")
            if display_text:
                silent_payload = prepare_audio_payload(
                    audio_path=None,
                    display_text=display_text,
                    actions=None,
                    forwarded=True,
                    turn_id=self._get_message_turn_id(data),
                )
                await self.broadcast_to_group(
                    group_members, silent_payload, exclude_uid=client_uid
                )

    async def _handle_group_info(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle group info request"""
        await self.send_group_update(websocket, client_uid)

    async def _handle_init_config_request(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle request for initialization configuration"""
        context = self.client_contexts.get(client_uid)
        if not context:
            context = self.default_context_cache

        await websocket.send_text(
            json.dumps(
                self._model_and_conf_payload(context, client_uid),
                ensure_ascii=False,
            )
        )

    async def _handle_heartbeat(
        self, websocket: WebSocket, client_uid: str, data: WSMessage
    ) -> None:
        """Handle heartbeat messages from clients"""
        try:
            await websocket.send_json({"type": "heartbeat-ack"})
        except Exception as e:
            logger.error(f"Error sending heartbeat acknowledgment: {e}")
