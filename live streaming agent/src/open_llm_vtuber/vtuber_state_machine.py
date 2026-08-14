"""
VTuber mode state machine.

There are three independent controls:
- sleep: whether the VTuber is sleeping
- punish: whether the VTuber is in punishment pose
- interaction mode: co_host or barrage

The effective public mode is idle whenever sleep or punish is active. This keeps
all input gates using `sm.mode == VTuberMode.IDLE` working as the single source
of truth for "input disabled".
"""

import json
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


class VTuberMode(str, Enum):
    IDLE = "idle"
    CO_HOST = "co_host"
    BARRAGE = "barrage"


class IdleSubMode(str, Enum):
    SLEEP = "sleep"
    PUNISH = "punish"


class VTuberStateMachine:
    def __init__(self, ws_handler=None):
        self._interaction_mode: VTuberMode = VTuberMode.CO_HOST
        self._sleeping: bool = True
        self._punished: bool = False
        self._mode: VTuberMode = VTuberMode.IDLE
        self._idle_sub_mode: IdleSubMode = IdleSubMode.SLEEP
        self._ws_handler = ws_handler
        self._on_mode_change_callbacks: List[Callable] = []

        logger.info("[VTuberSM] initialized, mode=idle interaction_mode=co_host")

    @property
    def mode(self) -> VTuberMode:
        return self._mode

    @property
    def idle_sub_mode(self) -> IdleSubMode:
        return self._idle_sub_mode

    @property
    def interaction_mode(self) -> VTuberMode:
        return self._interaction_mode

    @property
    def sleeping(self) -> bool:
        return self._sleeping

    @property
    def punished(self) -> bool:
        return self._punished

    def on_mode_change(self, callback: Callable):
        """Register a callback for mode changes: callback(old_mode, new_mode)."""
        self._on_mode_change_callbacks.append(callback)

    async def handle_switch(self, switch: str) -> Dict[str, Any]:
        handlers = {
            "sleep": self._on_sleep,
            "co_host": self._on_co_host,
            "barrage": self._on_barrage,
            "punish": self._on_punish,
        }

        handler = handlers.get(switch)
        if not handler:
            logger.warning(f"[VTuberSM] unknown switch: {switch}")
            return self._state_result(
                old_mode=self._mode,
                action="ignored",
                reason=f"unknown switch: {switch}",
            )

        return await handler()

    async def _on_sleep(self) -> Dict[str, Any]:
        old_mode = self._mode
        self._sleeping = not self._sleeping
        return await self._apply_state_change(old_mode, "sleep")

    async def enter_sleep(
        self,
        reason: str = "force_sleep",
        *,
        interrupt_current: bool = True,
        broadcast_mode_change: bool = True,
    ) -> Dict[str, Any]:
        old_mode = self._mode
        self._sleeping = True
        self._punished = False
        result = await self._apply_state_change(
            old_mode,
            reason,
            interrupt_current=interrupt_current,
        )
        if broadcast_mode_change:
            await self._broadcast_mode_changed(result)
        return result

    async def _on_punish(self) -> Dict[str, Any]:
        old_mode = self._mode
        self._punished = not self._punished
        return await self._apply_state_change(old_mode, "punish")

    async def _on_co_host(self) -> Dict[str, Any]:
        old_mode = self._mode
        self._interaction_mode = VTuberMode.CO_HOST
        return await self._apply_state_change(old_mode, "co_host")

    async def _on_barrage(self) -> Dict[str, Any]:
        old_mode = self._mode
        self._interaction_mode = VTuberMode.BARRAGE
        return await self._apply_state_change(old_mode, "barrage")

    async def _apply_state_change(
        self,
        old_mode: VTuberMode,
        action: str,
        *,
        interrupt_current: bool = True,
    ) -> Dict[str, Any]:
        new_mode = self._effective_mode()
        self._mode = new_mode
        self._idle_sub_mode = self._effective_idle_sub_mode()

        logger.info(
            "[VTuberSM] action={} old_mode={} new_mode={} interaction_mode={} "
            "sleeping={} punished={}",
            action,
            old_mode.value,
            new_mode.value,
            self._interaction_mode.value,
            self._sleeping,
            self._punished,
        )

        if (
            interrupt_current
            and old_mode != VTuberMode.IDLE
            and new_mode == VTuberMode.IDLE
        ):
            await self._interrupt_current_conversations()
            if self._ws_handler and hasattr(
                self._ws_handler,
                "cancel_pending_wake_animation",
            ):
                await self._ws_handler.cancel_pending_wake_animation(
                    f"state-machine:{action}"
                )

        await self._broadcast_animation(self._current_animation())
        await self._fire_callbacks(old_mode, new_mode)
        return self._state_result(old_mode=old_mode, action=action)

    def _effective_mode(self) -> VTuberMode:
        if self._sleeping or self._punished:
            return VTuberMode.IDLE
        return self._interaction_mode

    def _effective_idle_sub_mode(self) -> IdleSubMode:
        if self._punished:
            return IdleSubMode.PUNISH
        return IdleSubMode.SLEEP

    def _current_animation(self) -> str:
        if self._mode == VTuberMode.IDLE:
            return "breathing_only" if self._punished else "sleep_idle"
        return "normal"

    def _state_result(
        self,
        old_mode: VTuberMode,
        action: str,
        reason: str | None = None,
    ) -> Dict[str, Any]:
        result = {
            "old_mode": old_mode.value,
            "new_mode": self._mode.value,
            "sub_mode": (
                self._idle_sub_mode.value if self._mode == VTuberMode.IDLE else None
            ),
            "interaction_mode": self._interaction_mode.value,
            "sleeping": self._sleeping,
            "punished": self._punished,
            "action": action,
        }
        if reason:
            result["reason"] = reason
        return result

    async def _interrupt_current_conversations(self):
        if not self._ws_handler:
            return

        from .conversations.conversation_handler import handle_individual_interrupt

        for uid in list(self._ws_handler.client_connections.keys()):
            payload = {"type": "control", "text": "interrupt"}
            turn_id = self._ws_handler.current_turn_ids.get(uid)
            if turn_id:
                payload["turn_id"] = turn_id
            try:
                await self._ws_handler._send_text_to_client(
                    uid,
                    json.dumps(payload),
                )
            except Exception as e:
                logger.warning(f"[VTuberSM] failed to send interrupt to {uid}: {e}")

        for uid in list(self._ws_handler.client_contexts.keys()):
            task = self._ws_handler.current_conversation_tasks.get(uid)
            if not task or task.done():
                continue
            context = self._ws_handler.client_contexts.get(uid)
            if not context:
                continue
            turn_id = self._ws_handler.current_turn_ids.get(uid)
            try:
                await handle_individual_interrupt(
                    client_uid=uid,
                    current_conversation_tasks=self._ws_handler.current_conversation_tasks,
                    context=context,
                    heard_response="",
                    turn_id=turn_id,
                )
            except Exception as e:
                logger.warning(f"[VTuberSM] interrupt failed for {uid}: {e}")

    async def _broadcast_animation(self, animation: str):
        if not self._ws_handler:
            return

        msg = json.dumps(
            {
                "type": "control",
                "text": "idle-animation",
                "animation": animation,
            }
        )

        for uid in list(self._ws_handler.client_connections.keys()):
            try:
                await self._ws_handler._send_text_to_client(uid, msg)
            except Exception as e:
                logger.warning(f"[VTuberSM] failed to send animation to {uid}: {e}")

    async def _broadcast_mode_changed(self, result: Dict[str, Any]):
        if not self._ws_handler:
            return

        wake_animation_pending = bool(
            getattr(self._ws_handler, "display_wake_animation_pending", False)
        )
        detail = {**result, "wake_animation_pending": wake_animation_pending}

        msg = json.dumps(
            {
                "type": "mode-changed",
                "mode": result["new_mode"],
                "sub_mode": result.get("sub_mode"),
                "interaction_mode": result.get("interaction_mode"),
                "sleeping": result.get("sleeping", False),
                "punished": result.get("punished", False),
                "wake_animation_pending": wake_animation_pending,
                "detail": detail,
            },
            ensure_ascii=False,
        )

        for uid in list(self._ws_handler.client_connections.keys()):
            try:
                await self._ws_handler._send_text_to_client(uid, msg)
            except Exception as e:
                logger.warning(
                    f"[VTuberSM] failed to send mode change to {uid}: {e}"
                )

    async def _fire_callbacks(self, old_mode: VTuberMode, new_mode: VTuberMode):
        for cb in self._on_mode_change_callbacks:
            try:
                cb(old_mode, new_mode)
            except Exception as e:
                logger.error(f"[VTuberSM] callback error: {e}")


_instance: Optional[VTuberStateMachine] = None


def get_vtuber_state_machine() -> Optional[VTuberStateMachine]:
    return _instance


def init_vtuber_state_machine(ws_handler=None) -> VTuberStateMachine:
    global _instance
    _instance = VTuberStateMachine(ws_handler=ws_handler)
    return _instance
