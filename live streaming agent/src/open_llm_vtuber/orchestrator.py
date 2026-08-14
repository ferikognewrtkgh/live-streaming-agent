"""
orchestrator.py - Orchestrator + State Machine + Persona

Decision layer core logic:
  1. Dequeue message
  2. Read Persona state: decide whether to respond, emotion affects priority
  3. State machine check: is current state allowed
  4. Decide interrupt strategy: INSERT / MERGE / DISCARD
  5. Execute: inject text-input with Persona context
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Dict, Any
from loguru import logger

from .persona import Persona, get_persona


# ============================================================
# Priority & Message
# ============================================================

class MsgPriority(int, Enum):
    """Message priority (lower number = higher priority)"""
    PARTNER = 0     # partner streamer
    MIC = 1         # local microphone input
    VIP = 2         # VIP / high-value barrage
    CONTINUE = 3    # continuous conversation follow-up
    BARRAGE = 4     # normal barrage
    EVENT = 5       # Event Loop trigger


class MsgSource(str, Enum):
    MIC = "mic"
    BARRAGE = "barrage"
    PARTNER = "partner"
    EVENT_LOOP = "event_loop"
    CONTINUE = "continue"


@dataclass(order=True)
class OrchestratorMessage:
    """Unified message format for Orchestrator queue"""
    priority: int
    timestamp: float = field(compare=False, default_factory=time.time)
    source: str = field(compare=False, default=MsgSource.BARRAGE)
    text: str = field(compare=False, default="")
    user_id: str = field(compare=False, default="")
    nickname: str = field(compare=False, default="")
    extra: Dict[str, Any] = field(compare=False, default_factory=dict)


# ============================================================
# State Machine
# ============================================================

class State(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTING = "interrupting"


class InterruptStrategy(str, Enum):
    INSERT = "insert"       # stop current TTS immediately, process new message
    MERGE = "merge"         # finish current sentence, then process new message
    DISCARD = "discard"     # drop new message, continue current playback


class StateMachine:
    """
    Simple state machine, no third-party library dependency.

    Transition table:
      idle         + new_message  -> thinking
      thinking     + llm_done    -> speaking
      thinking     + interrupt   -> interrupting  (high priority only)
      speaking     + tts_done    -> idle
      speaking     + interrupt   -> interrupting  (high priority only)
      interrupting + done        -> idle          (ready to re-process)
    """

    _TRANSITIONS = {
        (State.IDLE, "new_message"):      State.THINKING,
        (State.THINKING, "llm_done"):     State.SPEAKING,
        (State.THINKING, "tts_done"):     State.IDLE,
        (State.THINKING, "interrupt"):    State.INTERRUPTING,
        (State.SPEAKING, "tts_done"):     State.IDLE,
        (State.SPEAKING, "interrupt"):    State.INTERRUPTING,
        (State.INTERRUPTING, "done"):     State.IDLE,
    }

    def __init__(self):
        self._state = State.IDLE
        self._callbacks: Dict[str, list] = {}

    @property
    def state(self) -> State:
        return self._state

    def can_trigger(self, trigger: str) -> bool:
        return (self._state, trigger) in self._TRANSITIONS

    def trigger(self, trigger: str) -> bool:
        """Execute state transition. Returns True if successful."""
        key = (self._state, trigger)
        if key not in self._TRANSITIONS:
            logger.debug(f"[StateMachine] ignored: {self._state.value} + {trigger}")
            return False
        old = self._state
        self._state = self._TRANSITIONS[key]
        logger.info(f"[StateMachine] {old.value} --{trigger}--> {self._state.value}")
        for cb in self._callbacks.get(trigger, []):
            cb(old, self._state)
        for cb in self._callbacks.get("*", []):
            cb(old, self._state)
        return True

    def on(self, trigger: str, callback: Callable):
        """Register state transition callback"""
        self._callbacks.setdefault(trigger, []).append(callback)

    def is_idle(self) -> bool:
        return self._state == State.IDLE

    def is_busy(self) -> bool:
        return self._state != State.IDLE


# ============================================================
# Orchestrator
# ============================================================

class Orchestrator:
    """
    Orchestrator - combines state machine + Persona to decide
    whether to respond, whether to interrupt, and how.
    """

    INTERRUPT_THRESHOLD = MsgPriority.MIC
    QUEUE_MAX_SIZE = 100

    def __init__(self, ws_handler, persona: Optional[Persona] = None):
        self.ws_handler = ws_handler
        self.sm = StateMachine()
        self.persona: Persona = persona or get_persona()
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=self.QUEUE_MAX_SIZE
        )

        self._current_msg: Optional[OrchestratorMessage] = None
        self._pending_merge: Optional[OrchestratorMessage] = None
        self._current_task: Optional[asyncio.Task] = None
        self._current_generation = 0
        self._running = False
        self._tasks: list = []

        # State machine callbacks
        self.sm.on("tts_done", self._on_idle)
        self.sm.on("new_message", self._on_thinking)
        self.sm.on("interrupt", self._on_interrupt)

    # --------------------------------------------------
    # Public interface
    # --------------------------------------------------

    async def start(self):
        """Start Orchestrator main loop"""
        if self._running:
            return
        self._running = True
        logger.info("[Orchestrator] started")
        self._tasks = [asyncio.create_task(self._main_loop())]

    async def stop(self):
        """Stop Orchestrator"""
        self._running = False
        self._pending_merge = None
        self._current_generation += 1
        current_task = self._current_task
        self._current_task = None
        self._current_msg = None
        for t in self._tasks:
            t.cancel()
        if current_task and not current_task.done():
            current_task.cancel()
        logger.info("[Orchestrator] stopped")

    async def put_message(self, msg: OrchestratorMessage):
        """Submit a message to the Orchestrator. All input sources call this."""
        # Persona affinity-driven priority adjustment
        if msg.user_id and msg.source == MsgSource.BARRAGE:
            boost = self.persona.interrupt_priority_boost(msg.user_id)
            adjusted = max(0, int(msg.priority) + boost)
            if adjusted != msg.priority:
                logger.debug(
                    f"[Orchestrator] affinity priority adjust: P{msg.priority}->P{adjusted} "
                    f"({msg.nickname})"
                )
                msg = OrchestratorMessage(
                    priority=adjusted,
                    timestamp=msg.timestamp,
                    source=msg.source,
                    text=msg.text,
                    user_id=msg.user_id,
                    nickname=msg.nickname,
                    extra=msg.extra,
                )

        try:
            self.queue.put_nowait(msg)
            logger.debug(
                f"[Orchestrator] enqueued [{msg.source}] P{msg.priority}: {msg.text[:40]}"
            )
        except asyncio.QueueFull:
            if msg.priority <= MsgPriority.VIP:
                await self._evict_lowest_priority()
                await self.queue.put(msg)
                logger.warning(f"[Orchestrator] evicted low-pri, force enqueued: {msg.text[:40]}")
            else:
                logger.debug(f"[Orchestrator] queue full, dropped: {msg.text[:40]}")

    @property
    def state(self) -> State:
        return self.sm.state

    def _source_value(self, msg: OrchestratorMessage) -> str:
        source = msg.source
        if isinstance(source, MsgSource):
            return source.value
        return str(source or "")

    def _resolve_target_connection(
        self,
        msg: OrchestratorMessage,
    ) -> tuple[str, Any | None]:
        requested_uid = str(msg.extra.get("client_uid") or "").strip()
        if requested_uid:
            connection = self.ws_handler.client_connections.get(requested_uid)
            if connection:
                return requested_uid, connection

        if not self.ws_handler.client_connections:
            return "", None

        target_uid = next(iter(self.ws_handler.client_connections))
        return target_uid, self.ws_handler.client_connections.get(target_uid)

    # --------------------------------------------------
    # Main loop
    # --------------------------------------------------

    async def _main_loop(self):
        """Orchestrator main loop: dequeue messages and dispatch"""
        while self._running:
            try:
                msg: OrchestratorMessage = await asyncio.wait_for(
                    self.queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                return

            try:
                await self._dispatch(msg)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"[Orchestrator] dispatch error: {e}")
                logger.exception(e)
                # Recovery: ensure state machine returns to IDLE so we can
                # continue processing future messages
                if self.sm.state != State.IDLE:
                    logger.warning(
                        f"[Orchestrator] state machine stuck in "
                        f"{self.sm.state.value}, forcing reset to idle"
                    )
                    self.sm._state = State.IDLE
                self._current_task = None
                self._current_msg = None

    # --------------------------------------------------
    # Decision layer (Persona + State Machine)
    # --------------------------------------------------

    async def _dispatch(self, msg: OrchestratorMessage):
        """
        Decision layer: combine Persona state + state machine
        to decide whether to respond and how to handle conflicts.

        Decision order:
          1. Response rate limit check (关键词弹幕跳过)
          2. Persona.should_respond() - personality level filter (关键词弹幕跳过)
          3. State machine state - can we execute now
          4. Interrupt strategy - how to handle conflict
        """
        is_keyword = msg.extra.get("is_keyword", False)

        # -- Step 0: Response rate limit check --
        # 关键词弹幕跳过频率限制（用户主动触发，不应被静默丢弃）
        if msg.source == MsgSource.BARRAGE and msg.user_id and not is_keyword:
            from .barrage_filter import get_response_rate_limiter
            limiter = get_response_rate_limiter()
            if limiter and not limiter.can_respond(msg.user_id):
                logger.info(
                    f"[Orchestrator] 回复频率限制: {msg.nickname} 5分钟内已回复上限"
                )
                return

        # -- Step 1: Persona decides whether to respond --
        # 关键词弹幕跳过 persona 过滤（用户用了关键词说明想互动）
        if msg.source == MsgSource.BARRAGE and msg.user_id and not is_keyword:
            if not self.persona.should_respond(msg.user_id, msg.text):
                logger.info(
                    f"[Orchestrator] Persona decided not to respond to {msg.nickname}: {msg.text[:30]}"
                )
                return

        # -- Step 2: Update Persona emotion --
        if msg.source == MsgSource.BARRAGE and msg.user_id:
            self.persona.on_barrage(
                user_id=msg.user_id,
                nickname=msg.nickname,
                content=msg.text,
            )

        state = self.sm.state

        # -- Step 3: idle -> process directly --
        if state == State.IDLE:
            await self._process(msg)
            return

        # -- Step 4: non-idle -> decide interrupt strategy --
        if state in (State.THINKING, State.SPEAKING):
            strategy = self._decide_interrupt_strategy(msg)

            if strategy == InterruptStrategy.DISCARD:
                logger.debug(f"[Orchestrator] discarded: {msg.text[:40]}")
                return

            if strategy == InterruptStrategy.INSERT:
                logger.info(f"[Orchestrator] inserting: {msg.text[:40]}")
                await self._interrupt_and_process(msg)
                return

            if strategy == InterruptStrategy.MERGE:
                logger.info(f"[Orchestrator] merging: {msg.text[:40]}")
                self._pending_merge = msg
                return

        # -- Step 5: interrupting state -> store for later --
        if state == State.INTERRUPTING:
            self._pending_merge = msg

    def _decide_interrupt_strategy(
        self, msg: OrchestratorMessage
    ) -> InterruptStrategy:
        """
        Decide interrupt strategy.

        Rules:
          During gift processing: only MIC (P1) can INSERT, all else DISCARD
          mic/partner (P0/P1)  -> INSERT  regardless of emotion
          VIP (P2)             -> MERGE   (upgrade to INSERT when excited)
          keyword barrage (P3) -> MERGE   (always, user explicitly triggered)
          fan club (P3)        -> MERGE   (only when happy/excited)
          normal barrage (P4+) -> DISCARD
        """
        p = msg.priority
        is_keyword = msg.extra.get("is_keyword", False)

        # 礼物模式打断保护：只有 MIC/编导 (P0/P1) 可以打断
        if self._current_msg and self._current_msg.extra.get("is_gift", False):
            if p <= int(MsgPriority.MIC):
                return InterruptStrategy.INSERT
            return InterruptStrategy.DISCARD

        # Mic and partner: unconditional insert
        if p <= int(MsgPriority.MIC):
            return InterruptStrategy.INSERT

        # VIP gift: usually merge, insert when excited
        if p <= int(MsgPriority.VIP):
            from .persona import Emotion
            if self.persona.emotion == Emotion.EXCITED:
                return InterruptStrategy.INSERT
            return InterruptStrategy.MERGE

        # 关键词弹幕：无条件 MERGE（用户主动触发，不应被丢弃）
        if is_keyword:
            return InterruptStrategy.MERGE

        # Fan club continuous conversation: merge when happy/excited
        if p <= int(MsgPriority.CONTINUE):
            from .persona import Emotion
            if self.persona.emotion in (Emotion.HAPPY, Emotion.EXCITED):
                return InterruptStrategy.MERGE

        # Normal barrage: discard
        return InterruptStrategy.DISCARD

    # --------------------------------------------------
    # Execution layer
    # --------------------------------------------------

    async def _process(self, msg: OrchestratorMessage):
        """Trigger state machine and start one conversation without blocking queue dispatch."""
        if not self.sm.trigger("new_message"):
            logger.warning("[Orchestrator] state machine rejected new_message")
            return

        self._current_generation += 1
        generation = self._current_generation
        self._current_msg = msg
        self._current_task = asyncio.create_task(self._run_conversation(msg))
        self._current_task.add_done_callback(
            lambda task, expected_generation=generation: asyncio.create_task(
                self._handle_conversation_task_done(task, expected_generation)
            )
        )

    async def _handle_conversation_task_done(
        self,
        task: asyncio.Task,
        expected_generation: int,
    ) -> None:
        """Clean up the active task if it is still the current generation."""
        try:
            task.result()
        except asyncio.CancelledError:
            logger.info("[Orchestrator] conversation task cancelled")
        except Exception as e:
            logger.error(f"[Orchestrator] conversation task error: {e}")
            logger.exception(e)

        if (
            self._current_task is not task
            or self._current_generation != expected_generation
        ):
            logger.debug("[Orchestrator] ignoring stale conversation task cleanup")
            return

        self._current_task = None
        self._current_msg = None

        # Ensure state machine returns to IDLE so future messages can be processed.
        if not self.sm.is_idle():
            logger.warning(
                f"[Orchestrator] state stuck in {self.sm.state.value} "
                f"after task ended, triggering tts_done"
            )
            self.sm.trigger("tts_done")
            if not self.sm.is_idle():
                logger.warning(
                    f"[Orchestrator] forcing state reset from {self.sm.state.value} to idle"
                )
                self.sm._state = State.IDLE

        if self._pending_merge and self.sm.is_idle():
            pending = self._pending_merge
            self._pending_merge = None
            if self._running:
                await self._process(pending)

    async def _run_conversation(self, msg: OrchestratorMessage):
        """
        Execute conversation:
          1. Get Persona context
          2. Prepend context to message (let LLM sense current state)
          3. Inject text-input
          4. Wait for completion
          5. Update Persona
        """
        if not self.ws_handler.client_connections:
            logger.warning("[Orchestrator] no active client, skipping")
            self.sm.trigger("tts_done")
            return

        target_uid, websocket = self._resolve_target_connection(msg)
        if not websocket:
            self.sm.trigger("tts_done")
            return

        source_value = self._source_value(msg)
        prepared_data = msg.extra.get("conversation_data")
        if isinstance(prepared_data, dict):
            fake_msg = dict(prepared_data)
            metadata = fake_msg.get("metadata")
            metadata = dict(metadata) if isinstance(metadata, dict) else {}
            metadata.update(
                {
                    "orchestrator_dispatched": True,
                    "orchestrator_source": source_value,
                    "orchestrator_priority": int(msg.priority),
                }
            )
            fake_msg["metadata"] = metadata
            turn_id = str(msg.extra.get("turn_id") or "").strip() or None
            logger.info(
                "[Orchestrator] dispatching prepared conversation "
                "(source={} priority={} turn_id={}): {}",
                source_value,
                int(msg.priority),
                turn_id,
                msg.text[:50],
            )
            await self.ws_handler._dispatch_prepared_conversation(
                websocket,
                target_uid,
                fake_msg,
                turn_id,
            )
            await self._wait_task_done(target_uid)
            self.persona.on_ai_spoke(response_text="")
            self.sm.trigger("tts_done")
            return

        # Get Persona context (inject to LLM)
        persona_context = self.persona.to_prompt_context(
            current_user_id=msg.user_id or None
        )

        # Prepend Persona context to message
        enriched_text = f"{persona_context}\n\n{msg.text}"

        fake_msg = {"type": "text-input", "text": enriched_text}
        if msg.source == MsgSource.BARRAGE:
            metadata = {
                "human_name": "礼物" if msg.extra.get("is_gift") else "弹幕",
                "input_source": "barrage",
            }
            extra_metadata = msg.extra.get("metadata")
            if isinstance(extra_metadata, dict):
                metadata.update(extra_metadata)
            fake_msg["metadata"] = metadata
        logger.info(
            f"[Orchestrator] injecting conversation (emotion:{self.persona.emotion.value}): "
            f"{msg.text[:50]}"
        )

        # 把"被回复的这条弹幕"推给前端字幕窗口 (弹幕字幕界面用).
        # 只在弹幕来源时发: 用户抖音 id (nickname) + 弹幕原文 (纯内容).
        # msg.text 被格式化成 "[barrage] {nickname}: {content}", 这里剥掉前缀,
        # 只把弹幕内容本身送给前端.
        if msg.source == MsgSource.BARRAGE and (msg.text or msg.nickname):
            barrage_content = msg.text or ""
            prefix = f"[barrage] {msg.nickname}: "
            if msg.nickname and barrage_content.startswith(prefix):
                barrage_content = barrage_content[len(prefix):]
            elif barrage_content.startswith("[barrage]"):
                # 兜底: nickname 含特殊字符时按第一个 ": " 切
                _, sep, tail = barrage_content.partition(": ")
                if sep:
                    barrage_content = tail
            try:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "barrage-display",
                            "nickname": msg.nickname or "",
                            "user_id": msg.user_id or "",
                            "avatar_url": msg.extra.get("avatar_url") or "",
                            "avatar_path": msg.extra.get("avatar_path") or "",
                            "content": barrage_content,
                        },
                        ensure_ascii=False,
                    )
                )
            except Exception as exc:
                logger.debug(f"[Orchestrator] send barrage-display failed: {exc}")

        await self.ws_handler._handle_conversation_trigger(
            websocket, target_uid, fake_msg
        )

        await self._wait_task_done(target_uid)

        # Update Persona after conversation
        self.persona.on_ai_spoke(response_text="")

        # 回复频率计数
        if msg.source == MsgSource.BARRAGE and msg.user_id:
            from .barrage_filter import get_response_rate_limiter
            limiter = get_response_rate_limiter()
            if limiter:
                limiter.record_response(msg.user_id)

        self.sm.trigger("tts_done")

    async def _wait_task_done(self, uid: str, timeout: float = 120.0):
        """Wait for ws_handler conversation task to complete"""
        start = time.time()
        # Wait for task to appear
        while time.time() - start < 3.0:
            if uid in self.ws_handler.current_conversation_tasks:
                task = self.ws_handler.current_conversation_tasks[uid]
                if task and not task.done():
                    break
            await asyncio.sleep(0.1)

        # Wait for task to finish
        while time.time() - start < timeout:
            task = self.ws_handler.current_conversation_tasks.get(uid)
            if not task or task.done():
                return
            await asyncio.sleep(0.3)

        logger.warning("[Orchestrator] wait for conversation done timed out")

    async def _interrupt_and_process(self, msg: OrchestratorMessage):
        """INSERT strategy: stop current conversation, immediately process new message"""
        if not self.sm.trigger("interrupt"):
            logger.warning("[Orchestrator] state machine rejected interrupt")
            return

        current_task = self._current_task
        if current_task and not current_task.done():
            self._current_generation += 1
            self._current_task = None
            self._current_msg = None
            current_task.cancel()
            try:
                await current_task
            except asyncio.CancelledError:
                pass

        # Send interrupt signal to frontend
        target_uid, ws = self._resolve_target_connection(msg)
        if ws:
            try:
                await ws.send_text(
                    json.dumps({"type": "control", "text": "interrupt"})
                )
            except Exception as e:
                logger.warning(f"[Orchestrator] failed to send interrupt signal: {e}")

        if self.sm.state == State.INTERRUPTING:
            self.sm.trigger("done")
        if not self.sm.is_idle():
            logger.warning(
                "[Orchestrator] forcing state reset after interrupt: {}",
                self.sm.state.value,
            )
            self.sm._state = State.IDLE

        await self._process(msg)

    # --------------------------------------------------
    # State machine callbacks
    # --------------------------------------------------

    def _on_thinking(self, old, new):
        logger.debug(f"[Orchestrator] entering thinking | Persona: {self.persona.to_dict()}")

    def _on_idle(self, old, new):
        logger.debug("[Orchestrator] entering idle")

    def _on_interrupt(self, old, new):
        logger.info(f"[Orchestrator] interrupt: {old.value} -> {new.value}")

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    async def _evict_lowest_priority(self):
        """Remove lowest priority message from queue to make room"""
        items = []
        while not self.queue.empty():
            try:
                items.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if not items:
            return
        items.sort()
        items.pop()
        for item in items:
            try:
                self.queue.put_nowait(item)
            except asyncio.QueueFull:
                break


# ============================================================
# Start/stop entry points (called from server.py)
# ============================================================

_orchestrator_instance: Optional[Orchestrator] = None


async def start_orchestrator(ws_handler) -> Orchestrator:
    """
    Start Orchestrator. Call from server.py startup event.

    Example:
        @self.app.on_event("startup")
        async def _startup():
            self._orchestrator = await start_orchestrator(self.ws_handler)
    """
    global _orchestrator_instance
    orch = Orchestrator(ws_handler)
    _orchestrator_instance = orch
    await orch.start()
    return orch


async def stop_orchestrator():
    """Stop the Orchestrator"""
    global _orchestrator_instance
    if _orchestrator_instance:
        await _orchestrator_instance.stop()
        _orchestrator_instance = None


def get_orchestrator() -> Optional[Orchestrator]:
    """Get global Orchestrator instance"""
    return _orchestrator_instance
