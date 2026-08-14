import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    AsyncIterator,
    List,
    Dict,
    Any,
    Callable,
    Literal,
    Union,
    Optional,
)
from loguru import logger
from ..events import (
    is_web_search_event,
    make_llm_final_error_fallback_event,
    make_llm_first_token_event,
)
from .agent_interface import AgentInterface
from ..output_types import SentenceOutput, DisplayText
from ..stateless_llm.stateless_llm_interface import StatelessLLMInterface
from ..stateless_llm.claude_llm import AsyncLLM as ClaudeAsyncLLM
from ..stateless_llm.openai_compatible_llm import AsyncLLM as OpenAICompatibleAsyncLLM
from ..stateless_llm.provider_runtime_llm import ProviderRuntimeLLM
from ...chat_history_manager import (
    get_history,
    get_latest_summary_payload,
    get_summary_storage_dir,
    record_summary_file,
    update_metadate,
)
from ...blocked_words_loader import sanitize_blocked_words_text_with_matches
from ...conversations.conversation_utils import format_human_input_for_llm
from ..transformers import (
    sentence_divider,
    actions_extractor,
    tts_filter,
    display_processor,
)
from ...config_manager import TTSPreprocessorConfig
from ..input_types import BatchInput, TextSource
from prompts import prompt_loader
from ...mcpp.tool_manager import ToolManager
from ...mcpp.json_detector import StreamJSONDetector
from ...mcpp.types import ToolCallObject
from ...mcpp.tool_executor import ToolExecutor
from ...painting import get_paint_manager, paint_capability_prompt
from ...utils.prompt_trigger_registry import get_prompt_trigger_registry
from ...utils.sentence_divider import paint_command_spans


SUMMARY_TRIGGER_ROUNDS = 20
SUMMARY_ROUNDS_TO_COMPRESS = 10
SUMMARY_MESSAGE_PREFIX = "[Conversation summary of earlier turns]"
SUMMARY_API_ERROR_PREFIX = "Error calling the chat endpoint"
INITIAL_LLM_CALL_MAX_ATTEMPTS = 2
ERROR_TRIGGER_NAME = "error"
DEFAULT_ERROR_FALLBACK_TEXT = "\u6211\u6709\u70b9\u56f0\u4e86\uff0c\u6211\u53bb\u7761\u4e00\u4f1a\u3002"


class BasicMemoryAgent(AgentInterface):
    """Agent with basic chat memory and tool calling support."""

    _system: str = "You are a helpful assistant."

    def __init__(
        self,
        llm: StatelessLLMInterface,
        system: str,
        live2d_model,
        tts_preprocessor_config: TTSPreprocessorConfig = None,
        faster_first_response: bool = True,
        segment_method: str = "pysbd",
        use_mcpp: bool = False,
        interrupt_method: Literal["system", "user"] = "user",
        tool_prompts: Dict[str, str] = None,
        tool_manager: Optional[ToolManager] = None,
        tool_executor: Optional[ToolExecutor] = None,
        mcp_prompt_string: str = "",
        scene_prompts: Optional[Dict[str, Any]] = None,
        vision_llms: Optional[Dict[str, StatelessLLMInterface]] = None,
        default_vision_llm_provider: Optional[str] = None,
    ):
        """Initialize agent with LLM and configuration."""
        super().__init__()
        self._memory: List[Dict[str, Any]] = []
        self._summary_sequence = 0
        self._current_conf_uid = ""
        self._current_history_uid = ""
        self._loaded_summary_files: List[str] = []
        self._live2d_model = live2d_model
        self._tts_preprocessor_config = tts_preprocessor_config
        self._faster_first_response = faster_first_response
        self._segment_method = segment_method
        self._use_mcpp = use_mcpp
        self.interrupt_method = interrupt_method
        self._tool_prompts = tool_prompts or {}
        self._interrupt_handled = False
        self.prompt_mode_flag = False
        self._vision_llms = vision_llms or {}
        self._default_vision_llm_provider = default_vision_llm_provider

        self._tool_manager = tool_manager
        self._tool_executor = tool_executor
        self._mcp_prompt_string = mcp_prompt_string
        self._json_detector = StreamJSONDetector()

        # Scene-based prompt injection: append a scene-specific suffix to the
        # resident persona/system prompt, chosen per-message by input source.
        self._scene_enabled = bool((scene_prompts or {}).get("enabled", False))
        self._scene_sources: Dict[str, str] = (scene_prompts or {}).get(
            "sources", {}
        ) or {}
        self._scene_cache: Dict[str, str] = {}

        # Content/keyword-triggered scene modules: when the input text (mic ASR
        # or barrage) matches a rule's keywords, stack that scene module after
        # the source-based suffix. Keywords are lowercased once here.
        ct = (scene_prompts or {}).get("content_triggers", {}) or {}
        self._content_enabled = bool(ct.get("enabled", False))
        self._content_sources: List[str] = ct.get("sources", []) or []
        self._content_rules: List[tuple] = [
            (r.get("scene"), [str(k).lower() for k in (r.get("keywords") or [])])
            for r in (ct.get("rules") or [])
            if r.get("scene")
        ]

        self._formatted_tools_openai = []
        self._formatted_tools_claude = []
        if self._tool_manager:
            self._formatted_tools_openai = self._tool_manager.get_formatted_tools(
                "OpenAI"
            )
            self._formatted_tools_claude = self._tool_manager.get_formatted_tools(
                "Claude"
            )
            logger.debug(
                f"Agent received pre-formatted tools - OpenAI: {len(self._formatted_tools_openai)}, Claude: {len(self._formatted_tools_claude)}"
            )
        else:
            logger.debug(
                "ToolManager not provided, agent will not have pre-formatted tools."
            )

        self._set_llm(llm)
        self.set_system(system if system else self._system)

        if self._use_mcpp and not all(
            [
                self._tool_manager,
                self._tool_executor,
                self._json_detector,
            ]
        ):
            logger.warning(
                "use_mcpp is True, but some MCP components are missing in the agent. Tool calling might not work as expected."
            )
        elif not self._use_mcpp and any(
            [
                self._tool_manager,
                self._tool_executor,
                self._json_detector,
            ]
        ):
            logger.warning(
                "use_mcpp is False, but some MCP components were passed to the agent."
            )

        logger.info("BasicMemoryAgent initialized.")

    def _llm_for_input(self, input_data: BatchInput) -> StatelessLLMInterface:
        """Use a dedicated visual model only when the turn contains images."""
        if not input_data.images:
            return self._llm

        metadata = input_data.metadata or {}
        requested_provider = metadata.get("vision_model_provider")
        if requested_provider in self._vision_llms:
            logger.info("Using visual LLM provider: {}", requested_provider)
            return self._vision_llms[requested_provider]

        if self._default_vision_llm_provider in self._vision_llms:
            logger.info(
                "Using default visual LLM provider: {}",
                self._default_vision_llm_provider,
            )
            return self._vision_llms[self._default_vision_llm_provider]

        if self._vision_llms:
            fallback_provider = next(iter(self._vision_llms))
            logger.warning(
                "Requested visual provider '{}' is unavailable; using '{}'.",
                requested_provider,
                fallback_provider,
            )
            return self._vision_llms[fallback_provider]

        logger.warning(
            "Image input received but no visual LLM is configured; using the main LLM."
        )
        return self._llm

    def _set_llm(self, llm: StatelessLLMInterface):
        """Set the LLM for chat completion."""
        self._llm = llm
        self.chat = self._chat_function_factory()

    def set_system(self, system: str):
        """Set the system prompt."""
        # logger.debug(f"Memory Agent: Setting system prompt: '''{system}'''")

        if self.interrupt_method == "user":
            system = f"{system}\n\nIf you received `[interrupted by user]` signal, you were interrupted."

        self._system = system

    def _load_scene_cached(self, scene_name: str) -> str:
        """Load a scene prompt by name, caching the result in-process.

        Fails soft: any load error logs and caches an empty string so we don't
        repeatedly hit the disk for a missing/broken scene file.
        """
        if scene_name in self._scene_cache:
            return self._scene_cache[scene_name]

        content = ""
        try:
            content = prompt_loader.load_scene(scene_name).strip()
        except Exception as e:
            logger.warning(f"Failed to load scene prompt '{scene_name}': {e}")
            content = ""

        self._scene_cache[scene_name] = content
        return content

    def resolve_scene_suffix(self, metadata: Optional[Dict[str, Any]]) -> str:
        """Resolve the scene prompt suffix for the given message metadata.

        Returns an empty string when scene injection is disabled, no metadata is
        present, the input source is unknown, or the mapped scene file is empty.
        """
        if not self._scene_enabled or not metadata:
            return ""

        input_source = metadata.get("input_source")
        if not input_source:
            return ""

        scene_name = self._scene_sources.get(input_source)
        if not scene_name:
            return ""

        return self._load_scene_cached(scene_name)

    def resolve_content_suffixes(self, input_data: BatchInput) -> List[str]:
        """Resolve keyword-triggered scene modules for this message.

        Returns scene prompt bodies (in config order, de-duplicated) for every
        rule whose keywords substring-match the input text. Only fires when the
        message comes from a configured content source (e.g. barrage/mic).
        Returns an empty list when disabled, source not gated in, no text, or
        nothing matches.
        """
        if not self._content_enabled:
            return []

        meta = getattr(input_data, "metadata", None) or {}
        if meta.get("input_source") not in self._content_sources:
            return []

        texts = getattr(input_data, "texts", None) or []
        text_lower = " ".join(
            t.content
            for t in texts
            if t.source == TextSource.INPUT and t.content
        ).lower()
        if not text_lower:
            return []

        out: List[str] = []
        seen: set = set()
        for scene, kws in self._content_rules:
            if scene in seen:
                continue
            if any(kw in text_lower for kw in kws):
                content = self._load_scene_cached(scene)
                if content:
                    out.append(content)
                    seen.add(scene)
        return out

    def _effective_system(self, input_data: BatchInput) -> str:
        """Compute the system prompt for this message.

        Resident persona/system prompt, plus the source-based scene suffix,
        plus any keyword-triggered content modules, all appended in order.
        """
        parts: List[str] = [self._system]
        suffix = self.resolve_scene_suffix(getattr(input_data, "metadata", None))
        if suffix:
            parts.append(suffix)
        parts.extend(self.resolve_content_suffixes(input_data))
        if get_paint_manager().enabled:
            parts.append(paint_capability_prompt())
        return "\n\n".join(p for p in parts if p)

    def _add_message(
        self,
        message: Union[str, List[Dict[str, Any]]],
        role: str,
        display_text: DisplayText | None = None,
        skip_memory: bool = False,
    ) -> bool:
        """Add message to memory."""
        if skip_memory:
            return False

        text_content = ""
        if isinstance(message, list):
            for item in message:
                if item.get("type") == "text":
                    text_content += item["text"] + " "
            text_content = text_content.strip()
        elif isinstance(message, str):
            text_content = message
        else:
            logger.warning(
                f"_add_message received unexpected message type: {type(message)}"
            )
            text_content = str(message)

        if role == "assistant":
            sanitized_text_content, blocked_words = (
                sanitize_blocked_words_text_with_matches(
                    text_content,
                    ignored_spans=paint_command_spans(text_content),
                )
            )
            if sanitized_text_content != text_content:
                logger.info(
                    "Blocked word sanitized in assistant memory: "
                    f"words={blocked_words!r}"
                )
            text_content = sanitized_text_content

        if not text_content and role == "assistant":
            return False

        message_data = {
            "role": role,
            "content": text_content,
        }

        if display_text:
            if display_text.name:
                message_data["name"] = display_text.name
            if display_text.avatar:
                message_data["avatar"] = display_text.avatar

        if (
            self._memory
            and self._memory[-1]["role"] == role
            and self._memory[-1]["content"] == text_content
        ):
            return False

        self._memory.append(message_data)
        return True

    def _coerce_memory_message(
        self,
        message: Dict[str, Any],
    ) -> Dict[str, str] | None:
        """Normalize persisted history or summary-tail messages for LLM memory."""
        source_role = message.get("role")
        if source_role in {"user", "assistant"}:
            role = source_role
        elif source_role == "human":
            role = "user"
        elif source_role == "ai":
            role = "assistant"
        elif source_role == "system":
            role = "user"
        else:
            logger.warning(f"Skipping memory message with unknown role: {message}")
            return None

        content = message.get("content", "")
        if isinstance(content, str):
            text_content = content.strip()
        elif isinstance(content, list):
            text_content = " ".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ).strip()
        else:
            text_content = str(content).strip()

        if not text_content:
            logger.warning(f"Skipping invalid message from history: {message}")
            return None

        if role == "user":
            text_content = format_human_input_for_llm(
                text_content,
                message.get("name"),
            )

        return {"role": role, "content": text_content}

    def remove_recent_user_input_from_memory(
        self,
        input_text: str,
        human_name: str | None = None,
    ) -> bool:
        """Remove the latest user memory item when it is the current pre-stored input."""
        plain_text = str(input_text or "").strip()
        formatted_text = format_human_input_for_llm(plain_text, human_name)
        candidates = {plain_text, formatted_text}

        for index in range(len(self._memory) - 1, -1, -1):
            message = self._memory[index]
            if message.get("role") != "user":
                continue

            content = str(message.get("content") or "").strip()
            if content in candidates:
                del self._memory[index]
                logger.info(
                    "Removed current pre-stored user input from runtime memory "
                    "after history reload."
                )
                return True

            logger.debug(
                "Latest user memory did not match current pre-stored input; "
                "leaving memory unchanged."
            )
            return False

        return False

    @staticmethod
    def _merge_summary_tail_with_history(
        tail_memory: List[Dict[str, Any]],
        history_memory: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], int, int]:
        """Append history written after a summary's uncompressed tail snapshot."""
        if not tail_memory:
            return [message.copy() for message in history_memory], 0, len(
                history_memory
            )

        minimum_match = 1 if len(tail_memory) == 1 else 2
        for matched_count in range(len(tail_memory), minimum_match - 1, -1):
            prefix = tail_memory[:matched_count]
            latest_start = len(history_memory) - matched_count
            for start in range(latest_start, -1, -1):
                end = start + matched_count
                if history_memory[start:end] != prefix:
                    continue

                newer_messages = history_memory[end:]
                merged = [message.copy() for message in prefix]
                merged.extend(message.copy() for message in newer_messages)
                return merged, matched_count, len(newer_messages)

        return [message.copy() for message in history_memory], 0, len(
            history_memory
        )

    def set_memory_from_history(self, conf_uid: str, history_uid: str) -> None:
        """Load full history plus the latest compressed summary for runtime memory."""
        self._current_conf_uid = conf_uid
        self._current_history_uid = history_uid

        history_memory = []
        for msg in get_history(conf_uid, history_uid):
            memory_item = self._coerce_memory_message(msg)
            if memory_item:
                history_memory.append(memory_item)

        self._memory = history_memory
        self._summary_sequence = 0
        self._loaded_summary_files = []

        summary_info = get_latest_summary_payload(conf_uid, history_uid)
        if summary_info:
            payload = summary_info.get("payload", {})
            summary_text = payload.get("summary")
            uncompressed_messages = payload.get("uncompressed_messages", [])
            self._loaded_summary_files = list(summary_info.get("summary_files", []))
            self._summary_sequence = int(payload.get("summary_sequence") or 0)
            update_metadate(
                conf_uid,
                history_uid,
                {
                    "summary_files": self._loaded_summary_files,
                    "active_summary_file": summary_info.get("relative_path"),
                },
            )

            tail_memory = []
            if isinstance(uncompressed_messages, list):
                for message in uncompressed_messages:
                    if isinstance(message, dict):
                        memory_item = self._coerce_memory_message(message)
                        if memory_item:
                            tail_memory.append(memory_item)

            merged_tail, matched_count, newer_count = (
                self._merge_summary_tail_with_history(
                    tail_memory,
                    history_memory,
                )
            )
            if tail_memory and not matched_count:
                logger.warning(
                    "Could not align the latest summary tail with raw history; "
                    "using full raw history after the summary to avoid losing messages."
                )
            elif newer_count:
                logger.info(
                    "Appended {} raw history messages written after the summary "
                    "tail (matched {}/{} tail messages).",
                    newer_count,
                    matched_count,
                    len(tail_memory),
                )
            tail_memory = merged_tail

            if isinstance(summary_text, str) and summary_text.strip():
                if not tail_memory and history_memory:
                    logger.warning(
                        "Latest memory summary has no uncompressed tail; "
                        "falling back to full raw history after the summary."
                    )
                    tail_memory = history_memory
                self._memory = [
                    self._build_summary_memory_message(summary_text.strip()),
                    *tail_memory,
                ]
                logger.info(
                    "Loaded memory from latest summary {} with {} uncompressed "
                    "messages.",
                    summary_info.get("relative_path"),
                    len(tail_memory),
                )

        logger.info(
            f"Loaded {len(history_memory)} raw history messages and "
            f"{len(self._memory)} runtime memory messages."
        )

    def _is_summary_message(self, message: Dict[str, Any]) -> bool:
        """Return True when a memory item is a compressed summary marker."""
        content = message.get("content", "")
        return isinstance(content, str) and content.startswith(SUMMARY_MESSAGE_PREFIX)

    def _message_content_to_text(self, content: Any) -> str:
        """Convert stored LLM message content to plain text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
            return "\n".join(part for part in text_parts if part)
        return str(content)

    def _completed_dialogue_rounds(self) -> List[tuple[int, int]]:
        """Return memory indexes for completed user/assistant turns."""
        rounds = []
        index = 0
        while index < len(self._memory) - 1:
            current = self._memory[index]
            next_message = self._memory[index + 1]
            if self._is_summary_message(current):
                index += 1
                continue
            if (
                current.get("role") == "user"
                and next_message.get("role") == "assistant"
                and not self._is_summary_message(next_message)
            ):
                rounds.append((index, index + 1))
                index += 2
                continue
            index += 1
        return rounds

    def _build_summary_messages(
        self,
        round_pairs: List[tuple[int, int]],
        previous_summary_messages: List[Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        """Build messages used to compress old conversation turns."""
        previous_summary_messages = previous_summary_messages or []
        messages: List[Dict[str, Any]] = []
        for summary_number, message in enumerate(previous_summary_messages, 1):
            summary_text = self._message_content_to_text(message.get("content", ""))
            if summary_text.strip():
                messages.append(
                    {
                        "role": "user",
                        "content": f"[已有压缩记忆 {summary_number}]\n{summary_text}",
                    }
                )

        for user_index, assistant_index in round_pairs:
            user_text = self._message_content_to_text(
                self._memory[user_index].get("content", "")
            )
            assistant_text = self._message_content_to_text(
                self._memory[assistant_index].get("content", "")
            )
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})

        messages.append(
            {
                "role": "user",
                "content": (
                    "请总结以上对话，用中文输出简洁的长期记忆。"
                    "如果有已有压缩记忆，请和新对话合并成从开头到现在的连续总结。"
                    "保留角色身份、用户偏好、重要事实、未完成事项和情绪关系。"
                    "不要编造信息；方括号标签如 [black]、[happy] 是表情或动作标签，不是名字。"
                    "请以“[压缩记忆]”开头。"
                ),
            }
        )
        return messages

    async def _collect_llm_text(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        call_source: str = "unknown",
    ) -> str:
        """Collect text from a direct LLM call without changing memory."""
        pieces = []
        stream = self._llm.chat_completion(
            messages, system_prompt, call_source=call_source
        )
        async for event in stream:
            if isinstance(event, dict):
                if event.get("type") == "text_delta":
                    pieces.append(event.get("text", ""))
                elif event.get("type") == "error":
                    raise RuntimeError(event.get("message", "Unknown LLM error"))
            elif isinstance(event, str):
                pieces.append(event)
        return "".join(pieces).strip()

    def _looks_like_llm_error(self, text: str) -> bool:
        """Detect error strings yielded by compatible LLM wrappers."""
        stripped = text.strip()
        return stripped.startswith(SUMMARY_API_ERROR_PREFIX) or stripped.startswith(
            "[Error from LLM"
        )

    def _get_error_fallback_text(self) -> str:
        """Pick the final user-facing fallback after the initial retry also fails."""
        prompt = get_prompt_trigger_registry().get_next(ERROR_TRIGGER_NAME)
        if prompt:
            return prompt.text
        logger.warning(
            "No error trigger prompt available; using built-in fallback text."
        )
        return DEFAULT_ERROR_FALLBACK_TEXT

    async def _final_error_fallback_stream(
        self,
        *,
        first_token_event_sent: bool,
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        fallback_text = self._get_error_fallback_text()
        logger.warning(
            "LLM initial call still failed after retry; using error fallback text."
        )
        if not first_token_event_sent:
            yield make_llm_first_token_event()
        yield fallback_text
        yield make_llm_final_error_fallback_event()
        if self._add_message(fallback_text, "assistant"):
            await self._maybe_summarize_memory()

    def _build_summary_memory_message(self, summary_text: str) -> Dict[str, str]:
        """Create a portable memory item for the compressed conversation summary."""
        return {
            "role": "user",
            "content": (
                f"{SUMMARY_MESSAGE_PREFIX}\n"
                "This is compressed memory from earlier dialogue, not a new user "
                f"request.\n{summary_text}"
            ),
        }

    async def _warm_summary_cache(self, proposed_memory: List[Dict[str, Any]]) -> str:
        """Send the proposed memory once so the upstream API can build prompt cache."""
        warm_messages = proposed_memory.copy()
        warm_messages.append(
            {
                "role": "user",
                "content": (
                    "[Cache warm-up check] Reply with exactly OK. "
                    "Do not update memory from this request."
                ),
            }
        )
        return await self._collect_llm_text(
            warm_messages,
            self._system,
            call_source="memory_summary_cache_warm",
        )

    def _store_summary_file(
        self,
        *,
        summary_text: str,
        source_messages: List[Dict[str, Any]],
        uncompressed_messages: List[Dict[str, Any]],
        uncompressed_round_count: int,
        included_previous_summary_count: int,
        used_summary_files: List[str],
        cache_warm_success: bool,
        cache_warm_response: str,
        replacement_applied: bool,
    ) -> Path:
        """Persist one memory summary into its own JSON file."""
        conf_uid = self._current_conf_uid or "_unknown"
        history_uid = self._current_history_uid or ""
        summary_dir = get_summary_storage_dir(conf_uid)
        self._summary_sequence += 1
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        file_path = (
            summary_dir
            / f"basic_memory_summary_{timestamp}_{uuid.uuid4().hex[:8]}.json"
        )
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "conf_uid": conf_uid,
            "history_uid": history_uid,
            "summary_sequence": self._summary_sequence,
            "trigger_rounds": SUMMARY_TRIGGER_ROUNDS,
            "compressed_rounds": SUMMARY_ROUNDS_TO_COMPRESS,
            "included_previous_summary_count": included_previous_summary_count,
            "used_summary_files": used_summary_files,
            "summary_message_prefix": SUMMARY_MESSAGE_PREFIX,
            "summary": summary_text,
            "source_messages": source_messages,
            "uncompressed_round_count": uncompressed_round_count,
            "uncompressed_messages": uncompressed_messages,
            "cache_warm_success": cache_warm_success,
            "cache_warm_response_preview": cache_warm_response[:500],
            "replacement_applied": replacement_applied,
        }
        with file_path.open("w", encoding="utf-8") as summary_file:
            json.dump(payload, summary_file, ensure_ascii=False, indent=2)
        return file_path

    async def _maybe_summarize_memory(self) -> None:
        """Summarize and replace old dialogue once memory reaches the threshold."""
        completed_rounds = self._completed_dialogue_rounds()
        if len(completed_rounds) < SUMMARY_TRIGGER_ROUNDS:
            return

        round_pairs = completed_rounds[:SUMMARY_ROUNDS_TO_COMPRESS]
        start_index = round_pairs[0][0]
        end_index = round_pairs[-1][1] + 1
        previous_summary_indexes = [
            index
            for index in range(start_index)
            if self._is_summary_message(self._memory[index])
        ]
        replacement_start_index = (
            previous_summary_indexes[0] if previous_summary_indexes else start_index
        )
        previous_summary_messages = [
            self._memory[index].copy() for index in previous_summary_indexes
        ]
        source_messages = [
            message.copy()
            for message in self._memory[replacement_start_index:end_index]
        ]
        uncompressed_messages = [message.copy() for message in self._memory[end_index:]]
        uncompressed_round_count = max(
            len(completed_rounds) - SUMMARY_ROUNDS_TO_COMPRESS,
            0,
        )

        logger.info(
            "BasicMemoryAgent memory reached "
            f"{len(completed_rounds)} rounds; summarizing first "
            f"{SUMMARY_ROUNDS_TO_COMPRESS} rounds with "
            f"{len(previous_summary_messages)} previous summaries."
        )

        summary_messages = self._build_summary_messages(
            round_pairs,
            previous_summary_messages=previous_summary_messages,
        )
        try:
            summary_text = await self._collect_llm_text(
                summary_messages,
                self._system,
                call_source="memory_summary",
            )
        except Exception:
            logger.exception("Failed to generate memory summary.")
            return

        if not summary_text or self._looks_like_llm_error(summary_text):
            logger.error(f"Memory summary failed: {summary_text}")
            return

        summary_message = self._build_summary_memory_message(summary_text)
        proposed_memory = (
            self._memory[:replacement_start_index]
            + [summary_message]
            + self._memory[end_index:]
        )

        cache_warm_success = False
        cache_warm_response = ""
        try:
            cache_warm_response = await self._warm_summary_cache(proposed_memory)
            cache_warm_success = bool(
                cache_warm_response
            ) and not self._looks_like_llm_error(cache_warm_response)
        except Exception:
            logger.exception("Failed to warm memory summary cache.")

        replacement_applied = cache_warm_success
        summary_file_path = self._store_summary_file(
            summary_text=summary_text,
            source_messages=source_messages,
            uncompressed_messages=uncompressed_messages,
            uncompressed_round_count=uncompressed_round_count,
            included_previous_summary_count=len(previous_summary_messages),
            used_summary_files=self._loaded_summary_files.copy(),
            cache_warm_success=cache_warm_success,
            cache_warm_response=cache_warm_response,
            replacement_applied=replacement_applied,
        )

        if not cache_warm_success:
            logger.error(
                "Memory summary cache warm-up failed; leaving original memory in place. "
                f"Summary saved to {summary_file_path}."
            )
            return

        self._memory = proposed_memory
        if self._current_conf_uid and self._current_history_uid:
            record_summary_file(
                self._current_conf_uid,
                self._current_history_uid,
                summary_file_path,
                active=True,
            )
            summary_info = get_latest_summary_payload(
                self._current_conf_uid,
                self._current_history_uid,
            )
            if summary_info:
                self._loaded_summary_files = list(
                    summary_info.get("summary_files", [])
                )
        logger.info(
            "Replaced first "
            f"{SUMMARY_ROUNDS_TO_COMPRESS} memory rounds with summary file "
            f"{summary_file_path}."
        )

    def handle_interrupt(self, heard_response: str) -> None:
        """Handle user interruption."""
        if self._interrupt_handled:
            return

        self._interrupt_handled = True

        if self._memory and self._memory[-1]["role"] == "assistant":
            if not self._memory[-1]["content"].endswith("..."):
                self._memory[-1]["content"] = heard_response + "..."
            else:
                self._memory[-1]["content"] = heard_response + "..."
        else:
            if heard_response:
                self._memory.append(
                    {
                        "role": "assistant",
                        "content": heard_response + "...",
                    }
                )

        interrupt_role = "system" if self.interrupt_method == "system" else "user"
        self._memory.append(
            {
                "role": interrupt_role,
                "content": "[Interrupted by user]",
            }
        )
        logger.info(f"Handled interrupt with role '{interrupt_role}'.")

    def _to_text_prompt(self, input_data: BatchInput) -> str:
        """Format input data to text prompt."""
        message_parts = []

        for text_data in input_data.texts:
            if text_data.source == TextSource.INPUT:
                message_parts.append(text_data.content)
            elif text_data.source == TextSource.CLIPBOARD:
                message_parts.append(
                    f"[User shared content from clipboard: {text_data.content}]"
                )

        if input_data.images:
            message_parts.append("\n[User has also provided images]")

        return "\n".join(message_parts).strip()

    def _to_messages(self, input_data: BatchInput) -> List[Dict[str, Any]]:
        """Prepare messages for LLM API call."""
        messages = self._memory.copy()
        user_content = []
        text_prompt = self._to_text_prompt(input_data)
        if text_prompt:
            user_content.append({"type": "text", "text": text_prompt})

        if input_data.images:
            image_added = False
            for img_data in input_data.images:
                if isinstance(img_data.data, str) and img_data.data.startswith(
                    "data:image"
                ):
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": img_data.data, "detail": "auto"},
                        }
                    )
                    image_added = True
                else:
                    logger.error(
                        f"Invalid image data format: {type(img_data.data)}. Skipping image."
                    )

            if not image_added and not text_prompt:
                logger.warning(
                    "User input contains images but none could be processed."
                )

        if user_content:
            user_message = {"role": "user", "content": user_content}
            messages.append(user_message)

            skip_memory = False
            if input_data.metadata and input_data.metadata.get("skip_memory", False):
                skip_memory = True

            if not skip_memory:
                memory_text = text_prompt if text_prompt else "[User provided image(s)]"
                if input_data.metadata and input_data.metadata.get("memory_input_text"):
                    memory_text = str(input_data.metadata["memory_input_text"])
                self._add_message(
                    memory_text,
                    "user",
                )
        else:
            logger.warning("No content generated for user message.")

        return messages

    async def _claude_tool_interaction_loop(
        self,
        initial_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: Optional[str] = None,
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """Handle Claude interaction loop with tool support."""
        system = system if system is not None else self._system
        messages = initial_messages.copy()
        current_turn_text = ""
        pending_tool_calls = []
        current_assistant_message_content = []
        first_token_event_sent = False
        llm_call_count = 0

        while True:
            llm_call_count += 1
            retry_initial_call = False
            stream = self._llm.chat_completion(
                messages,
                system,
                tools=tools,
                call_source=f"chat_claude_tools_call_{llm_call_count}",
            )
            pending_tool_calls.clear()
            current_assistant_message_content.clear()

            try:
                async for event in stream:
                    if event["type"] == "text_delta":
                        text = event["text"]
                        current_turn_text += text
                        if text and not first_token_event_sent:
                            first_token_event_sent = True
                            yield make_llm_first_token_event()
                        yield text
                        if (
                            not current_assistant_message_content
                            or current_assistant_message_content[-1]["type"] != "text"
                        ):
                            current_assistant_message_content.append(
                                {"type": "text", "text": text}
                            )
                        else:
                            current_assistant_message_content[-1]["text"] += text
                    elif event["type"] == "tool_use_complete":
                        tool_call_data = event["data"]
                        logger.info(
                            f"Tool request: {tool_call_data['name']} (ID: {tool_call_data['id']})"
                        )
                        pending_tool_calls.append(tool_call_data)
                        current_assistant_message_content.append(
                            {
                                "type": "tool_use",
                                "id": tool_call_data["id"],
                                "name": tool_call_data["name"],
                                "input": tool_call_data["input"],
                            }
                        )
                    # elif event["type"] == "message_delta":
                    #     if event["data"]["delta"].get("stop_reason"):
                    #         stop_reason = event["data"]["delta"].get("stop_reason")
                    elif event["type"] == "message_stop":
                        break
                    elif event["type"] == "error":
                        logger.error(f"LLM API Error: {event['message']}")
                        if not current_turn_text.strip() and llm_call_count == 1:
                            logger.warning("Initial Claude LLM call failed; retrying once.")
                            retry_initial_call = True
                            break
                        if not current_turn_text.strip():
                            async for output in self._final_error_fallback_stream(
                                first_token_event_sent=first_token_event_sent,
                            ):
                                yield output
                            return
                        yield f"[Error from LLM: {event['message']}]"
                        return
            except Exception as exc:
                logger.exception("Claude LLM stream raised an exception.")
                if not current_turn_text.strip() and llm_call_count == 1:
                    logger.warning("Initial Claude LLM call failed; retrying once.")
                    retry_initial_call = True
                elif not current_turn_text.strip():
                    async for output in self._final_error_fallback_stream(
                        first_token_event_sent=first_token_event_sent,
                    ):
                        yield output
                    return
                else:
                    yield f"[Error from LLM: {exc}]"
                    return

            if retry_initial_call:
                pending_tool_calls.clear()
                current_assistant_message_content.clear()
                continue

            if pending_tool_calls:
                filtered_assistant_content = [
                    block
                    for block in current_assistant_message_content
                    if not (
                        block.get("type") == "text"
                        and not block.get("text", "").strip()
                    )
                ]

                if filtered_assistant_content:
                    messages.append(
                        {"role": "assistant", "content": filtered_assistant_content}
                    )
                    assistant_text_for_memory = "".join(
                        [
                            c["text"]
                            for c in filtered_assistant_content
                            if c["type"] == "text"
                        ]
                    ).strip()
                    if assistant_text_for_memory:
                        self._add_message(assistant_text_for_memory, "assistant")

                tool_results_for_llm = []
                if not self._tool_executor:
                    logger.error(
                        "Claude Tool interaction requested but ToolExecutor is not available."
                    )
                    yield "[Error: ToolExecutor not configured]"
                    return

                tool_executor_iterator = self._tool_executor.execute_tools(
                    tool_calls=pending_tool_calls,
                    caller_mode="Claude",
                )
                try:
                    while True:
                        update = await anext(tool_executor_iterator)
                        if update.get("type") == "final_tool_results":
                            tool_results_for_llm = update.get("results", [])
                            break
                        else:
                            yield update
                except StopAsyncIteration:
                    logger.warning(
                        "Tool executor finished without final results marker."
                    )

                if tool_results_for_llm:
                    messages.append({"role": "user", "content": tool_results_for_llm})

                # stop_reason = None
                continue
            else:
                if current_turn_text:
                    if self._add_message(current_turn_text, "assistant"):
                        await self._maybe_summarize_memory()
                elif llm_call_count == 1:
                    logger.warning("Initial Claude LLM call returned empty output; retrying once.")
                    continue
                else:
                    async for output in self._final_error_fallback_stream(
                        first_token_event_sent=first_token_event_sent,
                    ):
                        yield output
                return

    async def _openai_tool_interaction_loop(
        self,
        initial_messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        system: Optional[str] = None,
    ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
        """Handle OpenAI interaction with tool support."""
        system = system if system is not None else self._system
        messages = initial_messages.copy()
        current_turn_text = ""
        pending_tool_calls: Union[List[ToolCallObject], List[Dict[str, Any]]] = []
        current_system_prompt = system
        first_token_event_sent = False
        llm_call_count = 0

        while True:
            llm_call_count += 1
            if self.prompt_mode_flag:
                if self._mcp_prompt_string:
                    current_system_prompt = (
                        f"{system}\n\n{self._mcp_prompt_string}"
                    )
                else:
                    logger.warning("Prompt mode active but mcp_prompt_string is empty!")
                    current_system_prompt = system
                tools_for_api = None
            else:
                current_system_prompt = system
                tools_for_api = tools

            stream = self._llm.chat_completion(
                messages,
                current_system_prompt,
                tools=tools_for_api,
                call_source=f"chat_openai_tools_call_{llm_call_count}",
            )
            pending_tool_calls.clear()
            current_turn_text = ""
            assistant_message_for_api = None
            detected_prompt_json = None
            goto_next_while_iteration = False
            retry_initial_call = False
            final_error_fallback = False

            async for event in stream:
                if self.prompt_mode_flag:
                    if isinstance(event, str):
                        if (
                            self._looks_like_llm_error(event)
                            and not current_turn_text.strip()
                        ):
                            if llm_call_count == 1:
                                logger.warning(
                                    "Initial OpenAI prompt-mode LLM call failed; retrying once."
                                )
                                retry_initial_call = True
                            else:
                                final_error_fallback = True
                            break
                        current_turn_text += event
                        if event and not first_token_event_sent:
                            first_token_event_sent = True
                            yield make_llm_first_token_event()
                        if self._json_detector:
                            potential_json = self._json_detector.process_chunk(event)
                            if potential_json:
                                try:
                                    if isinstance(potential_json, list):
                                        detected_prompt_json = potential_json
                                    elif isinstance(potential_json, dict):
                                        detected_prompt_json = [potential_json]

                                    if detected_prompt_json:
                                        break
                                except Exception as e:
                                    logger.error(f"Error parsing detected JSON: {e}")
                                    if self._json_detector:
                                        self._json_detector.reset()
                                    yield f"[Error parsing tool JSON: {e}]"
                                    goto_next_while_iteration = True
                                    break
                        yield event
                else:
                    if isinstance(event, str):
                        if (
                            self._looks_like_llm_error(event)
                            and not current_turn_text.strip()
                        ):
                            if llm_call_count == 1:
                                logger.warning("Initial OpenAI LLM call failed; retrying once.")
                                retry_initial_call = True
                            else:
                                final_error_fallback = True
                            break
                        current_turn_text += event
                        if event and not first_token_event_sent:
                            first_token_event_sent = True
                            yield make_llm_first_token_event()
                        yield event
                    elif isinstance(event, list) and all(
                        isinstance(tc, ToolCallObject) for tc in event
                    ):
                        pending_tool_calls = event
                        assistant_message_for_api = {
                            "role": "assistant",
                            "content": current_turn_text if current_turn_text else None,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": tc.type,
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments,
                                    },
                                }
                                for tc in pending_tool_calls
                            ],
                        }
                        break
                    elif event == "__API_NOT_SUPPORT_TOOLS__":
                        logger.warning(
                            f"LLM {getattr(self._llm, 'model', '')} has no native tool support. Switching to prompt mode."
                        )
                        self.prompt_mode_flag = True
                        if self._tool_manager:
                            self._tool_manager.disable()
                        if self._json_detector:
                            self._json_detector.reset()
                        goto_next_while_iteration = True
                        break
            if retry_initial_call:
                current_turn_text = ""
                pending_tool_calls.clear()
                if self._json_detector:
                    self._json_detector.reset()
                continue
            if final_error_fallback:
                async for output in self._final_error_fallback_stream(
                    first_token_event_sent=first_token_event_sent,
                ):
                    yield output
                return
            if goto_next_while_iteration:
                continue

            if detected_prompt_json:
                logger.info("Processing tools detected via prompt mode JSON.")
                self._add_message(current_turn_text, "assistant")

                parsed_tools = self._tool_executor.process_tool_from_prompt_json(
                    detected_prompt_json
                )
                if parsed_tools:
                    tool_results_for_llm = []
                    if not self._tool_executor:
                        logger.error(
                            "Prompt Tool interaction requested but ToolExecutor/MCPClient is not available."
                        )
                        yield "[Error: ToolExecutor/MCPClient not configured for prompt mode]"
                        continue

                    tool_executor_iterator = self._tool_executor.execute_tools(
                        tool_calls=parsed_tools,
                        caller_mode="Prompt",
                    )
                    try:
                        while True:
                            update = await anext(tool_executor_iterator)
                            if update.get("type") == "final_tool_results":
                                tool_results_for_llm = update.get("results", [])
                                break
                            else:
                                yield update
                    except StopAsyncIteration:
                        logger.warning(
                            "Prompt mode tool executor finished without final results marker."
                        )

                    if tool_results_for_llm:
                        result_strings = [
                            res.get("content", "Error: Malformed result")
                            for res in tool_results_for_llm
                        ]
                        combined_results_str = "\n".join(result_strings)
                        messages.append(
                            {"role": "user", "content": combined_results_str}
                        )
                continue

            elif pending_tool_calls and assistant_message_for_api:
                messages.append(assistant_message_for_api)
                if current_turn_text:
                    self._add_message(current_turn_text, "assistant")

                tool_results_for_llm = []
                if not self._tool_executor:
                    logger.error(
                        "OpenAI Tool interaction requested but ToolExecutor/MCPClient is not available."
                    )
                    yield "[Error: ToolExecutor/MCPClient not configured for OpenAI mode]"
                    continue

                tool_executor_iterator = self._tool_executor.execute_tools(
                    tool_calls=pending_tool_calls,
                    caller_mode="OpenAI",
                )
                try:
                    while True:
                        update = await anext(tool_executor_iterator)
                        if update.get("type") == "final_tool_results":
                            tool_results_for_llm = update.get("results", [])
                            break
                        else:
                            yield update
                except StopAsyncIteration:
                    logger.warning(
                        "OpenAI tool executor finished without final results marker."
                    )

                if tool_results_for_llm:
                    messages.extend(tool_results_for_llm)
                continue

            else:
                if current_turn_text:
                    if self._add_message(current_turn_text, "assistant"):
                        await self._maybe_summarize_memory()
                elif llm_call_count == 1:
                    logger.warning("Initial OpenAI LLM call returned empty output; retrying once.")
                    continue
                else:
                    async for output in self._final_error_fallback_stream(
                        first_token_event_sent=first_token_event_sent,
                    ):
                        yield output
                return

    def _chat_function_factory(
        self,
    ) -> Callable[[BatchInput], AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]]:
        """Create the chat pipeline function."""

        @tts_filter(self._tts_preprocessor_config)
        @display_processor()
        @actions_extractor(self._live2d_model)
        @sentence_divider(
            faster_first_response=self._faster_first_response,
            segment_method=self._segment_method,
            valid_tags=["think"],
        )
        async def chat_with_memory(
            input_data: BatchInput,
        ) -> AsyncIterator[Union[str, Dict[str, Any]]]:
            """Process chat with memory and tools."""
            self.reset_interrupt()
            self.prompt_mode_flag = False

            messages = self._to_messages(input_data)
            effective_system = self._effective_system(input_data)
            active_llm = self._llm_for_input(input_data)
            tools = None
            tool_mode = None
            llm_supports_native_tools = False

            if self._use_mcpp and self._tool_manager and not input_data.images:
                tools = None
                if isinstance(self._llm, ClaudeAsyncLLM):
                    tool_mode = "Claude"
                    tools = self._formatted_tools_claude
                    llm_supports_native_tools = True
                elif isinstance(
                    self._llm,
                    (OpenAICompatibleAsyncLLM, ProviderRuntimeLLM),
                ):
                    tool_mode = "OpenAI"
                    tools = self._formatted_tools_openai
                    llm_supports_native_tools = True
                else:
                    logger.warning(
                        f"LLM type {type(self._llm)} not explicitly handled for tool mode determination."
                    )

                if llm_supports_native_tools and not tools:
                    logger.warning(
                        f"No tools available/formatted for '{tool_mode}' mode, despite MCP being enabled."
                    )

            if self._use_mcpp and tool_mode == "Claude":
                logger.debug(
                    f"Starting Claude tool interaction loop with {len(tools)} tools."
                )
                async for output in self._claude_tool_interaction_loop(
                    messages, tools if tools else [], system=effective_system
                ):
                    yield output
                return
            elif self._use_mcpp and tool_mode == "OpenAI":
                logger.debug(
                    f"Starting OpenAI tool interaction loop with {len(tools)} tools."
                )
                async for output in self._openai_tool_interaction_loop(
                    messages, tools if tools else [], system=effective_system
                ):
                    yield output
                return
            else:
                first_token_event_sent = False
                for attempt in range(1, INITIAL_LLM_CALL_MAX_ATTEMPTS + 1):
                    logger.info("Starting simple chat completion.")
                    logger.info(
                        "Chat messages prepared: count={}, contains_images={}",
                        len(messages),
                        bool(input_data.images),
                    )
                    token_stream = active_llm.chat_completion(
                        messages,
                        effective_system,
                        call_source=f"chat_simple_attempt_{attempt}",
                    )
                    complete_response = ""
                    retry_initial_call = False
                    final_error_fallback = False
                    try:
                        async for event in token_stream:
                            text_chunk = ""
                            if is_web_search_event(event):
                                yield event
                                continue
                            if (
                                isinstance(event, dict)
                                and event.get("type") == "text_delta"
                            ):
                                text_chunk = event.get("text", "")
                            elif isinstance(event, str):
                                text_chunk = event
                            else:
                                continue
                            if text_chunk:
                                if (
                                    self._looks_like_llm_error(text_chunk)
                                    and not complete_response.strip()
                                ):
                                    if attempt < INITIAL_LLM_CALL_MAX_ATTEMPTS:
                                        logger.warning(
                                            "Initial LLM call failed; retrying once."
                                        )
                                        retry_initial_call = True
                                    else:
                                        final_error_fallback = True
                                    break
                                if not first_token_event_sent:
                                    first_token_event_sent = True
                                    yield make_llm_first_token_event()
                                yield text_chunk
                                complete_response += text_chunk
                    except Exception:
                        logger.exception("LLM stream raised an exception.")
                        if complete_response.strip():
                            logger.warning(
                                "LLM stream failed after producing partial output."
                            )
                        elif attempt < INITIAL_LLM_CALL_MAX_ATTEMPTS:
                            logger.warning("Initial LLM call failed; retrying once.")
                            retry_initial_call = True
                        else:
                            final_error_fallback = True
                    if retry_initial_call:
                        continue
                    if final_error_fallback:
                        async for output in self._final_error_fallback_stream(
                            first_token_event_sent=first_token_event_sent,
                        ):
                            yield output
                        return
                    if complete_response.strip():
                        if self._add_message(complete_response, "assistant"):
                            await self._maybe_summarize_memory()
                        break
                    if attempt < INITIAL_LLM_CALL_MAX_ATTEMPTS:
                        logger.warning("Initial LLM call returned empty output; retrying once.")
                        continue
                    logger.warning("LLM call returned empty output after retry.")
                    async for output in self._final_error_fallback_stream(
                        first_token_event_sent=first_token_event_sent,
                    ):
                        yield output

        return chat_with_memory

    async def chat(
        self,
        input_data: BatchInput,
    ) -> AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]:
        """Run chat pipeline."""
        chat_func_decorated = self._chat_function_factory()
        async for output in chat_func_decorated(input_data):
            yield output

    def reset_interrupt(self) -> None:
        """Reset interrupt flag."""
        self._interrupt_handled = False

    def start_group_conversation(
        self, human_name: str, ai_participants: List[str]
    ) -> None:
        """Start a group conversation."""
        if not self._tool_prompts:
            logger.warning("Tool prompts dictionary is not set.")
            return

        other_ais = ", ".join(name for name in ai_participants)
        prompt_name = self._tool_prompts.get("group_conversation_prompt", "")

        if not prompt_name:
            logger.warning("No group conversation prompt name found.")
            return

        try:
            group_context = prompt_loader.load_util(prompt_name).format(
                human_name=human_name, other_ais=other_ais
            )
            self._memory.append({"role": "user", "content": group_context})
        except FileNotFoundError:
            logger.error(f"Group conversation prompt file not found: {prompt_name}")
        except KeyError as e:
            logger.error(f"Missing formatting key in group conversation prompt: {e}")
        except Exception as e:
            logger.error(f"Failed to load group conversation prompt: {e}")
