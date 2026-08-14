from typing import Type

from loguru import logger

from .stateless_llm.stateless_llm_interface import StatelessLLMInterface
from .stateless_llm.stateless_llm_with_template import (
    AsyncLLMWithTemplate as StatelessLLMWithTemplate,
)
from .stateless_llm.openai_compatible_llm import AsyncLLM as OpenAICompatibleLLM
from .stateless_llm.ollama_llm import OllamaLLM
from .stateless_llm.claude_llm import AsyncLLM as ClaudeLLM


class LLMFactory:
    @staticmethod
    def create_llm(llm_provider, **kwargs) -> Type[StatelessLLMInterface]:
        """Create an LLM based on the configuration.

        Args:
            llm_provider: The type of LLM to create
            **kwargs: Additional arguments
        """
        logger.info(f"Initializing LLM: {llm_provider}")

        if llm_provider == "gemini_llm":
            # Google 原生 google-genai SDK, 真 SSE 流式
            # 推荐用于 flash / lite 系列 (可完全关 thinking, 首块最快)
            return GeminiNativeLLM(
                model=kwargs.get("model"),
                llm_api_key=kwargs.get("llm_api_key"),
                temperature=kwargs.get("temperature", 1.0),
                top_p=kwargs.get("top_p", 0.8),
                thinking_enabled=kwargs.get("thinking_enabled", False),
                thinking_level=kwargs.get("thinking_level", "low"),
                max_output_tokens=kwargs.get("max_output_tokens"),
            )

        if llm_provider == "gemini_openai_llm":
            # Gemini 通过 OpenAI 兼容端点访问
            # 推荐用于 pro 系列 (Pro 强制 thinking, 在 OpenAI 兼容层有时
            # 能跳过部分 thinking 等待, 首块感知比原生 SDK 略快)
            # base_url 默认指向 Google 官方 OpenAI 兼容端点
            return OpenAICompatibleLLM(
                model=kwargs.get("model"),
                base_url=kwargs.get(
                    "base_url",
                    "https://generativelanguage.googleapis.com/v1beta/openai/",
                ),
                llm_api_key=kwargs.get("llm_api_key"),
                organization_id=kwargs.get("organization_id"),
                project_id=kwargs.get("project_id"),
                temperature=kwargs.get("temperature"),
            )

        if (
            llm_provider == "openai_compatible_llm"
            or llm_provider == "openai_llm"
            or llm_provider == "zhipu_llm"
            or llm_provider == "deepseek_llm"
            or llm_provider == "groq_llm"
            or llm_provider == "mistral_llm"
            or llm_provider == "lmstudio_llm"
            or llm_provider == "doubao_vision_llm"
            or llm_provider == "qwen3_vl_llm"
            or llm_provider == "glm_5v_turbo_llm"
        ):
            is_visual_provider = llm_provider in {
                "doubao_vision_llm",
                "qwen3_vl_llm",
                "glm_5v_turbo_llm",
            }
            # Doubao visual models can be latency-sensitive in game vision mode.
            # Send an explicit non-thinking flag to Doubao, while leaving other
            # visual OpenAI-compatible providers untouched for compatibility.
            include_thinking_config = (
                llm_provider == "doubao_vision_llm" or not is_visual_provider
            )

            return OpenAICompatibleLLM(
                model=kwargs.get("model"),
                base_url=kwargs.get("base_url"),
                llm_api_key=kwargs.get("llm_api_key"),
                organization_id=kwargs.get("organization_id"),
                project_id=kwargs.get("project_id"),
                temperature=kwargs.get("temperature"),
                request_timeout_seconds=kwargs.get(
                    "request_timeout_seconds",
                    60.0 if is_visual_provider else 5.0,
                ),
                stream_idle_timeout_seconds=kwargs.get(
                    "stream_idle_timeout_seconds",
                    30.0 if is_visual_provider else 5.0,
                ),
                include_thinking_config=include_thinking_config,
            )
        if llm_provider == "stateless_llm_with_template":
            return StatelessLLMWithTemplate(
                model=kwargs.get("model"),
                base_url=kwargs.get("base_url"),
                llm_api_key=kwargs.get("llm_api_key"),
                organization_id=kwargs.get("organization_id"),
                template=kwargs.get("template"),
                project_id=kwargs.get("project_id"),
            )
        if llm_provider == "ollama_llm":
            return OllamaLLM(
                model=kwargs.get("model"),
                base_url=kwargs.get("base_url"),
                llm_api_key=kwargs.get("llm_api_key"),
                organization_id=kwargs.get("organization_id"),
                project_id=kwargs.get("project_id"),
                temperature=kwargs.get("temperature"),
                keep_alive=kwargs.get("keep_alive"),
                unload_at_exit=kwargs.get("unload_at_exit"),
            )

        elif llm_provider == "llama_cpp_llm":
            from .stateless_llm.llama_cpp_llm import LLM as LlamaLLM

            return LlamaLLM(
                model_path=kwargs.get("model_path"),
            )
        elif llm_provider == "claude_llm":
            return ClaudeLLM(
                system=kwargs.get("system_prompt"),
                base_url=kwargs.get("base_url"),
                model=kwargs.get("model"),
                llm_api_key=kwargs.get("llm_api_key"),
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")


# Creating an LLM instance using a factory
# llm_instance = LLMFactory.create_llm("ollama", **config_dict)
