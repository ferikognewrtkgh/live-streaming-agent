from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from .agent.stateless_llm.provider_runtime_llm import ProviderRuntimeLLM


PROJECT_MODEL_CONFIG_PATH = Path("logs/config/project_model_config.json")
DEFAULT_PROVIDER_BASE_URLS = {
    "deepseek": "https://api.deepseek.com/v1",
    "doubao": "https://ark.cn-beijing.volces.com/api/v3",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "kimi": "https://api.moonshot.cn/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "tencent-yuanbao": "https://tokenhub.tencentmaas.com/v1",
}
PROVIDER_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "id": "deepseek",
        "name": "DeepSeek",
        "default_model": "deepseek-v4-pro",
        "models": (
            ("deepseek-v4-pro", "DeepSeek V4 Pro"),
            ("deepseek-v4-flash", "DeepSeek V4 Flash"),
        ),
    },
    {
        "id": "doubao",
        "name": "豆包 / 火山方舟",
        "default_model": "doubao-seed-2-0-lite-260215",
        "models": (
            ("doubao-seed-evolving", "Doubao Seed Evolving"),
            ("doubao-seed-2-1-turbo-260628", "Doubao Seed 2.1 Turbo"),
            ("doubao-seed-2-1-pro-260628", "Doubao Seed 2.1 Pro"),
            ("doubao-seed-2-0-mini-260428", "Doubao Seed 2.0 Mini"),
            ("doubao-seed-2-0-pro-260215", "Doubao Seed 2.0 Pro"),
            ("doubao-seed-2-0-lite-260215", "Doubao Seed 2.0 Lite"),
        ),
    },
    {
        "id": "qwen",
        "name": "通义千问 / 百炼",
        "default_model": "qwen3.6-flash",
        "models": (
            ("qwen3.7-max", "Qwen3.7 Max"),
            ("qwen3.7-plus", "Qwen3.7 Plus"),
            ("qwen3.6-max-preview", "Qwen3.6 Max Preview"),
            ("qwen3.6-plus", "Qwen3.6 Plus"),
            ("qwen3.6-flash", "Qwen3.6 Flash"),
            ("qwen3.5-plus", "Qwen3.5 Plus"),
            ("qwen3.5-flash", "Qwen3.5 Flash"),
        ),
    },
    {
        "id": "kimi",
        "name": "Kimi / Moonshot",
        "default_model": "kimi-k2.6",
        "models": (("kimi-k3", "Kimi K3"), ("kimi-k2.6", "Kimi K2.6")),
    },
    {
        "id": "zhipu",
        "name": "GLM 智谱",
        "default_model": "glm-5.2",
        "models": (
            ("glm-5.2", "GLM-5.2"),
            ("glm-5.1", "GLM-5.1"),
            ("glm-5", "GLM-5"),
            ("glm-5-turbo", "GLM-5-Turbo"),
            ("glm-4.7", "GLM-4.7"),
            ("glm-4.7-flashx", "GLM-4.7-FlashX"),
            ("glm-4.7-flash", "GLM-4.7-Flash"),
            ("glm-4.6", "GLM-4.6"),
            ("glm-4.5-air", "GLM-4.5-Air"),
            ("glm-4.5-airx", "GLM-4.5-AirX"),
        ),
    },
    {
        "id": "tencent-yuanbao",
        "name": "腾讯元宝 / TokenHub",
        "default_model": "hy3",
        "models": (
            ("hy3", "Hy3"),
            ("hy3-preview", "Hy3 Preview"),
            ("hy-mt2-pro", "Hy-MT2 Pro"),
            ("hy-mt2-plus", "Hy-MT2 Plus"),
            ("hy-mt2-lite", "Hy-MT2 Lite"),
            ("hunyuan-role-latest", "Hy-Role Latest"),
            ("hy-role", "Hy-Role"),
        ),
    },
)
PROVIDER_BY_ID = {item["id"]: item for item in PROVIDER_CATALOG}
PROVIDER_LLM_CONFIG_KEYS = {
    "deepseek": ("deepseek_llm",),
    "doubao": ("doubao_vision_llm",),
    "qwen": ("qwen3_vl_llm",),
    "kimi": ("kimi_llm",),
    "zhipu": ("zhipu_llm", "glm_5v_turbo_llm"),
    "tencent-yuanbao": ("tencent_yuanbao_llm",),
}
PROVIDER_ENV_KEYS = {
    "deepseek": ("LIVE_STREAMING_AGENT_DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY"),
    "doubao": ("LIVE_STREAMING_AGENT_DOUBAO_API_KEY", "DOUBAO_API_KEY"),
    "qwen": ("LIVE_STREAMING_AGENT_QWEN_API_KEY", "QWEN_API_KEY", "DASHSCOPE_API_KEY"),
    "kimi": ("LIVE_STREAMING_AGENT_KIMI_API_KEY", "KIMI_API_KEY", "MOONSHOT_API_KEY"),
    "zhipu": ("LIVE_STREAMING_AGENT_ZHIPU_API_KEY", "ZHIPU_API_KEY"),
    "tencent-yuanbao": (
        "LIVE_STREAMING_AGENT_TOKENHUB_API_KEY",
        "LIVE_STREAMING_AGENT_TENCENT_YUANBAO_API_KEY",
        "TOKENHUB_API_KEY",
        "TENCENT_YUANBAO_API_KEY",
    ),
}


@dataclass(frozen=True)
class RuntimeProjectModelConfig:
    provider: str
    model: str
    temperature: float
    web_search_enabled: bool
    web_search_forced: bool
    web_search_max_tool_calls: int
    web_search_result_limit: int
    base_url: str
    api_key: str


class ProjectModelConfigManager:
    """Persist safe UI choices and resolve credentials only on the backend."""

    def __init__(self, config: Any, path: Path = PROJECT_MODEL_CONFIG_PATH) -> None:
        self.config = config
        self.path = path
        self._live_streaming_agent_credentials = self._read_live_streaming_agent_env_values()
        self._document = self._load_document()
        self._ensure_defaults()

    def public_state(self) -> dict[str, Any]:
        active = self.active_config()
        catalog = []
        for item in PROVIDER_CATALOG:
            provider_id = item["id"]
            saved_config = self._public_provider_config(provider_id)
            catalog.append(
                {
                    "id": provider_id,
                    "name": item["name"],
                    "default_model": item["default_model"],
                    "models": [
                        {"id": model_id, "label": label}
                        for model_id, label in item["models"]
                    ],
                    "web_search_supported": provider_id in {"doubao", "qwen"},
                    "has_api_key": bool(self._resolve_api_key(provider_id)),
                    "saved_config": saved_config,
                }
            )
        return {
            "type": "project-config-state",
            "provider": active.provider,
            "model": active.model,
            "temperature": active.temperature,
            "web_search_enabled": active.web_search_enabled,
            "web_search_forced": active.web_search_forced,
            "web_search_max_tool_calls": active.web_search_max_tool_calls,
            "web_search_result_limit": active.web_search_result_limit,
            "has_api_key": bool(active.api_key),
            "catalog": catalog,
        }

    def active_config(self) -> RuntimeProjectModelConfig:
        provider = str(self._document.get("active_provider") or "deepseek")
        if provider not in PROVIDER_BY_ID:
            provider = "deepseek"
        provider_state = (self._document.get("providers") or {}).get(provider) or {}
        catalog = PROVIDER_BY_ID[provider]
        model = str(provider_state.get("model") or catalog["default_model"])
        valid_models = {model_id for model_id, _label in catalog["models"]}
        if model not in valid_models:
            model = str(catalog["default_model"])
        temperature = self._validated_temperature(
            provider_state.get("temperature", 0.8)
        )
        web_search_enabled = bool(provider_state.get("web_search_enabled"))
        web_search_forced = bool(provider_state.get("web_search_forced"))
        web_search_max_tool_calls = self._validated_integer(
            provider_state.get("web_search_max_tool_calls", 1),
            minimum=1,
            maximum=10,
            field_name="最大搜索次数",
        )
        web_search_result_limit = self._validated_integer(
            provider_state.get("web_search_result_limit", 3),
            minimum=1,
            maximum=20,
            field_name="搜索网页数",
        )
        if provider not in {"doubao", "qwen"}:
            web_search_enabled = False
            web_search_forced = False
        if not web_search_enabled:
            web_search_forced = False
        return RuntimeProjectModelConfig(
            provider=provider,
            model=model,
            temperature=temperature,
            web_search_enabled=web_search_enabled,
            web_search_forced=web_search_forced,
            web_search_max_tool_calls=web_search_max_tool_calls,
            web_search_result_limit=web_search_result_limit,
            base_url=self._resolve_base_url(provider),
            api_key=self._resolve_api_key(provider),
        )

    def save(self, payload: dict[str, Any]) -> RuntimeProjectModelConfig:
        provider = str(payload.get("provider") or "").strip()
        if provider not in PROVIDER_BY_ID:
            raise ValueError(f"不支持的模型供应商: {provider}")
        model = str(payload.get("model") or "").strip()
        valid_models = {
            model_id for model_id, _label in PROVIDER_BY_ID[provider]["models"]
        }
        if model not in valid_models:
            raise ValueError(f"{provider} 不支持模型: {model}")
        temperature = self._validated_temperature(payload.get("temperature", 0.8))
        web_search_enabled = bool(payload.get("web_search_enabled"))
        web_search_forced = bool(payload.get("web_search_forced"))
        web_search_max_tool_calls = self._validated_integer(
            payload.get("web_search_max_tool_calls", 1),
            minimum=1,
            maximum=10,
            field_name="最大搜索次数",
        )
        web_search_result_limit = self._validated_integer(
            payload.get("web_search_result_limit", 3),
            minimum=1,
            maximum=20,
            field_name="搜索网页数",
        )
        if web_search_enabled and provider not in {"doubao", "qwen"}:
            raise ValueError("当前供应商不支持联网搜索")
        if not web_search_enabled:
            web_search_forced = False
        if not self._resolve_api_key(provider):
            raise ValueError(f"服务端未配置 {provider} API Key")
        providers = self._document.setdefault("providers", {})
        providers[provider] = {
            "model": model,
            "temperature": temperature,
            "web_search_enabled": web_search_enabled,
            "web_search_forced": web_search_forced,
            "web_search_max_tool_calls": web_search_max_tool_calls,
            "web_search_result_limit": web_search_result_limit,
        }
        self._document["active_provider"] = provider
        self._write_document()
        return self.active_config()

    def build_llm(
        self,
        runtime: RuntimeProjectModelConfig | None = None,
        *,
        connection_test: bool = False,
    ) -> ProviderRuntimeLLM:
        runtime = runtime or self.active_config()
        if not runtime.api_key:
            raise ValueError(f"服务端未配置 {runtime.provider} API Key")
        return ProviderRuntimeLLM(
            provider=runtime.provider,
            model=runtime.model,
            base_url=runtime.base_url,
            api_key=runtime.api_key,
            temperature=runtime.temperature,
            web_search_enabled=runtime.web_search_enabled,
            web_search_forced=runtime.web_search_forced,
            web_search_max_tool_calls=runtime.web_search_max_tool_calls,
            web_search_result_limit=runtime.web_search_result_limit,
            request_timeout_seconds=25.0 if connection_test else 5.0,
            stream_idle_timeout_seconds=25.0 if connection_test else 5.0,
        )

    async def test_connection(self, payload: dict[str, Any]) -> dict[str, Any]:
        provider = str(payload.get("provider") or "").strip()
        model = str(payload.get("model") or "").strip()
        temperature = self._validated_temperature(payload.get("temperature", 0.8))
        if provider not in PROVIDER_BY_ID:
            raise ValueError(f"不支持的模型供应商: {provider}")
        if model not in {
            model_id for model_id, _label in PROVIDER_BY_ID[provider]["models"]
        }:
            raise ValueError(f"{provider} 不支持模型: {model}")
        runtime = RuntimeProjectModelConfig(
            provider=provider,
            model=model,
            temperature=temperature,
            web_search_enabled=False,
            web_search_forced=False,
            web_search_max_tool_calls=1,
            web_search_result_limit=3,
            base_url=self._resolve_base_url(provider),
            api_key=self._resolve_api_key(provider),
        )
        started_at = time.perf_counter()
        try:
            llm = self.build_llm(runtime, connection_test=True)
            async for chunk in llm.chat_completion(
                [{"role": "user", "content": "用中文回复：连接成功"}],
                call_source="project_config_connection_test",
            ):
                if not isinstance(chunk, str) or not chunk.strip():
                    continue
                if chunk.startswith("Error calling the chat endpoint"):
                    raise RuntimeError(chunk)
                return {
                    "ok": True,
                    "message": "连接成功",
                    "latency_ms": round((time.perf_counter() - started_at) * 1000, 1),
                    "provider": provider,
                    "model": model,
                }
            raise RuntimeError("模型返回为空")
        except Exception as exc:
            logger.warning(
                "Project model connection test failed: provider={} model={} error={}",
                provider,
                model,
                exc,
            )
            return {
                "ok": False,
                "message": str(exc),
                "provider": provider,
                "model": model,
            }

    def _ensure_defaults(self) -> None:
        if self._document.get("active_provider") in PROVIDER_BY_ID:
            return
        provider, model, temperature = self._configured_base_model()
        self._document = {
            "active_provider": provider,
            "providers": {
                provider: {
                    "model": model,
                    "temperature": temperature,
                    "web_search_enabled": False,
                    "web_search_forced": False,
                    "web_search_max_tool_calls": 1,
                    "web_search_result_limit": 3,
                }
            },
        }
        self._write_document()

    def _configured_base_model(self) -> tuple[str, str, float]:
        settings = (
            self.config.character_config.agent_config.agent_settings.basic_memory_agent
        )
        provider_key = str(settings.llm_provider or "deepseek_llm")
        provider = {
            "deepseek_llm": "deepseek",
            "doubao_vision_llm": "doubao",
            "qwen3_vl_llm": "qwen",
            "kimi_llm": "kimi",
            "zhipu_llm": "zhipu",
            "tencent_yuanbao_llm": "tencent-yuanbao",
        }.get(provider_key, "deepseek")
        llm_config = getattr(
            self.config.character_config.agent_config.llm_configs,
            provider_key,
            None,
        )
        model = str(getattr(llm_config, "model", "") or "")
        valid_models = {
            model_id for model_id, _label in PROVIDER_BY_ID[provider]["models"]
        }
        if model not in valid_models:
            model = str(PROVIDER_BY_ID[provider]["default_model"])
        temperature = self._validated_temperature(
            getattr(llm_config, "temperature", 0.8)
        )
        return provider, model, temperature

    def _public_provider_config(self, provider: str) -> dict[str, Any]:
        catalog = PROVIDER_BY_ID[provider]
        provider_state = (self._document.get("providers") or {}).get(provider) or {}
        model = str(provider_state.get("model") or catalog["default_model"])
        if model not in {model_id for model_id, _label in catalog["models"]}:
            model = str(catalog["default_model"])
        try:
            temperature = self._validated_temperature(
                provider_state.get("temperature", 0.8)
            )
        except ValueError:
            temperature = 0.8
        try:
            max_tool_calls = self._validated_integer(
                provider_state.get("web_search_max_tool_calls", 1),
                minimum=1,
                maximum=10,
                field_name="最大搜索次数",
            )
        except ValueError:
            max_tool_calls = 1
        try:
            result_limit = self._validated_integer(
                provider_state.get("web_search_result_limit", 3),
                minimum=1,
                maximum=20,
                field_name="搜索网页数",
            )
        except ValueError:
            result_limit = 3
        search_supported = provider in {"doubao", "qwen"}
        search_enabled = search_supported and bool(
            provider_state.get("web_search_enabled")
        )
        return {
            "model": model,
            "temperature": temperature,
            "web_search_enabled": search_enabled,
            "web_search_forced": search_enabled
            and bool(provider_state.get("web_search_forced")),
            "web_search_max_tool_calls": max_tool_calls,
            "web_search_result_limit": result_limit,
        }

    def _resolve_base_url(self, provider: str) -> str:
        llm_config = self._provider_llm_config(provider)
        configured = str(getattr(llm_config, "base_url", "") or "").strip()
        return configured or DEFAULT_PROVIDER_BASE_URLS[provider]

    def _resolve_api_key(self, provider: str) -> str:
        for env_name in PROVIDER_ENV_KEYS.get(provider, ()):
            value = str(os.getenv(env_name) or "").strip()
            if value:
                return value
        llm_config = self._provider_llm_config(provider)
        value = str(getattr(llm_config, "llm_api_key", "") or "").strip()
        if self._looks_configured_secret(value):
            return value
        return self._live_streaming_agent_credentials.get(provider, "")

    def _provider_llm_config(self, provider: str) -> Any | None:
        configs = self.config.character_config.agent_config.llm_configs
        for config_key in PROVIDER_LLM_CONFIG_KEYS.get(provider, ()):
            value = getattr(configs, config_key, None)
            api_key = str(getattr(value, "llm_api_key", "") or "").strip()
            if value is not None and self._looks_configured_secret(api_key):
                return value
        keys = PROVIDER_LLM_CONFIG_KEYS.get(provider, ())
        return getattr(configs, keys[0], None) if keys else None

    @staticmethod
    def _looks_configured_secret(value: str) -> bool:
        lowered = value.lower()
        return bool(value) and not lowered.startswith(("your ", "your_", "<"))

    @staticmethod
    def _read_live_streaming_agent_env_values() -> dict[str, str]:
        path = Path("web-test/backend/.env")
        if not path.exists():
            return {}
        raw_values: dict[str, str] = {}
        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                raw_values[name.strip()] = value.strip().strip('"\'')
        except OSError:
            logger.exception("Failed to read web-test backend environment file.")
            return {}
        result: dict[str, str] = {}
        for provider, env_names in PROVIDER_ENV_KEYS.items():
            for env_name in env_names:
                value = raw_values.get(env_name, "")
                if value:
                    result[provider] = value
                    break
        return result

    def _load_document(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            logger.exception("Failed to load project model config: {}", self.path)
            return {}

    def _write_document(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(self._document, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(self.path)

    @staticmethod
    def _validated_temperature(value: Any) -> float:
        try:
            temperature = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("温度必须是 0 到 2 之间的数字") from exc
        if not 0 <= temperature <= 2:
            raise ValueError("温度必须在 0 到 2 之间")
        return temperature

    @staticmethod
    def _validated_integer(
        value: Any,
        *,
        minimum: int,
        maximum: int,
        field_name: str,
    ) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name}必须是整数") from exc
        if not minimum <= number <= maximum:
            raise ValueError(
                f"{field_name}必须在 {minimum} 到 {maximum} 之间"
            )
        return number
