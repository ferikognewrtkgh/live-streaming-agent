import json
from datetime import UTC, datetime
from typing import Any

from .config import BACKEND_DIR, Settings
from .schemas import (
    ModelConfig,
    ModelConfigResponse,
    ModelConfigSaveRequest,
    ModelConfigTestRequest,
    ModelProviderConfig,
    RuntimeModelConfig,
    SavedProviderConfig,
)

DEFAULT_PROVIDER_BASE_URLS = {
    "deepseek": "https://api.deepseek.com/v1",
    "doubao": "https://ark.cn-beijing.volces.com/api/v3",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "kimi": "https://api.moonshot.cn/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "tencent-yuanbao": "https://tokenhub.tencentmaas.com/v1",
}
SUPPORTED_PROVIDERS = frozenset(DEFAULT_PROVIDER_BASE_URLS)

RESPONSES_STYLE_PROVIDERS = {"doubao"}

LEGACY_PROVIDER_BASE_URLS = {
    (
        "tencent-yuanbao",
        "https://api.hunyuan.cloud.tencent.com/v1",
    ): DEFAULT_PROVIDER_BASE_URLS["tencent-yuanbao"],
}


def default_provider_api_base_url(provider: str) -> str:
    return DEFAULT_PROVIDER_BASE_URLS.get(
        provider,
        DEFAULT_PROVIDER_BASE_URLS["deepseek"],
    )


def default_provider_api_style(provider: str) -> str:
    if provider in RESPONSES_STYLE_PROVIDERS:
        return "responses"
    return "chat_completions"


class ModelConfigStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        configured_path = settings.llm_config_path
        if configured_path.is_absolute():
            self.path = configured_path
        else:
            self.path = BACKEND_DIR / configured_path

    def get_config(self) -> ModelConfigResponse:
        document = self._read_document()
        providers = self._read_provider_configs(document)
        active_provider = str(document.get("active_provider") or "")
        if not active_provider:
            active_provider = self.settings.llm_provider
        if active_provider not in SUPPORTED_PROVIDERS:
            active_provider = (
                self.settings.llm_provider
                if self.settings.llm_provider in SUPPORTED_PROVIDERS
                else "deepseek"
            )

        active = providers.get(active_provider)
        if active is None:
            active = self._settings_config(active_provider)
            providers[active_provider] = active

        return self._response_for(active, providers)

    def save_config(self, payload: ModelConfigSaveRequest) -> ModelConfigResponse:
        document = self._read_document()
        providers = self._read_provider_configs(document)
        existing = providers.get(payload.provider)
        settings_match = payload.provider == self.settings.llm_provider

        providers[payload.provider] = ModelProviderConfig(
            provider=payload.provider,
            model=payload.model,
            api_base_url=(
                existing.api_base_url
                if existing is not None
                else self.settings.llm_api_base_url
                if settings_match
                else default_provider_api_base_url(payload.provider)
            ),
            api_key=(
                existing.api_key
                if existing is not None
                else None
            ),
        )

        self._write_document(
            {
                "active_provider": payload.provider,
                "providers": {
                    provider: config.model_dump()
                    for provider, config in sorted(providers.items())
                },
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )
        return self.get_config()

    def resolve_runtime_config(self, model_config: ModelConfig) -> RuntimeModelConfig:
        provider = model_config.provider or self.settings.llm_provider
        saved = self._provider_config(provider)
        settings_match = provider == self.settings.llm_provider
        environment_api_key = self._environment_api_key(provider)

        api_base_url = (
            saved.api_base_url
            if saved is not None
            else self.settings.llm_api_base_url
            if settings_match
            else default_provider_api_base_url(provider)
        )
        if environment_api_key:
            api_key = environment_api_key
        elif saved is not None and saved.api_key:
            api_key = saved.api_key
        else:
            api_key = None

        if model_config.model:
            model = model_config.model
        elif saved is not None:
            model = saved.model
        else:
            model = self.settings.llm_model

        return RuntimeModelConfig(
            provider=provider,
            model=model,
            web_search_enabled=(
                model_config.web_search_enabled
                and provider in {"doubao", "qwen"}
            ),
            web_search_forced=(
                model_config.web_search_forced
                and model_config.web_search_enabled
                and provider in {"doubao", "qwen"}
            ),
            web_search_max_tool_calls=model_config.web_search_max_tool_calls,
            web_search_result_limit=model_config.web_search_result_limit,
            temperature=model_config.temperature,
            api_base_url=api_base_url,
            api_key=api_key,
            api_style=default_provider_api_style(provider),
        )

    def resolve_test_config(
        self,
        payload: ModelConfigTestRequest,
    ) -> RuntimeModelConfig:
        return self.resolve_runtime_config(
            ModelConfig(
                provider=payload.provider,
                model=payload.model,
                web_search_enabled=False,
                web_search_forced=False,
                web_search_max_tool_calls=1,
                web_search_result_limit=3,
                temperature=0,
            )
        )

    def _provider_config(self, provider: str) -> ModelProviderConfig | None:
        return self._read_provider_configs(self._read_document()).get(provider)

    def _settings_config(self, provider: str) -> ModelProviderConfig:
        is_settings_provider = provider == self.settings.llm_provider
        return ModelProviderConfig(
            provider=provider,
            model=self.settings.llm_model,
            api_base_url=(
                self.settings.llm_api_base_url
                if is_settings_provider
                else default_provider_api_base_url(provider)
            ),
            api_key=self._environment_api_key(provider),
        )

    def _environment_api_key(self, provider: str) -> str | None:
        provider_keys = {
            "deepseek": self.settings.deepseek_api_key,
            "doubao": self.settings.doubao_api_key,
            "qwen": self.settings.qwen_api_key,
            "kimi": self.settings.kimi_api_key,
            "zhipu": self.settings.zhipu_api_key,
            "tencent-yuanbao": (
                self.settings.tokenhub_api_key
                or self.settings.tencent_yuanbao_api_key
            ),
        }
        return provider_keys.get(provider) or (
            self.settings.llm_api_key
            if provider == self.settings.llm_provider
            else None
        )

    def _response_for(
        self,
        active: ModelProviderConfig,
        providers: dict[str, ModelProviderConfig],
    ) -> ModelConfigResponse:
        saved_providers = {
            provider: SavedProviderConfig(
                provider=provider,
                model=config.model,
                has_api_key=bool(
                    config.api_key
                    or self._environment_api_key(provider)
                ),
            )
            for provider, config in sorted(providers.items())
            if config.model
        }

        return ModelConfigResponse(
            provider=active.provider,
            model=active.model,
            has_api_key=bool(
                active.api_key
                or self._environment_api_key(active.provider)
            ),
            providers=saved_providers,
        )

    def _read_document(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _read_provider_configs(
        self,
        document: dict[str, Any],
    ) -> dict[str, ModelProviderConfig]:
        raw_providers = document.get("providers")
        if not isinstance(raw_providers, dict):
            return {}

        providers: dict[str, ModelProviderConfig] = {}
        for provider, raw_config in raw_providers.items():
            if (
                provider not in SUPPORTED_PROVIDERS
                or not isinstance(raw_config, dict)
            ):
                continue
            try:
                api_base_url = str(raw_config.get("api_base_url") or "")
                migrated_api_base_url = LEGACY_PROVIDER_BASE_URLS.get(
                    (provider, api_base_url),
                    api_base_url,
                )
                providers[provider] = ModelProviderConfig.model_validate(
                    {
                        **raw_config,
                        "provider": provider,
                        "api_base_url": migrated_api_base_url,
                    }
                )
            except ValueError:
                continue
        return providers

    def _write_document(self, document: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(self.path)
