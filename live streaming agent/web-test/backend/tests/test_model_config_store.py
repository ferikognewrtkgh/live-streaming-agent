import json

from backend.app.config import Settings
from backend.app.model_config_store import ModelConfigStore
from backend.app.schemas import ModelConfig, ModelConfigSaveRequest


def test_model_selection_preserves_server_only_provider_key(tmp_path) -> None:
    config_path = tmp_path / "model_config.json"
    config_path.write_text(
        json.dumps(
            {
                "active_provider": "qwen",
                "providers": {
                    "qwen": {
                        "model": "qwen-plus",
                        "api_base_url": (
                            "https://dashscope.aliyuncs.com/compatible-mode/v1"
                        ),
                        "api_key": "sk-test-secret",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    store = ModelConfigStore(
        Settings(
            _env_file=None,
            llm_config_path=config_path,
            llm_api_key=None,
        )
    )

    response = store.save_config(
        ModelConfigSaveRequest(
            provider="qwen",
            model="qwen-max",
        )
    )

    assert response.provider == "qwen"
    assert response.has_api_key is True
    assert "api_key" not in response.model_dump()
    assert "api_base_url" not in response.model_dump()

    runtime = store.resolve_runtime_config(
        ModelConfig(provider="qwen", model="qwen-max")
    )
    assert runtime.model == "qwen-max"
    assert runtime.api_key == "sk-test-secret"
    assert runtime.api_base_url == (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def test_qwen_web_search_setting_is_preserved(tmp_path) -> None:
    store = ModelConfigStore(
        Settings(
            _env_file=None,
            llm_config_path=tmp_path / "model_config.json",
            qwen_api_key="qwen-key",
        )
    )

    runtime = store.resolve_runtime_config(
        ModelConfig(
            provider="qwen",
            model="qwen3.7-max",
            web_search_enabled=True,
            web_search_forced=True,
        )
    )

    assert runtime.web_search_enabled is True
    assert runtime.web_search_forced is True

    multimodal_runtime = store.resolve_runtime_config(
        ModelConfig(
            provider="qwen",
            model="qwen3.6-flash",
            web_search_enabled=True,
        )
    )
    assert multimodal_runtime.web_search_enabled is True


def test_model_selection_uses_env_key_and_reports_missing_provider_key(
    tmp_path,
) -> None:
    store = ModelConfigStore(
        Settings(
            _env_file=None,
            llm_config_path=tmp_path / "model_config.json",
            llm_provider="deepseek",
            llm_api_key="env-secret",
        )
    )

    deepseek = store.save_config(
        ModelConfigSaveRequest(
            provider="deepseek",
            model="deepseek-v4-pro",
        )
    )
    assert deepseek.has_api_key is True

    qwen = store.save_config(
        ModelConfigSaveRequest(
            provider="qwen",
            model="qwen-plus",
        )
    )
    assert qwen.has_api_key is False

    runtime = store.resolve_runtime_config(
        ModelConfig(provider="qwen", model="qwen-plus")
    )
    assert runtime.api_key is None
    assert runtime.api_base_url == (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def test_provider_specific_env_keys_override_saved_keys(tmp_path) -> None:
    config_path = tmp_path / "model_config.json"
    config_path.write_text(
        json.dumps(
            {
                "active_provider": "qwen",
                "providers": {
                    "qwen": {
                        "model": "qwen-plus",
                        "api_base_url": (
                            "https://dashscope.aliyuncs.com/compatible-mode/v1"
                        ),
                        "api_key": "stale-saved-key",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    store = ModelConfigStore(
        Settings(
            _env_file=None,
            llm_config_path=config_path,
            llm_provider="deepseek",
            llm_api_key="legacy-deepseek-key",
            qwen_api_key="qwen-env-key",
        )
    )

    runtime = store.resolve_runtime_config(
        ModelConfig(provider="qwen", model="qwen-plus")
    )

    assert runtime.api_key == "qwen-env-key"
    assert store.get_config().has_api_key is True


def test_tencent_yuanbao_uses_tokenhub_env_key(tmp_path) -> None:
    store = ModelConfigStore(
        Settings(
            _env_file=None,
            llm_config_path=tmp_path / "model_config.json",
            tokenhub_api_key="tokenhub-env-key",
        )
    )

    runtime = store.resolve_runtime_config(
        ModelConfig(provider="tencent-yuanbao", model="hy3")
    )

    assert runtime.api_key == "tokenhub-env-key"
    assert runtime.api_base_url == "https://tokenhub.tencentmaas.com/v1"


def test_tencent_yuanbao_accepts_legacy_env_key(tmp_path) -> None:
    store = ModelConfigStore(
        Settings(
            _env_file=None,
            llm_config_path=tmp_path / "model_config.json",
            tencent_yuanbao_api_key="legacy-tokenhub-key",
        )
    )

    runtime = store.resolve_runtime_config(
        ModelConfig(provider="tencent-yuanbao", model="hy3")
    )

    assert runtime.api_key == "legacy-tokenhub-key"


def test_tencent_yuanbao_migrates_legacy_hunyuan_endpoint(tmp_path) -> None:
    config_path = tmp_path / "model_config.json"
    config_path.write_text(
        json.dumps(
            {
                "active_provider": "tencent-yuanbao",
                "providers": {
                    "tencent-yuanbao": {
                        "model": "hy3-preview",
                        "api_base_url": (
                            "https://api.hunyuan.cloud.tencent.com/v1"
                        ),
                        "api_key": "",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    store = ModelConfigStore(
        Settings(
            _env_file=None,
            llm_config_path=config_path,
            tokenhub_api_key="tokenhub-key",
        )
    )

    runtime = store.resolve_runtime_config(
        ModelConfig(provider="tencent-yuanbao", model="hy3")
    )

    assert runtime.api_base_url == "https://tokenhub.tencentmaas.com/v1"


def test_removed_provider_in_old_config_falls_back_to_supported_provider(
    tmp_path,
) -> None:
    config_path = tmp_path / "model_config.json"
    config_path.write_text(
        json.dumps(
            {
                "active_provider": "ollama",
                "providers": {
                    "ollama": {
                        "model": "qwen3.6-flash",
                        "api_base_url": "http://localhost:11434/v1",
                        "api_key": "",
                    },
                    "deepseek": {
                        "model": "deepseek-v4-flash",
                        "api_base_url": "https://api.deepseek.com/v1",
                        "api_key": "server-key",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    store = ModelConfigStore(
        Settings(
            _env_file=None,
            llm_config_path=config_path,
            llm_provider="deepseek",
            llm_api_key=None,
        )
    )

    response = store.get_config()

    assert response.provider == "deepseek"
    assert response.model == "deepseek-v4-flash"
    assert "ollama" not in response.providers
