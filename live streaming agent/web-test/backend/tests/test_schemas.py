import pytest
from backend.app.schemas import (
    ChatRequest,
    ConversationCreate,
    MessageCreate,
    ModelConfig,
    ModelConfigSaveRequest,
    UsernameRequest,
)
from pydantic import ValidationError


def test_username_is_normalized() -> None:
    payload = UsernameRequest(username="  莱叔   测试  ")
    assert payload.username == "莱叔 测试"


def test_empty_username_is_rejected() -> None:
    with pytest.raises(ValidationError):
        UsernameRequest(username="   ")


def test_conversation_defaults() -> None:
    payload = ConversationCreate(username="莱叔")
    assert payload.title == "新对话"


def test_message_metadata_defaults_to_empty_dict() -> None:
    payload = MessageCreate(username="莱叔", content="你好")
    assert payload.metadata == {}


def test_model_config_rejects_browser_supplied_credentials() -> None:
    with pytest.raises(ValidationError):
        ModelConfig.model_validate(
            {
                "provider": "deepseek",
                "model": "deepseek-chat",
                "api_key": "must-stay-on-server",
            }
        )


def test_model_temperature_accepts_range_and_rejects_out_of_range() -> None:
    assert ModelConfig(temperature=0).temperature == 0
    assert ModelConfig(temperature=1.25).temperature == 1.25
    assert ModelConfig(temperature=2).temperature == 2
    with pytest.raises(ValidationError):
        ModelConfig(temperature=-0.1)
    with pytest.raises(ValidationError):
        ModelConfig(temperature=2.1)


@pytest.mark.parametrize("provider", ["openai-compatible", "ollama"])
def test_removed_model_providers_are_rejected(provider: str) -> None:
    with pytest.raises(ValidationError):
        ModelConfig(provider=provider, model="some-model")


def test_model_save_rejects_browser_supplied_url_and_key() -> None:
    with pytest.raises(ValidationError):
        ModelConfigSaveRequest.model_validate(
            {
                "provider": "deepseek",
                "model": "deepseek-chat",
                "api_base_url": "https://untrusted.example/v1",
                "api_key": "must-stay-on-server",
            }
        )


def test_chat_request_normalizes_speaker_identity() -> None:
    payload = ChatRequest(
        username="莱叔",
        content="你好",
        speaker_identity="  新   身份  ",
    )
    assert payload.speaker_identity == "新 身份"
