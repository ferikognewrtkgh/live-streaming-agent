from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

MessageRole = Literal["user", "assistant", "system", "live_viewer"]
ConversationStatus = Literal["active", "archived"]
ModelProvider = Literal[
    "deepseek",
    "doubao",
    "qwen",
    "kimi",
    "zhipu",
    "tencent-yuanbao",
]


class UsernameRequest(BaseModel):
    username: str = Field(min_length=1, max_length=80)

    @field_validator("username")
    @classmethod
    def normalize_username(cls, value: str) -> str:
        normalized = " ".join(value.strip().split())
        if not normalized:
            raise ValueError("username cannot be empty")
        return normalized


class LiveCaptureStart(UsernameRequest):
    room_id: str = Field(min_length=1, max_length=200)

    @field_validator("room_id")
    @classmethod
    def normalize_room_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("room_id cannot be empty")
        return normalized


class LiveCaptureStatus(BaseModel):
    username: str
    room_id: str
    status: Literal["starting", "running", "stopped", "error"]
    message: str


class LiveLoginStatus(BaseModel):
    status: Literal["idle", "waiting_scan", "ready", "error"]
    message: str
    qr_image: str | None = None


class UserWebSearchConfig(BaseModel):
    enabled: bool = False
    forced: bool = False
    max_tool_calls: int = Field(default=1, ge=1, le=10)
    result_limit: int = Field(default=3, ge=1, le=20)


class UserRecord(BaseModel):
    username: str
    speaker_identity: str = "莱叔"
    model_provider: ModelProvider | None = None
    model: str | None = None
    provider_models: dict[str, str] = Field(default_factory=dict)
    provider_temperatures: dict[str, float] = Field(default_factory=dict)
    provider_web_search_configs: dict[str, UserWebSearchConfig] = Field(
        default_factory=dict
    )
    web_search_enabled: bool = False
    web_search_forced: bool = False
    web_search_max_tool_calls: int = Field(default=1, ge=1, le=10)
    web_search_result_limit: int = Field(default=3, ge=1, le=20)
    temperature: float = Field(default=0.8, ge=0, le=2)
    created_at: datetime
    last_seen_at: datetime


class ConversationCreate(UsernameRequest):
    title: str = Field(default="新对话", min_length=1, max_length=120)


class ConversationUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=120)
    status: ConversationStatus | None = None


class ConversationOrderUpdate(UsernameRequest):
    conversation_ids: list[str] = Field(min_length=1, max_length=100)


class ConversationRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    conversation_id: str
    username: str
    title: str
    status: ConversationStatus
    sort_order: int = 0
    completed_rounds: int = 0
    last_turn_message_ids: list[str] = Field(default_factory=list)
    effective_char_count: int = 0
    effective_char_count_version: int = 5
    effective_memory_through_round: int = 0
    memory_compression_count: int = 0
    memory_through_round: int = 0
    short_term_memory: str = ""
    memory_status: Literal["idle", "compressing", "failed"] = "idle"
    memory_target_round: int = 0
    created_at: datetime
    updated_at: datetime


class MessageCreate(UsernameRequest):
    role: MessageRole = "user"
    content: str = Field(min_length=1, max_length=30_000)
    reasoning_content: str = Field(default="", max_length=500_000)
    emotion: str | None = Field(default=None, max_length=40)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageRecord(BaseModel):
    message_id: str
    conversation_id: str
    username: str
    role: MessageRole
    content: str
    reasoning_content: str = ""
    emotion: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ModelProvider = "deepseek"
    model: str = Field(default="deepseek-v4-pro", min_length=1, max_length=120)
    web_search_enabled: bool = False
    web_search_forced: bool = False
    web_search_max_tool_calls: int = Field(default=1, ge=1, le=10)
    web_search_result_limit: int = Field(default=3, ge=1, le=20)
    temperature: float = Field(default=0.8, ge=0, le=2)


class RuntimeModelConfig(ModelConfig):
    model_config = ConfigDict(extra="forbid")

    api_base_url: str = Field(default="https://api.deepseek.com/v1", max_length=500)
    api_key: str | None = Field(default=None, max_length=4096)
    api_style: Literal["chat_completions", "responses"] = "chat_completions"


class ModelProviderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ModelProvider
    model: str = Field(default="deepseek-v4-pro", min_length=1, max_length=120)
    api_base_url: str = Field(min_length=1, max_length=500)
    api_key: str | None = Field(default=None, max_length=4096)

    @field_validator("provider", "model", "api_base_url")
    @classmethod
    def normalize_model_config_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value cannot be empty")
        return normalized

    @field_validator("api_key")
    @classmethod
    def normalize_optional_api_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class ModelConfigSaveRequest(ModelConfig):
    pass


class ModelConfigTestRequest(ModelConfig):
    pass


class UserModelConfigUpdate(UsernameRequest):
    provider: ModelProvider
    model: str = Field(min_length=1, max_length=120)
    web_search_enabled: bool = False
    web_search_forced: bool = False
    web_search_max_tool_calls: int = Field(default=1, ge=1, le=10)
    web_search_result_limit: int = Field(default=3, ge=1, le=20)
    temperature: float = Field(default=0.8, ge=0, le=2)


class SavedProviderConfig(BaseModel):
    provider: str
    model: str
    has_api_key: bool = False


class ModelConfigResponse(BaseModel):
    provider: str
    model: str
    has_api_key: bool = False
    providers: dict[str, SavedProviderConfig] = Field(default_factory=dict)


class ModelConnectionTestResponse(BaseModel):
    ok: bool
    message: str
    latency_ms: float | None = None
    provider: str
    model: str


class PersonaPromptVersion(BaseModel):
    version: str
    title: str
    content: str


class SpeakerPromptVersion(PersonaPromptVersion):
    speaker_identity: str = ""


class PromptConfigResponse(BaseModel):
    active_version: str
    persona_prompt: str
    speaker_prompt: str
    speaker_identity: str
    versions: list[PersonaPromptVersion]
    active_speaker_version: str
    speaker_versions: list[SpeakerPromptVersion]


class PromptConfigUpdate(BaseModel):
    active_version: str | None = Field(default=None, min_length=1, max_length=255)
    active_speaker_version: str | None = Field(
        default=None,
        min_length=1,
        max_length=40,
    )
    speaker_prompt: str | None = Field(default=None, max_length=10_000)
    speaker_identity: str | None = Field(default=None, max_length=40)
    create_speaker_version: bool = False
    update_speaker_version: str | None = Field(
        default=None,
        min_length=1,
        max_length=40,
    )
    delete_speaker_version: str | None = Field(
        default=None,
        min_length=1,
        max_length=40,
    )
    rename_speaker_version: str | None = Field(
        default=None,
        min_length=1,
        max_length=40,
    )
    speaker_version_title: str | None = Field(
        default=None,
        min_length=1,
        max_length=40,
    )

    @field_validator(
        "active_version",
        "active_speaker_version",
        "speaker_prompt",
        "speaker_identity",
        "update_speaker_version",
        "delete_speaker_version",
        "rename_speaker_version",
        "speaker_version_title",
    )
    @classmethod
    def normalize_prompt_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip()


class ChatRequest(UsernameRequest):
    trace_id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=8,
        max_length=80,
    )
    content: str = Field(min_length=1, max_length=30_000)
    knowledge_query: str | None = Field(default=None, max_length=30_000)
    speaker_identity: str | None = Field(default=None, max_length=40)
    save_speaker_identity: bool = False
    system_prompt: str = Field(default="", max_length=200_000)
    knowledge_enabled: bool = True
    llm_config: ModelConfig = Field(default_factory=ModelConfig)

    @field_validator("speaker_identity")
    @classmethod
    def normalize_speaker_identity(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return " ".join(value.strip().split())


class FrontendLatencyReport(BaseModel):
    trace_id: str = Field(min_length=8, max_length=80)
    conversation_id: str = Field(min_length=1, max_length=120)
    click_timestamp: datetime
    click_to_request_start_ms: float = Field(ge=0, le=300_000)
    click_to_response_headers_ms: float = Field(ge=0, le=300_000)
    click_to_stream_start_ms: float = Field(ge=0, le=300_000)
    click_to_first_chunk_ms: float = Field(ge=0, le=300_000)
    click_to_first_paint_ms: float = Field(ge=0, le=300_000)
    response_headers_to_first_chunk_ms: float = Field(ge=0, le=300_000)
    first_chunk_to_first_paint_ms: float = Field(ge=0, le=300_000)


class ChatResponse(BaseModel):
    user_message: MessageRecord
    assistant_message: MessageRecord
    mode: Literal["model"]
    knowledge_hit_count: int = 0
    completed_rounds: int
    effective_char_count: int
    memory_status: Literal["idle", "compressing", "failed"]
    memory_through_round: int
    memory_target_round: int


class WorkspaceResponse(BaseModel):
    username: str
    speaker_identity: str
    llm_config: ModelConfig
    provider_models: dict[str, str] = Field(default_factory=dict)
    provider_temperatures: dict[str, float] = Field(default_factory=dict)
    provider_web_search_configs: dict[str, UserWebSearchConfig] = Field(
        default_factory=dict
    )
    conversations: list[ConversationRecord]
    messages: list[MessageRecord]


class MessagePage(BaseModel):
    items: list[MessageRecord]
    next_before: datetime | None = None


class ConversationSearchMatch(BaseModel):
    message_id: str
    role: MessageRole
    source: Literal["message", "knowledge", "web_search"] = "message"
    snippet: str
    created_at: datetime


class ConversationSearchResult(BaseModel):
    conversation_id: str
    title: str
    match_count: int
    matches: list[ConversationSearchMatch]


class RewindResponse(BaseModel):
    message_ids: list[str]
    deleted_count: int
    completed_rounds: int
    effective_char_count: int
    memory_compression_count: int
    memory_through_round: int
    short_term_memory: str
    memory_status: Literal["idle", "compressing", "failed"]
    memory_target_round: int


class ShortTermMemoryRecord(BaseModel):
    memory_id: str
    conversation_id: str
    username: str
    compression_number: int
    through_round: int
    summary: str
    created_at: datetime


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    size: int = Field(default=5, ge=1, le=20)


class KnowledgeHit(BaseModel):
    document_id: str
    score: float | None
    source: dict[str, Any]
