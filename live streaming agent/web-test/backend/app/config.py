from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_prefix="LIVE_STREAMING_AGENT_",
        extra="ignore",
    )

    app_name: str = "LiveStreamingAgent Live Backend"
    environment: str = "development"
    api_prefix: str = "/api"

    es_url: str = "http://localhost:9200"
    es_username: str | None = None
    es_password: str | None = None
    es_api_key: str | None = None
    es_index_prefix: str = "live_streaming_agent"
    knowledge_index: str = "vtuber_knowledge"
    knowledge_vector_index: str = "vtuber_knowledge_vectors"
    knowledge_embedding_model: str = "BAAI/bge-small-zh-v1.5"
    knowledge_query_prefix: str = "为这个句子生成表示以用于检索相关文章："
    knowledge_semantic_min_score: float = 0.75
    knowledge_top_k: int = 3
    knowledge_min_injection_interval_rounds: int = 10
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    llm_provider: str = "deepseek"
    llm_model: str = "deepseek-v4-pro"
    llm_api_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str | None = None
    deepseek_api_key: str | None = None
    doubao_api_key: str | None = None
    qwen_api_key: str | None = None
    kimi_api_key: str | None = None
    zhipu_api_key: str | None = None
    tokenhub_api_key: str | None = None
    # Kept temporarily so existing local .env files continue to work.
    tencent_yuanbao_api_key: str | None = None
    llm_config_path: Path = BACKEND_DIR / "model_config.json"
    douyin_profile_root: Path = BACKEND_DIR / "data" / "douyin_profiles"
    prompt_resources_path: Path = BACKEND_DIR.parent / "resources" / "prompt"
    short_term_memory_prompt_path: Path = (
        BACKEND_DIR.parent / "resources" / "短期记忆总结提示词.txt"
    )

    @property
    def users_index(self) -> str:
        return f"{self.es_index_prefix}_users"

    @property
    def conversations_index(self) -> str:
        return f"{self.es_index_prefix}_conversations"

    @property
    def messages_index(self) -> str:
        return f"{self.es_index_prefix}_messages"

    @property
    def memories_index(self) -> str:
        return f"{self.es_index_prefix}_memories"

    @property
    def prompt_configs_index(self) -> str:
        return f"{self.es_index_prefix}_prompt_configs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
