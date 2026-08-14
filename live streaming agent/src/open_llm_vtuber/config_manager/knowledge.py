from typing import ClassVar

from pydantic import Field

from .i18n import Description, I18nMixin


class KnowledgeConfig(I18nMixin):
    """Configuration for local Elasticsearch knowledge retrieval."""

    enabled: bool = Field(False, alias="enabled")
    es_url: str = Field("auto", alias="es_url")
    api_key: str = Field("", alias="api_key")
    username: str = Field("", alias="username")
    password: str = Field("", alias="password")
    verify_certs: bool = Field(False, alias="verify_certs")
    request_timeout: float = Field(30.0, alias="request_timeout")
    embedding_model: str = Field("", alias="embedding_model")
    knowledge_index: str = Field("vtuber_knowledge", alias="knowledge_index")
    vector_index: str = Field("vtuber_knowledge_vectors", alias="vector_index")
    performance_storage_enabled: bool = Field(
        False,
        alias="performance_storage_enabled",
    )
    performance_index: str = Field(
        "vtuber_performance_metrics",
        alias="performance_index",
    )
    query_prefix: str = Field(
        "为这个句子生成表示以用于检索相关文章：",
        alias="query_prefix",
    )
    semantic_min_score: float = Field(0.75, alias="semantic_min_score")
    top_k: int = Field(3, alias="top_k")
    min_injection_interval_rounds: int = Field(
        10,
        alias="min_injection_interval_rounds",
    )
    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "enabled": Description(
            en="Enable local Elasticsearch knowledge retrieval",
            zh="是否启用本地 Elasticsearch 知识库检索",
        ),
        "es_url": Description(
            en="Elasticsearch URL. Use auto to probe local HTTP/HTTPS",
            zh="Elasticsearch 地址。auto 会自动探测本机 HTTP/HTTPS",
        ),
        "embedding_model": Description(
            en="Local embedding model path or sentence-transformers model id",
            zh="本地 embedding 模型路径或 sentence-transformers 模型 ID",
        ),
        "min_injection_interval_rounds": Description(
            en="Minimum conversation turns between knowledge injections",
            zh="两次知识库注入之间的最少对话轮数",
        ),
        "performance_storage_enabled": Description(
            en="Store voice-turn performance metrics in Elasticsearch",
            zh="是否将语音轮次性能指标存入 Elasticsearch",
        ),
        "performance_index": Description(
            en="Elasticsearch index for voice-turn performance metrics",
            zh="语音轮次性能指标使用的 Elasticsearch 索引",
        ),
    }
