from pydantic import Field
from typing import ClassVar, Dict

from .i18n import Description, I18nMixin


class PaintConfig(I18nMixin):
    """Configuration for the drawing assistant model."""

    provider: str = Field("glm", alias="provider")
    api_key: str = Field("", alias="api_key")
    base_url: str = Field("", alias="base_url")
    model: str = Field("", alias="model")
    timeout_seconds: float = Field(90.0, alias="timeout_seconds")

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "provider": Description(
            en="Paint model provider. Built-in values: glm or ark",
            zh="画图代码生成模型提供商。内置值：glm 或 ark",
        ),
        "api_key": Description(
            en="API key for the paint model provider",
            zh="画图模型服务的 API key",
        ),
        "base_url": Description(
            en="Optional OpenAI-compatible base URL override",
            zh="可选的 OpenAI 兼容接口地址覆盖",
        ),
        "model": Description(
            en="Optional paint model name override",
            zh="可选的画图模型名称覆盖",
        ),
        "timeout_seconds": Description(
            en="Timeout for paint model requests in seconds",
            zh="画图模型请求超时时间（秒）",
        ),
    }
