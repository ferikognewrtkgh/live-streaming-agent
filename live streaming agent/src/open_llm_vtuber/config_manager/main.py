# config_manager/main.py
from pydantic import BaseModel, Field
from typing import ClassVar, Dict

from .system import SystemConfig
from .character import CharacterConfig
from .live import LiveConfig
from .paint import PaintConfig
from .knowledge import KnowledgeConfig
from .i18n import Description, I18nMixin


class Config(I18nMixin, BaseModel):
    """
    Main configuration for the application.
    """

    system_config: SystemConfig = Field(default=None, alias="system_config")
    character_config: CharacterConfig = Field(..., alias="character_config")
    live_config: LiveConfig = Field(default_factory=LiveConfig, alias="live_config")
    paint_config: PaintConfig = Field(default_factory=PaintConfig, alias="paint_config")
    knowledge_config: KnowledgeConfig = Field(
        default_factory=KnowledgeConfig,
        alias="knowledge_config",
    )

    DESCRIPTIONS: ClassVar[Dict[str, Description]] = {
        "system_config": Description(
            en="System configuration settings",
            zh="系统配置设置",
        ),
        "character_config": Description(
            en="Character configuration settings",
            zh="角色配置设置",
        ),
        "live_config": Description(
            en="Live streaming platform integration settings",
            zh="直播平台集成设置",
        ),
        "paint_config": Description(
            en="Drawing assistant model settings",
            zh="画图助手模型设置",
        ),
        "knowledge_config": Description(
            en="Local Elasticsearch knowledge retrieval settings",
            zh="本地 Elasticsearch 知识库检索设置",
        ),
    }
