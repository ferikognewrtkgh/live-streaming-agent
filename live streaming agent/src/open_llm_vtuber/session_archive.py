"""
Session Archive — 后端进程结束时，读取本轮对话记录并生成纪要。

对话记录已存在于 logs/chat_history/ 中，无需重复保存。
本模块直接扫描今天的对话文件，调用 LLM 生成纪要：
  logs/session_archives/<日期>_Live Streaming Agent/summary.md
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from loguru import logger

from .agent.stateless_llm_factory import LLMFactory
from .chat_history_manager import CHAT_HISTORY_ROOT

SESSION_ARCHIVE_ROOT = os.path.join("logs", "session_archives")

SUMMARY_SYSTEM_PROMPT = """\
你是一位直播复盘助手。根据提供的本场直播对话记录，生成一份简洁的直播纪要。
纪要必须严格包含以下四个部分，每部分 2-4 句话：

## 核心事件
概括本场直播中发生的主要事件和话题。

## 关系进展
概括主播与观众/角色之间的关系变化、情感互动。

## 观众反应
概括观众的主要反应、弹幕互动特点、礼物情况。

## 下场延续
基于本场内容，提出下一场直播可以延续或深入的话题、悬念。

注意：
- 用中文输出
- 简洁精炼，不要罗列原文
- 如果对话内容较少，可适当缩短每部分内容
- 直接输出纪要内容，不要加额外的开头或结尾
"""


def _build_transcript(messages: List[Dict[str, Any]]) -> str:
    """将对话记录拼成可读文本供 LLM 摘要。"""
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        name = msg.get("name", "")
        content = msg.get("content", "")
        if role == "system" or role == "metadata":
            continue
        label = name if name else ("用户" if role == "human" else "AI")
        lines.append(f"[{label}]: {content}")
    return "\n".join(lines)


def _find_today_histories(conf_uid: str) -> List[str]:
    """
    扫描 chat_history/<conf_uid>/ 目录，找到文件名以今天日期开头的对话文件。
    文件名格式: 2026-05-25_14-21-36_xxxxx.json
    """
    today_prefix = datetime.now().strftime("%Y-%m-%d")
    conf_dir = os.path.join(CHAT_HISTORY_ROOT, conf_uid)
    if not os.path.isdir(conf_dir):
        return []

    results = []
    for filename in os.listdir(conf_dir):
        if filename.startswith(today_prefix) and filename.endswith(".json"):
            results.append(os.path.join(conf_dir, filename))

    # 按文件名排序（即按时间排序）
    results.sort()
    return results


def _load_messages_from_file(filepath: str) -> List[Dict[str, Any]]:
    """从单个历史文件加载对话消息（过滤 metadata）。"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [msg for msg in data if msg.get("role") != "metadata"]
    except Exception as e:
        logger.error(f"Failed to load history file {filepath}: {e}")
        return []


async def _generate_summary(transcript: str, llm_provider: str, llm_kwargs: dict) -> str:
    """调用 LLM 生成纪要文本。"""
    try:
        llm = LLMFactory.create_llm(llm_provider, **llm_kwargs)
        messages = [
            {
                "role": "user",
                "content": f"以下是本场直播的对话记录：\n\n{transcript}",
            }
        ]
        pieces = []
        stream = llm.chat_completion(messages, SUMMARY_SYSTEM_PROMPT)
        async for event in stream:
            if isinstance(event, dict):
                if event.get("type") == "text_delta":
                    pieces.append(event.get("text", ""))
                elif event.get("type") == "error":
                    logger.error(f"LLM summary error: {event.get('message')}")
                    break
            elif isinstance(event, str):
                pieces.append(event)
        return "".join(pieces).strip()
    except Exception as e:
        logger.error(f"Failed to generate session summary: {e}")
        return f"[纪要生成失败: {e}]"


def _get_llm_config_from_context(ctx) -> tuple[str, dict]:
    """从 ServiceContext 中提取 LLM provider 和参数。"""
    agent_config = ctx.character_config.agent_config
    llm_provider = agent_config.agent_settings.basic_memory_agent.llm_provider
    llm_configs = agent_config.llm_configs
    llm_kwargs = getattr(llm_configs, llm_provider, None)
    if llm_kwargs is None:
        return llm_provider, {}
    return llm_provider, llm_kwargs.model_dump()


async def archive_session(
    default_context,
    character_name: str = "Live Streaming Agent",
) -> Optional[str]:
    """
    读取今天的对话记录，调用 LLM 生成纪要并保存。

    Args:
        default_context: 服务器的 default_context_cache (ServiceContext)，
                         用于获取 conf_uid 和 LLM 配置。
        character_name: 归档文件夹中使用的角色名。

    Returns:
        纪要文件路径，如果没有可归档内容则返回 None。
    """
    conf_uid = default_context.character_config.conf_uid

    # 找今天的所有对话文件
    today_files = _find_today_histories(conf_uid)
    if not today_files:
        logger.info("[session_archive] No conversation files found for today.")
        return None

    # 合并今天所有对话
    flat_messages = []
    for filepath in today_files:
        messages = _load_messages_from_file(filepath)
        flat_messages.extend(messages)

    if not flat_messages:
        logger.info("[session_archive] Today's conversations are empty, skipping.")
        return None

    transcript = _build_transcript(flat_messages)
    if not transcript.strip():
        logger.info("[session_archive] Transcript is empty, skipping summary.")
        return None

    logger.info(
        f"[session_archive] Found {len(today_files)} file(s), "
        f"{len(flat_messages)} messages, generating summary..."
    )

    # 调用 LLM 生成纪要
    llm_provider, llm_kwargs = _get_llm_config_from_context(default_context)
    summary_text = await _generate_summary(transcript, llm_provider, llm_kwargs)

    if not summary_text:
        summary_text = "[未能生成纪要]"

    # 写入纪要文件
    date_str = datetime.now().strftime("%Y-%m-%d")
    archive_name = f"{date_str}_{character_name}"
    archive_dir = os.path.join(SESSION_ARCHIVE_ROOT, archive_name)
    os.makedirs(archive_dir, exist_ok=True)

    summary_path = os.path.join(archive_dir, "summary.md")
    header = f"# 直播纪要 — {date_str} {character_name}\n\n"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(header + summary_text + "\n")
    logger.info(f"[session_archive] Summary saved to {summary_path}")

    return summary_path
