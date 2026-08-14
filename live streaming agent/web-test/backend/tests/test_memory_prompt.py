import pytest
from backend.app.memory_prompt import (
    build_short_term_memory_input,
    build_short_term_memory_prompt,
)
from backend.app.schemas import MessageRecord


def test_memory_prompt_uses_only_file_content(tmp_path) -> None:
    prompt_path = tmp_path / "短期记忆总结提示词.txt"
    prompt_path.write_text("只提取新增信息。", encoding="utf-8")

    prompt = build_short_term_memory_prompt(prompt_path)

    assert prompt == "只提取新增信息。"


def test_memory_input_combines_previous_summary_and_dialogue_in_one_message() -> None:
    messages = [
        MessageRecord(
            message_id="user-11",
            conversation_id="conversation-1",
            username="测试用户",
            role="user",
            content="莱叔：“你好”",
            metadata={"round_number": 11},
            created_at="2026-07-29T06:00:00+00:00",
        ),
        MessageRecord(
            message_id="assistant-11",
            conversation_id="conversation-1",
            username="测试用户",
            role="assistant",
            content="[开心] 你好呀",
            metadata={"round_number": 11},
            created_at="2026-07-29T06:00:01+00:00",
        ),
    ]

    memory_input = build_short_term_memory_input("已有总结", messages)

    assert memory_input == (
        "# 之前总结的短期记忆\n"
        "已有总结\n\n"
        "# 本次待总结对话\n"
        "用户：莱叔：“你好”\n"
        "AI：[开心] 你好呀"
    )


def test_memory_prompt_rejects_missing_or_empty_file(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="无法读取"):
        build_short_term_memory_prompt(tmp_path / "missing.txt")

    empty_path = tmp_path / "empty.txt"
    empty_path.write_text("  ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="提示词为空"):
        build_short_term_memory_prompt(empty_path)
