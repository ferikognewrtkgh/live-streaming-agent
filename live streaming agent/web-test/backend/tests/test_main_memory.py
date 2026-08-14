from unittest.mock import AsyncMock

import pytest
from backend.app.main import (
    create_conversation,
    next_default_conversation_title,
    next_short_term_memory_target,
    resolve_short_term_memory_context,
)
from backend.app.schemas import (
    ConversationCreate,
    ConversationRecord,
    MessageRecord,
    ShortTermMemoryRecord,
)


def conversation(*, through_round: int, summary: str) -> ConversationRecord:
    return ConversationRecord(
        conversation_id="conversation-1",
        username="测试用户",
        title="新对话 1",
        status="active",
        completed_rounds=through_round,
        memory_compression_count=through_round // 10,
        memory_through_round=through_round,
        short_term_memory=summary,
        created_at="2026-07-30T06:00:00+00:00",
        updated_at="2026-07-30T06:00:00+00:00",
    )


def message(message_id: str, round_number: int) -> MessageRecord:
    return MessageRecord(
        message_id=message_id,
        conversation_id="conversation-1",
        username="测试用户",
        role="assistant",
        content=f"第 {round_number} 轮",
        metadata={"round_number": round_number},
        created_at="2026-07-30T06:00:00+00:00",
    )


def test_compression_is_prepared_every_10_completed_rounds() -> None:
    first_window = conversation(through_round=0, summary="")
    second_window = conversation(through_round=10, summary="前十轮总结")

    assert next_short_term_memory_target(first_window, 9) is None
    assert next_short_term_memory_target(first_window, 10) == 10
    assert next_short_term_memory_target(second_window, 19) is None
    assert next_short_term_memory_target(second_window, 20) == 20


def test_default_conversation_title_uses_highest_historical_number() -> None:
    conversations = [
        conversation(through_round=0, summary="").model_copy(
            update={"title": "新对话 1", "status": "archived"}
        ),
        conversation(through_round=0, summary="").model_copy(
            update={"title": "新对话2"}
        ),
        conversation(through_round=0, summary="").model_copy(
            update={"title": "已重命名"}
        ),
    ]

    assert next_default_conversation_title(conversations) == "新对话 3"


@pytest.mark.asyncio
async def test_create_default_conversation_includes_archived_titles() -> None:
    store = AsyncMock()
    history = [
        conversation(through_round=0, summary="").model_copy(
            update={"title": "新对话 1", "status": "archived"}
        ),
        conversation(through_round=0, summary="").model_copy(
            update={"title": "新对话 2"}
        ),
    ]
    created = history[-1].model_copy(
        update={"conversation_id": "conversation-3", "title": "新对话 3"}
    )
    store.list_conversations.return_value = history
    store.create_conversation.return_value = created

    result = await create_conversation(
        ConversationCreate(username="测试用户"),
        store,
    )

    store.list_conversations.assert_awaited_once_with(
        "测试用户",
        include_archived=True,
        size=1000,
    )
    store.create_conversation.assert_awaited_once_with(
        "测试用户",
        "新对话 3",
    )
    assert result.title == "新对话 3"


@pytest.mark.asyncio
async def test_first_memory_is_not_injected_before_round_20() -> None:
    store = AsyncMock()
    store.latest_short_term_memory_at_or_before.return_value = None
    history = [message("round-9", 9), message("round-10", 10)]

    summary, through_round, filtered_history = (
        await resolve_short_term_memory_context(
            store=store,
            conversation=conversation(through_round=10, summary="前十轮总结"),
            round_number=19,
            history=history,
        )
    )

    assert summary == ""
    assert through_round == 0
    assert filtered_history == history
    store.latest_short_term_memory_at_or_before.assert_not_awaited()


@pytest.mark.asyncio
async def test_first_memory_is_injected_at_round_20() -> None:
    store = AsyncMock()
    history = [message("round-10", 10), message("round-11", 11)]

    summary, through_round, filtered_history = (
        await resolve_short_term_memory_context(
            store=store,
            conversation=conversation(through_round=10, summary="前十轮总结"),
            round_number=20,
            history=history,
        )
    )

    assert summary == "前十轮总结"
    assert through_round == 10
    assert [item.message_id for item in filtered_history] == ["round-11"]
    store.latest_short_term_memory_at_or_before.assert_not_awaited()


@pytest.mark.asyncio
async def test_previous_eligible_memory_remains_active_until_next_window() -> None:
    store = AsyncMock()
    store.latest_short_term_memory_at_or_before.return_value = (
        ShortTermMemoryRecord(
            memory_id="memory-10",
            conversation_id="conversation-1",
            username="测试用户",
            compression_number=1,
            through_round=10,
            summary="前十轮总结",
            created_at="2026-07-30T06:00:00+00:00",
        )
    )

    summary, through_round, _ = await resolve_short_term_memory_context(
        store=store,
        conversation=conversation(through_round=20, summary="前二十轮总结"),
        round_number=21,
        history=[message("round-11", 11)],
    )

    assert summary == "前十轮总结"
    assert through_round == 10
    store.latest_short_term_memory_at_or_before.assert_awaited_once_with(
        "conversation-1",
        11,
    )
