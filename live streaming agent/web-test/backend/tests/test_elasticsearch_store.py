from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from backend.app.config import Settings
from backend.app.elasticsearch_store import (
    INDEX_MAPPINGS,
    ElasticsearchStore,
    count_message_characters,
    count_text_characters,
    user_document_id,
)
from backend.app.schemas import (
    ConversationRecord,
    ConversationUpdate,
    MessageCreate,
    MessageRecord,
    ShortTermMemoryRecord,
)


@pytest.mark.asyncio
async def test_preload_knowledge_embedder_runs_warmup_encoding() -> None:
    client = AsyncMock()
    store = ElasticsearchStore(client, Settings())
    embedder = MagicMock()
    embedder.encode_query.return_value = [0.1, 0.2]
    store._knowledge_embedder = embedder

    await store.preload_knowledge_embedder()

    embedder.encode_query.assert_called_once_with(
        "预热",
        store.settings.knowledge_query_prefix,
    )


@pytest.mark.asyncio
async def test_resolve_user_does_not_wait_for_index_refresh() -> None:
    client = AsyncMock()
    store = ElasticsearchStore(client, Settings())

    user = await store.resolve_user("测试用户")

    assert user.username == "测试用户"
    assert user.speaker_identity == "莱叔"
    assert user.model_provider == store.settings.llm_provider
    assert user.model == store.settings.llm_model
    assert user.provider_models == {
        store.settings.llm_provider: store.settings.llm_model
    }
    assert user.provider_web_search_configs[
        store.settings.llm_provider
    ].model_dump() == {
        "enabled": False,
        "forced": False,
        "max_tool_calls": 1,
        "result_limit": 3,
    }
    client.create.assert_awaited_once()
    assert client.create.await_args.kwargs["refresh"] is False


@pytest.mark.asyncio
async def test_update_user_speaker_identity_does_not_wait_for_refresh() -> None:
    client = AsyncMock()
    store = ElasticsearchStore(client, Settings())

    await store.update_user_speaker_identity("测试用户", "店长")

    update_call = client.update.await_args
    assert update_call.kwargs["doc"] == {"speaker_identity": "店长"}
    assert update_call.kwargs["refresh"] is False


@pytest.mark.asyncio
async def test_update_user_model_config_does_not_wait_for_refresh() -> None:
    client = AsyncMock()
    store = ElasticsearchStore(client, Settings())

    await store.update_user_model_config(
        "测试用户",
        "qwen",
        "qwen-max",
    )

    update_call = client.update.await_args
    assert update_call.kwargs["doc"] == {
        "model_provider": "qwen",
        "model": "qwen-max",
        "provider_models": {"qwen": "qwen-max"},
        "provider_temperatures": {"qwen": 0.8},
        "provider_web_search_configs": {
            "qwen": {
                "enabled": False,
                "forced": False,
                "max_tool_calls": 1,
                "result_limit": 3,
            }
        },
        "web_search_enabled": False,
        "web_search_forced": False,
        "web_search_max_tool_calls": 1,
        "web_search_result_limit": 3,
        "temperature": 0.8,
    }
    assert update_call.kwargs["refresh"] is False


@pytest.mark.asyncio
async def test_search_conversation_messages_returns_every_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = AsyncMock()
    client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_source": {
                        "conversation_id": "conversation-1",
                        "title": "第一次直播",
                    }
                },
                {
                    "_source": {
                        "conversation_id": "conversation-2",
                        "title": "第二次直播",
                    }
                },
            ]
        }
    }
    scan_kwargs = {}

    async def fake_scan(*_args, **kwargs):
        scan_kwargs.update(kwargs)
        for message_id, created_at in [
            ("message-2", "2026-08-01T03:00:00+00:00"),
            ("message-1", "2026-08-01T02:00:00+00:00"),
        ]:
            yield {
                "_source": {
                    "message_id": message_id,
                    "conversation_id": "conversation-2",
                    "role": "assistant",
                    "content": "Live Streaming Agent 记得这句话在这里出现过",
                    "created_at": created_at,
                }
            }

    monkeypatch.setattr(
        "backend.app.elasticsearch_store.async_scan",
        fake_scan,
    )
    store = ElasticsearchStore(client, Settings())

    results = await store.search_conversation_messages(
        "测试用户",
        "  这句话  ",
    )

    assert len(results) == 1
    assert results[0].conversation_id == "conversation-2"
    assert results[0].title == "第二次直播"
    assert results[0].match_count == 2
    assert [match.message_id for match in results[0].matches] == [
        "message-2",
        "message-1",
    ]
    assert results[0].matches[0].snippet == "Live Streaming Agent 记得这句话在这里出现过"
    conversation_search = client.search.await_args.kwargs
    assert {"term": {"status": "active"}} in conversation_search["query"][
        "bool"
    ]["filter"]
    message_query = scan_kwargs["query"]["query"]
    search_clause = message_query["bool"]["must"][0]["multi_match"]
    assert search_clause["query"] == "这句话"
    assert search_clause["type"] == "phrase"
    assert "content" in search_clause["fields"]
    assert "metadata.knowledge_injected_context" in search_clause["fields"]
    assert "metadata.web_search_sources.snippet" in search_clause["fields"]
    assert message_query["bool"]["must_not"] == [
        {"term": {"metadata.deleted": True}}
    ]


def test_conversation_search_matches_include_knowledge_and_web_sources() -> None:
    matches = ElasticsearchStore._conversation_search_matches(
        {
            "message_id": "message-1",
            "role": "assistant",
            "content": "正文里有目标词",
            "metadata": {
                "knowledge_injected_context": "知识库注入也有目标词",
                "web_search_sources": [
                    {
                        "title": "网页标题",
                        "snippet": "网页摘要包含目标词",
                        "url": "https://example.com/source",
                    },
                    {
                        "title": "无关网页",
                        "snippet": "没有命中",
                        "url": "https://example.com/other",
                    },
                ],
            },
            "created_at": "2026-08-04T06:00:00+00:00",
        },
        "目标词",
    )

    assert [match.source for match in matches] == [
        "message",
        "knowledge",
        "web_search",
    ]
    assert all("目标词" in match.snippet for match in matches)


@pytest.mark.asyncio
async def test_prompt_config_is_stored_in_elasticsearch_without_refresh() -> None:
    client = AsyncMock()
    store = ElasticsearchStore(client, Settings())
    document = {
        "active_version": "Live Streaming Agent.txt",
        "active_speaker_version": "v1.0",
        "speaker_versions": [],
    }

    await store.save_prompt_config_document("用户甲", document)

    index_call = client.index.await_args
    assert index_call.kwargs["index"] == "live_streaming_agent_prompt_configs"
    assert index_call.kwargs["id"] == user_document_id("用户甲")
    assert index_call.kwargs["document"] == {
        **document,
        "username": "用户甲",
    }
    assert index_call.kwargs["refresh"] is False


@pytest.mark.asyncio
async def test_prompt_config_is_loaded_from_elasticsearch() -> None:
    client = AsyncMock()
    client.get.return_value = {"_source": {"active_speaker_version": "__none__"}}
    store = ElasticsearchStore(client, Settings())

    document = await store.get_prompt_config_document("用户乙")

    assert document == {"active_speaker_version": "__none__"}
    client.get.assert_awaited_once_with(
        index="live_streaming_agent_prompt_configs",
        id=user_document_id("用户乙"),
    )


@pytest.mark.asyncio
async def test_short_term_memory_save_does_not_replace_character_count() -> None:
    client = AsyncMock()
    store = ElasticsearchStore(client, Settings())
    store.get_conversation = AsyncMock(
        return_value=ConversationRecord(
            conversation_id="conversation-1",
            username="测试用户",
            title="新对话 1",
            status="active",
            completed_rounds=20,
            effective_char_count=24,
            effective_memory_through_round=10,
            memory_through_round=10,
            short_term_memory="旧摘要",
            memory_status="compressing",
            memory_target_round=20,
            created_at="2026-07-27T06:00:00+00:00",
            updated_at="2026-07-27T06:00:00+00:00",
        )
    )
    memory = await store.create_short_term_memory(
        conversation_id="conversation-1",
        username="测试用户",
        compression_number=2,
        through_round=20,
        summary="新的压缩结果",
    )

    assert memory.through_round == 20
    assert client.index.await_args.kwargs["index"] == "live_streaming_agent_memories"
    update_call = client.update.await_args
    params = update_call.kwargs["script"]["params"]
    assert params["summary"] == "新的压缩结果"
    assert "character_delta" not in params
    assert "effective_char_count" not in update_call.kwargs["script"]["source"]
    assert "memory_status = 'idle'" in update_call.kwargs["script"]["source"]
    assert update_call.kwargs["refresh"] is False


@pytest.mark.asyncio
async def test_character_count_changes_only_when_memory_enters_context() -> None:
    client = AsyncMock()
    store = ElasticsearchStore(client, Settings())
    conversation = ConversationRecord(
        conversation_id="conversation-1",
        username="测试用户",
        title="新对话 1",
        status="active",
        completed_rounds=19,
        effective_char_count=100,
        effective_memory_through_round=0,
        memory_through_round=10,
        short_term_memory="前十轮摘要",
        created_at="2026-07-27T06:00:00+00:00",
        updated_at="2026-07-27T06:00:00+00:00",
    )
    active_messages = [
        MessageRecord(
            message_id="round-11",
            conversation_id="conversation-1",
            username="测试用户",
            role="assistant",
            content="第十一至十九轮原文",
            metadata={"round_number": 11},
            created_at="2026-07-27T06:00:00+00:00",
        )
    ]
    pending_message = MessageRecord(
        message_id="round-20-user",
        conversation_id="conversation-1",
        username="测试用户",
        role="user",
        content="第二十轮问题",
        metadata={"round_number": 20},
        created_at="2026-07-27T06:20:00+00:00",
    )
    store.list_messages_by_round_range = AsyncMock(
        return_value=active_messages
    )

    effective_count = (
        await store.activate_short_term_memory_character_count(
            conversation,
            through_round=10,
            summary="前十轮摘要",
            pending_message=pending_message,
        )
    )

    expected_count = sum(
        count_message_characters(message.content, message.metadata)
        for message in active_messages
    ) + count_message_characters(
        pending_message.content,
        pending_message.metadata,
    ) + count_text_characters("前十轮摘要")
    assert effective_count == expected_count
    update = client.update.await_args.kwargs["doc"]
    assert update["effective_char_count"] == expected_count
    assert update["effective_memory_through_round"] == 10
    assert update["effective_char_count_version"] == 5


@pytest.mark.asyncio
async def test_mark_memory_compressing_keeps_existing_summary_untouched() -> None:
    client = AsyncMock()
    client.update.return_value = {"result": "updated"}
    store = ElasticsearchStore(client, Settings())

    marked = await store.try_mark_memory_compressing("conversation-1", 20)

    assert marked is True
    script = client.update.await_args.kwargs["script"]
    assert script["params"] == {"target_round": 20}
    assert "short_term_memory" not in script["source"]


@pytest.mark.asyncio
async def test_mark_memory_compressing_is_idempotent() -> None:
    client = AsyncMock()
    client.update.return_value = {"result": "noop"}
    store = ElasticsearchStore(client, Settings())

    marked = await store.try_mark_memory_compressing("conversation-1", 10)

    assert marked is False


@pytest.mark.asyncio
async def test_create_conversation_does_not_wait_for_index_refresh() -> None:
    client = AsyncMock()
    store = ElasticsearchStore(client, Settings())

    conversation = await store.create_conversation("测试用户", "新对话")

    assert conversation.username == "测试用户"
    assert conversation.status == "active"
    client.index.assert_awaited_once()
    index_call = client.index.await_args
    assert index_call.kwargs["index"] == "live_streaming_agent_conversations"
    assert index_call.kwargs["refresh"] is False
    client.create.assert_not_awaited()
    client.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_reorder_conversations_persists_each_position_without_refresh() -> None:
    client = AsyncMock()
    store = ElasticsearchStore(client, Settings())
    store.list_conversations = AsyncMock(
        return_value=[
            ConversationRecord(
                conversation_id="conversation-1",
                username="测试用户",
                title="对话 1",
                status="active",
                created_at="2026-07-27T06:00:00+00:00",
                updated_at="2026-07-27T06:00:00+00:00",
            ),
            ConversationRecord(
                conversation_id="conversation-2",
                username="测试用户",
                title="对话 2",
                status="active",
                created_at="2026-07-27T06:01:00+00:00",
                updated_at="2026-07-27T06:01:00+00:00",
            ),
        ]
    )

    reordered = await store.reorder_conversations(
        "测试用户",
        ["conversation-2", "conversation-1"],
    )

    assert [item.conversation_id for item in reordered] == [
        "conversation-2",
        "conversation-1",
    ]
    assert [item.sort_order for item in reordered] == [0, 1]
    assert client.update.await_count == 2
    assert all(
        call.kwargs["refresh"] is False
        for call in client.update.await_args_list
    )


@pytest.mark.asyncio
async def test_rename_conversation_does_not_wait_for_index_refresh() -> None:
    client = AsyncMock()
    client.get.return_value = {
        "_source": {
            "conversation_id": "conversation-1",
            "username": "测试用户",
                "title": "新对话 1",
                "status": "active",
                "completed_rounds": 2,
                "effective_char_count": 0,
                "effective_char_count_version": 5,
                "effective_memory_through_round": 0,
                "created_at": "2026-07-27T06:00:00+00:00",
            "updated_at": "2026-07-27T06:00:00+00:00",
        }
    }
    store = ElasticsearchStore(client, Settings())

    conversation = await store.update_conversation(
        "conversation-1",
        ConversationUpdate(title="改名后的对话"),
    )

    assert conversation.title == "改名后的对话"
    update_call = client.update.await_args
    assert update_call.kwargs["doc"]["title"] == "改名后的对话"
    assert update_call.kwargs["refresh"] is False


@pytest.mark.asyncio
async def test_create_message_does_not_wait_for_index_refresh() -> None:
    client = AsyncMock()
    client.get.return_value = {
        "_source": {
            "conversation_id": "conversation-1",
            "username": "测试用户",
                "title": "新对话 1",
                "status": "active",
                "completed_rounds": 2,
                "effective_char_count": 0,
                "effective_char_count_version": 5,
                "effective_memory_through_round": 0,
                "created_at": "2026-07-27T06:00:00+00:00",
            "updated_at": "2026-07-27T06:00:00+00:00",
        }
    }
    store = ElasticsearchStore(client, Settings())

    await store.create_message(
        "conversation-1",
        MessageCreate(
            username="测试用户",
            content="你好",
            reasoning_content="internal reasoning",
            metadata={"knowledge_injected_context": "知识 内容"},
        ),
    )

    assert client.index.await_args.kwargs["refresh"] is False
    assert (
        client.index.await_args.kwargs["document"]["reasoning_content"]
        == "internal reasoning"
    )
    assert client.update.await_args.kwargs["refresh"] is False
    assert (
        client.update.await_args.kwargs["script"]["params"]["char_count"]
        == 6
    )


@pytest.mark.asyncio
async def test_discard_uncompleted_message_hides_it_and_restores_count() -> None:
    client = AsyncMock()
    store = ElasticsearchStore(client, Settings())
    now = datetime.now(UTC)
    conversation = ConversationRecord(
        conversation_id="conversation-1",
        username="测试用户",
        title="测试对话",
        status="active",
        completed_rounds=3,
        effective_char_count=42,
        effective_char_count_version=5,
        effective_memory_through_round=0,
        created_at=now,
        updated_at=now,
    )
    message = MessageRecord(
        message_id="user-pending",
        conversation_id=conversation.conversation_id,
        username=conversation.username,
        role="user",
        content="失败输入",
        metadata={"round_number": 4},
        created_at=now,
    )

    await store.discard_uncompleted_message(conversation, message)

    message_update, conversation_update = client.update.await_args_list
    assert message_update.kwargs["id"] == "user-pending"
    assert message_update.kwargs["doc"]["metadata"]["deleted"] is True
    assert (
        message_update.kwargs["doc"]["metadata"]["deleted_reason"]
        == "incomplete_turn"
    )
    assert conversation_update.kwargs["id"] == "conversation-1"
    assert conversation_update.kwargs["doc"]["effective_char_count"] == 42
    assert (
        conversation_update.kwargs["doc"]["effective_memory_through_round"]
        == 0
    )


def test_character_count_ignores_whitespace() -> None:
    assert count_text_characters("身份：“你好 世界”\n") == 9


def test_message_character_count_includes_injected_knowledge() -> None:
    assert count_message_characters(
        "回答",
        {"knowledge_injected_context": "知识 内容"},
    ) == 6


def test_message_character_count_ignores_web_search_sources() -> None:
    assert count_message_characters(
        "回答",
        {
            "web_search_sources": [
                {
                    "title": "来源 标题",
                    "snippet": "网页 摘要",
                    "url": "https://example.com/not-counted",
                },
                {
                    "title": "第二条",
                    "snippet": "",
                    "url": "https://example.com/second",
                },
            ]
        },
    ) == 2


@pytest.mark.asyncio
async def test_legacy_character_count_is_rebuilt_with_injected_knowledge(
    monkeypatch,
) -> None:
    client = AsyncMock()
    client.get.return_value = {
        "_source": {
            "conversation_id": "conversation-1",
            "username": "测试用户",
            "title": "旧对话",
            "status": "active",
            "effective_char_count": 4,
            "created_at": "2026-07-27T06:00:00+00:00",
            "updated_at": "2026-07-27T06:00:00+00:00",
        }
    }

    async def fake_scan(*_args, **_kwargs):
        yield {
            "_source": {
                "content": "回答",
                "metadata": {
                    "knowledge_injected_context": "知识 内容",
                    "web_search_sources": [
                        {
                            "title": "不计入字数的来源",
                            "snippet": "不计入字数的网页摘要",
                        }
                    ],
                },
            }
        }

    monkeypatch.setattr(
        "backend.app.elasticsearch_store.async_scan",
        fake_scan,
    )
    store = ElasticsearchStore(client, Settings())

    conversation = await store.get_conversation("conversation-1")

    assert conversation.effective_char_count == 6
    assert conversation.effective_char_count_version == 5
    assert conversation.effective_memory_through_round == 0
    assert client.update.await_args.kwargs["doc"] == {
        "effective_char_count": 6,
        "effective_char_count_version": 5,
        "effective_memory_through_round": 0,
    }


@pytest.mark.asyncio
async def test_ensure_indices_only_adds_missing_mapping_fields() -> None:
    client = AsyncMock()
    client.indices.exists.return_value = True
    store = ElasticsearchStore(client, Settings())

    mappings_by_index = {
        store.settings.users_index: INDEX_MAPPINGS["users"],
        store.settings.conversations_index: INDEX_MAPPINGS["conversations"],
        store.settings.messages_index: INDEX_MAPPINGS["messages"],
        store.settings.memories_index: INDEX_MAPPINGS["memories"],
        store.settings.prompt_configs_index: INDEX_MAPPINGS["prompt_configs"],
    }

    async def get_mapping(*, index: str):
        properties = dict(mappings_by_index[index]["properties"])
        if index == store.settings.users_index:
            properties["speaker_identity"] = {"type": "text"}
        if index == store.settings.conversations_index:
            properties.pop("effective_char_count")
            properties.pop("effective_char_count_version")
            properties.pop("effective_memory_through_round")
        return {index: {"mappings": {"properties": properties}}}

    client.indices.get_mapping.side_effect = get_mapping

    await store.ensure_indices()

    client.indices.put_mapping.assert_awaited_once_with(
        index=store.settings.conversations_index,
        properties={
            "effective_char_count": {"type": "integer"},
            "effective_char_count_version": {"type": "integer"},
            "effective_memory_through_round": {"type": "integer"},
        },
    )


@pytest.mark.asyncio
async def test_list_messages_excludes_rewound_messages() -> None:
    client = AsyncMock()
    client.search.return_value = {"hits": {"hits": []}}
    store = ElasticsearchStore(client, Settings())

    await store.list_messages("conversation-1")

    search_call = client.search.await_args
    assert {"term": {"metadata.deleted": True}} in search_call.kwargs["query"][
        "bool"
    ]["must_not"]


@pytest.mark.asyncio
async def test_user_performance_keeps_rewound_messages() -> None:
    client = AsyncMock()
    client.search.return_value = {"hits": {"hits": []}}
    store = ElasticsearchStore(client, Settings())

    await store.list_user_performance_messages("测试用户", size=20)

    search_call = client.search.await_args
    query = search_call.kwargs["query"]["bool"]
    assert {"term": {"username": "测试用户"}} in query["filter"]
    assert {"term": {"role": "assistant"}} in query["filter"]
    assert {
        "exists": {
            "field": "metadata.performance_metrics.model_first_token_ms"
        }
    } in query["filter"]
    assert "must_not" not in query
    assert search_call.kwargs["size"] == 20


@pytest.mark.asyncio
async def test_user_performance_can_filter_by_created_at_range() -> None:
    client = AsyncMock()
    client.search.return_value = {"hits": {"hits": []}}
    store = ElasticsearchStore(client, Settings())
    day_start = datetime(2026, 7, 29, 16, tzinfo=UTC)
    day_end = datetime(2026, 7, 30, 16, tzinfo=UTC)

    await store.list_user_performance_messages(
        "测试用户",
        size=500,
        created_at_gte=day_start,
        created_at_lt=day_end,
    )

    search_call = client.search.await_args
    filters = search_call.kwargs["query"]["bool"]["filter"]
    assert {
        "range": {
            "created_at": {
                "gte": day_start.isoformat(),
                "lt": day_end.isoformat(),
            }
        }
    } in filters
    assert search_call.kwargs["size"] == 500


@pytest.mark.asyncio
async def test_search_knowledge_merges_keyword_and_semantic_hits() -> None:
    client = AsyncMock()
    client.indices.exists.return_value = True
    client.indices.analyze.return_value = {
        "tokens": [
            {"token": "莱叔"},
            {"token": "你"},
            {"token": "知道"},
            {"token": "小心心"},
            {"token": "吗"},
        ]
    }
    client.search.side_effect = [
        {
            "hits": {
                "hits": [
                    {
                        "_id": "gift-1",
                        "_source": {"keywords": ["小心心", "礼物"]},
                    }
                ]
            }
        },
        {
            "hits": {
                "hits": [
                    {
                        "_score": 0.81,
                        "_source": {
                            "knowledge_id": "gift-1",
                            "description": "小心心是一种直播礼物",
                        },
                    },
                    {
                        "_score": 0.70,
                        "_source": {
                            "knowledge_id": "low-score",
                            "description": "低相关内容",
                        },
                    },
                ]
            }
        },
        {
            "hits": {
                "hits": [
                    {
                        "_id": "gift-1",
                        "_source": {
                            "title": "小心心",
                            "keywords": ["小心心", "礼物"],
                            "body": "1 个小心心等于 1 抖币",
                        },
                    }
                ]
            }
        },
    ]
    store = ElasticsearchStore(client, Settings())
    embedder = MagicMock()
    embedder.encode_query.return_value = [0.1, 0.2]
    store._knowledge_embedder = embedder

    results = await store.search_knowledge(
        "莱叔：“你知道小心心吗”",
        size=3,
    )

    assert len(results) == 1
    assert results[0].document_id == "gift-1"
    assert results[0].source["retrieval_reasons"] == [
        "keyword",
        "semantic",
    ]
    assert results[0].source["matched_keywords"] == ["小心心"]
    keyword_call = client.search.await_args_list[0]
    keyword_query = keyword_call.kwargs["query"]["bool"]
    assert keyword_query["should"][0] == {
        "terms": {
            "keyword_terms": ["你", "吗", "小心心", "知道", "莱叔"],
            "boost": 100.0,
        }
    }
    assert keyword_query["should"][1] == {
        "match": {
            "keyword_text": {
                "query": "莱叔：“你知道小心心吗”",
                "analyzer": "ik_max_word",
            }
        }
    }
    assert keyword_query["minimum_should_match"] == 1
    semantic_call = client.search.await_args_list[1]
    assert semantic_call.kwargs["index"] == "vtuber_knowledge_vectors"
    assert semantic_call.kwargs["knn"]["k"] == 15
    embedder.encode_query.assert_called_once_with(
        "莱叔：“你知道小心心吗”",
        store.settings.knowledge_query_prefix,
    )


@pytest.mark.asyncio
async def test_rewind_last_turn_soft_deletes_latest_question_and_answer() -> None:
    client = AsyncMock()
    client.get.return_value = {
        "_source": {
            "conversation_id": "conversation-1",
            "username": "测试用户",
                "title": "新对话 1",
                "status": "active",
                "completed_rounds": 2,
                "effective_char_count": 0,
                "effective_char_count_version": 5,
                "effective_memory_through_round": 0,
                "created_at": "2026-07-27T06:00:00+00:00",
            "updated_at": "2026-07-27T06:02:00+00:00",
        }
    }
    client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_id": "assistant-document-2",
                    "_source": {
                        "message_id": "assistant-2",
                        "role": "assistant",
                        "content": "好的",
                        "metadata": {
                            "model": "deepseek-chat",
                            "knowledge_injected_context": "知识内容",
                        },
                    },
                },
                {
                    "_id": "user-document-2",
                    "_source": {
                        "message_id": "user-2",
                        "role": "user",
                        "content": "你好",
                        "metadata": {},
                    },
                },
                {
                    "_id": "assistant-document-1",
                    "_source": {
                        "message_id": "assistant-1",
                        "role": "assistant",
                        "metadata": {},
                    },
                },
                {
                    "_id": "user-document-1",
                    "_source": {
                        "message_id": "user-1",
                        "role": "user",
                        "metadata": {},
                    },
                },
            ]
        }
    }
    store = ElasticsearchStore(client, Settings())

    message_ids = await store.rewind_last_turn(
        "conversation-1",
        "测试用户",
    )

    assert message_ids == ["assistant-2", "user-2"]
    client.update.assert_awaited_once()
    updated_conversation = client.update.await_args.kwargs["doc"]
    assert updated_conversation["completed_rounds"] == 1
    assert updated_conversation["effective_char_count"] == 0
    client.bulk.assert_awaited_once()
    bulk_call = client.bulk.await_args
    assert bulk_call.kwargs["refresh"] is False
    operations = bulk_call.kwargs["operations"]
    assert operations[0]["update"]["_id"] == "assistant-document-2"
    assert operations[1]["doc"]["metadata"]["model"] == "deepseek-chat"
    assert operations[1]["doc"]["metadata"]["deleted"] is True
    assert operations[2]["update"]["_id"] == "user-document-2"
    assert operations[3]["doc"]["metadata"]["deleted"] is True


@pytest.mark.asyncio
async def test_rewind_uses_realtime_last_turn_ids_without_search_refresh() -> None:
    client = AsyncMock()
    client.mget.return_value = {
        "docs": [
            {
                "_id": "user-document-2",
                "found": True,
                "_source": {
                    "message_id": "user-2",
                    "role": "user",
                    "content": "你好",
                    "metadata": {"round_number": 2},
                    "created_at": "2026-07-27T06:02:00+00:00",
                },
            },
            {
                "_id": "assistant-document-2",
                "found": True,
                "_source": {
                    "message_id": "assistant-2",
                    "role": "assistant",
                    "content": "你好呀",
                    "metadata": {"round_number": 2},
                    "created_at": "2026-07-27T06:02:01+00:00",
                },
            },
        ]
    }
    store = ElasticsearchStore(client, Settings())
    store.get_conversation = AsyncMock(
        return_value=ConversationRecord(
            conversation_id="conversation-1",
            username="测试用户",
            title="新对话 1",
            status="active",
            completed_rounds=2,
            last_turn_message_ids=["user-2", "assistant-2"],
            effective_char_count=8,
            created_at="2026-07-27T06:00:00+00:00",
            updated_at="2026-07-27T06:02:01+00:00",
        )
    )

    message_ids = await store.rewind_last_turn(
        "conversation-1",
        "测试用户",
    )

    assert message_ids == ["assistant-2", "user-2"]
    client.mget.assert_awaited_once_with(
        index="live_streaming_agent_messages",
        ids=["user-2", "assistant-2"],
    )
    client.search.assert_not_awaited()
    conversation_update = client.update.await_args.kwargs["doc"]
    assert conversation_update["completed_rounds"] == 1
    assert conversation_update["last_turn_message_ids"] == []


@pytest.mark.asyncio
async def test_rewind_triggering_round_cancels_inflight_compression() -> None:
    client = AsyncMock()
    client.mget.return_value = {
        "docs": [
            {
                "_id": "user-document-20",
                "found": True,
                "_source": {
                    "message_id": "user-20",
                    "role": "user",
                    "content": "问题",
                    "metadata": {"round_number": 20},
                    "created_at": "2026-07-27T06:20:00+00:00",
                },
            },
            {
                "_id": "assistant-document-20",
                "found": True,
                "_source": {
                    "message_id": "assistant-20",
                    "role": "assistant",
                    "content": "回复",
                    "metadata": {"round_number": 20},
                    "created_at": "2026-07-27T06:20:01+00:00",
                },
            },
        ]
    }
    store = ElasticsearchStore(client, Settings())
    store.get_conversation = AsyncMock(
        return_value=ConversationRecord(
            conversation_id="conversation-1",
            username="测试用户",
            title="新对话 1",
            status="active",
            completed_rounds=20,
            last_turn_message_ids=["user-20", "assistant-20"],
            effective_char_count=8,
            memory_status="compressing",
            memory_target_round=20,
            created_at="2026-07-27T06:00:00+00:00",
            updated_at="2026-07-27T06:20:01+00:00",
        )
    )

    await store.rewind_last_turn("conversation-1", "测试用户")

    updated = client.update.await_args.kwargs["doc"]
    assert updated["completed_rounds"] == 19
    assert updated["memory_status"] == "idle"
    assert updated["memory_target_round"] == 0


@pytest.mark.asyncio
async def test_memory_history_deduplicates_same_compressed_round() -> None:
    client = AsyncMock()
    store = ElasticsearchStore(client, Settings())
    store.get_conversation = AsyncMock(
        return_value=ConversationRecord(
            conversation_id="conversation-1",
            username="测试用户",
            title="新对话 1",
            status="active",
            completed_rounds=21,
            memory_compression_count=1,
            memory_through_round=10,
            short_term_memory="新摘要",
            created_at="2026-07-27T06:00:00+00:00",
            updated_at="2026-07-27T06:00:00+00:00",
        )
    )
    client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_source": {
                        "memory_id": "new",
                        "conversation_id": "conversation-1",
                        "username": "测试用户",
                        "compression_number": 1,
                        "through_round": 10,
                        "summary": "新摘要",
                        "created_at": "2026-07-27T06:02:00+00:00",
                    }
                },
                {
                    "_source": {
                        "memory_id": "old",
                        "conversation_id": "conversation-1",
                        "username": "测试用户",
                        "compression_number": 1,
                        "through_round": 10,
                        "summary": "旧摘要",
                        "created_at": "2026-07-27T06:01:00+00:00",
                    }
                },
            ]
        }
    }

    memories = await store.list_short_term_memories("conversation-1")

    assert [memory.memory_id for memory in memories] == ["new"]


@pytest.mark.asyncio
async def test_rewind_triggering_round_removes_first_compression() -> None:
    client = AsyncMock()
    store = ElasticsearchStore(client, Settings())
    store.get_conversation = AsyncMock(
        return_value=ConversationRecord(
            conversation_id="conversation-1",
            username="测试用户",
            title="新对话 1",
            status="active",
            completed_rounds=10,
            effective_char_count=8,
            memory_compression_count=1,
            memory_through_round=10,
            short_term_memory="压缩摘要",
            memory_target_round=10,
            created_at="2026-07-27T06:00:00+00:00",
            updated_at="2026-07-27T06:00:00+00:00",
        )
    )
    client.search.side_effect = [
        {
            "hits": {
                "hits": [
                    {
                        "_id": "assistant-10",
                        "_source": {
                            "message_id": "assistant-10",
                            "role": "assistant",
                            "content": "回复",
                            "metadata": {"round_number": 10},
                        },
                    },
                    {
                        "_id": "user-10",
                        "_source": {
                            "message_id": "user-10",
                            "role": "user",
                            "content": "问题",
                            "metadata": {"round_number": 10},
                        },
                    },
                ]
            }
        },
        {"hits": {"hits": []}},
    ]
    store.list_messages_by_round_range = AsyncMock(
        return_value=[
            MessageRecord(
                message_id="user-1",
                conversation_id="conversation-1",
                username="测试用户",
                role="user",
                content="原始 问题",
                metadata={"round_number": 1},
                created_at="2026-07-27T06:00:00+00:00",
            ),
            MessageRecord(
                message_id="assistant-1",
                conversation_id="conversation-1",
                username="测试用户",
                role="assistant",
                content="原始回复",
                metadata={"round_number": 1},
                created_at="2026-07-27T06:00:01+00:00",
            ),
        ]
    )

    await store.rewind_last_turn("conversation-1", "测试用户")

    updated = client.update.await_args.kwargs["doc"]
    assert updated["completed_rounds"] == 9
    assert updated["memory_compression_count"] == 0
    assert updated["memory_through_round"] == 0
    assert updated["short_term_memory"] == ""
    assert updated["memory_status"] == "idle"
    assert updated["effective_char_count"] == 4
    assert updated["effective_memory_through_round"] == 0


@pytest.mark.asyncio
async def test_rewind_later_triggering_round_restores_previous_summary() -> None:
    client = AsyncMock()
    store = ElasticsearchStore(client, Settings())
    store.get_conversation = AsyncMock(
        return_value=ConversationRecord(
            conversation_id="conversation-1",
            username="测试用户",
            title="新对话 1",
            status="active",
            completed_rounds=20,
            effective_char_count=20,
            effective_memory_through_round=10,
            memory_compression_count=2,
            memory_through_round=20,
            short_term_memory="截至二十轮",
            memory_target_round=20,
            created_at="2026-07-27T06:00:00+00:00",
            updated_at="2026-07-27T06:00:00+00:00",
        )
    )
    client.search.side_effect = [
        {
            "hits": {
                "hits": [
                    {
                        "_id": "assistant-20",
                        "_source": {
                            "message_id": "assistant-20",
                            "role": "assistant",
                            "content": "回复",
                            "metadata": {"round_number": 20},
                        },
                    },
                    {
                        "_id": "user-20",
                        "_source": {
                            "message_id": "user-20",
                            "role": "user",
                            "content": "问题",
                            "metadata": {"round_number": 20},
                        },
                    },
                ]
            }
        },
        {
            "hits": {
                "hits": [
                    {
                        "_source": ShortTermMemoryRecord(
                            memory_id="memory-10",
                            conversation_id="conversation-1",
                            username="测试用户",
                            compression_number=1,
                            through_round=10,
                            summary="截至十轮",
                            created_at="2026-07-27T06:00:00+00:00",
                        ).model_dump(mode="json")
                    }
                ]
            }
        },
        {"hits": {"hits": []}},
    ]
    store.list_messages_by_round_range = AsyncMock(
        return_value=[
            MessageRecord(
                message_id="active-raw-messages",
                conversation_id="conversation-1",
                username="测试用户",
                role="user",
                content="前十九轮原文",
                metadata={"round_number": 1},
                created_at="2026-07-27T06:00:00+00:00",
            )
        ]
    )

    await store.rewind_last_turn("conversation-1", "测试用户")

    updated = client.update.await_args.kwargs["doc"]
    assert updated["completed_rounds"] == 19
    assert updated["memory_compression_count"] == 1
    assert updated["memory_through_round"] == 10
    assert updated["short_term_memory"] == "截至十轮"
    assert updated["effective_char_count"] == 6
    assert updated["effective_memory_through_round"] == 0


@pytest.mark.asyncio
async def test_stale_compression_result_is_discarded_after_rewind() -> None:
    client = AsyncMock()
    store = ElasticsearchStore(client, Settings())
    store.get_conversation = AsyncMock(
        return_value=ConversationRecord(
            conversation_id="conversation-1",
            username="测试用户",
            title="新对话 1",
            status="active",
            completed_rounds=9,
            memory_status="idle",
            created_at="2026-07-27T06:00:00+00:00",
            updated_at="2026-07-27T06:00:00+00:00",
        )
    )

    memory = await store.create_short_term_memory(
        conversation_id="conversation-1",
        username="测试用户",
        compression_number=1,
        through_round=10,
        summary="过期总结",
    )

    assert memory is None
    client.index.assert_not_awaited()
