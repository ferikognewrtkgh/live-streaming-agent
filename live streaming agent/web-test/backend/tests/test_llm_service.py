import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import backend.app.llm_service as llm_module
import backend.app.main as main_module
import httpx
import pytest
from backend.app.config import Settings
from backend.app.llm_service import (
    LLMService,
    format_knowledge_context,
)
from backend.app.main import chat
from backend.app.schemas import (
    ChatRequest,
    ConversationRecord,
    MessageCreate,
    MessagePage,
    MessageRecord,
    ModelConfig,
    RuntimeModelConfig,
)
from fastapi import HTTPException


@pytest.mark.asyncio
async def test_health_reports_database_unavailable_without_retries() -> None:
    health_client = SimpleNamespace(ping=AsyncMock(return_value=False))
    client = MagicMock()
    client.options.return_value = health_client

    result = await main_module.health(SimpleNamespace(client=client))

    client.options.assert_called_once_with(request_timeout=1, max_retries=0)
    assert result == {
        "status": "degraded",
        "elasticsearch": "unavailable",
    }


def test_vtuber_knowledge_item_is_formatted_for_model_context() -> None:
    formatted = format_knowledge_context(
        [
            {
                "title": "抖音-小额礼物-小心心",
                "category": "平台相关",
                "keywords": ["小心心", "礼物"],
                "matched_keywords": ["小心心"],
                "body": "1 个小心心等于 1 抖币",
                "usage": "相关时自然引用",
            }
        ]
    )

    assert formatted.startswith("[知识库检索结果]")
    assert "标题：抖音-小额礼物-小心心" in formatted
    assert "内容：1 个小心心等于 1 抖币" in formatted
    assert "命中关键词：小心心" in formatted
    assert "用法：相关时自然引用" in formatted


def test_knowledge_context_does_not_truncate_long_content() -> None:
    long_body = "正文" * 1_500
    long_usage = "用法" * 300

    formatted = format_knowledge_context(
        [
            {
                "title": "长知识",
                "body": long_body,
                "usage": long_usage,
            }
        ]
    )

    assert long_body in formatted
    assert long_usage in formatted
    assert len(formatted) > 2_400


@pytest.mark.asyncio
async def test_missing_api_key_returns_configuration_error() -> None:
    service = LLMService(Settings(llm_api_key=None))
    with pytest.raises(HTTPException) as raised:
        await service.generate_reply(
            conversation_id="conversation-1",
            username="test-user",
            user_content="你好",
            history=[],
            system_prompt="你是 Live Streaming Agent",
            knowledge_context=[],
            model_config=ModelConfig(),
        )

    assert raised.value.status_code == 424
    assert "未配置 API Key" in str(raised.value.detail)


@pytest.mark.asyncio
async def test_model_reply_is_forwarded_as_real_stream_chunks(monkeypatch) -> None:
    captured_request: dict = {}
    logged_events: list[dict] = []
    latency_events: list[dict] = []
    monkeypatch.setattr(llm_module, "log_model_event", logged_events.append)
    monkeypatch.setattr(
        llm_module,
        "log_chat_latency_event",
        latency_events.append,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request.update(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=(
                'data: {"choices":[{"delta":{"reasoning_content":"internal "}}]}\n\n'
                'data: {"choices":[{"delta":{"reasoning_content":"reasoning"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"你"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"好"}}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    service = LLMService(
        Settings(llm_api_key="test-key"),
        transport=httpx.MockTransport(handler),
    )

    replies = [
        chunk
        async for chunk in service.stream_reply(
            conversation_id="conversation-1",
            username="test-user",
            user_content="你好",
            history=[],
            system_prompt="你是 Live Streaming Agent",
            knowledge_context=[],
            model_config=ModelConfig(),
            trace_id="trace-12345678",
        )
    ]
    chunks = [reply.content for reply in replies if reply.content]
    reasoning_chunks = [
        reply.reasoning_content for reply in replies if reply.reasoning_content
    ]

    assert captured_request["stream"] is True
    assert captured_request["thinking"] == {"type": "disabled"}
    assert chunks == ["你", "好"]
    assert reasoning_chunks == ["internal ", "reasoning"]
    first_chunk_event = next(
        event for event in logged_events if event["event"] == "model_first_chunk"
    )
    completed_event = next(
        event for event in logged_events if event["event"] == "model_call_completed"
    )
    assert first_chunk_event["ttft_ms"] >= 0
    assert completed_event["ttft_ms"] == first_chunk_event["ttft_ms"]
    assert completed_event["response"]["reasoning_content"] == (
        "internal reasoning"
    )
    assert completed_event["response"]["content_chunk_count"] == 2
    assert [
        event["event"] for event in latency_events
    ] == [
        "model_request_started",
        "model_response_headers_received",
        "model_first_chunk_received",
        "model_stream_completed",
    ]
    assert all(event["trace_id"] == "trace-12345678" for event in latency_events)


@pytest.mark.asyncio
async def test_doubao_web_search_forwards_responses_sse_deltas(
    monkeypatch,
) -> None:
    captured_request: dict = {}
    logged_events: list[dict] = []
    monkeypatch.setattr(llm_module, "log_model_event", logged_events.append)

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request.update(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=(
                'event: response.created\n'
                'data: {"type":"response.created","response":{"id":"r-1"}}\n\n'
                'event: response.web_search_call.in_progress\n'
                'data: {"type":"response.web_search_call.in_progress"}\n\n'
                'event: response.web_search_call.searching\n'
                'data: {"type":"response.web_search_call.searching"}\n\n'
                'event: response.web_search_call.completed\n'
                'data: {"type":"response.web_search_call.completed"}\n\n'
                'event: response.output_text.delta\n'
                'data: {"type":"response.output_text.delta","delta":"今"}\n\n'
                'event: response.output_text.annotation.added\n'
                'data: {"type":"response.output_text.annotation.added",'
                '"annotation":{"type":"url_citation","title":"示例来源",'
                '"url":"https://example.com/news","summary":"来源摘要"}}\n\n'
                'event: response.output_text.delta\n'
                'data: {"type":"response.output_text.delta","delta":"天"}\n\n'
                'event: response.completed\n'
                'data: {"type":"response.completed","response":{"output":['
                '{"type":"message","content":[{"type":"output_text",'
                '"text":"今天","annotations":[{"type":"url_citation",'
                '"title":"示例来源","url":"https://example.com/news",'
                '"summary":"来源摘要"}]}]}]}}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    service = LLMService(
        Settings(llm_api_key="test-key"),
        transport=httpx.MockTransport(handler),
    )
    replies = [
        chunk
        async for chunk in service.stream_reply(
            conversation_id="conversation-1",
            username="test-user",
            user_content="今天有什么新闻",
            history=[],
            system_prompt="你是 Live Streaming Agent",
            knowledge_context=[],
            model_config=RuntimeModelConfig(
                provider="doubao",
                model="doubao-seed-2-0-lite-260215",
                api_base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key="test-key",
                api_style="responses",
                web_search_enabled=True,
            ),
        )
    ]

    assert captured_request["stream"] is True
    assert "tool_choice" not in captured_request
    assert [reply.content for reply in replies if reply.content] == ["今", "天"]
    source_replies = [reply for reply in replies if reply.web_sources]
    timing_replies = [
        reply
        for reply in replies
        if reply.web_search_duration_ms is not None
    ]
    assert len(timing_replies) == 1
    assert timing_replies[0].web_search_duration_ms >= 0
    first_content_index = next(
        index for index, reply in enumerate(replies) if reply.content
    )
    assert replies[first_content_index - 1].web_search_duration_ms is not None
    assert len(source_replies) == 1
    assert list(source_replies[0].web_sources) == [
        {
            "title": "示例来源",
            "url": "https://example.com/news",
            "snippet": "来源摘要",
        }
    ]
    diagnostic = next(
        event
        for event in logged_events
        if event["event"] == "responses_stream_diagnostic"
    )
    assert diagnostic["http_status"] == 200
    assert diagnostic["content_type"] == "text/event-stream"
    assert diagnostic["event_counts"] == {
        "response.created": 1,
        "response.web_search_call.in_progress": 1,
        "response.web_search_call.searching": 1,
        "response.web_search_call.completed": 1,
        "response.output_text.delta": 2,
        "response.output_text.annotation.added": 1,
        "response.completed": 1,
    }
    assert diagnostic["terminal_event_type"] == "response.completed"
    assert diagnostic["terminal_payload"]["response"]["output"][0]["type"] == (
        "message"
    )
    assert diagnostic["non_sse_lines"] == []


@pytest.mark.asyncio
async def test_doubao_web_search_without_answer_returns_specific_error(
    monkeypatch,
) -> None:
    logged_events: list[dict] = []
    monkeypatch.setattr(llm_module, "log_model_event", logged_events.append)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=(
                'event: response.completed\n'
                'data: {"type":"response.completed","response":{"output":['
                '{"type":"web_search_call","status":"completed"},'
                '{"type":"message","content":[{"type":"output_text",'
                '"text":"","annotations":['
                '{"type":"url_citation","title":"来源一",'
                '"url":"https://example.com/one"},'
                '{"type":"url_citation","title":"来源二",'
                '"url":"https://example.com/two"},'
                '{"type":"url_citation","title":"来源三",'
                '"url":"https://example.com/three"}]}]}]}}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    service = LLMService(
        Settings(llm_api_key="test-key"),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(HTTPException) as raised:
        _ = [
            chunk
            async for chunk in service.stream_reply(
                conversation_id="conversation-1",
                username="test-user",
                user_content="搜索后回答",
                history=[],
                system_prompt="你是 Live Streaming Agent",
                knowledge_context=[],
                model_config=RuntimeModelConfig(
                    provider="doubao",
                    model="doubao-seed-evolving",
                    api_base_url="https://ark.cn-beijing.volces.com/api/v3",
                    api_key="test-key",
                    api_style="responses",
                    web_search_enabled=True,
                    web_search_forced=True,
                ),
            )
        ]

    expected_detail = (
        "豆包联网搜索已完成并返回 3 个来源，但模型没有生成回复正文。"
        "请重试，或关闭强制搜索后再试。"
    )
    assert raised.value.status_code == 502
    assert raised.value.detail == expected_detail
    failed_event = next(
        event for event in logged_events if event["event"] == "model_call_failed"
    )
    assert failed_event["error"] == expected_detail


@pytest.mark.asyncio
async def test_qwen_web_search_uses_auto_responses_tool_and_returns_sources() -> None:
    captured_request: dict = {}
    captured_url = ""
    captured_sse_header = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_url, captured_sse_header
        captured_request.update(json.loads(request.content))
        captured_url = str(request.url)
        captured_sse_header = request.headers.get("X-DashScope-SSE", "")
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=(
                'event: response.output_item.added\n'
                'data: {"type":"response.output_item.added","item":'
                '{"type":"web_search_call","action":{"sources":['
                '{"type":"url","url":"https://example.com/qwen"}]}}}\n\n'
                'event: response.output_text.delta\n'
                'data: {"type":"response.output_text.delta","delta":"最"}\n\n'
                'event: response.output_text.delta\n'
                'data: {"type":"response.output_text.delta","delta":"新"}\n\n'
                'event: response.completed\n'
                'data: {"type":"response.completed","response":{"output":['
                '{"type":"web_search_call","action":{"sources":['
                '{"type":"url","url":"https://example.com/qwen"}]}},'
                '{"type":"message","content":[{"type":"output_text",'
                '"text":"最新"}]}]}}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    service = LLMService(
        Settings(llm_api_key="test-key"),
        transport=httpx.MockTransport(handler),
    )
    replies = [
        chunk
        async for chunk in service.stream_reply(
            conversation_id="conversation-1",
            username="test-user",
            user_content="最新新闻",
            history=[],
            system_prompt="你是 Live Streaming Agent",
            knowledge_context=[],
            model_config=RuntimeModelConfig(
                provider="qwen",
                model="qwen3.7-max",
                api_base_url=(
                    "https://dashscope.aliyuncs.com/compatible-mode/v1"
                ),
                api_key="test-key",
                api_style="chat_completions",
                web_search_enabled=True,
                temperature=0.35,
            ),
        )
    ]

    assert captured_url == (
        "https://dashscope.aliyuncs.com/compatible-mode/v1/responses"
    )
    assert captured_sse_header == ""
    assert captured_request["input"][-1]["content"] == "最新新闻"
    assert captured_request["enable_thinking"] is False
    assert captured_request["temperature"] == 0.35
    assert captured_request["tools"] == [{"type": "web_search"}]
    assert "tool_choice" not in captured_request
    assert [reply.content for reply in replies if reply.content] == ["最", "新"]
    source_reply = next(reply for reply in replies if reply.web_sources)
    assert list(source_reply.web_sources) == [
        {
            "title": "https://example.com/qwen",
            "url": "https://example.com/qwen",
            "snippet": "",
        }
    ]


@pytest.mark.asyncio
async def test_qwen_without_search_still_uses_native_dashscope() -> None:
    captured_request: dict = {}
    captured_url = ""
    captured_sse_header = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_url, captured_sse_header
        captured_request.update(json.loads(request.content))
        captured_url = str(request.url)
        captured_sse_header = request.headers.get("X-DashScope-SSE", "")
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=(
                'data: {"output":{"choices":[{"message":'
                '{"content":"连"}}],"finish_reason":"null"}}\n\n'
                'data: {"output":{"choices":[{"message":'
                '{"content":"接"}}],"finish_reason":"stop"}}\n\n'
            ),
        )

    service = LLMService(
        Settings(llm_api_key="test-key"),
        transport=httpx.MockTransport(handler),
    )
    replies = [
        chunk
        async for chunk in service.stream_reply(
            conversation_id="conversation-1",
            username="test-user",
            user_content="你好",
            history=[],
            system_prompt="你是 Live Streaming Agent",
            knowledge_context=[],
            model_config=RuntimeModelConfig(
                provider="qwen",
                model="qwen3.6-flash",
                api_base_url=(
                    "https://dashscope.aliyuncs.com/compatible-mode/v1"
                ),
                api_key="test-key",
                web_search_enabled=False,
                temperature=0.45,
            ),
        )
    ]

    assert captured_url.endswith(
        "/api/v1/services/aigc/multimodal-generation/generation"
    )
    assert "enable_search" not in captured_request["parameters"]
    assert "search_options" not in captured_request["parameters"]
    assert captured_request["parameters"]["incremental_output"] is True
    assert captured_sse_header == "enable"
    assert captured_request["parameters"]["temperature"] == 0.45
    assert [reply.content for reply in replies if reply.content] == ["连", "接"]


def test_qwen_text_only_models_use_dashscope_text_endpoint() -> None:
    assert LLMService._dashscope_endpoint(
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "qwen3.7-max",
    ).endswith("/api/v1/services/aigc/text-generation/generation")


def test_qwen_multimodal_search_is_not_forced() -> None:
    body = LLMService._dashscope_request_body(
        model="qwen3.6-flash",
        messages=[{"role": "user", "content": "最新新闻"}],
        temperature=0.8,
        web_search_enabled=True,
    )

    assert body["parameters"]["search_options"]["forced_search"] is False
    assert body["parameters"]["search_options"]["search_strategy"] == "agent"
    assert LLMService._dashscope_endpoint(
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "qwen3.6-max-preview",
    ).endswith("/api/v1/services/aigc/text-generation/generation")


@pytest.mark.asyncio
async def test_responses_stream_records_non_sse_body(monkeypatch) -> None:
    logged_events: list[dict] = []
    monkeypatch.setattr(llm_module, "log_model_event", logged_events.append)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            json={
                "status": "incomplete",
                "incomplete_details": {"reason": "upstream_empty"},
            },
        )

    service = LLMService(
        Settings(llm_api_key="test-key"),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(HTTPException) as raised:
        _ = [
            chunk
            async for chunk in service.stream_reply(
                conversation_id="conversation-1",
                username="test-user",
                user_content="联网搜索",
                history=[],
                system_prompt="你是 Live Streaming Agent",
                knowledge_context=[],
                model_config=RuntimeModelConfig(
                    provider="doubao",
                    model="doubao-seed-2-0-lite-260215",
                    api_base_url="https://ark.cn-beijing.volces.com/api/v3",
                    api_key="test-key",
                    api_style="responses",
                    web_search_enabled=True,
                ),
            )
        ]

    assert raised.value.status_code == 502
    diagnostic = next(
        event
        for event in logged_events
        if event["event"] == "responses_stream_diagnostic"
    )
    assert diagnostic["content_type"] == "application/json"
    assert diagnostic["event_counts"] == {}
    assert diagnostic["terminal_payload"] is None
    assert '"status":"incomplete"' in diagnostic["non_sse_lines"][0]


@pytest.mark.asyncio
async def test_empty_user_content_does_not_add_memory_round_instruction() -> None:
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request.update(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=(
                'data: {"choices":[{"delta":{"content":"总结"}}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    service = LLMService(
        Settings(llm_api_key="test-key"),
        transport=httpx.MockTransport(handler),
    )
    history = [
        SimpleNamespace(
            role="user",
            content="你好",
            created_at=datetime(2026, 7, 29, 6, 32, 21, tzinfo=UTC),
        ),
        SimpleNamespace(
            role="assistant",
            content="你好呀",
            created_at=datetime(2026, 7, 29, 6, 32, 22, tzinfo=UTC),
        ),
    ]

    await service.generate_reply(
        conversation_id="conversation-1",
        username="test-user",
        user_content="",
        history=history,
        system_prompt="总结重要信息",
        knowledge_context=[],
        model_config=ModelConfig(),
    )

    assert captured_request["messages"] == [
        {"role": "system", "content": "总结重要信息"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好呀"},
    ]


@pytest.mark.asyncio
async def test_connection_without_api_key_returns_guidance() -> None:
    service = LLMService(Settings(llm_api_key=None))
    result = await service.test_connection(
        RuntimeModelConfig(
            provider="qwen",
            model="qwen-plus",
            api_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=None,
        )
    )
    assert result.ok is False
    assert "API Key" in result.message


@pytest.mark.asyncio
async def test_connection_returns_after_first_stream_character() -> None:
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request.update(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=(
                'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"连"}}]}\n\n'
                'data: {"choices":[{"delta":{"content":"接成功"}}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    service = LLMService(
        Settings(llm_api_key="test-key"),
        transport=httpx.MockTransport(handler),
    )
    result = await service.test_connection(
        RuntimeModelConfig(
            provider="deepseek",
            model="deepseek-chat",
            api_base_url="https://api.deepseek.com/v1",
            api_key="test-key",
        )
    )

    assert captured_request["stream"] is True
    assert captured_request["thinking"] == {"type": "disabled"}
    assert result.ok is True
    assert result.message == "连"
    assert result.latency_ms is not None


@pytest.mark.parametrize(
    ("provider", "api_style", "expected_field", "expected_value"),
    [
        ("deepseek", "chat_completions", "thinking", {"type": "disabled"}),
        ("qwen", "chat_completions", "enable_thinking", False),
        ("kimi", "chat_completions", "thinking", {"type": "disabled"}),
        ("zhipu", "chat_completions", "thinking", {"type": "disabled"}),
        (
            "tencent-yuanbao",
            "chat_completions",
            "thinking",
            {"type": "disabled"},
        ),
        ("doubao", "responses", "thinking", {"type": "disabled"}),
    ],
)
def test_request_body_disables_provider_thinking(
    provider: str,
    api_style: str,
    expected_field: str,
    expected_value,
) -> None:
    body = LLMService._request_body(
        provider=provider,
        model="test-model",
        messages=[{"role": "user", "content": "你好"}],
        api_style=api_style,
        temperature=0,
    )

    assert body[expected_field] == expected_value


def test_kimi_request_uses_required_temperature() -> None:
    body = LLMService._request_body(
        provider="kimi",
        model="kimi-k2.6",
        messages=[{"role": "user", "content": "你好"}],
        api_style="chat_completions",
        temperature=0,
    )

    assert body["temperature"] == 0.6
    assert body["thinking"] == {"type": "disabled"}


def test_kimi_k3_uses_minimum_supported_reasoning_effort() -> None:
    body = LLMService._request_body(
        provider="kimi",
        model="kimi-k3",
        messages=[{"role": "user", "content": "你好"}],
        api_style="chat_completions",
        temperature=0,
    )

    assert body["temperature"] == 1.0
    assert body["reasoning_effort"] == "low"
    assert "thinking" not in body


@pytest.mark.asyncio
async def test_connection_surfaces_stream_error_message() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=(
                'data: {"error":{"code":"ModelNotFound",'
                '"message":"requested model is unavailable"}}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    service = LLMService(
        Settings(llm_api_key="test-key"),
        transport=httpx.MockTransport(handler),
    )
    result = await service.test_connection(
        RuntimeModelConfig(
            provider="deepseek",
            model="deepseek-v4-pro",
            api_base_url="https://api.deepseek.com/v1",
            api_key="test-key",
        )
    )

    assert result.ok is False
    assert "ModelNotFound" in result.message
    assert "requested model is unavailable" in result.message


def test_responses_stream_extracts_output_text_delta() -> None:
    assert (
        LLMService._extract_stream_content(
            {
                "type": "response.output_text.delta",
                "delta": "连",
            },
            "responses",
        )
        == "连"
    )


def test_doubao_web_search_uses_default_limits() -> None:
    body = LLMService._request_body(
        provider="doubao",
        model="doubao-seed-2-0-lite-260215",
        messages=[{"role": "user", "content": "今天有什么新闻"}],
        api_style="responses",
        temperature=0.8,
        web_search_enabled=True,
    )

    assert "tool_choice" not in body
    assert body["max_tool_calls"] == 1
    assert body["tools"] == [
        {
            "type": "web_search",
            "limit": 3,
        }
    ]


def test_doubao_web_search_uses_configured_limits() -> None:
    body = LLMService._request_body(
        provider="doubao",
        model="doubao-seed-2-0-lite-260215",
        messages=[{"role": "user", "content": "今天有什么新闻"}],
        api_style="responses",
        temperature=0.8,
        web_search_enabled=True,
        web_search_max_tool_calls=4,
        web_search_result_limit=8,
    )

    assert body["max_tool_calls"] == 4
    assert body["tools"] == [
        {
            "type": "web_search",
            "limit": 8,
        }
    ]


def test_qwen_and_doubao_can_force_web_search() -> None:
    qwen_body = LLMService._request_body(
        provider="qwen",
        model="qwen3.7-plus",
        messages=[{"role": "user", "content": "今天有什么新闻"}],
        api_style="responses",
        temperature=0.8,
        web_search_enabled=True,
        web_search_forced=True,
    )
    doubao_body = LLMService._request_body(
        provider="doubao",
        model="doubao-seed-2-0-lite-260215",
        messages=[{"role": "user", "content": "今天有什么新闻"}],
        api_style="responses",
        temperature=0.8,
        web_search_enabled=True,
        web_search_forced=True,
    )

    assert qwen_body["tool_choice"] == "required"
    assert doubao_body["tool_choice"] == "required"


def test_web_search_sources_are_deduplicated_without_backend_limit() -> None:
    output = [
        {
            "content": [
                {
                    "annotations": [
                        {
                            "type": "url_citation",
                            "title": f"来源 {index}",
                            "url": f"https://example.com/{index}",
                        }
                        for index in range(7)
                    ]
                }
            ]
        }
    ]

    sources = LLMService._extract_web_sources({"output": output})

    assert len(sources) == 7
    assert sources[0] == {
        "title": "来源 0",
        "url": "https://example.com/0",
        "snippet": "",
    }


@pytest.mark.asyncio
async def test_chat_flow_persists_user_and_assistant_messages(monkeypatch) -> None:
    latency_events: list[dict] = []
    monkeypatch.setattr(
        main_module,
        "log_chat_latency_event",
        latency_events.append,
    )

    class FakeStore:
        def __init__(self) -> None:
            self.messages: list[MessageRecord] = []
            self.knowledge_queries: list[str] = []
            self.model_config: (
                tuple[str, str, str, bool, bool, int, int, float] | None
            ) = None

        async def update_user_model_config(
            self,
            username: str,
            provider: str,
            model: str,
            web_search_enabled: bool,
            web_search_forced: bool,
            web_search_max_tool_calls: int,
            web_search_result_limit: int,
            temperature: float,
        ) -> None:
            self.model_config = (
                username,
                provider,
                model,
                web_search_enabled,
                web_search_forced,
                web_search_max_tool_calls,
                web_search_result_limit,
                temperature,
            )

        async def get_conversation(
            self,
            conversation_id: str,
        ) -> ConversationRecord:
            from datetime import UTC, datetime

            now = datetime.now(UTC)
            return ConversationRecord(
                conversation_id=conversation_id,
                username="测试用户",
                title="测试对话",
                status="active",
                created_at=now,
                updated_at=now,
            )

        async def create_message(
            self,
            conversation_id: str,
            payload: MessageCreate,
        ) -> MessageRecord:
            from datetime import UTC, datetime

            message = MessageRecord(
                message_id=f"m-{len(self.messages) + 1}",
                conversation_id=conversation_id,
                username=payload.username,
                role=payload.role,
                content=payload.content,
                reasoning_content=payload.reasoning_content,
                emotion=payload.emotion,
                metadata=payload.metadata,
                created_at=datetime.now(UTC),
            )
            self.messages.append(message)
            return message

        async def list_messages(
            self,
            _conversation_id: str,
            *,
            size: int,
        ) -> MessagePage:
            return MessagePage(items=self.messages[-size:])

        async def search_knowledge(self, query: str, size: int):
            self.knowledge_queries.append(query)
            return [
                SimpleNamespace(
                    document_id="knowledge-1",
                    source={
                        "knowledge_id": "knowledge-1",
                        "title": "Live Streaming Agent 资料",
                        "body": "Live Streaming Agent 喜欢和观众互动",
                    }
                )
            ]

        async def advance_completed_rounds(
            self,
            _conversation_id: str,
            _message_ids: list[str],
        ) -> int:
            return 1

    store = FakeStore()
    class FakeLLMService:
        async def stream_reply(self, **_kwargs):
            yield SimpleNamespace(
                content="",
                mode="model",
                reasoning_content="internal reasoning",
            )
            yield SimpleNamespace(
                content="【开心】你好呀！",
                mode="model",
                reasoning_content="",
                web_search_duration_ms=321.0,
                web_sources=tuple(
                    {
                        "title": f"示例来源 {index}",
                        "url": f"https://example.com/source-{index}",
                        "snippet": f"来源摘要 {index}",
                    }
                    for index in range(4)
                ),
            )

    monkeypatch.setattr(main_module, "llm_service", FakeLLMService())

    class FakeModelConfigStore:
        def resolve_runtime_config(
            self,
            model_config: ModelConfig,
        ) -> RuntimeModelConfig:
            return RuntimeModelConfig(
                provider=model_config.provider,
                model=model_config.model,
                api_base_url="https://api.deepseek.com/v1",
                api_key=None,
            )

    monkeypatch.setattr(main_module, "model_config_store", FakeModelConfigStore())
    response = await chat(
        "conversation-1",
        ChatRequest(
            username="测试用户",
            content="莱叔：“你好”",
            knowledge_query="你好",
            speaker_identity="莱叔",
            save_speaker_identity=False,
            knowledge_enabled=True,
            llm_config=ModelConfig(),
        ),
        store,  # type: ignore[arg-type]
    )

    body = b"".join([chunk async for chunk in response.body_iterator])
    events = [json.loads(line) for line in body.decode().splitlines()]

    assert [event["type"] for event in events] == [
        "start",
        "metric",
        "web_search_sources",
        "metric",
        "delta",
        "metric",
        "done",
    ]
    start_event = events[0]
    source_event = next(
        event for event in events if event["type"] == "web_search_sources"
    )
    delta_event = next(event for event in events if event["type"] == "delta")
    done_event = events[-1]
    assert start_event["user_message"]["content"] == "莱叔：“你好”"
    assert start_event["knowledge_hit_count"] == 1
    assert start_event["knowledge_duration_ms"] >= 0
    assert len(source_event["sources"]) == 4
    assert source_event["sources"][0]["url"] == "https://example.com/source-0"
    assert "内容：Live Streaming Agent 喜欢和观众互动" in start_event[
        "knowledge_injected_context"
    ]
    assert "你好呀" in delta_event["content"]
    assert done_event["assistant_message"]["role"] == "assistant"
    assert "你好呀" in done_event["assistant_message"]["content"]
    assert (
        done_event["assistant_message"]["reasoning_content"]
        == "internal reasoning"
    )
    assert done_event["performance_metrics"]["model_first_token_ms"] >= 0
    assert done_event["performance_metrics"]["model_first_sentence_ms"] >= 0
    assert done_event["performance_metrics"]["model_complete_ms"] >= 0
    assert done_event["performance_metrics"]["web_search_duration_ms"] == 321.0
    assert "内容：Live Streaming Agent 喜欢和观众互动" in done_event["assistant_message"][
        "metadata"
    ]["knowledge_injected_context"]
    saved_web_sources = done_event["assistant_message"]["metadata"][
        "web_search_sources"
    ]
    assert len(saved_web_sources) == 4
    assert saved_web_sources[0]["url"] == "https://example.com/source-0"
    assert [message.role for message in store.messages] == ["user", "assistant"]
    assert store.messages[0].reasoning_content == ""
    assert store.messages[1].reasoning_content == "internal reasoning"
    assert store.knowledge_queries == ["莱叔：“你好”"]
    assert store.model_config == (
        "测试用户",
        "deepseek",
        "deepseek-v4-pro",
        False,
        False,
        1,
        3,
        0.8,
    )
    assert [
        event["event"] for event in latency_events
    ] == [
        "backend_request_received",
        "backend_history_loaded",
        "backend_short_term_memory_context_ready",
        "backend_knowledge_ready",
        "backend_stream_response_ready",
        "backend_stream_started",
        "backend_model_stream_started",
        "backend_first_chunk_forwarded",
        "backend_user_message_saved",
        "backend_assistant_message_saved",
        "backend_round_completed",
        "backend_stream_completed",
    ]


@pytest.mark.asyncio
async def test_failed_model_call_does_not_persist_or_complete_the_turn(
    monkeypatch,
) -> None:
    from datetime import UTC, datetime

    class FakeStore:
        def __init__(self) -> None:
            self.create_calls = 0
            self.advance_calls = 0

        async def update_user_model_config(self, *_args) -> None:
            return None

        async def get_conversation(
            self,
            conversation_id: str,
        ) -> ConversationRecord:
            now = datetime.now(UTC)
            return ConversationRecord(
                conversation_id=conversation_id,
                username="测试用户",
                title="测试对话",
                status="active",
                completed_rounds=4,
                effective_char_count=20,
                created_at=now,
                updated_at=now,
            )

        async def list_messages(
            self,
            _conversation_id: str,
            *,
            size: int,
        ) -> MessagePage:
            return MessagePage(items=[])

        async def create_message(self, *_args) -> MessageRecord:
            self.create_calls += 1
            raise AssertionError("failed model turns must not be persisted")

        async def advance_completed_rounds(self, *_args) -> int:
            self.advance_calls += 1
            raise AssertionError("failed model turns must not advance rounds")

    class FailingLLMService:
        async def stream_reply(self, **_kwargs):
            raise HTTPException(status_code=502, detail="upstream failed")
            yield

    class FakeModelConfigStore:
        def resolve_runtime_config(
            self,
            model_config: ModelConfig,
        ) -> RuntimeModelConfig:
            return RuntimeModelConfig(
                provider=model_config.provider,
                model=model_config.model,
                api_base_url="https://api.deepseek.com/v1",
                api_key="test-key",
            )

    store = FakeStore()
    monkeypatch.setattr(main_module, "llm_service", FailingLLMService())
    monkeypatch.setattr(
        main_module,
        "model_config_store",
        FakeModelConfigStore(),
    )

    response = await chat(
        "conversation-1",
        ChatRequest(
            username="测试用户",
            content="这次会失败",
            knowledge_enabled=False,
            llm_config=ModelConfig(),
        ),
        store,  # type: ignore[arg-type]
    )
    events = [
        json.loads(line)
        for line in (
            b"".join([chunk async for chunk in response.body_iterator])
            .decode()
            .splitlines()
        )
    ]

    assert [event["type"] for event in events] == ["start", "error"]
    assert events[-1]["status"] == 502
    assert store.create_calls == 0
    assert store.advance_calls == 0
