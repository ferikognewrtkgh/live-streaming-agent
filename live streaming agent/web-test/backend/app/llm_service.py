import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4

import httpx
from fastapi import HTTPException, status

from .config import Settings
from .logging_setup import log_chat_latency_event, log_model_event
from .schemas import (
    MessageRecord,
    ModelConfig,
    ModelConnectionTestResponse,
    RuntimeModelConfig,
)


@dataclass(frozen=True)
class GeneratedReply:
    content: str
    mode: str
    reasoning_content: str = ""
    web_sources: tuple[dict, ...] = ()
    web_search_duration_ms: float | None = None


def normalize_knowledge_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def format_knowledge_context(items: list[dict]) -> str:
    if not items:
        return ""
    lines = [
        "[知识库检索结果]",
        (
            "以下内容来自本地知识库，是当前这轮用户输入的补充背景。"
            "回答时只在相关时参考；如果无关，可以忽略。"
        ),
    ]
    for index, item in enumerate(items, start=1):
        keywords = "、".join(
            str(value) for value in item.get("keywords", []) if value
        )
        matched_keywords = "、".join(
            str(value) for value in item.get("matched_keywords", []) if value
        )
        body = normalize_knowledge_text(item.get("body"))
        usage = normalize_knowledge_text(item.get("usage"))
        lines.append(
            "\n".join(
                part
                for part in [
                    (
                        f"{index}. 标题："
                        f"{item.get('title') or item.get('knowledge_id')}"
                    ),
                    (
                        f"   分类：{item.get('category')}"
                        if item.get("category")
                        else ""
                    ),
                    f"   关键词：{keywords}" if keywords else "",
                    (
                        f"   命中关键词：{matched_keywords}"
                        if matched_keywords
                        else ""
                    ),
                    f"   用法：{usage}" if usage else "",
                    f"   内容：{body}" if body else "",
                ]
                if part
            )
        )
    return "\n".join(lines).strip()


class LLMService:
    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport

    async def test_connection(
        self,
        model_config: RuntimeModelConfig,
    ) -> ModelConnectionTestResponse:
        provider = model_config.provider
        model = model_config.model
        api_base_url = model_config.api_base_url.rstrip("/")
        api_key = model_config.api_key
        api_style = model_config.api_style

        if not api_key:
            return ModelConnectionTestResponse(
                ok=False,
                message=(
                    "服务端未配置 API Key。"
                    "请在 backend/model_config.json 或 .env 中配置。"
                ),
                provider=provider,
                model=model,
            )

        started = perf_counter()
        try:
            test_messages = [
                {"role": "user", "content": "用中文回复：连接成功"}
            ]
            if provider == "qwen":
                body = self._dashscope_request_body(
                    model=model,
                    messages=test_messages,
                    temperature=0,
                    max_tokens=256,
                    web_search_enabled=False,
                )
                endpoint = self._dashscope_endpoint(api_base_url, model)
                headers = self._dashscope_headers(api_key)
            else:
                body = self._request_body(
                    provider=provider,
                    model=model,
                    messages=test_messages,
                    api_style=api_style,
                    temperature=0,
                    max_tokens=256,
                    web_search_enabled=False,
                )
                body["stream"] = True
                endpoint = self._endpoint(api_base_url, api_style)
                headers = self._headers(api_key)
            async with httpx.AsyncClient(
                timeout=25,
                transport=self.transport,
            ) as client, client.stream(
                "POST",
                endpoint,
                headers=headers,
                json=body,
            ) as response:
                if response.is_error:
                    await response.aread()
                    response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    stream_error = self._extract_stream_error(payload)
                    if stream_error:
                        raise ValueError(stream_error)
                    content = (
                        self._extract_dashscope_stream_content(payload)
                        if provider == "qwen"
                        else self._extract_stream_content(payload, api_style)
                    )
                    if not content or not content.strip():
                        continue
                    first_character = next(
                        character
                        for character in content
                        if not character.isspace()
                    )
                    return ModelConnectionTestResponse(
                        ok=True,
                        message=first_character,
                        latency_ms=round((perf_counter() - started) * 1000, 2),
                        provider=provider,
                        model=model,
                    )
                raise ValueError("empty model stream")
        except httpx.HTTPStatusError as error:
            detail = error.response.text[:500]
            return ModelConnectionTestResponse(
                ok=False,
                message=f"连接失败：HTTP {error.response.status_code}，{detail}",
                latency_ms=round((perf_counter() - started) * 1000, 2),
                provider=provider,
                model=model,
            )
        except httpx.RequestError as error:
            return ModelConnectionTestResponse(
                ok=False,
                message=f"连接失败：{error}",
                latency_ms=round((perf_counter() - started) * 1000, 2),
                provider=provider,
                model=model,
            )
        except ValueError as error:
            return ModelConnectionTestResponse(
                ok=False,
                message=(
                    "连接失败：流式响应结束前未收到模型文本"
                    if str(error) == "empty model stream"
                    else f"连接失败：{error}"
                ),
                latency_ms=round((perf_counter() - started) * 1000, 2),
                provider=provider,
                model=model,
            )

    async def generate_reply(
        self,
        *,
        conversation_id: str,
        username: str,
        user_content: str,
        history: list[MessageRecord],
        system_prompt: str,
        knowledge_context: list[dict],
        model_config: ModelConfig | RuntimeModelConfig,
        trace_id: str | None = None,

    ) -> GeneratedReply:
        parts: list[str] = []
        async for chunk in self.stream_reply(
            conversation_id=conversation_id,
            username=username,
            user_content=user_content,
            history=history,
            system_prompt=system_prompt,
            knowledge_context=knowledge_context,
            model_config=model_config,
            trace_id=trace_id,
        ):
            parts.append(chunk.content)
        return GeneratedReply(content="".join(parts), mode="model")

    async def stream_reply(
        self,
        *,
        conversation_id: str,
        username: str,
        user_content: str,
        history: list[MessageRecord],
        system_prompt: str,
        knowledge_context: list[dict],
        model_config: ModelConfig,
        trace_id: str | None = None,
    ) -> AsyncIterator[GeneratedReply]:
        provider = model_config.provider or self.settings.llm_provider
        model = model_config.model or self.settings.llm_model
        api_base_url = getattr(
            model_config,
            "api_base_url",
            self.settings.llm_api_base_url,
        ).rstrip("/")
        api_style = getattr(model_config, "api_style", "chat_completions")
        qwen_web_search = (
            provider == "qwen" and model_config.web_search_enabled
        )
        effective_api_style = (
            "responses" if qwen_web_search else api_style
        )
        request_endpoint = (
            self._endpoint(api_base_url, "responses")
            if qwen_web_search
            else self._dashscope_endpoint(api_base_url, model)
            if provider == "qwen"
            else self._endpoint(api_base_url, api_style)
        )

        messages: list[dict[str, str]] = []
        combined_system_prompt = system_prompt.strip()
        if knowledge_context:
            knowledge_text = format_knowledge_context(knowledge_context)
            combined_system_prompt = (
                f"{combined_system_prompt}\n\n"
                "以下是可参考的知识库内容。仅在相关时使用：\n"
                f"{knowledge_text}"
            ).strip()
        if combined_system_prompt:
            messages.append({"role": "system", "content": combined_system_prompt})

        last_history_content: str | None = None
        for message in history[-20:]:
            if message.role not in {"user", "assistant", "system"}:
                continue
            last_history_content = message.content
            messages.append({"role": message.role, "content": message.content})

        if user_content and last_history_content != user_content:
            messages.append({"role": "user", "content": user_content})

        request_id = str(uuid4())
        started = perf_counter()
        log_model_event(
            {
                "event": "model_call_started",
                "timestamp": datetime.now(UTC).isoformat(),
                "request_id": request_id,
                "trace_id": trace_id,
                "conversation_id": conversation_id,
                "username": username,
                "provider": provider,
                "model": model,
                "endpoint": request_endpoint,
                "request": {
                    "system_prompt": system_prompt,
                    "knowledge_context": knowledge_context,
                    "messages": messages,
                    "temperature": self._request_temperature(
                        provider,
                        model,
                        model_config.temperature,
                    ),
                    "stream": True,
                    "web_search_enabled": (
                        provider in {"doubao", "qwen"}
                        and model_config.web_search_enabled
                    ),
                    "web_search_max_tool_calls": (
                        model_config.web_search_max_tool_calls
                    ),
                    "web_search_result_limit": (
                        model_config.web_search_result_limit
                    ),
                    "web_search_forced": (
                        model_config.web_search_enabled
                        and model_config.web_search_forced
                        and provider in {"doubao", "qwen"}
                    ),
                    "web_search_strategy": (
                        (
                            "required"
                            if model_config.web_search_forced
                            else "auto"
                        )
                        if qwen_web_search
                        else None
                    ),
                },
            }
        )
        if trace_id:
            log_chat_latency_event(
                {
                    "event": "model_request_started",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "trace_id": trace_id,
                    "conversation_id": conversation_id,
                    "model_elapsed_ms": 0,
                    "provider": provider,
                    "model": model,
                }
            )

        api_key = getattr(model_config, "api_key", None) or self.settings.llm_api_key
        if not api_key:
            self._log_result(
                event="model_call_failed",
                request_id=request_id,
                conversation_id=conversation_id,
                username=username,
                provider=provider,
                model=model,
                started=started,
                error="API Key is not configured on the server",
                trace_id=trace_id,
            )
            raise HTTPException(
                status_code=status.HTTP_424_FAILED_DEPENDENCY,
                detail=(
                    f"{provider} 未配置 API Key。"
                    "请在后端 model_config.json 或 .env 中配置。"
                ),
            )

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        web_sources: list[dict[str, str]] = []
        ttft_ms: float | None = None
        response_created_at: float | None = None
        web_search_triggered = False
        web_search_duration_emitted = False
        web_search_duration_ms = 0.0
        content_chunk_count = 0
        try:
            if provider == "qwen" and not qwen_web_search:
                body = self._dashscope_request_body(
                    model=model,
                    messages=messages,
                    temperature=model_config.temperature,
                    web_search_enabled=model_config.web_search_enabled,
                )
                async with httpx.AsyncClient(
                    timeout=60,
                    transport=self.transport,
                ) as client, client.stream(
                    "POST",
                    request_endpoint,
                    headers=self._dashscope_headers(api_key),
                    json=body,
                ) as response:
                    if response.is_error:
                        await response.aread()
                        response.raise_for_status()
                    if trace_id:
                        log_chat_latency_event(
                            {
                                "event": "model_response_headers_received",
                                "timestamp": datetime.now(UTC).isoformat(),
                                "trace_id": trace_id,
                                "conversation_id": conversation_id,
                                "model_elapsed_ms": round(
                                    (perf_counter() - started) * 1000,
                                    2,
                                ),
                            }
                        )

                    seen_source_urls: set[str] = set()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line.removeprefix("data:").strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            payload = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        stream_error = self._extract_stream_error(payload)
                        if stream_error:
                            raise HTTPException(
                                status_code=status.HTTP_502_BAD_GATEWAY,
                                detail=(
                                    "qwen model request failed: "
                                    f"{stream_error}"
                                ),
                            )

                        for source in self._extract_web_sources(payload):
                            source_url = source["url"]
                            if source_url in seen_source_urls:
                                continue
                            seen_source_urls.add(source_url)
                            web_sources.append(source)
                            yield GeneratedReply(
                                content="",
                                mode="model",
                                web_sources=(source,),
                            )

                        content = self._extract_dashscope_stream_content(
                            payload
                        )
                        if not content:
                            continue
                        content_chunk_count += 1
                        if ttft_ms is None:
                            ttft_ms = round(
                                (perf_counter() - started) * 1000,
                                2,
                            )
                            self._log_result(
                                event="model_first_chunk",
                                request_id=request_id,
                                conversation_id=conversation_id,
                                username=username,
                                provider=provider,
                                model=model,
                                started=started,
                                mode="model",
                                ttft_ms=ttft_ms,
                                trace_id=trace_id,
                            )
                            if trace_id:
                                log_chat_latency_event(
                                    {
                                        "event": "model_first_chunk_received",
                                        "timestamp": (
                                            datetime.now(UTC).isoformat()
                                        ),
                                        "trace_id": trace_id,
                                        "conversation_id": conversation_id,
                                        "model_elapsed_ms": ttft_ms,
                                        "ttft_ms": ttft_ms,
                                    }
                                )
                        content_parts.append(content)
                        yield GeneratedReply(content=content, mode="model")

            elif effective_api_style == "responses":
                body = self._request_body(
                    provider=provider,
                    model=model,
                    messages=messages,
                    api_style=effective_api_style,
                    temperature=model_config.temperature,
                    web_search_enabled=(
                        provider in {"doubao", "qwen"}
                        and model_config.web_search_enabled
                    ),
                    web_search_forced=model_config.web_search_forced,
                    web_search_max_tool_calls=(
                        model_config.web_search_max_tool_calls
                    ),
                    web_search_result_limit=(
                        model_config.web_search_result_limit
                    ),
                )
                body["stream"] = True
                completed_response: dict | None = None
                stream_event_counts: dict[str, int] = {}
                terminal_payload: dict | None = None
                non_sse_lines: list[str] = []
                stream_diagnostic_logged = False
                async with httpx.AsyncClient(
                    timeout=60,
                    transport=self.transport,
                ) as client, client.stream(
                    "POST",
                    request_endpoint,
                    headers=self._headers(api_key),
                    json=body,
                ) as response:
                    if response.is_error:
                        await response.aread()
                        response.raise_for_status()
                    if trace_id:
                        log_chat_latency_event(
                            {
                                "event": "model_response_headers_received",
                                "timestamp": datetime.now(UTC).isoformat(),
                                "trace_id": trace_id,
                                "conversation_id": conversation_id,
                                "model_elapsed_ms": round(
                                    (perf_counter() - started) * 1000,
                                    2,
                                ),
                            }
                        )

                    seen_source_urls: set[str] = set()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            if (
                                line.strip()
                                and not line.startswith(("event:", ":"))
                                and len(non_sse_lines) < 5
                            ):
                                non_sse_lines.append(line[:16_000])
                            continue
                        data = line.removeprefix("data:").strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            payload = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        event_type = payload.get("type")
                        event_name = (
                            event_type
                            if isinstance(event_type, str) and event_type
                            else "<missing>"
                        )
                        stream_event_counts[event_name] = (
                            stream_event_counts.get(event_name, 0) + 1
                        )
                        if event_name == "response.created":
                            response_created_at = perf_counter()
                            if trace_id:
                                log_chat_latency_event(
                                    {
                                        "event": "model_response_created",
                                        "timestamp": datetime.now(
                                            UTC
                                        ).isoformat(),
                                        "trace_id": trace_id,
                                        "conversation_id": conversation_id,
                                        "model_elapsed_ms": round(
                                            (response_created_at - started)
                                            * 1000,
                                            2,
                                        ),
                                    }
                                )
                        search_started, _search_completed = (
                            self._web_search_event_phase(payload)
                        )
                        if search_started:
                            web_search_triggered = True
                        if event_name in {
                            "error",
                            "response.completed",
                            "response.failed",
                            "response.incomplete",
                        }:
                            terminal_payload = payload

                        stream_error = self._extract_stream_error(payload)
                        if stream_error:
                            self._log_responses_stream_diagnostic(
                                request_id=request_id,
                                trace_id=trace_id,
                                conversation_id=conversation_id,
                                username=username,
                                provider=provider,
                                model=model,
                                response_status=response.status_code,
                                response_content_type=response.headers.get(
                                    "content-type",
                                    "",
                                ),
                                event_counts=stream_event_counts,
                                terminal_payload=terminal_payload,
                                non_sse_lines=non_sse_lines,
                            )
                            stream_diagnostic_logged = True
                            raise HTTPException(
                                status_code=status.HTTP_502_BAD_GATEWAY,
                                detail=(
                                    f"{provider} model request failed: "
                                    f"{stream_error}"
                                ),
                            )

                        for source in self._extract_web_sources(payload):
                            source_url = source["url"]
                            if source_url in seen_source_urls:
                                continue
                            seen_source_urls.add(source_url)
                            web_sources.append(source)
                            yield GeneratedReply(
                                content="",
                                mode="model",
                                web_sources=(source,),
                            )

                        if payload.get("type") == "response.completed":
                            response_value = payload.get("response")
                            if isinstance(response_value, dict):
                                completed_response = response_value

                        reasoning_content = (
                            self._extract_responses_stream_reasoning(payload)
                        )
                        if reasoning_content:
                            reasoning_parts.append(reasoning_content)
                            yield GeneratedReply(
                                content="",
                                mode="model",
                                reasoning_content=reasoning_content,
                            )

                        content = self._extract_stream_content(
                            payload,
                            effective_api_style,
                        )
                        if not content:
                            continue
                        content_chunk_count += 1
                        if ttft_ms is None:
                            first_content_at = perf_counter()
                            ttft_ms = round(
                                (first_content_at - started) * 1000,
                                2,
                            )
                            if web_search_triggered:
                                web_search_duration_ms = round(
                                    (
                                        first_content_at
                                        - (response_created_at or started)
                                    )
                                    * 1000,
                                    2,
                                )
                                web_search_duration_emitted = True
                            self._log_result(
                                event="model_first_chunk",
                                request_id=request_id,
                                conversation_id=conversation_id,
                                username=username,
                                provider=provider,
                                model=model,
                                started=started,
                                mode="model",
                                ttft_ms=ttft_ms,
                                trace_id=trace_id,
                            )
                            if trace_id:
                                log_chat_latency_event(
                                    {
                                        "event": "model_first_chunk_received",
                                        "timestamp": (
                                            datetime.now(UTC).isoformat()
                                        ),
                                        "trace_id": trace_id,
                                        "conversation_id": conversation_id,
                                        "model_elapsed_ms": ttft_ms,
                                        "ttft_ms": ttft_ms,
                                        "web_search_duration_ms": (
                                            web_search_duration_ms
                                        ),
                                    }
                                )
                            if web_search_duration_emitted:
                                yield GeneratedReply(
                                    content="",
                                    mode="model",
                                    web_search_duration_ms=(
                                        web_search_duration_ms
                                    ),
                                )
                        content_parts.append(content)
                        yield GeneratedReply(content=content, mode="model")

                    if not stream_diagnostic_logged:
                        self._log_responses_stream_diagnostic(
                            request_id=request_id,
                            trace_id=trace_id,
                            conversation_id=conversation_id,
                            username=username,
                            provider=provider,
                            model=model,
                            response_status=response.status_code,
                            response_content_type=response.headers.get(
                                "content-type",
                                "",
                            ),
                            event_counts=stream_event_counts,
                            terminal_payload=terminal_payload,
                            non_sse_lines=non_sse_lines,
                        )

                if not content_parts and completed_response is not None:
                    try:
                        content = self._extract_content(
                            completed_response,
                            effective_api_style,
                        )
                    except ValueError as error:
                        if not web_sources:
                            raise
                        detail = self._empty_web_search_response_detail(
                            provider,
                            len(web_sources),
                        )
                        self._log_result(
                            event="model_call_failed",
                            request_id=request_id,
                            conversation_id=conversation_id,
                            username=username,
                            provider=provider,
                            model=model,
                            started=started,
                            error=detail,
                            ttft_ms=ttft_ms,
                            trace_id=trace_id,
                        )
                        raise HTTPException(
                            status_code=status.HTTP_502_BAD_GATEWAY,
                            detail=detail,
                        ) from error
                    first_content_at = perf_counter()
                    ttft_ms = round(
                        (first_content_at - started) * 1000,
                        2,
                    )
                    if web_search_triggered:
                        web_search_duration_ms = round(
                            (
                                first_content_at
                                - (response_created_at or started)
                            )
                            * 1000,
                            2,
                        )
                        web_search_duration_emitted = True
                    self._log_result(
                        event="model_first_chunk",
                        request_id=request_id,
                        conversation_id=conversation_id,
                        username=username,
                        provider=provider,
                        model=model,
                        started=started,
                        mode="model",
                        ttft_ms=ttft_ms,
                        trace_id=trace_id,
                    )
                    if trace_id:
                        log_chat_latency_event(
                            {
                                "event": "model_first_chunk_received",
                                "timestamp": datetime.now(UTC).isoformat(),
                                "trace_id": trace_id,
                                "conversation_id": conversation_id,
                                "model_elapsed_ms": ttft_ms,
                                "ttft_ms": ttft_ms,
                                "web_search_duration_ms": (
                                    web_search_duration_ms
                                ),
                            }
                        )
                    if web_search_duration_emitted:
                        yield GeneratedReply(
                            content="",
                            mode="model",
                            web_search_duration_ms=web_search_duration_ms,
                        )
                    content_parts.append(content)
                    content_chunk_count += 1
                    yield GeneratedReply(content=content, mode="model")

            else:
                body = self._request_body(
                    provider=provider,
                    model=model,
                    messages=messages,
                    api_style=api_style,
                    temperature=model_config.temperature,
                    web_search_enabled=False,
                )
                body["stream"] = True
                async with httpx.AsyncClient(
                    timeout=60,
                    transport=self.transport,
                ) as client, client.stream(
                    "POST",
                    self._endpoint(api_base_url, api_style),
                    headers=self._headers(api_key),
                    json=body,
                ) as response:
                    if response.is_error:
                        await response.aread()
                        response.raise_for_status()
                    if trace_id:
                        log_chat_latency_event(
                            {
                                "event": "model_response_headers_received",
                                "timestamp": datetime.now(UTC).isoformat(),
                                "trace_id": trace_id,
                                "conversation_id": conversation_id,
                                "model_elapsed_ms": round(
                                    (perf_counter() - started) * 1000,
                                    2,
                                ),
                            }
                        )

                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line.removeprefix("data:").strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            payload = json.loads(data)
                            delta = payload["choices"][0]["delta"]
                        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                            continue
                        reasoning_content = delta.get("reasoning_content")
                        if (
                            isinstance(reasoning_content, str)
                            and reasoning_content
                        ):
                            reasoning_parts.append(reasoning_content)
                            yield GeneratedReply(
                                content="",
                                mode="model",
                                reasoning_content=reasoning_content,
                            )
                        content = delta.get("content")
                        if not isinstance(content, str) or not content:
                            continue
                        content_chunk_count += 1
                        if ttft_ms is None:
                            ttft_ms = round((perf_counter() - started) * 1000, 2)
                            self._log_result(
                                event="model_first_chunk",
                                request_id=request_id,
                                conversation_id=conversation_id,
                                username=username,
                                provider=provider,
                                model=model,
                                started=started,
                                mode="model",
                                ttft_ms=ttft_ms,
                                trace_id=trace_id,
                            )
                            if trace_id:
                                log_chat_latency_event(
                                    {
                                        "event": "model_first_chunk_received",
                                        "timestamp": datetime.now(UTC).isoformat(),
                                        "trace_id": trace_id,
                                        "conversation_id": conversation_id,
                                        "model_elapsed_ms": ttft_ms,
                                        "ttft_ms": ttft_ms,
                                    }
                                )
                        content_parts.append(content)
                        yield GeneratedReply(content=content, mode="model")
            content = "".join(content_parts).strip()
            reasoning_content = "".join(reasoning_parts)
            if not content:
                detail = (
                    self._empty_web_search_response_detail(
                        provider,
                        len(web_sources),
                    )
                    if web_sources
                    else f"{provider} returned an empty response"
                )
                self._log_result(
                    event="model_call_failed",
                    request_id=request_id,
                    conversation_id=conversation_id,
                    username=username,
                    provider=provider,
                    model=model,
                    started=started,
                    error=detail,
                    trace_id=trace_id,
                )
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=detail,
                )
            self._log_result(
                event="model_call_completed",
                request_id=request_id,
                conversation_id=conversation_id,
                username=username,
                provider=provider,
                model=model,
                started=started,
                mode="model",
                response={
                    "content": content,
                    "reasoning_content": reasoning_content,
                    "web_sources": web_sources,
                    "web_search_duration_ms": web_search_duration_ms,
                    "content_chunk_count": content_chunk_count,
                },
                ttft_ms=ttft_ms,
                trace_id=trace_id,
            )
            if trace_id:
                log_chat_latency_event(
                    {
                        "event": "model_stream_completed",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "trace_id": trace_id,
                        "conversation_id": conversation_id,
                        "model_elapsed_ms": round(
                            (perf_counter() - started) * 1000,
                            2,
                        ),
                        "ttft_ms": ttft_ms,
                    }
                )
        except httpx.HTTPStatusError as error:
            detail = error.response.text[:500]
            self._log_result(
                event="model_call_failed",
                request_id=request_id,
                conversation_id=conversation_id,
                username=username,
                provider=provider,
                model=model,
                started=started,
                error=f"HTTP {error.response.status_code}: {detail}",
                ttft_ms=ttft_ms,
                trace_id=trace_id,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"{provider} model request failed: {detail}",
            ) from error
        except httpx.RequestError as error:
            self._log_result(
                event="model_call_failed",
                request_id=request_id,
                conversation_id=conversation_id,
                username=username,
                provider=provider,
                model=model,
                started=started,
                error=str(error),
                ttft_ms=ttft_ms,
                trace_id=trace_id,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"{provider} model is unreachable: {error}",
            ) from error
        except (KeyError, IndexError, AttributeError, ValueError) as error:
            self._log_result(
                event="model_call_failed",
                request_id=request_id,
                conversation_id=conversation_id,
                username=username,
                provider=provider,
                model=model,
                started=started,
                error="Model returned an invalid response structure",
                ttft_ms=ttft_ms,
                trace_id=trace_id,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"{provider} returned an invalid response",
            ) from error

    @staticmethod
    def _empty_web_search_response_detail(
        provider: str,
        source_count: int,
    ) -> str:
        provider_name = "豆包" if provider == "doubao" else provider
        return (
            f"{provider_name}联网搜索已完成并返回 {source_count} 个来源，"
            "但模型没有生成回复正文。请重试，或关闭强制搜索后再试。"
        )

    @staticmethod
    def _log_result(
        *,
        event: str,
        request_id: str,
        conversation_id: str,
        username: str,
        provider: str,
        model: str,
        started: float,
        mode: str | None = None,
        response: dict | None = None,
        error: str | None = None,
        ttft_ms: float | None = None,
        trace_id: str | None = None,
    ) -> None:
        record = {
            "event": event,
            "timestamp": datetime.now(UTC).isoformat(),
            "request_id": request_id,
            "trace_id": trace_id,
            "conversation_id": conversation_id,
            "username": username,
            "provider": provider,
            "model": model,
            "duration_ms": round((perf_counter() - started) * 1000, 2),
        }
        if mode is not None:
            record["mode"] = mode
        if response is not None:
            record["response"] = response
        if error is not None:
            record["error"] = error
        if ttft_ms is not None:
            record["ttft_ms"] = ttft_ms
        log_model_event(record)

    @staticmethod
    def _log_responses_stream_diagnostic(
        *,
        request_id: str,
        trace_id: str | None,
        conversation_id: str,
        username: str,
        provider: str,
        model: str,
        response_status: int,
        response_content_type: str,
        event_counts: dict[str, int],
        terminal_payload: dict | None,
        non_sse_lines: list[str],
    ) -> None:
        terminal_response = (
            terminal_payload.get("response")
            if isinstance(terminal_payload, dict)
            else None
        )
        log_model_event(
            {
                "event": "responses_stream_diagnostic",
                "timestamp": datetime.now(UTC).isoformat(),
                "request_id": request_id,
                "trace_id": trace_id,
                "conversation_id": conversation_id,
                "username": username,
                "provider": provider,
                "model": model,
                "http_status": response_status,
                "content_type": response_content_type,
                "event_counts": event_counts,
                "terminal_event_type": (
                    terminal_payload.get("type")
                    if isinstance(terminal_payload, dict)
                    else None
                ),
                "terminal_response_status": (
                    terminal_response.get("status")
                    if isinstance(terminal_response, dict)
                    else None
                ),
                "terminal_response_error": (
                    terminal_response.get("error")
                    if isinstance(terminal_response, dict)
                    else None
                ),
                "terminal_incomplete_details": (
                    terminal_response.get("incomplete_details")
                    if isinstance(terminal_response, dict)
                    else None
                ),
                "terminal_payload": terminal_payload,
                "non_sse_lines": non_sse_lines,
            }
        )

    @staticmethod
    def _headers(api_key: str | None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    @classmethod
    def _dashscope_headers(cls, api_key: str | None) -> dict[str, str]:
        return {
            **cls._headers(api_key),
            "X-DashScope-SSE": "enable",
        }

    @staticmethod
    def _dashscope_endpoint(api_base_url: str, model: str) -> str:
        normalized = api_base_url.rstrip("/")
        generation_type = (
            "multimodal-generation"
            if LLMService._qwen_uses_multimodal_endpoint(model)
            else "text-generation"
        )
        native_suffix = f"/api/v1/services/aigc/{generation_type}/generation"
        if normalized.endswith((
            "/api/v1/services/aigc/text-generation/generation",
            "/api/v1/services/aigc/multimodal-generation/generation",
        )):
            native_root = normalized.split("/api/v1/services/aigc/", 1)[0]
            return f"{native_root}{native_suffix}"
        if normalized.endswith(native_suffix):
            return normalized
        for compatible_suffix in (
            "/compatible-mode/v1/chat/completions",
            "/compatible-mode/v1",
        ):
            if normalized.endswith(compatible_suffix):
                return f"{normalized.removesuffix(compatible_suffix)}{native_suffix}"
        return f"{normalized}{native_suffix}"

    @staticmethod
    def _qwen_uses_multimodal_endpoint(model: str) -> bool:
        return model not in {
            "qwen3.7-max",
            "qwen3.6-max-preview",
        }

    @staticmethod
    def _qwen_web_search_strategy(model: str) -> str:
        return (
            "agent"
            if LLMService._qwen_uses_multimodal_endpoint(model)
            else "turbo"
        )

    @staticmethod
    def _endpoint(api_base_url: str, api_style: str) -> str:
        if api_base_url.endswith("/chat/completions") or api_base_url.endswith(
            "/responses"
        ):
            return api_base_url
        suffix = "/responses" if api_style == "responses" else "/chat/completions"
        return f"{api_base_url.rstrip('/')}{suffix}"

    @staticmethod
    def _request_body(
        *,
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        api_style: str,
        temperature: float,
        max_tokens: int | None = None,
        web_search_enabled: bool = False,
        web_search_forced: bool = False,
        web_search_max_tool_calls: int = 1,
        web_search_result_limit: int = 3,
    ) -> dict:
        temperature = LLMService._request_temperature(provider, model, temperature)
        if api_style == "responses":
            body: dict = {
                "model": model,
                "input": messages,
                "temperature": temperature,
                "stream": False,
            }
            if provider == "qwen":
                body["enable_thinking"] = False
            else:
                body["thinking"] = {"type": "disabled"}
            if max_tokens is not None:
                body["max_output_tokens"] = max_tokens
            if provider == "qwen" and web_search_enabled:
                body["tools"] = [{"type": "web_search"}]
                if web_search_forced:
                    body["tool_choice"] = "required"
            elif provider == "doubao" and web_search_enabled:
                body["max_tool_calls"] = web_search_max_tool_calls
                body["tools"] = [
                    {
                        "type": "web_search",
                        "limit": web_search_result_limit,
                    }
                ]
                if web_search_forced:
                    body["tool_choice"] = "required"
            return body

        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if provider == "qwen":
            body["enable_thinking"] = False
        elif provider == "kimi" and model == "kimi-k3":
            body["reasoning_effort"] = "low"
        else:
            body["thinking"] = {"type": "disabled"}
        return body

    @staticmethod
    def _dashscope_request_body(
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None = None,
        web_search_enabled: bool = False,
    ) -> dict:
        parameters: dict = {
            "result_format": "message",
            "incremental_output": True,
            "enable_thinking": False,
            "temperature": temperature,
        }
        if max_tokens is not None:
            parameters["max_tokens"] = max_tokens
        if web_search_enabled:
            parameters["enable_search"] = True
            parameters["search_options"] = {
                "forced_search": False,
                "search_strategy": (
                    LLMService._qwen_web_search_strategy(model)
                ),
                "enable_source": True,
                "prepend_search_result": True,
            }
        return {
            "model": model,
            "input": {
                "messages": (
                    [
                        {
                            **message,
                            "content": [{"text": message["content"]}],
                        }
                        for message in messages
                    ]
                    if LLMService._qwen_uses_multimodal_endpoint(model)
                    else messages
                )
            },
            "parameters": parameters,
        }

    @staticmethod
    def _request_temperature(provider: str, model: str, requested: float) -> float:
        if provider == "kimi":
            return 1.0 if model == "kimi-k3" else 0.6
        return requested

    @classmethod
    def _extract_content(cls, payload: dict, api_style: str) -> str:
        if api_style != "responses":
            content = payload["choices"][0]["message"]["content"].strip()
            if not content:
                raise ValueError("empty model response")
            return content

        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        parts: list[str] = []
        for item in payload.get("output", []):
            content = item.get("content") if isinstance(item, dict) else None
            if isinstance(content, str):
                parts.append(content)
                continue
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                text = content_item.get("text") or content_item.get("content")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())

        if parts:
            return "\n".join(parts).strip()

        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            return cls._extract_content(payload, "chat_completions")

        raise ValueError("Model returned an invalid response structure")

    @staticmethod
    def _extract_web_sources(payload: dict) -> list[dict[str, str]]:
        sources: list[dict[str, str]] = []
        seen_urls: set[str] = set()

        def visit(value: object, *, allow_url_only: bool = False) -> None:
            if isinstance(value, dict):
                raw_url = value.get("url")
                has_source_metadata = any(
                    value.get(field)
                    for field in (
                        "title",
                        "name",
                        "site_name",
                        "snippet",
                        "summary",
                        "description",
                        "content",
                    )
                )
                if isinstance(raw_url, str) and (
                    has_source_metadata or allow_url_only
                ):
                    url = raw_url.strip()
                    if (
                        url.startswith(("http://", "https://"))
                        and url not in seen_urls
                    ):
                        title = str(
                            value.get("title")
                            or value.get("name")
                            or value.get("site_name")
                            or url
                        ).strip()
                        snippet = str(
                            value.get("snippet")
                            or value.get("summary")
                            or value.get("description")
                            or value.get("content")
                            or ""
                        ).strip()
                        sources.append(
                            {
                                "title": title[:500],
                                "url": url[:4000],
                                "snippet": snippet[:4000],
                            }
                        )
                        seen_urls.add(url)
                for key, child in value.items():
                    visit(
                        child,
                        allow_url_only=allow_url_only or key == "sources",
                    )
            elif isinstance(value, list):
                for child in value:
                    visit(child, allow_url_only=allow_url_only)

        visit(payload.get("output", payload))
        return sources

    @staticmethod
    def _web_search_event_phase(payload: dict) -> tuple[bool, bool]:
        event_type = str(payload.get("type") or "")
        started = event_type in {
            "response.web_search_call.in_progress",
            "response.web_search_call.searching",
        }
        completed = event_type == "response.web_search_call.completed"

        item = payload.get("item")
        if isinstance(item, dict) and item.get("type") == "web_search_call":
            if event_type == "response.output_item.added":
                started = True
            if (
                event_type == "response.output_item.done"
                or item.get("status") == "completed"
            ):
                completed = True

        if event_type == "response.completed" and not completed:
            response = payload.get("response")
            output = response.get("output") if isinstance(response, dict) else None
            if isinstance(output, list) and any(
                isinstance(output_item, dict)
                and output_item.get("type") == "web_search_call"
                for output_item in output
            ):
                started = True
                completed = True

        return started, completed

    @staticmethod
    def _extract_dashscope_stream_content(payload: dict) -> str | None:
        try:
            content = payload["output"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, AttributeError):
            return None
        if isinstance(content, str):
            return content or None
        if not isinstance(content, list):
            return None
        parts = [
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict) and item.get("text")
        ]
        return "".join(parts) or None

    @staticmethod
    def _extract_responses_stream_reasoning(payload: dict) -> str | None:
        if payload.get("type") not in {
            "response.reasoning_text.delta",
            "response.reasoning_summary_text.delta",
        }:
            return None
        delta = payload.get("delta")
        return delta if isinstance(delta, str) and delta else None

    @staticmethod
    def _extract_stream_content(payload: dict, api_style: str) -> str | None:
        if api_style == "responses":
            if payload.get("type") == "response.output_text.delta":
                delta = payload.get("delta")
                return delta if isinstance(delta, str) else None
            return None

        try:
            content = payload["choices"][0]["delta"].get("content")
        except (KeyError, IndexError, TypeError, AttributeError):
            return None
        return content if isinstance(content, str) else None

    @staticmethod
    def _extract_stream_error(payload: dict) -> str | None:
        error = payload.get("error")
        if isinstance(error, str) and error.strip():
            return error.strip()
        if isinstance(error, dict):
            message = error.get("message")
            code = error.get("code")
            if isinstance(message, str) and message.strip():
                return (
                    f"{code}: {message.strip()}"
                    if code
                    else message.strip()
                )

        if payload.get("type") in {"error", "response.failed"}:
            message = payload.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
            response = payload.get("response")
            if isinstance(response, dict):
                nested_error = response.get("error")
                if isinstance(nested_error, dict):
                    nested_message = nested_error.get("message")
                    if isinstance(nested_message, str) and nested_message.strip():
                        return nested_message.strip()
            return "模型流返回失败事件"
        code = payload.get("code")
        message = payload.get("message")
        if code and isinstance(message, str) and message.strip():
            return f"{code}: {message.strip()}"
        return None
