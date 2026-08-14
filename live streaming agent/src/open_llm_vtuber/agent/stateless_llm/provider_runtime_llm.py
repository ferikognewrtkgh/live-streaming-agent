from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator

import httpx
from loguru import logger

from ...mcpp.types import ToolCallObject
from .openai_compatible_llm import AsyncLLM as OpenAICompatibleLLM
from .stateless_llm_interface import StatelessLLMInterface


WEB_SEARCH_PROVIDERS = {"doubao", "qwen"}


class ProviderRuntimeLLM(StatelessLLMInterface):
    """Runtime-selectable LLM with Live Streaming Agent-style built-in web search support."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float,
        web_search_enabled: bool = False,
        web_search_forced: bool = False,
        web_search_max_tool_calls: int = 1,
        web_search_result_limit: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
        request_timeout_seconds: float = 5.0,
        stream_idle_timeout_seconds: float = 5.0,
    ) -> None:
        self.provider = provider
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.web_search_enabled = bool(
            web_search_enabled and provider in WEB_SEARCH_PROVIDERS
        )
        self.web_search_forced = bool(
            web_search_forced and self.web_search_enabled
        )
        self.web_search_max_tool_calls = max(
            1,
            min(10, int(web_search_max_tool_calls)),
        )
        self.web_search_result_limit = max(
            1,
            min(20, int(web_search_result_limit)),
        )
        self._api_key = api_key
        self._transport = transport
        request_temperature = self._request_temperature()
        request_extra_body: dict[str, Any] | None = None
        if provider == "qwen":
            request_extra_body = {"enable_thinking": False}
        elif provider == "kimi" and model == "kimi-k3":
            request_extra_body = {"reasoning_effort": "low"}
        elif provider == "kimi":
            request_extra_body = {}
        self._fallback = OpenAICompatibleLLM(
            model=model,
            base_url=self.base_url,
            llm_api_key=api_key,
            temperature=request_temperature,
            request_timeout_seconds=request_timeout_seconds,
            stream_idle_timeout_seconds=stream_idle_timeout_seconds,
            include_thinking_config=request_extra_body is None,
            request_extra_body=request_extra_body,
        )
        self.support_tools = True

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        system: str = "",
        tools: list[dict[str, Any]] | None = None,
        call_source: str = "unknown",
    ) -> AsyncIterator[str | list[ToolCallObject] | dict[str, Any]]:
        use_web_search = (
            self.web_search_enabled
            and call_source.startswith("chat_")
            and not tools
        )
        use_responses_api = not tools and (
            self.provider == "doubao" or use_web_search
        )
        if not use_responses_api:
            stream = (
                self._fallback.chat_completion(
                    messages,
                    system,
                    call_source=call_source,
                )
                if tools is None
                else self._fallback.chat_completion(
                    messages,
                    system,
                    tools=tools,
                    call_source=call_source,
                )
            )
            async for item in stream:
                yield item
            self.support_tools = self._fallback.support_tools
            return

        async for item in self._web_search_completion(
            messages,
            system=system,
            call_source=call_source,
            search_enabled=use_web_search,
        ):
            yield item

    async def _web_search_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str,
        call_source: str,
        search_enabled: bool,
    ) -> AsyncIterator[str | dict[str, Any]]:
        request_messages = list(messages)
        if system:
            request_messages = [{"role": "system", "content": system}, *messages]

        endpoint = self._responses_endpoint(self.base_url)
        body = self._responses_request_body(
            request_messages,
            search_enabled=search_enabled,
        )
        request_started_at = time.perf_counter()
        response_created_at: float | None = None
        search_triggered = False
        timing_emitted = False
        first_content_logged = False
        stream_event_counts: dict[str, int] = {}
        response_error_detail = ""
        logger.info(
            "Responses model request started: source={} provider={} model={} "
            "endpoint={} web_search={}",
            call_source,
            self.provider,
            self.model,
            endpoint,
            search_enabled,
        )

        timeout = httpx.Timeout(60.0, connect=10.0)
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                transport=self._transport,
            ) as client, client.stream(
                "POST",
                endpoint,
                headers=self._headers(self._api_key),
                json=body,
            ) as response:
                if response.is_error:
                    await response.aread()
                    response_error_detail = self._response_error_detail(response)
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

                    event_name = str(payload.get("type") or "<missing>")
                    stream_event_counts[event_name] = (
                        stream_event_counts.get(event_name, 0) + 1
                    )
                    if event_name == "response.created":
                        response_created_at = time.perf_counter()

                    search_started, search_completed = (
                        self._web_search_event_phase(payload)
                        if search_enabled
                        else (False, False)
                    )
                    if search_started and not search_triggered:
                        search_triggered = True
                        logger.info(
                            "Web search started: source={} provider={} model={}",
                            call_source,
                            self.provider,
                            self.model,
                        )
                        yield {"type": "web-search-start"}

                    stream_error = self._extract_stream_error(payload)
                    if stream_error:
                        raise RuntimeError(stream_error)

                    content = self._extract_responses_stream_content(payload)
                    if not content:
                        if search_completed and search_triggered:
                            logger.debug(
                                "Web search provider reported completion before text: "
                                "source={} provider={} model={}",
                                call_source,
                                self.provider,
                                self.model,
                            )
                        continue

                    if not first_content_logged:
                        first_content_at = time.perf_counter()
                        first_content_logged = True
                        latency = first_content_at - request_started_at
                        logger.info(
                            "LLM first token latency: {:.3f}s ({:.0f} ms), "
                            "source={}, model={}, base_url={}, web_search={}",
                            latency,
                            latency * 1000,
                            call_source,
                            self.model,
                            self.base_url,
                            search_triggered,
                        )
                        if search_triggered:
                            search_seconds = max(
                                0.0,
                                first_content_at
                                - (response_created_at or request_started_at),
                            )
                            timing_emitted = True
                            logger.info(
                                "Web search completed: source={} provider={} "
                                "model={} duration_ms={:.1f}",
                                call_source,
                                self.provider,
                                self.model,
                                search_seconds * 1000,
                            )
                            yield {
                                "type": "web-search-timing",
                                "seconds": search_seconds,
                            }
                    yield content

            if search_triggered and not timing_emitted:
                search_seconds = max(
                    0.0,
                    time.perf_counter()
                    - (response_created_at or request_started_at),
                )
                logger.info(
                    "Web search completed without text timing anchor: source={} "
                    "provider={} model={} duration_ms={:.1f}",
                    call_source,
                    self.provider,
                    self.model,
                    search_seconds * 1000,
                )
                yield {"type": "web-search-timing", "seconds": search_seconds}
        except (httpx.HTTPError, RuntimeError, asyncio.TimeoutError) as exc:
            logger.error(
                "Web search model request failed: source={} provider={} model={} "
                "events={} response_body={} error={}",
                call_source,
                self.provider,
                self.model,
                stream_event_counts,
                response_error_detail or "<unavailable>",
                exc,
            )
            yield (
                "Error calling the chat endpoint: Error occurred while generating "
                "a web-search response. See the logs for details."
            )

    def _responses_request_body(
        self,
        messages: list[dict[str, Any]],
        *,
        search_enabled: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "input": self._normalize_responses_messages(messages),
            "temperature": self._request_temperature(),
            "stream": True,
        }
        if self.provider == "qwen":
            body["enable_thinking"] = False
            if search_enabled:
                body["tools"] = [{"type": "web_search"}]
                if self.web_search_forced:
                    body["tool_choice"] = "required"
        else:
            body["thinking"] = {"type": "disabled"}
            if search_enabled:
                body["max_tool_calls"] = self.web_search_max_tool_calls
                body["tools"] = [
                    {
                        "type": "web_search",
                        "limit": self.web_search_result_limit,
                    }
                ]
                if self.web_search_forced:
                    body["tool_choice"] = "required"
        return body

    @staticmethod
    def _normalize_responses_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert Chat Completions message parts to Responses API input."""
        normalized_messages: list[dict[str, Any]] = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                text_parts = [
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, dict)
                    and item.get("type") in {"text", "input_text"}
                    and item.get("text") is not None
                ]
                if len(text_parts) == len(content):
                    content = "\n".join(text_parts)
                else:
                    normalized_parts: list[dict[str, Any]] = []
                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        item_type = item.get("type")
                        if item_type == "text":
                            normalized_parts.append(
                                {
                                    "type": "input_text",
                                    "text": str(item.get("text", "")),
                                }
                            )
                        elif item_type == "image_url":
                            image = item.get("image_url")
                            image_url = (
                                image.get("url")
                                if isinstance(image, dict)
                                else image
                            )
                            if image_url:
                                normalized_parts.append(
                                    {
                                        "type": "input_image",
                                        "image_url": image_url,
                                    }
                                )
                        elif item_type in {"input_text", "input_image"}:
                            normalized_parts.append(dict(item))
                    content = normalized_parts

            normalized_messages.append(
                {
                    "role": str(message.get("role") or "user"),
                    "content": content,
                }
            )
        return normalized_messages

    @staticmethod
    def _response_error_detail(response: httpx.Response) -> str:
        detail = " ".join(response.text.split())
        return detail[:2000] if detail else "<empty>"

    def _request_temperature(self) -> float:
        if self.provider == "kimi":
            return 1.0 if self.model == "kimi-k3" else 0.6
        return self.temperature

    @staticmethod
    def _responses_endpoint(base_url: str) -> str:
        normalized = base_url.rstrip("/")
        if normalized.endswith("/responses"):
            return normalized
        if normalized.endswith("/chat/completions"):
            normalized = normalized.removesuffix("/chat/completions")
        return f"{normalized}/responses"

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _extract_responses_stream_content(payload: dict[str, Any]) -> str | None:
        if payload.get("type") != "response.output_text.delta":
            return None
        delta = payload.get("delta")
        return delta if isinstance(delta, str) and delta else None

    @staticmethod
    def _web_search_event_phase(payload: dict[str, Any]) -> tuple[bool, bool]:
        event_type = str(payload.get("type") or "")
        started = event_type in {
            "response.web_search_call.in_progress",
            "response.web_search_call.searching",
        }
        completed = event_type == "response.web_search_call.completed"
        item = payload.get("item")
        if isinstance(item, dict) and item.get("type") == "web_search_call":
            started = started or event_type == "response.output_item.added"
            completed = completed or event_type == "response.output_item.done"
            completed = completed or item.get("status") == "completed"
        if event_type == "response.completed":
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
    def _extract_stream_error(payload: dict[str, Any]) -> str | None:
        error = payload.get("error")
        if isinstance(error, str) and error.strip():
            return error.strip()
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                code = error.get("code")
                return f"{code}: {message.strip()}" if code else message.strip()
        if payload.get("type") in {"error", "response.failed"}:
            response = payload.get("response")
            nested = response.get("error") if isinstance(response, dict) else None
            if isinstance(nested, dict) and nested.get("message"):
                return str(nested["message"])
            return str(payload.get("message") or "模型流返回失败事件")
        return None
