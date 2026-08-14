"""Description: This file contains the implementation of the `AsyncLLM` class.
This class is responsible for handling asynchronous interaction with OpenAI API compatible
endpoints for language generation.
"""

from typing import AsyncIterator, List, Dict, Any
import asyncio
import time
from openai import (
    AsyncStream,
    AsyncOpenAI,
    APIError,
    APIConnectionError,
    RateLimitError,
    NotGiven,
    NOT_GIVEN,
)
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from loguru import logger

from .stateless_llm_interface import StatelessLLMInterface
from ...mcpp.types import ToolCallObject


LLM_REQUEST_TIMEOUT_SECONDS = 5
LLM_STREAM_IDLE_TIMEOUT_SECONDS = 5


def _messages_log_view(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep image data URLs out of logs."""
    safe_messages = []
    for message in messages:
        safe_message = dict(message)
        content = safe_message.get("content")
        if isinstance(content, list):
            safe_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    image_url = item.get("image_url") or {}
                    url = image_url.get("url", "")
                    safe_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"<data-url omitted, chars={len(url)}>",
                                "detail": image_url.get("detail"),
                            },
                        }
                    )
                else:
                    safe_content.append(item)
            safe_message["content"] = safe_content
        safe_messages.append(safe_message)
    return safe_messages


class AsyncLLM(StatelessLLMInterface):
    def __init__(
        self,
        model: str,
        base_url: str,
        llm_api_key: str = "z",
        organization_id: str = "z",
        project_id: str = "z",
        temperature: float = 1.0,
        request_timeout_seconds: float = LLM_REQUEST_TIMEOUT_SECONDS,
        stream_idle_timeout_seconds: float = LLM_STREAM_IDLE_TIMEOUT_SECONDS,
        include_thinking_config: bool = True,
        request_extra_body: dict[str, Any] | None = None,
    ):
        """
        Initializes an instance of the `AsyncLLM` class.

        Parameters:
        - model (str): The model to be used for language generation.
        - base_url (str): The base URL for the OpenAI API.
        - organization_id (str, optional): The organization ID for the OpenAI API. Defaults to "z".
        - project_id (str, optional): The project ID for the OpenAI API. Defaults to "z".
        - llm_api_key (str, optional): The API key for the OpenAI API. Defaults to "z".
        - temperature (float, optional): What sampling temperature to use, between 0 and 2. Defaults to 1.0.
        """
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.request_timeout_seconds = request_timeout_seconds
        self.stream_idle_timeout_seconds = stream_idle_timeout_seconds
        self.include_thinking_config = include_thinking_config
        self.request_extra_body = request_extra_body
        self.client = AsyncOpenAI(
            base_url=base_url,
            organization=organization_id,
            project=project_id,
            api_key=llm_api_key,
        )
        self.support_tools = True

        logger.info(
            "Initialized AsyncLLM with the parameters: {}, {}, "
            "include_thinking_config={}",
            self.base_url,
            self.model,
            self.include_thinking_config,
        )

    async def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        system: str = "",
        tools: List[Dict[str, Any]] | NotGiven = NOT_GIVEN,
        call_source: str = "unknown",
    ) -> AsyncIterator[str | List[ChoiceDeltaToolCall]]:
        """
        Generates a chat completion using the OpenAI API asynchronously.

        Parameters:
        - messages (List[Dict[str, Any]]): The list of messages to send to the API.
        - system (str, optional): System prompt to use for this completion.
        - tools (List[Dict[str, str]], optional): List of tools to use for this completion.

        Yields:
        - str: The content of each chunk from the API response.
        - List[ChoiceDeltaToolCall]: The tool calls detected in the response.

        Raises:
        - APIConnectionError: When the server cannot be reached
        - RateLimitError: When a 429 status code is received
        - APIError: For other API-related errors
        """
        stream = None
        # Tool call related state variables
        accumulated_tool_calls = {}
        in_tool_call = False
        llm_request_started_at = time.perf_counter()
        first_stream_chunk_logged = False
        first_content_logged = False
        chunks_before_first_content = 0

        try:
            # If system prompt is provided, add it to the messages
            messages_with_system = messages
            if system:
                messages_with_system = [
                    {"role": "system", "content": system},
                    *messages,
                ]
            logger.debug(
                f"LLM call source: {call_source}\nSystem: {system[:20]}...\n"
                f"Messages: {_messages_log_view(messages)}"
            )
            available_tools = tools if self.support_tools else NOT_GIVEN
            request_extra_body = self.request_extra_body
            if request_extra_body is None:
                request_extra_body = (
                    {"thinking": {"type": "disabled"}}
                    if self.include_thinking_config
                    else {}
                )

            stream_create_started_at = time.perf_counter()
            stream: AsyncStream[ChatCompletionChunk] = await asyncio.wait_for(
                self.client.chat.completions.create(
                    messages=messages_with_system,
                    model=self.model,
                    stream=True,
                    extra_body=request_extra_body,


                    temperature=self.temperature,
                    top_p=0.8,
                    tools=available_tools,

                ),
                timeout=self.request_timeout_seconds,
            )
            stream_create_latency = time.perf_counter() - stream_create_started_at
            logger.info(
                "LLM stream create latency: {:.3f}s ({:.0f} ms), source={}, model={}, base_url={}",
                stream_create_latency,
                stream_create_latency * 1000,
                call_source,
                self.model,
                self.base_url,
            )
            logger.debug(
                f"Tool Support: {self.support_tools}, Available tools: {available_tools}"
            )

            stream_iterator = stream.__aiter__()
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        stream_iterator.__anext__(),
                        timeout=self.stream_idle_timeout_seconds,
                    )
                    # logger.info(f"chunk {chunk}")
                except StopAsyncIteration:
                    break

                if not first_stream_chunk_logged:
                    first_chunk_latency = time.perf_counter() - llm_request_started_at
                    logger.info(
                        "LLM first stream chunk latency: {:.3f}s ({:.0f} ms), source={}, model={}, base_url={}",
                        first_chunk_latency,
                        first_chunk_latency * 1000,
                        call_source,
                        self.model,
                        self.base_url,
                    )
                    first_stream_chunk_logged = True

                if len(chunk.choices) == 0:
                    logger.debug("Received an empty choices chunk; skipping it.")
                    continue

                if self.support_tools:
                    has_tool_calls = (
                        hasattr(chunk.choices[0].delta, "tool_calls")
                        and chunk.choices[0].delta.tool_calls
                    )

                    if has_tool_calls:
                        logger.debug(
                            f"Tool calls detected in chunk: {chunk.choices[0].delta.tool_calls}"
                        )
                        in_tool_call = True
                        # Process tool calls in the current chunk
                        for tool_call in chunk.choices[0].delta.tool_calls:
                            index = (
                                tool_call.index if hasattr(tool_call, "index") else 0
                            )

                            # Initialize tool call for this index if needed
                            if index not in accumulated_tool_calls:
                                accumulated_tool_calls[index] = {
                                    "index": index,
                                    "id": getattr(tool_call, "id", None),
                                    "type": getattr(tool_call, "type", None),
                                    "function": {"name": "", "arguments": ""},
                                }

                            # Update tool call information
                            if hasattr(tool_call, "id") and tool_call.id:
                                accumulated_tool_calls[index]["id"] = tool_call.id
                            if hasattr(tool_call, "type") and tool_call.type:
                                accumulated_tool_calls[index]["type"] = tool_call.type

                            # Update function information
                            if hasattr(tool_call, "function"):
                                if (
                                    hasattr(tool_call.function, "name")
                                    and tool_call.function.name
                                ):
                                    accumulated_tool_calls[index]["function"][
                                        "name"
                                    ] = tool_call.function.name
                                if (
                                    hasattr(tool_call.function, "arguments")
                                    and tool_call.function.arguments
                                ):
                                    accumulated_tool_calls[index]["function"][
                                        "arguments"
                                    ] += tool_call.function.arguments

                        continue

                    # If we were in a tool call but now we're not, yield the tool call result
                    elif in_tool_call and not has_tool_calls:
                        in_tool_call = False
                        # Convert accumulated tool calls to the required format and output
                        logger.info(f"Complete tool calls: {accumulated_tool_calls}")

                        # Use the from_dict method to create a ToolCallObject instance from a dictionary
                        complete_tool_calls = [
                            ToolCallObject.from_dict(tool_data)
                            for tool_data in accumulated_tool_calls.values()
                        ]

                        yield complete_tool_calls
                        accumulated_tool_calls = {}  # Reset for potential future tool calls

                # Process regular content chunks
                if chunk.choices[0].finish_reason == "stop":
                    try:
                        if hasattr(chunk, "model_dump"):
                            try:
                                chunk_info = chunk.model_dump(
                                    mode="json", exclude_none=True
                                )
                            except TypeError:
                                chunk_info = chunk.model_dump(exclude_none=True)
                        elif hasattr(chunk, "dict"):
                            chunk_info = chunk.dict(exclude_none=True)
                        else:
                            chunk_info = str(chunk)
                    except Exception:
                        chunk_info = str(chunk)
                    logger.info(
                        "LLM stop chunk api info: source={}, model={}, base_url={}, chunk={}",
                        call_source,
                        self.model,
                        self.base_url,
                        chunk_info,
                    )
                if chunk.choices[0].delta.content is None:
                    chunk.choices[0].delta.content = ""
                content = chunk.choices[0].delta.content
                if content and not first_content_logged: #and not await __import__("asyncio").sleep(1.6):
                    first_content_latency = time.perf_counter() - llm_request_started_at
                    logger.info(
                        "LLM first token latency: {:.3f}s ({:.0f} ms), source={}, model={}, base_url={}, empty_chunks_before_content={}",
                        first_content_latency,
                        first_content_latency * 1000,
                        call_source,
                        self.model,
                        self.base_url,
                        chunks_before_first_content,
                    )
                    first_content_logged = True
                elif not first_content_logged:
                    chunks_before_first_content += 1
                yield content

            # If stream ends while still in a tool call, make sure to yield the tool call
            if in_tool_call and accumulated_tool_calls:
                logger.info(f"Final tool call at stream end: {accumulated_tool_calls}")

                # Create a ToolCallObject instance from a dictionary using the from_dict method.
                complete_tool_calls = [
                    ToolCallObject.from_dict(tool_data)
                    for tool_data in accumulated_tool_calls.values()
                ]

                yield complete_tool_calls

        except APIConnectionError as e:
            logger.error(
                f"Error calling the chat endpoint: Connection error. Failed to connect to the LLM API. \nCheck the configurations and the reachability of the LLM backend. \nSee the logs for details. \nTroubleshooting with documentation: https://open-llm-vtuber.github.io/docs/faq#%E9%81%87%E5%88%B0-error-calling-the-chat-endpoint-%E9%94%99%E8%AF%AF%E6%80%8E%E4%B9%88%E5%8A%9E \n{e.__cause__}"
            )
            yield "Error calling the chat endpoint: Connection error. Failed to connect to the LLM API. Check the configurations and the reachability of the LLM backend. See the logs for details. Troubleshooting with documentation: [https://open-llm-vtuber.github.io/docs/faq#%E9%81%87%E5%88%B0-error-calling-the-chat-endpoint-%E9%94%99%E8%AF%AF%E6%80%8E%E4%B9%88%E5%8A%9E]"

        except RateLimitError as e:
            logger.error(
                f"Error calling the chat endpoint: Rate limit exceeded: {e.response}"
            )
            yield "Error calling the chat endpoint: Rate limit exceeded. Please try again later. See the logs for details."

        except asyncio.TimeoutError:
            logger.error(
                "Error calling the chat endpoint: timed out waiting for LLM response. "
                "Base URL: {}, Model: {}",
                self.base_url,
                self.model,
            )
            yield "Error calling the chat endpoint: timed out waiting for LLM response. Please check the LLM backend."

        except APIError as e:
            if "does not support tools" in str(e):
                self.support_tools = False
                logger.warning(
                    f"{self.model} does not support tools. Disabling tool support."
                )
                yield "__API_NOT_SUPPORT_TOOLS__"
                return
            logger.error(f"LLM API: Error occurred: {e}")
            logger.info(f"Base URL: {self.base_url}")
            logger.info(f"Model: {self.model}")
            logger.info(f"Messages: {_messages_log_view(messages)}")
            logger.info(f"temperature: {self.temperature}")
            yield "Error calling the chat endpoint: Error occurred while generating response. See the logs for details."

        finally:
            # make sure the stream is properly closed
            # so when interrupted, no more tokens will being generated.
            if stream:
                logger.debug("Chat completion finished.")
                await stream.close()
                logger.debug("Stream closed.")
