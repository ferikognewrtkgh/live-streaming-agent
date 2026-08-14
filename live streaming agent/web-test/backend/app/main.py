import json
import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, time, timedelta, timezone
from time import perf_counter
from typing import Annotated

from elasticsearch import ConnectionError as ElasticsearchConnectionError
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from .config import get_settings
from .douyin_live import DouyinLiveManager
from .elasticsearch_store import (
    ElasticsearchStore,
    count_message_characters,
    count_text_characters,
    elasticsearch_lifespan,
)
from .llm_service import LLMService, format_knowledge_context
from .logging_setup import configure_logging, log_chat_latency_event
from .memory_prompt import (
    build_short_term_memory_input,
    build_short_term_memory_prompt,
)
from .model_config_store import ModelConfigStore
from .prompt_config_store import PromptConfigStore
from .schemas import (
    ChatRequest,
    ConversationCreate,
    ConversationOrderUpdate,
    ConversationRecord,
    ConversationSearchResult,
    ConversationUpdate,
    FrontendLatencyReport,
    KnowledgeHit,
    KnowledgeSearchRequest,
    LiveCaptureStart,
    LiveCaptureStatus,
    LiveLoginStatus,
    MessageCreate,
    MessagePage,
    MessageRecord,
    ModelConfig,
    ModelConfigResponse,
    ModelConfigSaveRequest,
    ModelConfigTestRequest,
    ModelConnectionTestResponse,
    PromptConfigResponse,
    PromptConfigUpdate,
    RewindResponse,
    ShortTermMemoryRecord,
    UserModelConfigUpdate,
    UsernameRequest,
    WorkspaceResponse,
)

configure_logging()
logger = logging.getLogger(__name__)
SHANGHAI_TIMEZONE = timezone(timedelta(hours=8))

settings = get_settings()
llm_service = LLMService(settings)
model_config_store = ModelConfigStore(settings)
prompt_config_store = PromptConfigStore(settings)


def ndjson_event(payload: dict) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, default=str, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with elasticsearch_lifespan(settings) as store:
        app.state.store = store
        app.state.douyin_live_manager = DouyinLiveManager(
            settings.douyin_profile_root
        )
        try:
            yield
        finally:
            await app.state.douyin_live_manager.stop_all()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_store(request: Request) -> ElasticsearchStore:
    return request.app.state.store


Store = Annotated[ElasticsearchStore, Depends(get_store)]


def next_default_conversation_title(
    conversations: list[ConversationRecord],
) -> str:
    highest_number = 0
    for conversation in conversations:
        match = re.fullmatch(r"新对话\s*(\d+)", conversation.title.strip())
        if match:
            highest_number = max(highest_number, int(match.group(1)))
    return f"新对话 {highest_number + 1}"


def next_short_term_memory_target(
    conversation: ConversationRecord,
    completed_rounds: int,
) -> int | None:
    target_round = conversation.memory_through_round + 10
    return target_round if completed_rounds >= target_round else None


async def resolve_short_term_memory_context(
    *,
    store: ElasticsearchStore,
    conversation: ConversationRecord,
    round_number: int,
    history: list[MessageRecord],
) -> tuple[str, int, list[MessageRecord]]:
    summary = ""
    through_round = 0
    maximum_memory_round = round_number - 10
    if maximum_memory_round < 10:
        return summary, through_round, history

    if (
        conversation.short_term_memory
        and conversation.memory_through_round <= maximum_memory_round
    ):
        summary = conversation.short_term_memory
        through_round = conversation.memory_through_round
    elif conversation.memory_through_round > maximum_memory_round:
        memory = await store.latest_short_term_memory_at_or_before(
            conversation.conversation_id,
            maximum_memory_round,
        )
        if memory is not None:
            summary = memory.summary
            through_round = memory.through_round

    if through_round <= 0:
        return summary, through_round, history
    filtered_history = [
        message
        for message in history
        if not (
            isinstance(message.metadata.get("round_number"), int)
            and message.metadata["round_number"] <= through_round
        )
    ]
    return summary, through_round, filtered_history


async def compress_short_term_memory(
    *,
    store: ElasticsearchStore,
    conversation_id: str,
    username: str,
    target_round: int,
    previous_summary: str,
    model_config: ModelConfig,
    trace_id: str,
) -> None:
    start_round = target_round - 9
    try:
        memory_messages = await store.list_messages_by_round_range(
            conversation_id,
            start_round,
            target_round,
        )
        memory_prompt = build_short_term_memory_prompt(
            settings.short_term_memory_prompt_path
        )
        memory_input = build_short_term_memory_input(
            previous_summary,
            memory_messages,
        )
        compressed = await llm_service.generate_reply(
            conversation_id=conversation_id,
            username=username,
            user_content=memory_input,
            history=[],
            system_prompt=memory_prompt,
            knowledge_context=[],
            model_config=model_config.model_copy(
                update={"web_search_enabled": False}
            ),
            trace_id=trace_id,
        )
        memory = await store.create_short_term_memory(
            conversation_id=conversation_id,
            username=username,
            compression_number=target_round // 10,
            through_round=target_round,
            summary=compressed.content,
        )
        if memory is None:
            log_chat_latency_event(
                {
                    "event": "backend_short_term_memory_discarded",
                    "timestamp": datetime.now(UTC).isoformat(),
                    "trace_id": trace_id,
                    "conversation_id": conversation_id,
                    "through_round": target_round,
                }
            )
            return
        log_chat_latency_event(
            {
                "event": "backend_short_term_memory_saved",
                "timestamp": datetime.now(UTC).isoformat(),
                "trace_id": trace_id,
                "conversation_id": conversation_id,
                "compression_number": target_round // 10,
                "through_round": target_round,
            }
        )
    except Exception:
        logger.exception("Short-term memory compression failed")
        try:
            await store.mark_memory_compression_failed(conversation_id)
        except Exception:
            logger.exception("Failed to mark short-term memory compression failure")


@app.exception_handler(ElasticsearchConnectionError)
async def elasticsearch_connection_error_handler(
    _request: Request,
    _error: ElasticsearchConnectionError,
):
    logger.exception("Elasticsearch request failed")
    return _service_unavailable_response()


def _service_unavailable_response():
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Elasticsearch is unavailable"},
    )


@app.get("/health")
async def health(store: Store) -> dict[str, str]:
    health_client = store.client.options(request_timeout=1, max_retries=0)
    is_ready = await health_client.ping()
    return {
        "status": "ok" if is_ready else "degraded",
        "elasticsearch": "connected" if is_ready else "unavailable",
    }


@app.post(
    f"{settings.api_prefix}/live/login/start",
    response_model=LiveLoginStatus,
)
async def start_live_login(
    request: Request,
) -> LiveLoginStatus:
    manager: DouyinLiveManager = request.app.state.douyin_live_manager
    try:
        result = await manager.start_login()
    except Exception as exc:
        logger.exception("Failed to prepare shared Douyin login")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    return LiveLoginStatus.model_validate(result)


@app.get(
    f"{settings.api_prefix}/live/login/status",
    response_model=LiveLoginStatus,
)
async def get_live_login_status(request: Request) -> LiveLoginStatus:
    manager: DouyinLiveManager = request.app.state.douyin_live_manager
    try:
        result = await manager.login_status()
    except Exception as exc:
        logger.exception("Failed to read shared Douyin login status")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    return LiveLoginStatus.model_validate(result)


@app.post(
    f"{settings.api_prefix}/live/login/finish",
    response_model=LiveLoginStatus,
)
async def finish_live_login(
    request: Request,
) -> LiveLoginStatus:
    manager: DouyinLiveManager = request.app.state.douyin_live_manager
    result = await manager.finish_login()
    return LiveLoginStatus.model_validate(result)


@app.post(
    f"{settings.api_prefix}/live/start",
    response_model=LiveCaptureStatus,
)
async def start_live_capture(
    payload: LiveCaptureStart,
    request: Request,
) -> LiveCaptureStatus:
    manager: DouyinLiveManager = request.app.state.douyin_live_manager
    try:
        session = await manager.start(payload.username, payload.room_id)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return LiveCaptureStatus(
        username=payload.username,
        room_id=session.web_rid,
        status="starting",
        message="正在连接直播间",
    )


@app.post(
    f"{settings.api_prefix}/live/stop",
    response_model=LiveCaptureStatus,
)
async def stop_live_capture(
    payload: UsernameRequest,
    request: Request,
) -> LiveCaptureStatus:
    manager: DouyinLiveManager = request.app.state.douyin_live_manager
    session = await manager.stop(payload.username)
    return LiveCaptureStatus(
        username=payload.username,
        room_id=session.web_rid if session else "",
        status="stopped",
        message="已停止抓取",
    )


@app.post(
    f"{settings.api_prefix}/live/release",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def release_live_capture(
    request: Request,
    username: Annotated[str, Query(min_length=1, max_length=80)],
) -> Response:
    normalized_username = " ".join(username.strip().split())
    manager: DouyinLiveManager = request.app.state.douyin_live_manager
    await manager.release(normalized_username)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get(f"{settings.api_prefix}/live/events")
async def stream_live_events(
    request: Request,
    username: Annotated[str, Query(min_length=1, max_length=80)],
    after: Annotated[int, Query(ge=0)] = 0,
) -> StreamingResponse:
    normalized_username = " ".join(username.strip().split())
    manager: DouyinLiveManager = request.app.state.douyin_live_manager
    session = manager.get(normalized_username)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="尚未开始抓取")

    async def generate() -> AsyncIterator[bytes]:
        async for event in session.stream(after):
            if await request.is_disconnected():
                break
            yield ndjson_event(event)

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get(f"{settings.api_prefix}/model-config", response_model=ModelConfigResponse)
async def get_model_config() -> ModelConfigResponse:
    return model_config_store.get_config()


@app.put(f"{settings.api_prefix}/model-config", response_model=ModelConfigResponse)
async def save_model_config(
    payload: ModelConfigSaveRequest,
) -> ModelConfigResponse:
    return model_config_store.save_config(payload)


@app.post(
    f"{settings.api_prefix}/model-config/test",
    response_model=ModelConnectionTestResponse,
)
async def test_model_config(
    payload: ModelConfigTestRequest,
) -> ModelConnectionTestResponse:
    runtime_config = model_config_store.resolve_test_config(payload)
    return await llm_service.test_connection(runtime_config)


@app.get(f"{settings.api_prefix}/prompt-config", response_model=PromptConfigResponse)
async def get_prompt_config(
    store: Store,
    username: Annotated[str, Query(min_length=1, max_length=80)],
) -> PromptConfigResponse:
    document = await store.get_prompt_config_document(username)
    return prompt_config_store.get_config(document)


@app.put(f"{settings.api_prefix}/prompt-config", response_model=PromptConfigResponse)
async def save_prompt_config(
    payload: PromptConfigUpdate,
    store: Store,
    username: Annotated[str, Query(min_length=1, max_length=80)],
) -> PromptConfigResponse:
    await store.resolve_user(username)
    document = await store.get_prompt_config_document(username)
    config, updated_document = prompt_config_store.save_config(payload, document)
    await store.save_prompt_config_document(username, updated_document)
    return config


@app.post(f"{settings.api_prefix}/users/resolve", response_model=WorkspaceResponse)
async def resolve_user(payload: UsernameRequest, store: Store) -> WorkspaceResponse:
    user = await store.resolve_user(payload.username)
    conversations = await store.list_conversations(user.username)
    messages = []
    if conversations:
        messages = (
            await store.list_messages(conversations[0].conversation_id, size=50)
        ).items
    return WorkspaceResponse(
        username=user.username,
        speaker_identity=user.speaker_identity,
        llm_config=ModelConfig(
            provider=user.model_provider or settings.llm_provider,
            model=user.model or settings.llm_model,
            web_search_enabled=user.web_search_enabled,
            web_search_forced=user.web_search_forced,
            web_search_max_tool_calls=user.web_search_max_tool_calls,
            web_search_result_limit=user.web_search_result_limit,
            temperature=user.temperature,
        ),
        provider_models=user.provider_models,
        provider_temperatures=user.provider_temperatures,
        provider_web_search_configs=user.provider_web_search_configs,
        conversations=conversations,
        messages=messages,
    )


@app.put(
    f"{settings.api_prefix}/users/model-config",
    response_model=ModelConfig,
)
async def save_user_model_config(
    payload: UserModelConfigUpdate,
    store: Store,
) -> ModelConfig:
    await store.resolve_user(payload.username)
    await store.update_user_model_config(
        payload.username,
        payload.provider,
        payload.model,
        payload.web_search_enabled,
        payload.web_search_forced,
        payload.web_search_max_tool_calls,
        payload.web_search_result_limit,
        payload.temperature,
    )
    return ModelConfig(
        provider=payload.provider,
        model=payload.model,
        web_search_enabled=payload.web_search_enabled,
        web_search_forced=payload.web_search_forced,
        web_search_max_tool_calls=payload.web_search_max_tool_calls,
        web_search_result_limit=payload.web_search_result_limit,
        temperature=payload.temperature,
    )


@app.get(
    f"{settings.api_prefix}/conversations",
    response_model=list[ConversationRecord],
)
async def list_conversations(
    store: Store,
    username: Annotated[str, Query(min_length=1, max_length=80)],
    include_archived: bool = False,
) -> list[ConversationRecord]:
    return await store.list_conversations(
        " ".join(username.strip().split()),
        include_archived=include_archived,
    )


@app.post(
    f"{settings.api_prefix}/conversations",
    response_model=ConversationRecord,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    payload: ConversationCreate,
    store: Store,
) -> ConversationRecord:
    title = payload.title
    if title == "新对话":
        history = await store.list_conversations(
            payload.username,
            include_archived=True,
            size=1000,
        )
        title = next_default_conversation_title(history)
    return await store.create_conversation(payload.username, title)


@app.put(
    f"{settings.api_prefix}/conversations/order",
    response_model=list[ConversationRecord],
)
async def reorder_conversations(
    payload: ConversationOrderUpdate,
    store: Store,
) -> list[ConversationRecord]:
    return await store.reorder_conversations(
        payload.username,
        payload.conversation_ids,
    )


@app.get(
    f"{settings.api_prefix}/conversations/{{conversation_id}}",
    response_model=ConversationRecord,
)
async def get_conversation(
    conversation_id: str,
    store: Store,
) -> ConversationRecord:
    return await store.get_conversation(conversation_id)


@app.patch(
    f"{settings.api_prefix}/conversations/{{conversation_id}}",
    response_model=ConversationRecord,
)
async def update_conversation(
    conversation_id: str,
    payload: ConversationUpdate,
    store: Store,
) -> ConversationRecord:
    if payload.title is None and payload.status is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="at least one field must be supplied",
        )
    return await store.update_conversation(conversation_id, payload)


@app.get(
    f"{settings.api_prefix}/conversations/{{conversation_id}}/messages",
    response_model=MessagePage,
)
async def list_messages(
    conversation_id: str,
    store: Store,
    before: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> MessagePage:
    await store.get_conversation(conversation_id)
    return await store.list_messages(conversation_id, before=before, size=limit)


@app.get(
    f"{settings.api_prefix}/users/{{username}}/performance",
    response_model=list[MessageRecord],
)
async def list_user_performance_messages(
    username: str,
    store: Store,
    limit: Annotated[int, Query(ge=1, le=500)] = 20,
    day: date | None = None,
) -> list[MessageRecord]:
    day_start: datetime | None = None
    day_end: datetime | None = None
    if day is not None:
        local_start = datetime.combine(
            day,
            time.min,
            tzinfo=SHANGHAI_TIMEZONE,
        )
        day_start = local_start.astimezone(UTC)
        day_end = (local_start + timedelta(days=1)).astimezone(UTC)
    return await store.list_user_performance_messages(
        username,
        size=limit,
        created_at_gte=day_start,
        created_at_lt=day_end,
    )


@app.get(
    f"{settings.api_prefix}/users/{{username}}/conversation-search",
    response_model=list[ConversationSearchResult],
)
async def search_user_conversations(
    username: str,
    store: Store,
    q: Annotated[str, Query(min_length=1, max_length=200)],
) -> list[ConversationSearchResult]:
    normalized_username = " ".join(username.strip().split())
    normalized_phrase = " ".join(q.strip().split())
    if not normalized_username or not normalized_phrase:
        return []
    return await store.search_conversation_messages(
        normalized_username,
        normalized_phrase,
    )


@app.get(
    f"{settings.api_prefix}/conversations/{{conversation_id}}/memories",
    response_model=list[ShortTermMemoryRecord],
)
async def list_short_term_memories(
    conversation_id: str,
    store: Store,
) -> list[ShortTermMemoryRecord]:
    return await store.list_short_term_memories(conversation_id)


@app.post(
    f"{settings.api_prefix}/conversations/{{conversation_id}}/messages",
    response_model=MessageRecord,
    status_code=status.HTTP_201_CREATED,
)
async def create_message(
    conversation_id: str,
    payload: MessageCreate,
    store: Store,
) -> MessageRecord:
    return await store.create_message(conversation_id, payload)


@app.post(
    f"{settings.api_prefix}/conversations/{{conversation_id}}/rewind",
    response_model=RewindResponse,
)
async def rewind_conversation(
    conversation_id: str,
    payload: UsernameRequest,
    store: Store,
) -> RewindResponse:
    message_ids = await store.rewind_last_turn(
        conversation_id,
        payload.username,
    )
    conversation = await store.get_conversation(conversation_id)
    return RewindResponse(
        message_ids=message_ids,
        deleted_count=len(message_ids),
        completed_rounds=conversation.completed_rounds,
        effective_char_count=conversation.effective_char_count,
        memory_compression_count=conversation.memory_compression_count,
        memory_through_round=conversation.memory_through_round,
        short_term_memory=conversation.short_term_memory,
        memory_status=conversation.memory_status,
        memory_target_round=conversation.memory_target_round,
    )


@app.post(
    f"{settings.api_prefix}/conversations/{{conversation_id}}/chat",
    response_class=StreamingResponse,
)
async def chat(
    conversation_id: str,
    payload: ChatRequest,
    store: Store,
) -> StreamingResponse:
    request_started = perf_counter()
    conversation = await store.get_conversation(conversation_id)
    if conversation.username != payload.username:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="username does not own this conversation",
        )
    await store.update_user_model_config(
        payload.username,
        payload.llm_config.provider,
        payload.llm_config.model,
        payload.llm_config.web_search_enabled,
        payload.llm_config.web_search_forced,
        payload.llm_config.web_search_max_tool_calls,
        payload.llm_config.web_search_result_limit,
        payload.llm_config.temperature,
    )
    round_number = conversation.completed_rounds + 1

    def log_stage(event: str, **details) -> None:
        log_chat_latency_event(
            {
                "event": event,
                "timestamp": datetime.now(UTC).isoformat(),
                "trace_id": payload.trace_id,
                "conversation_id": conversation_id,
                "backend_elapsed_ms": round(
                    (perf_counter() - request_started) * 1000,
                    2,
                ),
                **details,
            }
        )

    log_stage("backend_request_received")
    pending_user_message = MessageRecord(
        message_id=f"pending-{payload.trace_id}",
        conversation_id=conversation_id,
        username=payload.username,
        role="user",
        content=payload.content,
        metadata={"round_number": round_number},
        created_at=datetime.now(UTC),
    )
    history = (await store.list_messages(conversation_id, size=20)).items
    log_stage("backend_history_loaded", history_message_count=len(history))

    (
        context_memory_summary,
        context_memory_through_round,
        history,
    ) = await resolve_short_term_memory_context(
        store=store,
        conversation=conversation,
        round_number=round_number,
        history=history,
    )
    pending_effective_char_count_before_assistant = (
        conversation.effective_char_count
        + count_text_characters(pending_user_message.content)
    )
    log_stage(
        "backend_short_term_memory_context_ready",
        memory_available=bool(conversation.short_term_memory),
        memory_injected=bool(context_memory_summary),
        memory_through_round=context_memory_through_round,
    )

    knowledge_started = perf_counter()
    knowledge_sources: list[dict] = []
    knowledge_ids: list[str] = []
    if payload.knowledge_enabled:
        try:
            desired_top_k = max(1, min(3, settings.knowledge_top_k))
            search_top_k = min(
                20,
                max(desired_top_k * 4, desired_top_k + 6),
            )
            knowledge_query = payload.knowledge_query or payload.content
            if payload.speaker_identity:
                attributed_prefix = f"{payload.speaker_identity}：“"
                if not (
                    knowledge_query.startswith(attributed_prefix)
                    and knowledge_query.endswith("”")
                ):
                    knowledge_query = (
                        f"{payload.speaker_identity}：“{knowledge_query}”"
                    )
            knowledge_hits = await store.search_knowledge(
                knowledge_query,
                size=search_top_k,
            )
            last_injected_round_by_id: dict[str, int] = {}
            for message in history:
                if message.role != "assistant":
                    continue
                injected_round = message.metadata.get("round_number")
                injected_ids = message.metadata.get("knowledge_ids")
                if not isinstance(injected_round, int) or not isinstance(
                    injected_ids, list
                ):
                    continue
                for knowledge_id in injected_ids:
                    last_injected_round_by_id[str(knowledge_id)] = (
                        injected_round
                    )
            min_interval = max(
                1,
                settings.knowledge_min_injection_interval_rounds,
            )
            filtered_hits = [
                hit
                for hit in knowledge_hits
                if (
                    hit.document_id not in last_injected_round_by_id
                    or round_number
                    - last_injected_round_by_id[hit.document_id]
                    >= min_interval
                )
            ][:desired_top_k]
            knowledge_sources = [hit.source for hit in filtered_hits]
            knowledge_ids = [hit.document_id for hit in filtered_hits]
        except HTTPException as error:
            if error.status_code != status.HTTP_503_SERVICE_UNAVAILABLE:
                raise
    knowledge_duration_ms = round(
        (perf_counter() - knowledge_started) * 1000,
        2,
    )
    log_stage(
        "backend_knowledge_ready",
        knowledge_hit_count=len(knowledge_sources),
        knowledge_duration_ms=knowledge_duration_ms,
    )
    runtime_model_config = model_config_store.resolve_runtime_config(
        payload.llm_config,
    )
    knowledge_injected_context = format_knowledge_context(knowledge_sources)
    model_user_content = payload.content
    if knowledge_injected_context:
        model_user_content = (
            f"{knowledge_injected_context}\n\n"
            f"[用户当前提问]\n{payload.content}"
        )
    compression_job: dict[str, int | str] = {}
    effective_system_prompt = payload.system_prompt
    if context_memory_summary:
        effective_system_prompt = (
            f"{effective_system_prompt}\n\n"
            "以下是已总结的短期记忆。"
            "请自然参考，不要向用户复述记忆管理过程：\n"
            f"{context_memory_summary}"
        ).strip()

    async def stream_events() -> AsyncIterator[bytes]:
        log_stage("backend_stream_started")
        yield ndjson_event(
            {
                "type": "start",
                "trace_id": payload.trace_id,
                "user_message": pending_user_message.model_dump(mode="json"),
                "knowledge_hit_count": len(knowledge_sources),
                "knowledge_injected_context": knowledge_injected_context,
                "round_number": round_number,
                "knowledge_duration_ms": knowledge_duration_ms,
            }
        )

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        web_sources: list[dict] = []
        reply_mode = "model"
        first_chunk_forwarded = False
        first_sentence_forwarded = False
        model_first_token_ms = 0.0
        model_first_sentence_ms = 0.0
        model_complete_ms = 0.0
        web_search_duration_ms = 0.0
        saved_messages: list[MessageRecord] = []
        turn_completed = False

        async def discard_incomplete_turn() -> None:
            if turn_completed:
                return
            for saved_message in reversed(saved_messages):
                try:
                    await store.discard_uncompleted_message(
                        conversation,
                        saved_message,
                    )
                except Exception:
                    logger.exception("Failed to discard incomplete chat turn")

        try:
            log_stage("backend_model_stream_started")
            model_started = perf_counter()
            async for chunk in llm_service.stream_reply(
                conversation_id=conversation_id,
                username=payload.username,
                user_content=model_user_content,
                history=history,
                system_prompt=effective_system_prompt,
                knowledge_context=[],
                model_config=runtime_model_config,
                trace_id=payload.trace_id,
            ):
                chunk_reasoning_content = getattr(
                    chunk,
                    "reasoning_content",
                    "",
                )
                if chunk_reasoning_content:
                    reasoning_parts.append(chunk_reasoning_content)
                chunk_web_search_duration_ms = getattr(
                    chunk,
                    "web_search_duration_ms",
                    None,
                )
                if isinstance(chunk_web_search_duration_ms, (int, float)):
                    web_search_duration_ms = round(
                        float(chunk_web_search_duration_ms),
                        2,
                    )
                    yield ndjson_event(
                        {
                            "type": "metric",
                            "metrics": {
                                "web_search_duration_ms": (
                                    web_search_duration_ms
                                ),
                            },
                        }
                    )
                chunk_web_sources = getattr(chunk, "web_sources", ())
                if chunk_web_sources:
                    existing_source_urls = {
                        str(source.get("url") or "") for source in web_sources
                    }
                    sources_changed = False
                    for source in chunk_web_sources:
                        source_url = str(source.get("url") or "")
                        if (
                            not source_url
                            or source_url in existing_source_urls
                        ):
                            continue
                        web_sources.append(dict(source))
                        existing_source_urls.add(source_url)
                        sources_changed = True
                    if sources_changed:
                        yield ndjson_event(
                            {
                                "type": "web_search_sources",
                                "sources": web_sources,
                            }
                        )
                if not chunk.content:
                    continue
                content_parts.append(chunk.content)
                reply_mode = chunk.mode
                if not first_chunk_forwarded:
                    first_chunk_forwarded = True
                    model_first_token_ms = round(
                        (perf_counter() - model_started) * 1000,
                        2,
                    )
                    log_stage("backend_first_chunk_forwarded")
                    yield ndjson_event(
                        {
                            "type": "metric",
                            "metrics": {
                                "model_first_token_ms": model_first_token_ms,
                            },
                        }
                    )
                yield ndjson_event(
                    {
                        "type": "delta",
                        "content": chunk.content,
                    }
                )
                if (
                    not first_sentence_forwarded
                    and re.search(r"[。！？!?；;\n]", "".join(content_parts))
                ):
                    first_sentence_forwarded = True
                    model_first_sentence_ms = round(
                        (perf_counter() - model_started) * 1000,
                        2,
                    )
                    yield ndjson_event(
                        {
                            "type": "metric",
                            "metrics": {
                                "model_first_sentence_ms": (
                                    model_first_sentence_ms
                                ),
                            },
                        }
                    )

            model_complete_ms = round(
                (perf_counter() - model_started) * 1000,
                2,
            )
            if not first_sentence_forwarded:
                model_first_sentence_ms = model_complete_ms
                yield ndjson_event(
                    {
                        "type": "metric",
                        "metrics": {
                            "model_first_sentence_ms": model_first_sentence_ms,
                        },
                    }
                )

            assistant_content = "".join(content_parts)
            if not assistant_content.strip():
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail="model returned an empty response",
                )
            saved_user_message = await store.create_message(
                conversation_id,
                MessageCreate(
                    username=payload.username,
                    role="user",
                    content=payload.content,
                    metadata={"round_number": round_number},
                ),
            )
            saved_messages.append(saved_user_message)
            log_stage("backend_user_message_saved")
            effective_char_count_before_assistant = (
                pending_effective_char_count_before_assistant
            )
            if context_memory_summary:
                effective_char_count_before_assistant = (
                    await store.activate_short_term_memory_character_count(
                        conversation,
                        through_round=context_memory_through_round,
                        summary=context_memory_summary,
                        pending_message=saved_user_message,
                    )
                )
            assistant_message = await store.create_message(
                conversation_id,
                MessageCreate(
                    username=payload.username,
                    role="assistant",
                    content=assistant_content,
                    reasoning_content="".join(reasoning_parts),
                    metadata={
                        "provider": runtime_model_config.provider,
                        "model": runtime_model_config.model,
                        "temperature": LLMService._request_temperature(
                            runtime_model_config.provider,
                            runtime_model_config.model,
                            runtime_model_config.temperature,
                        ),
                        "reply_mode": reply_mode,
                        "knowledge_hit_count": len(knowledge_sources),
                        "knowledge_ids": knowledge_ids,
                        "knowledge_injected_context": knowledge_injected_context,
                        "web_search_enabled": (
                            runtime_model_config.web_search_enabled
                        ),
                        "web_search_max_tool_calls": (
                            runtime_model_config.web_search_max_tool_calls
                        ),
                        "web_search_result_limit": (
                            runtime_model_config.web_search_result_limit
                        ),
                        "web_search_forced": (
                            runtime_model_config.web_search_forced
                        ),
                        "web_search_strategy": (
                            (
                                "required"
                                if runtime_model_config.web_search_forced
                                else "auto"
                            )
                            if runtime_model_config.provider == "qwen"
                            and runtime_model_config.web_search_enabled
                            else None
                        ),
                        "web_search_sources": web_sources,
                        "round_number": round_number,
                        "performance_metrics": {
                            "knowledge_duration_ms": knowledge_duration_ms,
                            "web_search_duration_ms": web_search_duration_ms,
                            "model_first_token_ms": model_first_token_ms,
                            "model_first_sentence_ms": model_first_sentence_ms,
                            "model_complete_ms": model_complete_ms,
                        },
                    },
                ),
            )
            saved_messages.append(assistant_message)
            log_stage("backend_assistant_message_saved")
            completed_rounds = await store.advance_completed_rounds(
                conversation_id,
                [saved_user_message.message_id, assistant_message.message_id],
            )
            turn_completed = True
            log_stage("backend_round_completed", completed_rounds=completed_rounds)

            through_round = next_short_term_memory_target(
                conversation,
                completed_rounds,
            )
            if through_round is not None:
                try:
                    should_compress = await store.try_mark_memory_compressing(
                        conversation_id,
                        through_round,
                    )
                    if should_compress:
                        compression_job.update(
                            {
                                "target_round": through_round,
                                "previous_summary": conversation.short_term_memory,
                            }
                        )
                        log_stage(
                            "backend_short_term_memory_scheduled",
                            through_round=through_round,
                        )
                except Exception:
                    logger.exception("Failed to schedule short-term memory")
            if (
                payload.save_speaker_identity
                and payload.speaker_identity
            ):
                try:
                    await store.update_user_speaker_identity(
                        payload.username,
                        payload.speaker_identity,
                    )
                    log_stage("backend_speaker_identity_saved")
                except Exception:
                    logger.exception("Failed to save speaker identity")
            yield ndjson_event(
                {
                    "type": "done",
                    "trace_id": payload.trace_id,
                    "user_message": saved_user_message.model_dump(mode="json"),
                    "assistant_message": assistant_message.model_dump(mode="json"),
                    "mode": reply_mode,
                    "knowledge_hit_count": len(knowledge_sources),
                    "completed_rounds": completed_rounds,
                    "effective_char_count": (
                        effective_char_count_before_assistant
                        + count_message_characters(
                            assistant_message.content,
                            assistant_message.metadata,
                        )
                    ),
                    "memory_status": (
                        "compressing" if compression_job else conversation.memory_status
                    ),
                    "memory_through_round": conversation.memory_through_round,
                    "memory_target_round": (
                        int(compression_job["target_round"])
                        if compression_job
                        else conversation.memory_target_round
                    ),
                    "performance_metrics": {
                        "knowledge_duration_ms": knowledge_duration_ms,
                        "web_search_duration_ms": web_search_duration_ms,
                        "model_first_token_ms": model_first_token_ms,
                        "model_first_sentence_ms": model_first_sentence_ms,
                        "model_complete_ms": model_complete_ms,
                    },
                }
            )
            log_stage("backend_stream_completed")
        except HTTPException as error:
            await discard_incomplete_turn()
            log_stage(
                "backend_stream_failed",
                status=error.status_code,
                error=str(error.detail),
            )
            yield ndjson_event(
                {
                    "type": "error",
                    "status": error.status_code,
                    "message": str(error.detail),
                }
            )
        except Exception:
            await discard_incomplete_turn()
            logger.exception("Streaming chat failed")
            log_stage(
                "backend_stream_failed",
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                error="streaming chat failed",
            )
            yield ndjson_event(
                {
                    "type": "error",
                    "status": status.HTTP_500_INTERNAL_SERVER_ERROR,
                    "message": "streaming chat failed",
                }
            )

    async def run_compression_job() -> None:
        target_round = compression_job.get("target_round")
        if not isinstance(target_round, int):
            return
        previous_summary = compression_job.get("previous_summary")
        await compress_short_term_memory(
            store=store,
            conversation_id=conversation_id,
            username=payload.username,
            target_round=target_round,
            previous_summary=(
                previous_summary if isinstance(previous_summary, str) else ""
            ),
            model_config=runtime_model_config,
            trace_id=payload.trace_id,
        )

    response = StreamingResponse(
        stream_events(),
        media_type="application/x-ndjson",
        background=BackgroundTask(run_compression_job),
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
    log_stage("backend_stream_response_ready")
    return response


@app.post(
    f"{settings.api_prefix}/telemetry/chat-latency",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def record_frontend_chat_latency(
    payload: FrontendLatencyReport,
) -> Response:
    log_chat_latency_event(
        {
            "event": "frontend_first_paint",
            "timestamp": datetime.now(UTC).isoformat(),
            **payload.model_dump(mode="json"),
        }
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post(
    f"{settings.api_prefix}/knowledge/search",
    response_model=list[KnowledgeHit],
)
async def search_knowledge(
    payload: KnowledgeSearchRequest,
    store: Store,
) -> list[KnowledgeHit]:
    return await store.search_knowledge(payload.query, payload.size)
