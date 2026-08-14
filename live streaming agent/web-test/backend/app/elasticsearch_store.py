import asyncio
import hashlib
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from elasticsearch import AsyncElasticsearch, ConflictError, NotFoundError
from elasticsearch.helpers import async_scan
from fastapi import HTTPException, status

from .config import Settings
from .schemas import (
    ConversationRecord,
    ConversationSearchMatch,
    ConversationSearchResult,
    ConversationUpdate,
    KnowledgeHit,
    MessageCreate,
    MessagePage,
    MessageRecord,
    ShortTermMemoryRecord,
    UserRecord,
)

logger = logging.getLogger(__name__)
KNOWLEDGE_KEYWORD_ANALYZER = "ik_max_word"


class BGEEmbedder:
    def __init__(self, model_name_or_path: str) -> None:
        from sentence_transformers import SentenceTransformer

        resolved_model = self._resolve_local_model(model_name_or_path)
        self.model = SentenceTransformer(
            str(resolved_model),
            local_files_only=resolved_model.exists(),
        )

    @staticmethod
    def _resolve_local_model(model_name_or_path: str) -> Path:
        configured_path = Path(model_name_or_path)
        if configured_path.exists():
            return configured_path

        if model_name_or_path != "BAAI/bge-small-zh-v1.5":
            return configured_path

        project_root = Path(__file__).resolve().parents[3]
        snapshots_root = (
            project_root
            / "models"
            / "hub"
            / "models--BAAI--bge-small-zh-v1.5"
            / "snapshots"
        )
        snapshots = sorted(
            path for path in snapshots_root.glob("*") if path.is_dir()
        )
        if not snapshots:
            return configured_path
        return snapshots[-1]

    def encode_query(self, text: str, query_prefix: str) -> list[float]:
        query_text = f"{query_prefix}{text}" if query_prefix else text
        vector = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector[0].tolist()


INDEX_MAPPINGS: dict[str, dict[str, Any]] = {
    "users": {
        "properties": {
            "username": {"type": "keyword"},
            "speaker_identity": {"type": "keyword"},
            "model_provider": {"type": "keyword"},
            "model": {"type": "keyword"},
            "provider_models": {"type": "flattened"},
            "provider_temperatures": {"type": "flattened"},
            "provider_web_search_configs": {"type": "flattened"},
            "web_search_enabled": {"type": "boolean"},
            "web_search_forced": {"type": "boolean"},
            "web_search_max_tool_calls": {"type": "integer"},
            "web_search_result_limit": {"type": "integer"},
            "temperature": {"type": "float"},
            "created_at": {"type": "date"},
            "last_seen_at": {"type": "date"},
        }
    },
    "conversations": {
        "properties": {
            "conversation_id": {"type": "keyword"},
            "username": {"type": "keyword"},
            "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
            "status": {"type": "keyword"},
            "sort_order": {"type": "long"},
            "completed_rounds": {"type": "integer"},
            "last_turn_message_ids": {"type": "keyword"},
            "effective_char_count": {"type": "integer"},
            "effective_char_count_version": {"type": "integer"},
            "effective_memory_through_round": {"type": "integer"},
            "memory_compression_count": {"type": "integer"},
            "memory_through_round": {"type": "integer"},
            "short_term_memory": {"type": "text"},
            "memory_status": {"type": "keyword"},
            "memory_target_round": {"type": "integer"},
            "created_at": {"type": "date"},
            "updated_at": {"type": "date"},
        }
    },
    "messages": {
        "properties": {
            "message_id": {"type": "keyword"},
            "conversation_id": {"type": "keyword"},
            "username": {"type": "keyword"},
            "role": {"type": "keyword"},
            "content": {"type": "text"},
            "reasoning_content": {"type": "text", "index": False},
            "emotion": {"type": "keyword"},
            "metadata": {"type": "object", "enabled": True},
            "created_at": {"type": "date"},
        }
    },
    "memories": {
        "properties": {
            "memory_id": {"type": "keyword"},
            "conversation_id": {"type": "keyword"},
            "username": {"type": "keyword"},
            "compression_number": {"type": "integer"},
            "through_round": {"type": "integer"},
            "summary": {"type": "text"},
            "created_at": {"type": "date"},
        }
    },
    "prompt_configs": {
        "properties": {
            "username": {"type": "keyword"},
            "active_version": {"type": "keyword"},
            "active_speaker_version": {"type": "keyword"},
            "speaker_versions": {
                "type": "nested",
                "properties": {
                    "version": {"type": "keyword"},
                    "title": {"type": "keyword"},
                    "content": {"type": "text"},
                    "speaker_identity": {"type": "keyword"},
                },
            },
            "updated_at": {"type": "date"},
        }
    },
}


def utc_now() -> datetime:
    return datetime.now(UTC)


def user_document_id(username: str) -> str:
    return hashlib.sha256(username.encode("utf-8")).hexdigest()


def count_text_characters(text: str) -> int:
    return sum(not character.isspace() for character in text)


def count_message_characters(
    content: str,
    metadata: dict[str, Any] | None = None,
) -> int:
    message_metadata = metadata or {}
    injected_context = message_metadata.get("knowledge_injected_context")
    return count_text_characters(content) + (
        count_text_characters(injected_context)
        if isinstance(injected_context, str)
        else 0
    )


class ElasticsearchStore:
    def __init__(self, client: AsyncElasticsearch, settings: Settings) -> None:
        self.client = client
        self.settings = settings
        self._knowledge_embedder: BGEEmbedder | None = None
        self._knowledge_embedder_lock = asyncio.Lock()

    async def ensure_indices(self) -> None:
        indices = {
            self.settings.users_index: INDEX_MAPPINGS["users"],
            self.settings.conversations_index: INDEX_MAPPINGS["conversations"],
            self.settings.messages_index: INDEX_MAPPINGS["messages"],
            self.settings.memories_index: INDEX_MAPPINGS["memories"],
            self.settings.prompt_configs_index: INDEX_MAPPINGS["prompt_configs"],
        }
        for index_name, mapping in indices.items():
            if not await self.client.indices.exists(index=index_name):
                await self.client.indices.create(
                    index=index_name,
                    mappings=mapping,
                    settings={"number_of_shards": 1, "number_of_replicas": 0},
                )
            else:
                current = await self.client.indices.get_mapping(index=index_name)
                existing_properties = (
                    current.get(index_name, {})
                    .get("mappings", {})
                    .get("properties", {})
                )
                missing_properties = {
                    field: definition
                    for field, definition in mapping["properties"].items()
                    if field not in existing_properties
                }
                if missing_properties:
                    await self.client.indices.put_mapping(
                        index=index_name,
                        properties=missing_properties,
                    )

    async def resolve_user(self, username: str) -> UserRecord:
        now = utc_now()
        document_id = user_document_id(username)
        record = {
            "username": username,
            "speaker_identity": "莱叔",
            "model_provider": self.settings.llm_provider,
            "model": self.settings.llm_model,
            "provider_models": {
                self.settings.llm_provider: self.settings.llm_model
            },
            "provider_temperatures": {self.settings.llm_provider: 0.8},
            "provider_web_search_configs": {
                self.settings.llm_provider: {
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
            "created_at": now.isoformat(),
            "last_seen_at": now.isoformat(),
        }
        try:
            await self.client.create(
                index=self.settings.users_index,
                id=document_id,
                document=record,
                refresh=False,
            )
        except ConflictError:
            await self.client.update(
                index=self.settings.users_index,
                id=document_id,
                doc={"last_seen_at": now.isoformat()},
                refresh=False,
            )
            existing = await self.client.get(
                index=self.settings.users_index,
                id=document_id,
            )
            record = existing["_source"]
            record["last_seen_at"] = now.isoformat()
        if not record.get("provider_models"):
            provider = str(
                record.get("model_provider") or self.settings.llm_provider
            )
            model = str(record.get("model") or self.settings.llm_model)
            record["provider_models"] = {provider: model}
        if not record.get("provider_temperatures"):
            provider = str(
                record.get("model_provider") or self.settings.llm_provider
            )
            record["provider_temperatures"] = {
                provider: float(record.get("temperature", 0.8))
            }
        if not record.get("provider_web_search_configs"):
            provider = str(
                record.get("model_provider") or self.settings.llm_provider
            )
            record["provider_web_search_configs"] = {
                provider: {
                    "enabled": bool(record.get("web_search_enabled", False)),
                    "forced": bool(record.get("web_search_forced", False)),
                    "max_tool_calls": int(
                        record.get("web_search_max_tool_calls", 1)
                    ),
                    "result_limit": int(
                        record.get("web_search_result_limit", 3)
                    ),
                }
            }
        return UserRecord.model_validate(record)

    async def update_user_speaker_identity(
        self,
        username: str,
        speaker_identity: str,
    ) -> None:
        await self.client.update(
            index=self.settings.users_index,
            id=user_document_id(username),
            doc={"speaker_identity": speaker_identity},
            refresh=False,
        )

    async def update_user_model_config(
        self,
        username: str,
        provider: str,
        model: str,
        web_search_enabled: bool = False,
        web_search_forced: bool = False,
        web_search_max_tool_calls: int = 1,
        web_search_result_limit: int = 3,
        temperature: float = 0.8,
    ) -> None:
        await self.client.update(
            index=self.settings.users_index,
            id=user_document_id(username),
            doc={
                "model_provider": provider,
                "model": model,
                "provider_models": {provider: model},
                "provider_temperatures": {provider: temperature},
                "provider_web_search_configs": {
                    provider: {
                        "enabled": web_search_enabled,
                        "forced": web_search_forced,
                        "max_tool_calls": web_search_max_tool_calls,
                        "result_limit": web_search_result_limit,
                    }
                },
                "web_search_enabled": web_search_enabled,
                "web_search_forced": web_search_forced,
                "web_search_max_tool_calls": web_search_max_tool_calls,
                "web_search_result_limit": web_search_result_limit,
                "temperature": temperature,
            },
            refresh=False,
        )

    async def list_conversations(
        self,
        username: str,
        *,
        include_archived: bool = False,
        size: int = 100,
    ) -> list[ConversationRecord]:
        filters: list[dict[str, Any]] = [{"term": {"username": username}}]
        if not include_archived:
            filters.append({"term": {"status": "active"}})
        result = await self.client.search(
            index=self.settings.conversations_index,
            query={"bool": {"filter": filters}},
            sort=[
                {"sort_order": {"order": "asc", "missing": "_last"}},
                {"updated_at": {"order": "desc"}},
            ],
            size=size,
        )
        return list(
            await asyncio.gather(
                *(
                    self._hydrate_conversation_character_count(
                        hit["_id"],
                        hit["_source"],
                    )
                    for hit in result["hits"]["hits"]
                )
            )
        )

    async def create_conversation(
        self,
        username: str,
        title: str,
    ) -> ConversationRecord:
        now = utc_now()
        conversation_id = str(uuid4())
        record = ConversationRecord(
            conversation_id=conversation_id,
            username=username,
            title=title.strip(),
            status="active",
            sort_order=-int(now.timestamp() * 1_000_000),
            completed_rounds=0,
            effective_char_count=0,
            effective_char_count_version=5,
            effective_memory_through_round=0,
            memory_compression_count=0,
            memory_through_round=0,
            short_term_memory="",
            memory_status="idle",
            memory_target_round=0,
            created_at=now,
            updated_at=now,
        )
        await self.client.index(
            index=self.settings.conversations_index,
            id=conversation_id,
            document=record.model_dump(mode="json"),
            refresh=False,
        )
        return record

    async def reorder_conversations(
        self,
        username: str,
        conversation_ids: list[str],
    ) -> list[ConversationRecord]:
        current = await self.list_conversations(username, size=100)
        current_ids = {conversation.conversation_id for conversation in current}
        requested_ids = set(conversation_ids)
        if (
            len(conversation_ids) != len(requested_ids)
            or requested_ids != current_ids
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="conversation order is stale",
            )

        await asyncio.gather(
            *(
                self.client.update(
                    index=self.settings.conversations_index,
                    id=conversation_id,
                    doc={"sort_order": index},
                    refresh=False,
                )
                for index, conversation_id in enumerate(conversation_ids)
            )
        )
        by_id = {
            conversation.conversation_id: conversation for conversation in current
        }
        return [
            by_id[conversation_id].model_copy(
                update={"sort_order": index},
            )
            for index, conversation_id in enumerate(conversation_ids)
        ]

    async def get_conversation(self, conversation_id: str) -> ConversationRecord:
        try:
            result = await self.client.get(
                index=self.settings.conversations_index,
                id=conversation_id,
            )
        except NotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="conversation not found",
            ) from error
        return await self._hydrate_conversation_character_count(
            conversation_id,
            result["_source"],
        )

    async def _hydrate_conversation_character_count(
        self,
        conversation_id: str,
        source: dict[str, Any],
    ) -> ConversationRecord:
        if source.get("effective_char_count_version") == 5:
            return ConversationRecord.model_validate(source)

        completed_rounds = int(source.get("completed_rounds") or 0)
        memory_through_round = int(source.get("memory_through_round") or 0)
        maximum_effective_memory_round = min(
            max(completed_rounds - 10, 0),
            memory_through_round,
        )
        effective_memory = await self.latest_short_term_memory_at_or_before(
            conversation_id,
            maximum_effective_memory_round,
        )
        effective_memory_through_round = (
            effective_memory.through_round if effective_memory else 0
        )
        effective_char_count = count_text_characters(
            effective_memory.summary if effective_memory else ""
        )
        filters: list[dict[str, Any]] = [
            {"term": {"conversation_id": conversation_id}}
        ]
        if effective_memory_through_round:
            filters.append(
                {
                    "range": {
                        "metadata.round_number": {
                            "gt": effective_memory_through_round
                        }
                    }
                }
            )
        async for hit in async_scan(
            self.client,
            index=self.settings.messages_index,
            query={
                "_source": [
                    "content",
                    "metadata.knowledge_injected_context",
                ],
                "query": {
                    "bool": {
                        "filter": filters,
                        "must_not": [{"term": {"metadata.deleted": True}}],
                    }
                },
            },
        ):
            message_source = hit.get("_source") or {}
            effective_char_count += count_message_characters(
                str(message_source.get("content") or ""),
                message_source.get("metadata"),
            )

        source = {
            **source,
            "effective_char_count": effective_char_count,
            "effective_char_count_version": 5,
            "effective_memory_through_round": (
                effective_memory_through_round
            ),
        }
        await self.client.update(
            index=self.settings.conversations_index,
            id=conversation_id,
            doc={
                "effective_char_count": effective_char_count,
                "effective_char_count_version": 5,
                "effective_memory_through_round": (
                    effective_memory_through_round
                ),
            },
            refresh=False,
        )
        return ConversationRecord.model_validate(source)

    async def get_prompt_config_document(self, username: str) -> dict[str, Any]:
        try:
            result = await self.client.get(
                index=self.settings.prompt_configs_index,
                id=user_document_id(username),
            )
        except NotFoundError:
            return {}
        return dict(result["_source"])

    async def save_prompt_config_document(
        self,
        username: str,
        document: dict[str, Any],
    ) -> None:
        await self.client.index(
            index=self.settings.prompt_configs_index,
            id=user_document_id(username),
            document={**document, "username": username},
            refresh=False,
        )

    async def update_conversation(
        self,
        conversation_id: str,
        update: ConversationUpdate,
    ) -> ConversationRecord:
        current = await self.get_conversation(conversation_id)
        changes = update.model_dump(exclude_none=True)
        changes["updated_at"] = utc_now().isoformat()
        await self.client.update(
            index=self.settings.conversations_index,
            id=conversation_id,
            doc=changes,
            refresh=False,
        )
        merged = current.model_dump(mode="json")
        merged.update(changes)
        return ConversationRecord.model_validate(merged)

    async def list_messages(
        self,
        conversation_id: str,
        *,
        before: datetime | None = None,
        size: int = 50,
    ) -> MessagePage:
        filters: list[dict[str, Any]] = [
            {"term": {"conversation_id": conversation_id}}
        ]
        if before:
            filters.append({"range": {"created_at": {"lt": before.isoformat()}}})
        result = await self.client.search(
            index=self.settings.messages_index,
            query={
                "bool": {
                    "filter": filters,
                    "must_not": [{"term": {"metadata.deleted": True}}],
                }
            },
            sort=[{"created_at": {"order": "desc"}}],
            size=size + 1,
        )
        hits = result["hits"]["hits"]
        has_more = len(hits) > size
        selected = hits[:size]
        records = [
            MessageRecord.model_validate(hit["_source"]) for hit in reversed(selected)
        ]
        next_before = records[0].created_at if has_more and records else None
        return MessagePage(items=records, next_before=next_before)

    async def search_conversation_messages(
        self,
        username: str,
        phrase: str,
    ) -> list[ConversationSearchResult]:
        normalized_phrase = " ".join(phrase.strip().split())
        if not normalized_phrase:
            return []

        conversation_result = await self.client.search(
            index=self.settings.conversations_index,
            query={
                "bool": {
                    "filter": [
                        {"term": {"username": username}},
                        {"term": {"status": "active"}},
                    ]
                }
            },
            sort=[
                {"sort_order": {"order": "asc", "missing": "_last"}},
                {"updated_at": {"order": "desc"}},
            ],
            size=1000,
            source=["conversation_id", "title"],
        )
        conversations = [
            (
                str(hit["_source"]["conversation_id"]),
                str(hit["_source"].get("title") or "未命名对话"),
            )
            for hit in conversation_result["hits"]["hits"]
        ]
        if not conversations:
            return []

        conversation_ids = [conversation_id for conversation_id, _ in conversations]
        matches_by_conversation: dict[str, list[ConversationSearchMatch]] = {}
        async for hit in async_scan(
            self.client,
            index=self.settings.messages_index,
            query={
                "_source": [
                    "message_id",
                    "conversation_id",
                    "role",
                    "content",
                    "metadata.knowledge_injected_context",
                    "metadata.web_search_sources",
                    "created_at",
                ],
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"username": username}},
                            {"terms": {"conversation_id": conversation_ids}},
                        ],
                        "must": [
                            {
                                "multi_match": {
                                    "query": normalized_phrase,
                                    "type": "phrase",
                                    "fields": [
                                        "content",
                                        "metadata.knowledge_injected_context",
                                        "metadata.web_search_sources.title",
                                        "metadata.web_search_sources.snippet",
                                        "metadata.web_search_sources.url",
                                    ],
                                }
                            },
                        ],
                        "must_not": [
                            {"term": {"metadata.deleted": True}}
                        ],
                    }
                },
            },
        ):
            source = hit["_source"]
            conversation_id = str(source["conversation_id"])
            matches_by_conversation.setdefault(conversation_id, []).extend(
                self._conversation_search_matches(
                    source,
                    normalized_phrase,
                )
            )
        results: list[ConversationSearchResult] = []
        for conversation_id, title in conversations:
            matches = matches_by_conversation.get(conversation_id, [])
            if not matches:
                continue
            matches.sort(key=lambda match: match.created_at, reverse=True)
            results.append(
                ConversationSearchResult(
                    conversation_id=conversation_id,
                    title=title,
                    match_count=len(matches),
                    matches=matches,
                )
            )
        return results

    @classmethod
    def _conversation_search_matches(
        cls,
        source: dict[str, Any],
        phrase: str,
    ) -> list[ConversationSearchMatch]:
        candidates: list[tuple[str, str]] = [
            ("message", str(source.get("content") or "")),
        ]
        metadata = source.get("metadata")
        if isinstance(metadata, dict):
            knowledge_context = metadata.get("knowledge_injected_context")
            if isinstance(knowledge_context, str):
                candidates.append(("knowledge", knowledge_context))

            web_sources = metadata.get("web_search_sources")
            if isinstance(web_sources, list):
                for web_source in web_sources:
                    if not isinstance(web_source, dict):
                        continue
                    source_text = " ".join(
                        str(web_source.get(field) or "").strip()
                        for field in ("title", "snippet", "url")
                    ).strip()
                    if source_text:
                        candidates.append(("web_search", source_text))

        normalized_phrase = phrase.casefold()
        matches = [
            ConversationSearchMatch(
                message_id=str(source["message_id"]),
                role=source["role"],
                source=match_source,
                snippet=cls._conversation_search_snippet(text, phrase),
                created_at=source["created_at"],
            )
            for match_source, text in candidates
            if normalized_phrase in " ".join(text.split()).casefold()
        ]
        if matches:
            return matches

        # Elasticsearch analyzers may produce a phrase hit without preserving an
        # exact substring. Keep that hit navigable instead of silently dropping it.
        return [
            ConversationSearchMatch(
                message_id=str(source["message_id"]),
                role=source["role"],
                source="message",
                snippet=cls._conversation_search_snippet(
                    str(source.get("content") or ""),
                    phrase,
                ),
                created_at=source["created_at"],
            )
        ]

    @staticmethod
    def _conversation_search_snippet(
        content: str,
        phrase: str,
        radius: int = 42,
    ) -> str:
        normalized_content = " ".join(content.split())
        index = normalized_content.casefold().find(phrase.casefold())
        if index < 0:
            index = 0
        start = max(0, index - radius)
        end = min(len(normalized_content), index + len(phrase) + radius)
        prefix = "…" if start > 0 else ""
        suffix = "…" if end < len(normalized_content) else ""
        return f"{prefix}{normalized_content[start:end]}{suffix}"

    async def list_user_performance_messages(
        self,
        username: str,
        *,
        size: int = 20,
        created_at_gte: datetime | None = None,
        created_at_lt: datetime | None = None,
    ) -> list[MessageRecord]:
        filters: list[dict[str, Any]] = [
            {"term": {"username": username}},
            {"term": {"role": "assistant"}},
            {
                "exists": {
                    "field": (
                        "metadata.performance_metrics."
                        "model_first_token_ms"
                    )
                }
            },
        ]
        if created_at_gte is not None or created_at_lt is not None:
            created_at_range: dict[str, str] = {}
            if created_at_gte is not None:
                created_at_range["gte"] = created_at_gte.isoformat()
            if created_at_lt is not None:
                created_at_range["lt"] = created_at_lt.isoformat()
            filters.append({"range": {"created_at": created_at_range}})
        result = await self.client.search(
            index=self.settings.messages_index,
            query={
                "bool": {
                    "filter": filters
                }
            },
            sort=[{"created_at": {"order": "desc"}}],
            size=size,
        )
        return [
            MessageRecord.model_validate(hit["_source"])
            for hit in reversed(result["hits"]["hits"])
        ]

    async def list_messages_by_round_range(
        self,
        conversation_id: str,
        start_round: int,
        end_round: int,
    ) -> list[MessageRecord]:
        result = await self.client.search(
            index=self.settings.messages_index,
            query={
                "bool": {
                    "filter": [
                        {"term": {"conversation_id": conversation_id}},
                        {
                            "range": {
                                "metadata.round_number": {
                                    "gte": start_round,
                                    "lte": end_round,
                                }
                            }
                        },
                    ],
                    "must_not": [{"term": {"metadata.deleted": True}}],
                }
            },
            sort=[{"created_at": {"order": "asc"}}],
            size=(end_round - start_round + 1) * 3,
        )
        return [
            MessageRecord.model_validate(hit["_source"])
            for hit in result["hits"]["hits"]
        ]

    async def create_message(
        self,
        conversation_id: str,
        payload: MessageCreate,
    ) -> MessageRecord:
        conversation = await self.get_conversation(conversation_id)
        if conversation.username != payload.username:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="username does not own this conversation",
            )

        now = utc_now()
        message_id = str(uuid4())
        record = MessageRecord(
            message_id=message_id,
            conversation_id=conversation_id,
            username=payload.username,
            role=payload.role,
            content=payload.content,
            reasoning_content=payload.reasoning_content,
            emotion=payload.emotion,
            metadata=payload.metadata,
            created_at=now,
        )
        await self.client.index(
            index=self.settings.messages_index,
            id=message_id,
            document=record.model_dump(mode="json"),
            refresh=False,
        )
        await self.client.update(
            index=self.settings.conversations_index,
            id=conversation_id,
            script={
                "source": (
                    "ctx._source.updated_at = params.updated_at; "
                    "ctx._source.effective_char_count = "
                    "(ctx._source.effective_char_count ?: 0) + params.char_count; "
                    "ctx._source.effective_char_count_version = 5"
                ),
                "params": {
                    "updated_at": now.isoformat(),
                    "char_count": count_message_characters(
                        record.content,
                        record.metadata,
                    ),
                },
            },
            refresh=False,
        )
        return record

    async def advance_completed_rounds(
        self,
        conversation_id: str,
        message_ids: list[str],
    ) -> int:
        result = await self.client.update(
            index=self.settings.conversations_index,
            id=conversation_id,
            script={
                "source": (
                    "ctx._source.completed_rounds = "
                    "(ctx._source.completed_rounds ?: 0) + 1; "
                    "ctx._source.last_turn_message_ids = params.message_ids"
                ),
                "params": {"message_ids": message_ids},
            },
            source=True,
            refresh=False,
        )
        return int(result["get"]["_source"]["completed_rounds"])

    async def discard_uncompleted_message(
        self,
        conversation: ConversationRecord,
        message: MessageRecord,
    ) -> None:
        deleted_at = utc_now().isoformat()
        await self.client.update(
            index=self.settings.messages_index,
            id=message.message_id,
            doc={
                "metadata": {
                    **message.metadata,
                    "deleted": True,
                    "deleted_at": deleted_at,
                    "deleted_reason": "incomplete_turn",
                }
            },
            refresh=False,
        )
        await self.client.update(
            index=self.settings.conversations_index,
            id=conversation.conversation_id,
            doc={
                "updated_at": conversation.updated_at.isoformat(),
                "effective_char_count": conversation.effective_char_count,
                "effective_char_count_version": 5,
                "effective_memory_through_round": (
                    conversation.effective_memory_through_round
                ),
            },
            refresh=False,
        )

    async def list_short_term_memories(
        self,
        conversation_id: str,
    ) -> list[ShortTermMemoryRecord]:
        conversation = await self.get_conversation(conversation_id)
        if conversation.memory_through_round <= 0:
            return []
        result = await self.client.search(
            index=self.settings.memories_index,
            query={
                "bool": {
                    "filter": [
                        {"term": {"conversation_id": conversation_id}},
                        {
                            "range": {
                                "through_round": {
                                    "lte": conversation.memory_through_round
                                }
                            }
                        },
                    ]
                }
            },
            sort=[
                {"through_round": {"order": "asc"}},
                {"created_at": {"order": "desc"}},
            ],
            size=100,
        )
        memories_by_round: dict[int, ShortTermMemoryRecord] = {}
        for hit in result["hits"]["hits"]:
            memory = ShortTermMemoryRecord.model_validate(hit["_source"])
            memories_by_round.setdefault(memory.through_round, memory)
        return list(memories_by_round.values())

    async def latest_short_term_memory_at_or_before(
        self,
        conversation_id: str,
        through_round: int,
    ) -> ShortTermMemoryRecord | None:
        if through_round <= 0:
            return None
        result = await self.client.search(
            index=self.settings.memories_index,
            query={
                "bool": {
                    "filter": [
                        {"term": {"conversation_id": conversation_id}},
                        {"range": {"through_round": {"lte": through_round}}},
                    ]
                }
            },
            sort=[
                {"through_round": {"order": "desc"}},
                {"created_at": {"order": "desc"}},
            ],
            size=1,
        )
        hits = result["hits"]["hits"]
        if not hits:
            return None
        return ShortTermMemoryRecord.model_validate(hits[0]["_source"])

    async def activate_short_term_memory_character_count(
        self,
        conversation: ConversationRecord,
        *,
        through_round: int,
        summary: str,
        pending_message: MessageRecord,
    ) -> int:
        if through_round == conversation.effective_memory_through_round:
            return (
                conversation.effective_char_count
                + count_message_characters(
                    pending_message.content,
                    pending_message.metadata,
                )
            )
        active_messages = await self.list_messages_by_round_range(
            conversation.conversation_id,
            through_round + 1,
            conversation.completed_rounds,
        )
        effective_char_count = (
            count_text_characters(summary)
            + sum(
                count_message_characters(message.content, message.metadata)
                for message in active_messages
            )
            + count_message_characters(
                pending_message.content,
                pending_message.metadata,
            )
        )
        await self.client.update(
            index=self.settings.conversations_index,
            id=conversation.conversation_id,
            doc={
                "effective_char_count": effective_char_count,
                "effective_char_count_version": 5,
                "effective_memory_through_round": through_round,
            },
            refresh=False,
        )
        return effective_char_count

    async def create_short_term_memory(
        self,
        *,
        conversation_id: str,
        username: str,
        compression_number: int,
        through_round: int,
        summary: str,
    ) -> ShortTermMemoryRecord | None:
        conversation = await self.get_conversation(conversation_id)
        if (
            conversation.completed_rounds < through_round
            or conversation.memory_through_round >= through_round
            or conversation.memory_status != "compressing"
            or conversation.memory_target_round != through_round
        ):
            return None
        record = ShortTermMemoryRecord(
            memory_id=hashlib.sha256(
                f"{conversation_id}:{through_round}".encode()
            ).hexdigest(),
            conversation_id=conversation_id,
            username=username,
            compression_number=compression_number,
            through_round=through_round,
            summary=summary,
            created_at=utc_now(),
        )
        await self.client.index(
            index=self.settings.memories_index,
            id=record.memory_id,
            document=record.model_dump(mode="json"),
            refresh=False,
        )
        await self.client.update(
            index=self.settings.conversations_index,
            id=conversation_id,
            script={
                "source": (
                    "ctx._source.memory_compression_count = "
                    "params.compression_number; "
                    "ctx._source.memory_through_round = params.through_round; "
                    "ctx._source.short_term_memory = params.summary; "
                    "ctx._source.memory_status = 'idle'; "
                    "ctx._source.memory_target_round = params.through_round"
                ),
                "params": {
                    "compression_number": compression_number,
                    "through_round": through_round,
                    "summary": summary,
                },
            },
            refresh=False,
        )
        return record

    async def try_mark_memory_compressing(
        self,
        conversation_id: str,
        target_round: int,
    ) -> bool:
        result = await self.client.update(
            index=self.settings.conversations_index,
            id=conversation_id,
            script={
                "source": (
                    "if ((ctx._source.memory_through_round ?: 0) >= "
                    "params.target_round || "
                    "(ctx._source.memory_status == 'compressing' && "
                    "(ctx._source.memory_target_round ?: 0) >= "
                    "params.target_round)) { "
                    "ctx.op = 'noop'; "
                    "} else { "
                    "ctx._source.memory_status = 'compressing'; "
                    "ctx._source.memory_target_round = params.target_round; "
                    "}"
                ),
                "params": {"target_round": target_round},
            },
            refresh=False,
        )
        return result.get("result") != "noop"

    async def mark_memory_compression_failed(
        self,
        conversation_id: str,
    ) -> None:
        await self.client.update(
            index=self.settings.conversations_index,
            id=conversation_id,
            doc={"memory_status": "failed"},
            refresh=False,
        )

    async def rewind_last_turn(
        self,
        conversation_id: str,
        username: str,
    ) -> list[str]:
        conversation = await self.get_conversation(conversation_id)
        if conversation.username != username:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="username does not own this conversation",
            )

        stored_message_ids = conversation.last_turn_message_ids
        if stored_message_ids:
            realtime_result = await self.client.mget(
                index=self.settings.messages_index,
                ids=stored_message_ids,
            )
            hits = [
                {"_id": document["_id"], "_source": document["_source"]}
                for document in realtime_result["docs"]
                if document.get("found")
                and not document["_source"].get("metadata", {}).get("deleted")
            ]
        else:
            result = await self.client.search(
                index=self.settings.messages_index,
                query={
                    "bool": {
                        "filter": [
                            {"term": {"conversation_id": conversation_id}},
                            {
                                "term": {
                                    "metadata.round_number": (
                                        conversation.completed_rounds
                                    )
                                }
                            },
                        ],
                        "must_not": [{"term": {"metadata.deleted": True}}],
                    }
                },
                sort=[{"created_at": {"order": "desc"}}],
                size=3,
            )
            hits = result["hits"]["hits"]
        selected: list[dict[str, Any]] = []
        assistant_before_user: dict[str, Any] | None = None
        for hit in sorted(
            hits,
            key=lambda item: str(item["_source"].get("created_at") or ""),
            reverse=True,
        ):
            role = hit["_source"].get("role")
            if role == "assistant" and assistant_before_user is None:
                assistant_before_user = hit
            if role == "user":
                if assistant_before_user is not None:
                    selected.append(assistant_before_user)
                selected.append(hit)
                break

        if not selected:
            return []

        new_completed_rounds = max(conversation.completed_rounds - 1, 0)
        deleted_character_count = sum(
            count_message_characters(
                str(hit["_source"].get("content") or ""),
                hit["_source"].get("metadata"),
            )
            for hit in selected
        )
        deleted_at = utc_now().isoformat()
        operations: list[dict[str, Any]] = []
        for hit in selected:
            metadata = dict(hit["_source"].get("metadata") or {})
            metadata.update(
                {
                    "deleted": True,
                    "deleted_at": deleted_at,
                    "deleted_reason": "rewind",
                }
            )
            operations.extend(
                [
                    {
                        "update": {
                            "_index": self.settings.messages_index,
                            "_id": hit["_id"],
                        }
                    },
                    {"doc": {"metadata": metadata}},
                ]
            )
        await self.client.bulk(
            operations=operations,
            refresh=False,
        )

        memory_compression_count = conversation.memory_compression_count
        memory_through_round = conversation.memory_through_round
        short_term_memory = conversation.short_term_memory
        effective_char_count = max(
            conversation.effective_char_count - deleted_character_count,
            0,
        )
        if (
            conversation.memory_through_round > 0
            and new_completed_rounds < conversation.memory_through_round
        ):
            restored_memory = await self.latest_short_term_memory_at_or_before(
                conversation_id,
                new_completed_rounds,
            )
            memory_compression_count = (
                restored_memory.compression_number if restored_memory else 0
            )
            memory_through_round = (
                restored_memory.through_round if restored_memory else 0
            )
            short_term_memory = restored_memory.summary if restored_memory else ""

        maximum_effective_memory_round = min(
            max(new_completed_rounds - 10, 0),
            memory_through_round,
        )
        effective_memory = await self.latest_short_term_memory_at_or_before(
            conversation_id,
            maximum_effective_memory_round,
        )
        effective_memory_through_round = (
            effective_memory.through_round if effective_memory else 0
        )
        if (
            effective_memory_through_round
            != conversation.effective_memory_through_round
        ):
            active_messages = (
                await self.list_messages_by_round_range(
                    conversation_id,
                    effective_memory_through_round + 1,
                    new_completed_rounds,
                )
                if new_completed_rounds > effective_memory_through_round
                else []
            )
            effective_char_count = count_text_characters(
                effective_memory.summary if effective_memory else ""
            ) + sum(
                count_message_characters(message.content, message.metadata)
                for message in active_messages
            )

        memory_status = conversation.memory_status
        memory_target_round = conversation.memory_target_round
        if memory_through_round != conversation.memory_through_round:
            memory_status = "idle"
            memory_target_round = memory_through_round
        if (
            memory_status == "compressing"
            and new_completed_rounds < memory_target_round
        ):
            memory_status = "idle"
            memory_target_round = memory_through_round

        await self.client.update(
            index=self.settings.conversations_index,
            id=conversation_id,
            doc={
                "completed_rounds": new_completed_rounds,
                "last_turn_message_ids": [],
                "effective_char_count": effective_char_count,
                "effective_char_count_version": 5,
                "effective_memory_through_round": (
                    effective_memory_through_round
                ),
                "memory_compression_count": memory_compression_count,
                "memory_through_round": memory_through_round,
                "short_term_memory": short_term_memory,
                "memory_status": memory_status,
                "memory_target_round": memory_target_round,
            },
            refresh=False,
        )
        return [hit["_source"]["message_id"] for hit in selected]

    async def search_knowledge(
        self,
        query: str,
        size: int,
    ) -> list[KnowledgeHit]:
        if not await self.client.indices.exists(
            index=self.settings.knowledge_index
        ) or not await self.client.indices.exists(
            index=self.settings.knowledge_vector_index
        ):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="knowledge indices are unavailable",
            )

        normalized_query = str(query or "").strip()
        if not normalized_query:
            return []
        top_k = max(1, min(20, int(size)))
        candidate_size = max(top_k * 5, 10)

        keyword_hits: dict[str, dict[str, Any]] = {}
        analysis = await self.client.indices.analyze(
            index=self.settings.knowledge_index,
            analyzer=KNOWLEDGE_KEYWORD_ANALYZER,
            text=normalized_query,
        )
        query_terms = {
            str(token.get("token") or "").strip().casefold()
            for token in analysis.get("tokens", [])
            if str(token.get("token") or "").strip()
        }
        keyword_response = (
            await self.client.search(
                index=self.settings.knowledge_index,
                query={
                    "bool": {
                        "should": [
                            {
                                "terms": {
                                    "keyword_terms": sorted(query_terms),
                                    "boost": 100.0,
                                }
                            },
                            {
                                "match": {
                                    "keyword_text": {
                                        "query": normalized_query,
                                        "analyzer": KNOWLEDGE_KEYWORD_ANALYZER,
                                    }
                                }
                            },
                        ],
                        "minimum_should_match": 1,
                    }
                },
                source=["keywords"],
                size=max(candidate_size * 5, 100),
            )
            if query_terms
            else {"hits": {"hits": []}}
        )
        for hit in keyword_response["hits"]["hits"]:
            keywords = [
                str(keyword).strip()
                for keyword in (hit.get("_source") or {}).get("keywords", [])
                if str(keyword).strip()
            ]
            matched_keywords = [
                keyword
                for keyword in keywords
                if keyword.casefold() in normalized_query.casefold()
            ]
            if not matched_keywords:
                continue
            longest_keyword = max(len(keyword) for keyword in matched_keywords)
            keyword_hits[str(hit["_id"])] = {
                "score": float(
                    1000 + longest_keyword * 10 + len(matched_keywords)
                ),
                "reasons": ["keyword"],
                "matched_keywords": matched_keywords,
                "matched_descriptions": [],
            }
        keyword_hits = dict(
            sorted(
                keyword_hits.items(),
                key=lambda item: item[1]["score"],
                reverse=True,
            )[:candidate_size]
        )

        embedder = await self._get_knowledge_embedder()
        query_vector = await asyncio.to_thread(
            embedder.encode_query,
            query,
            self.settings.knowledge_query_prefix,
        )
        semantic_response = await self.client.search(
            index=self.settings.knowledge_vector_index,
            knn={
                "field": "embedding",
                "query_vector": query_vector,
                "k": candidate_size,
                "num_candidates": max(candidate_size * 20, 20),
            },
            size=candidate_size,
        )
        semantic_hits: dict[str, dict[str, Any]] = {}
        for hit in semantic_response["hits"]["hits"]:
            score = float(hit.get("_score") or 0.0)
            if score < self.settings.knowledge_semantic_min_score:
                continue
            source = hit.get("_source") or {}
            knowledge_id = str(source.get("knowledge_id") or "")
            if not knowledge_id:
                continue
            semantic_hits[knowledge_id] = {
                "score": score,
                "reasons": ["semantic"],
                "matched_keywords": [],
                "matched_descriptions": [str(source.get("description") or "")],
            }

        merged: dict[str, dict[str, Any]] = {}
        for source_hits in (keyword_hits, semantic_hits):
            for knowledge_id, item in source_hits.items():
                current = merged.setdefault(
                    knowledge_id,
                    {
                        "score": 0.0,
                        "reasons": [],
                        "matched_keywords": [],
                        "matched_descriptions": [],
                    },
                )
                current["score"] = max(current["score"], item["score"])
                current["reasons"].extend(item["reasons"])
                current["matched_keywords"].extend(item["matched_keywords"])
                current["matched_descriptions"].extend(
                    item["matched_descriptions"]
                )

        if not merged:
            return []
        docs_response = await self.client.search(
            index=self.settings.knowledge_index,
            query={"ids": {"values": list(merged)}},
            size=len(merged),
        )
        docs = {
            str(hit["_id"]): hit.get("_source") or {}
            for hit in docs_response["hits"]["hits"]
        }

        results: list[KnowledgeHit] = []
        for knowledge_id, match in sorted(
            merged.items(),
            key=lambda item: item[1]["score"],
            reverse=True,
        )[:top_k]:
            source = dict(docs.get(knowledge_id) or {})
            source.update(
                {
                    "knowledge_id": knowledge_id,
                    "retrieval_reasons": sorted(set(match["reasons"])),
                    "matched_keywords": sorted(
                        set(match["matched_keywords"])
                    ),
                    "matched_descriptions": match[
                        "matched_descriptions"
                    ][:3],
                }
            )
            results.append(
                KnowledgeHit(
                    document_id=knowledge_id,
                    score=match["score"],
                    source=source,
                )
            )
        return results

    async def _get_knowledge_embedder(self) -> BGEEmbedder:
        if self._knowledge_embedder is not None:
            return self._knowledge_embedder
        async with self._knowledge_embedder_lock:
            if self._knowledge_embedder is None:
                self._knowledge_embedder = await asyncio.to_thread(
                    BGEEmbedder,
                    self.settings.knowledge_embedding_model,
                )
        return self._knowledge_embedder

    async def preload_knowledge_embedder(self) -> None:
        """Load and warm up the knowledge embedder before serving requests."""
        embedder = await self._get_knowledge_embedder()
        await asyncio.to_thread(
            embedder.encode_query,
            "预热",
            self.settings.knowledge_query_prefix,
        )


def build_elasticsearch_client(settings: Settings) -> AsyncElasticsearch:
    kwargs: dict[str, Any] = {
        "hosts": [settings.es_url],
        "request_timeout": 10,
        "retry_on_timeout": True,
        "max_retries": 2,
    }
    if settings.es_api_key:
        kwargs["api_key"] = settings.es_api_key
    elif settings.es_username:
        kwargs["basic_auth"] = (settings.es_username, settings.es_password or "")
    return AsyncElasticsearch(**kwargs)


@asynccontextmanager
async def elasticsearch_lifespan(
    settings: Settings,
) -> AsyncIterator[ElasticsearchStore]:
    client = build_elasticsearch_client(settings)
    store = ElasticsearchStore(client, settings)
    try:
        logger.info("Loading knowledge embedding model")
        await store.preload_knowledge_embedder()
        logger.info("Knowledge embedding model is ready")
        if await client.ping():
            await store.ensure_indices()
        else:
            logger.warning("Elasticsearch is not reachable; API requests will return 503")
        yield store
    finally:
        await client.close()
