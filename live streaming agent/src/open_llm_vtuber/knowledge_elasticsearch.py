from __future__ import annotations

import asyncio
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from elasticsearch_dsl import Search, connections
from loguru import logger

from .utils.turn_trace import record_turn_event

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BGE_MODEL_PATH = PROJECT_ROOT / "models" / "bge-small-zh-v1.5"
DEFAULT_BGE_MODEL_ID = "BAAI/bge-small-zh-v1.5"
DEFAULT_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
DEFAULT_KEYWORD_ANALYZER = "ik_max_word"


@dataclass
class KnowledgeRuntime:
    client: Any
    embedder: BGEEmbedder
    embedding_dims: int
    knowledge_index: str
    vector_index: str


class BGEEmbedder:
    def __init__(self, model_path: str | Path) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: sentence-transformers. Run `uv sync` first."
            ) from exc

        model_path_text, local_files_only = resolve_embedding_model(model_path)
        try:
            self.model = SentenceTransformer(
                model_path_text,
                local_files_only=local_files_only,
            )
        except TypeError:
            self.model = SentenceTransformer(model_path_text)

    def encode_query(self, text: str, query_prefix: str) -> list[float]:
        query_text = f"{query_prefix}{text}" if query_prefix else text
        return self._encode([query_text])[0]

    def embedding_dims(self) -> int:
        return len(self._encode(["dimension probe"])[0])

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors.tolist()


def looks_like_model_id(value: str) -> bool:
    return (
        "/" in value
        and not value.startswith(("/", "\\", "."))
        and not re.match(r"^[a-zA-Z]:[\\/]", value)
    )


def resolve_embedding_model(model_path: str | Path) -> tuple[str, bool]:
    model_path_text = os.path.expandvars(str(model_path)).strip()
    candidate_path = Path(model_path_text).expanduser()
    if candidate_path.exists():
        return str(candidate_path), True

    if looks_like_model_id(model_path_text):
        return model_path_text, False

    raise FileNotFoundError(
        "Embedding model path does not exist: "
        f"{candidate_path}. Put the local model there, or use a model id such as "
        f"{DEFAULT_BGE_MODEL_ID!r}."
    )


def default_embedding_model() -> str | Path:
    configured = os.getenv("KNOWLEDGE_EMBEDDING_MODEL", "").strip()
    if configured:
        return configured
    if DEFAULT_BGE_MODEL_PATH.exists():
        return DEFAULT_BGE_MODEL_PATH
    return DEFAULT_BGE_MODEL_ID


def _alternate_es_scheme(es_url: str) -> str | None:
    if es_url.startswith("https://"):
        return f"http://{es_url.removeprefix('https://')}"
    if es_url.startswith("http://"):
        return f"https://{es_url.removeprefix('http://')}"
    return None


def _probe_es_url(es_url: str, *, verify_certs: bool, timeout: float) -> tuple[bool, str]:
    context = None
    if es_url.startswith("https://") and not verify_certs:
        context = ssl._create_unverified_context()

    try:
        request = urllib.request.Request(es_url, method="GET")
        with urllib.request.urlopen(request, timeout=timeout, context=context):
            return True, "ok"
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            return True, f"http-{exc.code}"
        return False, f"http-{exc.code}: {exc.reason}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def resolve_es_url(
    es_url: str,
    *,
    verify_certs: bool,
    request_timeout: float,
) -> str:
    probe_timeout = min(max(request_timeout, 1.0), 5.0)
    if es_url.lower() == "auto":
        candidates = ["http://127.0.0.1:9200", "https://127.0.0.1:9200"]
    else:
        candidates = [es_url]
        alternate = _alternate_es_scheme(es_url)
        if alternate:
            candidates.append(alternate)

    errors: list[str] = []
    for candidate in candidates:
        ok, reason = _probe_es_url(
            candidate,
            verify_certs=verify_certs,
            timeout=probe_timeout,
        )
        if ok:
            if candidate != es_url:
                logger.info(
                    "Using Elasticsearch endpoint {} instead of {!r}: {}",
                    candidate,
                    es_url,
                    reason,
                )
            return candidate
        errors.append(f"{candidate}: {reason}")

    raise ConnectionError(
        "Unable to detect a working Elasticsearch HTTP endpoint:\n"
        + "\n".join(f"- {error}" for error in errors)
    )


def create_es_connection(
    *,
    es_url: str,
    api_key: str,
    username: str,
    password: str,
    request_timeout: float,
    verify_certs: bool,
):
    es_url = resolve_es_url(
        es_url,
        verify_certs=verify_certs,
        request_timeout=request_timeout,
    )
    kwargs: dict[str, Any] = {
        "hosts": [es_url],
        "request_timeout": request_timeout,
        "verify_certs": verify_certs,
    }
    if not verify_certs:
        kwargs["ssl_show_warn"] = False
    if api_key:
        kwargs["api_key"] = api_key
    elif username and password:
        kwargs["basic_auth"] = (username, password or "")
    return connections.create_connection(alias="knowledge_runtime", **kwargs)


_RUNTIME_CACHE: dict[tuple[Any, ...], KnowledgeRuntime] = {}
_RUNTIME_LOCK = Lock()


def _runtime_cache_key(config: Any) -> tuple[Any, ...]:
    embedding_model = str(getattr(config, "embedding_model", "") or default_embedding_model())
    return (
        getattr(config, "es_url", "auto"),
        bool(getattr(config, "verify_certs", False)),
        float(getattr(config, "request_timeout", 30.0)),
        getattr(config, "api_key", ""),
        getattr(config, "username", ""),
        bool(getattr(config, "password", "")),
        embedding_model,
        getattr(config, "knowledge_index", "vtuber_knowledge"),
        getattr(config, "vector_index", "vtuber_knowledge_vectors"),
    )


def preload_knowledge_runtime(config: Any) -> KnowledgeRuntime:
    if not getattr(config, "enabled", False):
        raise RuntimeError("Knowledge base is disabled.")

    key = _runtime_cache_key(config)
    with _RUNTIME_LOCK:
        cached = _RUNTIME_CACHE.get(key)
        if cached is not None:
            return cached

        started = time.perf_counter()
        embedding_model = getattr(config, "embedding_model", "") or default_embedding_model()
        client = create_es_connection(
            es_url=getattr(config, "es_url", "auto"),
            api_key=getattr(config, "api_key", ""),
            username=getattr(config, "username", ""),
            password=getattr(config, "password", ""),
            request_timeout=float(getattr(config, "request_timeout", 30.0)),
            verify_certs=bool(getattr(config, "verify_certs", False)),
        )
        embedder = BGEEmbedder(embedding_model)
        runtime = KnowledgeRuntime(
            client=client,
            embedder=embedder,
            embedding_dims=embedder.embedding_dims(),
            knowledge_index=getattr(config, "knowledge_index", "vtuber_knowledge"),
            vector_index=getattr(config, "vector_index", "vtuber_knowledge_vectors"),
        )
        _RUNTIME_CACHE[key] = runtime
        logger.info(
            "Knowledge runtime preloaded: knowledge_index={} vector_index={} "
            "dims={} elapsed_ms={:.3f}",
            runtime.knowledge_index,
            runtime.vector_index,
            runtime.embedding_dims,
            (time.perf_counter() - started) * 1000,
        )
        return runtime


def keyword_search(
    client: Any,
    *,
    index_name: str,
    query: str,
    candidate_size: int,
) -> dict[str, dict[str, Any]]:
    normalized_query = str(query or "").strip()
    hits: dict[str, dict[str, Any]] = {}
    if not normalized_query:
        return hits

    analysis = client.indices.analyze(
        index=index_name,
        analyzer=DEFAULT_KEYWORD_ANALYZER,
        text=normalized_query,
    )
    query_terms = {
        str(token.get("token") or "").strip().casefold()
        for token in analysis.get("tokens", [])
        if str(token.get("token") or "").strip()
    }
    if not query_terms:
        return hits
    response = client.search(
        index=index_name,
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
                                "analyzer": DEFAULT_KEYWORD_ANALYZER,
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
    for hit in response.get("hits", {}).get("hits", []):
        source = hit.get("_source") or {}
        keywords = [
            str(keyword).strip()
            for keyword in source.get("keywords", [])
            if str(keyword).strip()
        ]
        matched_keywords = [
            keyword
            for keyword in keywords
            if keyword.casefold() in normalized_query.casefold()
        ]
        if not matched_keywords:
            continue

        knowledge_id = str(hit.get("_id"))
        longest_keyword = max(len(keyword) for keyword in matched_keywords)
        hits[knowledge_id] = {
            "score": float(1000 + longest_keyword * 10 + len(matched_keywords)),
            "reason": ["keyword"],
            "matched_keywords": matched_keywords,
        }

    return dict(
        sorted(
            hits.items(),
            key=lambda item: item[1]["score"],
            reverse=True,
        )[:candidate_size]
    )


def semantic_search(
    client: Any,
    *,
    embedder: BGEEmbedder,
    index_name: str,
    query: str,
    query_prefix: str,
    candidate_size: int,
    min_score: float,
) -> dict[str, dict[str, Any]]:
    query_vector = embedder.encode_query(query, query_prefix)
    response = (
        Search(using=client, index=index_name)
        .extra(
            knn={
                "field": "embedding",
                "query_vector": query_vector,
                "k": candidate_size,
                "num_candidates": max(candidate_size * 20, 20),
            },
            size=candidate_size,
        )
        .execute()
    )
    hits: dict[str, dict[str, Any]] = {}
    for hit in response:
        score = float(hit.meta.score or 0.0)
        if score < min_score:
            continue
        knowledge_id = str(hit.knowledge_id)
        hits[knowledge_id] = {
            "score": score,
            "reason": ["semantic"],
            "matched_description": str(hit.description),
        }
    return hits


def fetch_knowledge_docs(
    client: Any,
    *,
    index_name: str,
    knowledge_ids: list[str],
) -> dict[str, dict[str, Any]]:
    if not knowledge_ids:
        return {}
    response = (
        Search(using=client, index=index_name)
        .query("ids", values=knowledge_ids)[: len(knowledge_ids)]
        .execute()
    )
    return {str(hit.meta.id): hit.to_dict() for hit in response}


def search_knowledge(
    *,
    runtime: KnowledgeRuntime,
    query: str,
    top_k: int,
    query_prefix: str,
    semantic_min_score: float,
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    top_k = max(1, min(20, int(top_k)))
    candidate_size = max(top_k * 5, 10)

    keyword_started = time.perf_counter()
    keyword_hits = keyword_search(
        runtime.client,
        index_name=runtime.knowledge_index,
        query=query,
        candidate_size=candidate_size,
    )
    keyword_seconds = time.perf_counter() - keyword_started

    semantic_started = time.perf_counter()
    semantic_hits = semantic_search(
        runtime.client,
        embedder=runtime.embedder,
        index_name=runtime.vector_index,
        query=query,
        query_prefix=query_prefix,
        candidate_size=candidate_size,
        min_score=semantic_min_score,
    )
    semantic_seconds = time.perf_counter() - semantic_started

    merged: dict[str, dict[str, Any]] = {}
    for source_hits in [keyword_hits, semantic_hits]:
        for knowledge_id, hit in source_hits.items():
            current = merged.setdefault(
                knowledge_id,
                {
                    "score": 0.0,
                    "reasons": [],
                    "matched_keywords": [],
                    "matched_descriptions": [],
                },
            )
            current["score"] = max(current["score"], hit["score"])
            current["reasons"].extend(hit.get("reason", []))
            if hit.get("matched_keywords"):
                current["matched_keywords"].extend(hit["matched_keywords"])
            if hit.get("matched_description"):
                current["matched_descriptions"].append(hit["matched_description"])

    fetch_started = time.perf_counter()
    docs = fetch_knowledge_docs(
        runtime.client,
        index_name=runtime.knowledge_index,
        knowledge_ids=list(merged.keys()),
    )
    fetch_seconds = time.perf_counter() - fetch_started

    results = []
    for knowledge_id, merged_hit in sorted(
        merged.items(),
        key=lambda item: item[1]["score"],
        reverse=True,
    )[:top_k]:
        doc = docs.get(knowledge_id, {})
        results.append(
            {
                "knowledge_id": knowledge_id,
                "score": merged_hit["score"],
                "reasons": sorted(set(merged_hit["reasons"])),
                "matched_keywords": sorted(set(merged_hit["matched_keywords"])),
                "matched_descriptions": merged_hit["matched_descriptions"][:3],
                "title": doc.get("title"),
                "category": doc.get("category"),
                "keywords": doc.get("keywords", []),
                "semantic_descriptions": doc.get("semantic_descriptions", []),
                "body": doc.get("body"),
                "usage": doc.get("usage"),
                "priority": doc.get("priority", ""),
                "updated_at": doc.get("updated_at"),
            }
        )

    elapsed_seconds = time.perf_counter() - started
    logger.info(
        "Knowledge search timing: {}",
        json.dumps(
            {
                "query": query,
                "top_k": top_k,
                "keyword_hits": len(keyword_hits),
                "semantic_hits": len(semantic_hits),
                "semantic_min_score": semantic_min_score,
                "merged_hits": len(merged),
                "result_count": len(results),
                "elapsed_ms": round(elapsed_seconds * 1000, 3),
                "keyword_ms": round(keyword_seconds * 1000, 3),
                "semantic_ms": round(semantic_seconds * 1000, 3),
                "fetch_ms": round(fetch_seconds * 1000, 3),
            },
            ensure_ascii=False,
        ),
    )
    return results


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def format_knowledge_injection(results: list[dict[str, Any]]) -> str:
    lines = [
        "[知识库检索结果]",
        "以下内容来自本地知识库，是当前这轮用户输入的补充背景。回答时只在相关时参考；如果无关，可以忽略。",
    ]
    for index, item in enumerate(results, start=1):
        keywords = "、".join(str(value) for value in item.get("keywords", []) if value)
        matched_keywords = "、".join(
            str(value) for value in item.get("matched_keywords", []) if value
        )
        body = _normalize_text(item.get("body"))
        usage = _normalize_text(item.get("usage"))
        lines.append(
            "\n".join(
                part
                for part in [
                    f"{index}. 标题：{item.get('title') or item.get('knowledge_id')}",
                    f"   分类：{item.get('category')}" if item.get("category") else "",
                    f"   关键词：{keywords}" if keywords else "",
                    f"   命中关键词：{matched_keywords}" if matched_keywords else "",
                    f"   用法：{usage}" if usage else "",
                    f"   内容：{body}" if body else "",
                ]
                if part
            )
        )

    return "\n".join(lines).strip()


async def maybe_enhance_input_with_knowledge(
    *,
    context: Any,
    input_text: str,
    metadata: dict[str, Any] | None,
    turn_id: str | None,
    client_uid: str | None = None,
) -> tuple[str, dict[str, Any] | None]:
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    if metadata.get("knowledge_processed_for_turn"):
        return input_text, metadata

    metadata["knowledge_processed_for_turn"] = True
    config = getattr(getattr(context, "config", None), "knowledge_config", None)
    if not config or not getattr(config, "enabled", False):
        return input_text, metadata

    context.knowledge_turn_count = int(getattr(context, "knowledge_turn_count", 0)) + 1
    current_round = context.knowledge_turn_count
    min_interval = max(1, int(getattr(config, "min_injection_interval_rounds", 10)))
    runtime = getattr(context, "knowledge_runtime", None)
    if runtime is None:
        record_turn_event(
            turn_id,
            "knowledge_elasticsearch",
            "injection_skipped_runtime_missing",
            client_uid=client_uid,
            current_round=current_round,
        )
        return input_text, metadata

    query = str(input_text or "").strip()
    if not query:
        return input_text, metadata

    try:
        desired_top_k = max(1, min(3, int(getattr(config, "top_k", 3))))
        search_top_k = min(20, max(desired_top_k * 4, desired_top_k + 6))
        results = await asyncio.to_thread(
            search_knowledge,
            runtime=runtime,
            query=query,
            top_k=search_top_k,
            query_prefix=getattr(config, "query_prefix", DEFAULT_QUERY_PREFIX),
            semantic_min_score=float(getattr(config, "semantic_min_score", 0.75)),
        )
    except Exception as exc:
        logger.exception("Knowledge search failed.")
        record_turn_event(
            turn_id,
            "knowledge_elasticsearch",
            "search_failed",
            client_uid=client_uid,
            current_round=current_round,
            error=str(exc),
        )
        return input_text, metadata

    if not results:
        record_turn_event(
            turn_id,
            "knowledge_elasticsearch",
            "no_results",
            client_uid=client_uid,
            current_round=current_round,
        )
        return input_text, metadata

    last_round_by_id = getattr(context, "knowledge_last_injected_round_by_id", None)
    if not isinstance(last_round_by_id, dict):
        last_round_by_id = {}
        context.knowledge_last_injected_round_by_id = last_round_by_id

    filtered_results: list[dict[str, Any]] = []
    skipped_items: list[dict[str, Any]] = []
    for item in results:
        knowledge_id = str(item.get("knowledge_id") or "").strip()
        last_item_round = last_round_by_id.get(knowledge_id) if knowledge_id else None
        if (
            knowledge_id
            and last_item_round is not None
            and current_round - int(last_item_round) < min_interval
        ):
            skipped_items.append(
                {
                    "knowledge_id": knowledge_id,
                    "last_injected_round": int(last_item_round),
                    "remaining_rounds": min_interval
                    - (current_round - int(last_item_round)),
                }
            )
            continue
        filtered_results.append(item)

    if skipped_items:
        logger.debug(
            "Skipped recently injected knowledge ids: round={} skipped={}",
            current_round,
            skipped_items,
        )
        record_turn_event(
            turn_id,
            "knowledge_elasticsearch",
            "knowledge_ids_skipped_interval",
            client_uid=client_uid,
            current_round=current_round,
            min_interval_rounds=min_interval,
            skipped_items=skipped_items,
        )

    results = filtered_results[:desired_top_k]
    if not results:
        record_turn_event(
            turn_id,
            "knowledge_elasticsearch",
            "all_results_skipped_interval",
            client_uid=client_uid,
            current_round=current_round,
            min_interval_rounds=min_interval,
            skipped_items=skipped_items,
        )
        return input_text, metadata

    injection = format_knowledge_injection(results)
    context.knowledge_last_injected_turn = current_round
    injected_ids = [
        str(item.get("knowledge_id") or "").strip()
        for item in results
        if str(item.get("knowledge_id") or "").strip()
    ]
    for knowledge_id in injected_ids:
        last_round_by_id[knowledge_id] = current_round
    enhanced_input = f"{injection}\n\n[用户当前提问]\n{query}"
    metadata.update(
        {
            "knowledge_injected": True,
            "knowledge_result_count": len(results),
            "knowledge_ids": injected_ids,
            "knowledge_round": current_round,
            "knowledge_skipped_ids_due_to_interval": [
                item["knowledge_id"] for item in skipped_items
            ],
        }
    )
    logger.info(
        "Knowledge injected into current turn: round={} result_count={} ids={}",
        current_round,
        len(results),
        metadata["knowledge_ids"],
    )
    record_turn_event(
        turn_id,
        "knowledge_elasticsearch",
        "injected",
        client_uid=client_uid,
        current_round=current_round,
        result_count=len(results),
        knowledge_ids=metadata["knowledge_ids"],
        injected_chars=len(injection),
    )
    return enhanced_input, metadata
