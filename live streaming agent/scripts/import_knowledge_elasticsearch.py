from __future__ import annotations

import csv
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from elasticsearch.helpers import bulk
from elasticsearch_dsl import (
    DenseVector,
    Document,
    Index,
    Keyword,
    Search,
    Text,
    connections,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "resource" / "knowledge.csv"
DEFAULT_BGE_MODEL_PATH = PROJECT_ROOT / "models" / "bge-small-zh-v1.5"
DEFAULT_BGE_MODEL_ID = "BAAI/bge-small-zh-v1.5"
DEFAULT_ES_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD", "")
DEFAULT_ES_API_KEY = os.getenv("ELASTICSEARCH_API_KEY", "")
DEFAULT_ES_USERNAME = os.getenv(
    "ELASTICSEARCH_USERNAME",
    "elastic" if DEFAULT_ES_PASSWORD else "",
)
DEFAULT_ES_URL = os.getenv("ELASTICSEARCH_URL", "auto")
DEFAULT_ES_VERIFY_CERTS = os.getenv("ELASTICSEARCH_VERIFY_CERTS", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_KNOWLEDGE_INDEX = "vtuber_knowledge"
DEFAULT_VECTOR_INDEX = "vtuber_knowledge_vectors"
DEFAULT_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
DEFAULT_SEMANTIC_MIN_SCORE = float(os.getenv("KNOWLEDGE_SEMANTIC_MIN_SCORE", "0.75"))
DEFAULT_KEYWORD_ANALYZER = "ik_max_word"
IK_DICTIONARY_FILENAME = "vtuber_knowledge.dic"


CSV_COLUMNS = {
    "usage": "使用方法",
    "knowledge_id": "知识唯一ID",
    "category": "知识分类",
    "title": "知识标题",
    "keywords": "关键词",
    "semantic_descriptions": "语义描述",
    "body": "知识正文",
    "priority": "优先级",
    "updated_at": "更新时间",
}


class KnowledgeDocument(Document):
    knowledge_id = Keyword(required=True)
    category = Keyword()
    title = Text(fields={"raw": Keyword()})
    usage = Text()
    keywords = Keyword()
    keyword_terms = Keyword()
    keyword_text = Text(
        analyzer="ik_max_word",
        search_analyzer="ik_smart",
    )
    semantic_descriptions = Text()
    body = Text()
    priority = Keyword()
    updated_at = Keyword()


def build_vector_document_class(dims: int) -> type[Document]:
    class KnowledgeVectorDocument(Document):
        knowledge_id = Keyword(required=True)
        description_index = Keyword(required=True)
        description = Text()
        title = Text(fields={"raw": Keyword()})
        category = Keyword()
        keywords = Keyword()
        embedding = DenseVector(dims=dims, index=True, similarity="cosine")

    return KnowledgeVectorDocument


@dataclass
class KnowledgeRow:
    knowledge_id: str
    usage: str
    category: str
    title: str
    keywords: list[str]
    semantic_descriptions: list[str]
    body: str
    priority: str
    updated_at: str

    @property
    def keyword_text(self) -> str:
        return " ".join(self.keywords)

    @property
    def keyword_terms(self) -> list[str]:
        return [keyword.casefold() for keyword in self.keywords]


class BGEEmbedder:
    def __init__(self, model_path: str | Path) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency: sentence-transformers. "
                "Run `uv sync` after updating pyproject.toml."
            ) from exc

        model_path_text, local_files_only = resolve_embedding_model(model_path)
        try:
            self.model = SentenceTransformer(
                model_path_text,
                local_files_only=local_files_only,
            )
        except TypeError:
            self.model = SentenceTransformer(model_path_text)

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def encode_query(self, text: str, query_prefix: str) -> list[float]:
        query_text = f"{query_prefix}{text}" if query_prefix else text
        return self._encode([query_text])[0]

    def embedding_dims(self) -> int:
        return len(self.encode_documents(["dimension probe"])[0])

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


@dataclass
class KnowledgeRuntime:
    client: Any
    embedder: BGEEmbedder
    embedding_dims: int
    knowledge_index: str
    vector_index: str


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
        f"{candidate_path}\n"
        "Put the local model there, or pass a model id such as "
        f"{DEFAULT_BGE_MODEL_ID!r} to let sentence-transformers use its cache/download."
    )


def default_embedding_model() -> str | Path:
    configured = os.getenv("KNOWLEDGE_EMBEDDING_MODEL", "").strip()
    if configured:
        return configured
    if DEFAULT_BGE_MODEL_PATH.exists():
        return DEFAULT_BGE_MODEL_PATH
    return DEFAULT_BGE_MODEL_ID


def split_slash_text(value: str) -> list[str]:
    parts = re.split(r"[/／]", str(value or ""))
    return [part.strip() for part in parts if part.strip()]


def compact_text(value: str, limit: int = 700) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit].strip()


def detect_csv_encoding(csv_path: Path) -> str:
    raw = csv_path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"

    try:
        import chardet

        detected = chardet.detect(raw)
        encoding = detected.get("encoding")
        if encoding:
            try:
                raw.decode(str(encoding))
            except (LookupError, UnicodeDecodeError):
                pass
            else:
                return str(encoding)
    except Exception:
        pass

    for encoding in ("utf-8-sig", "gb18030"):
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "utf-8-sig"


def semantic_descriptions_for_row(
    row: dict[str, str],
    fallback: str,
) -> list[str]:
    descriptions = split_slash_text(row.get(CSV_COLUMNS["semantic_descriptions"], ""))
    if descriptions or fallback == "none":
        return descriptions

    title = compact_text(row.get(CSV_COLUMNS["title"], ""), limit=160)
    body = compact_text(row.get(CSV_COLUMNS["body"], ""), limit=700)
    if fallback == "title" and title:
        return [title]
    if fallback == "body" and body:
        return [body]
    if fallback == "title_body":
        combined = "\n".join(part for part in [title, body] if part)
        return [combined] if combined else []
    return []


def read_knowledge_csv(csv_path: Path, semantic_fallback: str) -> list[KnowledgeRow]:
    rows: list[KnowledgeRow] = []
    encoding = detect_csv_encoding(csv_path)
    with csv_path.open("r", encoding=encoding, newline="") as file:
        reader = csv.DictReader(file)
        missing = [
            column for column in CSV_COLUMNS.values() if column not in (reader.fieldnames or [])
        ]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        for line_number, raw_row in enumerate(reader, start=2):
            knowledge_id = str(raw_row.get(CSV_COLUMNS["knowledge_id"], "")).strip()
            if not knowledge_id:
                raise ValueError(f"CSV row {line_number} has empty knowledge id.")

            rows.append(
                KnowledgeRow(
                    knowledge_id=knowledge_id,
                    usage=str(raw_row.get(CSV_COLUMNS["usage"], "")).strip(),
                    category=str(raw_row.get(CSV_COLUMNS["category"], "")).strip(),
                    title=str(raw_row.get(CSV_COLUMNS["title"], "")).strip(),
                    keywords=split_slash_text(raw_row.get(CSV_COLUMNS["keywords"], "")),
                    semantic_descriptions=semantic_descriptions_for_row(
                        raw_row,
                        semantic_fallback,
                    ),
                    body=str(raw_row.get(CSV_COLUMNS["body"], "")).strip(),
                    priority=str(raw_row.get(CSV_COLUMNS["priority"], "")).strip(),
                    updated_at=str(raw_row.get(CSV_COLUMNS["updated_at"], "")).strip(),
                )
            )
    return rows


def collect_knowledge_keywords(rows: Iterable[KnowledgeRow]) -> list[str]:
    return sorted(
        {
            keyword.strip()
            for row in rows
            for keyword in row.keywords
            if keyword.strip()
        },
        key=lambda keyword: (keyword.casefold(), keyword),
    )


def discover_local_ik_config_dir(client: Any) -> Path:
    configured = os.getenv("ELASTICSEARCH_IK_CONFIG_DIR", "").strip()
    if configured:
        config_dir = Path(os.path.expandvars(configured)).expanduser().resolve()
        if config_dir.is_dir():
            return config_dir
        raise FileNotFoundError(
            f"ELASTICSEARCH_IK_CONFIG_DIR does not exist: {config_dir}"
        )

    response = client.nodes.info(metric="settings")
    candidates: set[Path] = set()
    for node in response.get("nodes", {}).values():
        home = str(node.get("settings", {}).get("path", {}).get("home", "")).strip()
        if not home:
            continue
        config_dir = Path(home) / "config" / "analysis-ik"
        if config_dir.is_dir():
            candidates.add(config_dir.resolve())
    if len(candidates) == 1:
        return candidates.pop()
    raise RuntimeError(
        "Unable to locate a single local IK config directory. Set "
        "ELASTICSEARCH_IK_CONFIG_DIR to Elasticsearch's config/analysis-ik path."
    )


def _enable_ik_extension_dictionary(config_path: Path, dictionary_name: str) -> bool:
    config_text = config_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r'(<entry\s+key=["\']ext_dict["\']\s*>)(.*?)(</entry>)',
        flags=re.DOTALL,
    )
    match = pattern.search(config_text)
    if match is None:
        raise RuntimeError(f"IK config has no ext_dict entry: {config_path}")
    configured_names = [
        value.strip() for value in match.group(2).split(";") if value.strip()
    ]
    if dictionary_name in configured_names:
        return False
    configured_names.append(dictionary_name)
    updated_text = pattern.sub(
        lambda current: (
            f"{current.group(1)}{';'.join(configured_names)}{current.group(3)}"
        ),
        config_text,
        count=1,
    )
    config_path.write_text(updated_text, encoding="utf-8")
    return True


def sync_ik_dictionary(
    client: Any,
    rows: Iterable[KnowledgeRow],
) -> dict[str, Any]:
    keywords = collect_knowledge_keywords(rows)
    if not keywords:
        raise RuntimeError("Knowledge CSV contains no keywords for the IK dictionary.")

    config_dir = discover_local_ik_config_dir(client)
    dictionary_path = config_dir / IK_DICTIONARY_FILENAME
    dictionary_text = "\n".join(keywords) + "\n"
    dictionary_changed = (
        not dictionary_path.exists()
        or dictionary_path.read_text(encoding="utf-8") != dictionary_text
    )
    if dictionary_changed:
        dictionary_path.write_text(dictionary_text, encoding="utf-8")

    config_changed = _enable_ik_extension_dictionary(
        config_dir / "IKAnalyzer.cfg.xml",
        IK_DICTIONARY_FILENAME,
    )
    return {
        "dictionary_path": str(dictionary_path),
        "keyword_count": len(keywords),
        "dictionary_changed": dictionary_changed,
        "config_changed": config_changed,
        "restart_required": dictionary_changed or config_changed,
    }


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
                print(
                    f"[elasticsearch] using {candidate} "
                    f"instead of {es_url!r} ({reason})",
                    file=sys.stderr,
                )
            return candidate
        errors.append(f"{candidate}: {reason}")

    raise ConnectionError(
        "Unable to detect a working Elasticsearch HTTP endpoint:\n"
        + "\n".join(f"- {error}" for error in errors)
    )


def create_es_connection(
    *,
    es_url: str = DEFAULT_ES_URL,
    api_key: str = DEFAULT_ES_API_KEY,
    username: str = DEFAULT_ES_USERNAME,
    password: str = DEFAULT_ES_PASSWORD,
    request_timeout: float = 30.0,
    verify_certs: bool = DEFAULT_ES_VERIFY_CERTS,
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
    return connections.create_connection(alias="knowledge", **kwargs)


def preload_knowledge_runtime(
    *,
    embedding_model: str | Path | None = None,
    es_url: str = DEFAULT_ES_URL,
    api_key: str = DEFAULT_ES_API_KEY,
    username: str = DEFAULT_ES_USERNAME,
    password: str = DEFAULT_ES_PASSWORD,
    request_timeout: float = 30.0,
    verify_certs: bool = DEFAULT_ES_VERIFY_CERTS,
    knowledge_index: str = DEFAULT_KNOWLEDGE_INDEX,
    vector_index: str = DEFAULT_VECTOR_INDEX,
) -> KnowledgeRuntime:
    client = create_es_connection(
        es_url=es_url,
        api_key=api_key,
        username=username,
        password=password,
        request_timeout=request_timeout,
        verify_certs=verify_certs,
    )
    embedder = BGEEmbedder(embedding_model or default_embedding_model())
    return KnowledgeRuntime(
        client=client,
        embedder=embedder,
        embedding_dims=embedder.embedding_dims(),
        knowledge_index=knowledge_index,
        vector_index=vector_index,
    )


def recreate_index(client: Any, index_name: str, document_class: type[Document]) -> None:
    index = Index(index_name, using=client)
    index.document(document_class)
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
    index.create()


def row_to_source(row: KnowledgeRow) -> dict[str, Any]:
    return {
        "knowledge_id": row.knowledge_id,
        "usage": row.usage,
        "category": row.category,
        "title": row.title,
        "keywords": row.keywords,
        "keyword_terms": row.keyword_terms,
        "keyword_text": row.keyword_text,
        "semantic_descriptions": row.semantic_descriptions,
        "body": row.body,
        "priority": row.priority,
        "updated_at": row.updated_at,
    }


def iter_knowledge_actions(rows: Iterable[KnowledgeRow], index_name: str):
    for row in rows:
        yield {
            "_op_type": "index",
            "_index": index_name,
            "_id": row.knowledge_id,
            "_source": row_to_source(row),
        }


def iter_vector_actions(
    rows: list[KnowledgeRow],
    embeddings_by_key: dict[tuple[str, int], list[float]],
    index_name: str,
):
    for row in rows:
        for index, description in enumerate(row.semantic_descriptions):
            yield {
                "_op_type": "index",
                "_index": index_name,
                "_id": f"{row.knowledge_id}#{index}",
                "_source": {
                    "knowledge_id": row.knowledge_id,
                    "description_index": str(index),
                    "description": description,
                    "title": row.title,
                    "category": row.category,
                    "keywords": row.keywords,
                    "embedding": embeddings_by_key[(row.knowledge_id, index)],
                },
            }


def import_knowledge(
    *,
    csv_path: str | Path = DEFAULT_CSV_PATH,
    semantic_fallback: str = "title_body",
    runtime: KnowledgeRuntime | None = None,
    embedding_model: str | Path | None = None,
    es_url: str = DEFAULT_ES_URL,
    api_key: str = DEFAULT_ES_API_KEY,
    username: str = DEFAULT_ES_USERNAME,
    password: str = DEFAULT_ES_PASSWORD,
    request_timeout: float = 30.0,
    verify_certs: bool = DEFAULT_ES_VERIFY_CERTS,
    knowledge_index: str | None = None,
    vector_index: str | None = None,
) -> dict[str, Any]:
    csv_path = Path(csv_path).resolve()
    rows = read_knowledge_csv(csv_path, semantic_fallback)
    if not rows:
        raise RuntimeError(f"No knowledge rows found in {csv_path}")

    if runtime is None:
        runtime = preload_knowledge_runtime(
            embedding_model=embedding_model,
            es_url=es_url,
            api_key=api_key,
            username=username,
            password=password,
            request_timeout=request_timeout,
            verify_certs=verify_certs,
        )

    client = runtime.client
    embedder = runtime.embedder
    knowledge_index = knowledge_index or runtime.knowledge_index
    vector_index = vector_index or runtime.vector_index
    ik_dictionary = sync_ik_dictionary(client, rows)
    if ik_dictionary["restart_required"]:
        raise RuntimeError(
            "IK keyword dictionary was updated. Restart every Elasticsearch node "
            "and run the import again so indexing uses the new dictionary. "
            f"Dictionary: {ik_dictionary['dictionary_path']}"
        )
    vector_document = build_vector_document_class(runtime.embedding_dims)
    recreate_index(client, knowledge_index, KnowledgeDocument)
    recreate_index(client, vector_index, vector_document)

    semantic_items = [
        (row.knowledge_id, index, description)
        for row in rows
        for index, description in enumerate(row.semantic_descriptions)
    ]
    embeddings_by_key: dict[tuple[str, int], list[float]] = {}
    if semantic_items:
        vectors = embedder.encode_documents([item[2] for item in semantic_items])
        for (knowledge_id, index, _), vector in zip(semantic_items, vectors):
            embeddings_by_key[(knowledge_id, index)] = vector

    knowledge_count, _ = bulk(
        client,
        iter_knowledge_actions(rows, knowledge_index),
        refresh=True,
    )
    vector_count = 0
    if semantic_items:
        vector_count, _ = bulk(
            client,
            iter_vector_actions(rows, embeddings_by_key, vector_index),
            refresh=True,
        )

    return {
        "csv": str(csv_path),
        "knowledge_index": knowledge_index,
        "vector_index": vector_index,
        "knowledge_rows": len(rows),
        "knowledge_documents_indexed": knowledge_count,
        "semantic_vectors_indexed": vector_count,
        "embedding_dims": runtime.embedding_dims,
        "ik_dictionary": ik_dictionary,
        "cleared_before_import": True,
    }


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
    min_score: float = DEFAULT_SEMANTIC_MIN_SCORE,
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
    top_k: int = 3,
    query_prefix: str = DEFAULT_QUERY_PREFIX,
    semantic_min_score: float = DEFAULT_SEMANTIC_MIN_SCORE,
    knowledge_index: str | None = None,
    vector_index: str | None = None,
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    top_k = max(1, min(3, int(top_k)))
    candidate_size = max(top_k * 5, 10)
    client = runtime.client
    embedder = runtime.embedder
    knowledge_index = knowledge_index or runtime.knowledge_index
    vector_index = vector_index or runtime.vector_index

    keyword_started = time.perf_counter()
    keyword_hits = keyword_search(
        client,
        index_name=knowledge_index,
        query=query,
        candidate_size=candidate_size,
    )
    keyword_seconds = time.perf_counter() - keyword_started

    semantic_started = time.perf_counter()
    semantic_hits = semantic_search(
        client,
        embedder=embedder,
        index_name=vector_index,
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
        client,
        index_name=knowledge_index,
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
    print("[search-timing]")
    print(
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
        )
    )
    return results


if __name__ == "__main__":
    # The default URL is "auto": the script probes HTTP and HTTPS on localhost:9200.
    # If your Elasticsearch enables authentication, set one of:
    #   $env:ELASTICSEARCH_PASSWORD="your elastic password"
    #   $env:ELASTICSEARCH_API_KEY="your api key"
    # Optional overrides:
    #   $env:ELASTICSEARCH_URL="auto"
    #   $env:ELASTICSEARCH_URL="http://127.0.0.1:9200"
    #   $env:ELASTICSEARCH_URL="https://127.0.0.1:9200"
    #   $env:ELASTICSEARCH_VERIFY_CERTS="false"
    #
    preload_started = time.perf_counter()
    runtime = preload_knowledge_runtime(
        es_url=DEFAULT_ES_URL,
        username=DEFAULT_ES_USERNAME,
        password=DEFAULT_ES_PASSWORD,
        api_key=DEFAULT_ES_API_KEY,
        verify_certs=DEFAULT_ES_VERIFY_CERTS,
        embedding_model=default_embedding_model(),
        knowledge_index=DEFAULT_KNOWLEDGE_INDEX,
        vector_index=DEFAULT_VECTOR_INDEX,
    )
    print("[preload]")
    print(
        json.dumps(
            {
                "embedding_dims": runtime.embedding_dims,
                "knowledge_index": runtime.knowledge_index,
                "vector_index": runtime.vector_index,
                "elapsed_ms": round((time.perf_counter() - preload_started) * 1000, 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    # Example 1: full import resource/knowledge.csv into Elasticsearch.
    # This always deletes and recreates DEFAULT_KNOWLEDGE_INDEX and
    # DEFAULT_VECTOR_INDEX first, so old documents/vectors will not remain.
    import_summary = import_knowledge(
        runtime=runtime,
        csv_path=DEFAULT_CSV_PATH,
    )
    print("[import]")
    print(json.dumps(import_summary, ensure_ascii=False, indent=2))

    # Example 2: hybrid keyword + semantic search. The query returns 1-3 rows.
    while True:
        search_str = input()
        search_results = search_knowledge(
            runtime=runtime,
            query=search_str,
            top_k=3,
            semantic_min_score=DEFAULT_SEMANTIC_MIN_SCORE,
        )
        print("[search]")
        print(json.dumps(search_results, ensure_ascii=False, indent=2))
