"""文件用途：使用 OpenSearch BM25 和 HNSW 实现 RAG 第一阶段混合召回。

OpenSearch owns the scalable indexes: Lucene BM25 for lexical retrieval and
Lucene HNSW for approximate vector retrieval. SQLite remains the source for
parent chunks, provenance, claims, and graph traversal after candidate IDs
have been selected.
"""

from __future__ import annotations

import base64
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable


INDEX_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class OpenSearchError(RuntimeError):
    """OpenSearch 拒绝索引或搜索操作时抛出的统一异常。"""


@dataclass(frozen=True)
class SearchHit:
    """OpenSearch 第一阶段召回返回的分块标识和分数。"""
    chunk_id: str
    score: float


def _normalized_filter_values(value: Any) -> Iterable[str]:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    for item in values:
        if item is None:
            continue
        if isinstance(item, (dict, list, tuple, set)):
            item = json.dumps(item, ensure_ascii=False, sort_keys=True)
        yield str(item).strip().casefold()


def filter_token(key: str, value: Any) -> list[str]:
    """把元数据过滤条件编码为 OpenSearch 可精确匹配的 token。"""
    normalized_key = str(key).strip().casefold()
    return [f"{normalized_key}={item}" for item in _normalized_filter_values(value)]


class OpenSearchHybridRetriever:
    """不依赖第三方客户端库的 OpenSearch REST 混合检索器。"""

    def __init__(self) -> None:
        self.base_url = os.getenv("RAG_OPENSEARCH_URL", "http://localhost:9200").rstrip("/")
        self.alias = os.getenv("RAG_OPENSEARCH_INDEX", "knowledge-rag").strip().casefold()
        if not INDEX_NAME.fullmatch(self.alias):
            raise ValueError("RAG_OPENSEARCH_INDEX 只能包含小写字母、数字、下划线和连字符")
        self.username = os.getenv("RAG_OPENSEARCH_USERNAME", "").strip()
        self.password = os.getenv("RAG_OPENSEARCH_PASSWORD", "")
        self.timeout = float(os.getenv("RAG_OPENSEARCH_TIMEOUT_SECONDS", "10"))
        self.verify_tls = os.getenv("RAG_OPENSEARCH_VERIFY_TLS", "true").casefold() in {
            "1", "true", "yes", "on",
        }

    def _request(
        self,
        method: str,
        path: str,
        payload: Any | None = None,
        *,
        content_type: str = "application/json",
        allowed: tuple[int, ...] = (200,),
    ) -> tuple[int, Any]:
        if isinstance(payload, bytes):
            body = payload
        elif payload is None:
            body = None
        else:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = content_type
        if self.username:
            token = base64.b64encode(f"{self.username}:{self.password}".encode("utf-8")).decode("ascii")
            headers["Authorization"] = f"Basic {token}"
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=body, headers=headers, method=method,
        )
        context = None
        if self.base_url.startswith("https://") and not self.verify_tls:
            context = ssl._create_unverified_context()  # noqa: SLF001 - explicitly configured for local dev.
        try:
            with urllib.request.urlopen(request, timeout=self.timeout, context=context) as response:
                raw = response.read()
                parsed = json.loads(raw.decode("utf-8")) if raw else {}
                status = int(response.status)
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                parsed = json.loads(raw.decode("utf-8")) if raw else {}
            except (UnicodeDecodeError, json.JSONDecodeError):
                parsed = {"error": raw.decode("utf-8", errors="replace")[:1000]}
            status = int(exc.code)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise OpenSearchError(f"无法连接 OpenSearch: {exc}") from exc
        if status not in allowed:
            raise OpenSearchError(f"OpenSearch {method} {path} 返回 {status}: {parsed}")
        return status, parsed

    def status(self, expected_fingerprint: str) -> dict[str, Any]:
        """检查索引是否存在、是否有数据且对应当前语料指纹。"""
        status, mapping = self._request("GET", f"/{self.alias}/_mapping", allowed=(200, 404))
        if status == 404:
            return {"ready": False, "reason": "opensearch_index_missing", "documents": 0}
        indexes = list(mapping.values())
        fingerprints = {
            str(item.get("mappings", {}).get("_meta", {}).get("corpus_fingerprint", ""))
            for item in indexes
        }
        _, count = self._request("GET", f"/{self.alias}/_count")
        documents = int(count.get("count", 0))
        ready = fingerprints == {expected_fingerprint} and documents > 0
        return {
            "ready": ready,
            "reason": "ready" if ready else "opensearch_index_stale",
            "documents": documents,
            "indexes": sorted(mapping),
        }

    def rebuild(
        self,
        rows: list[dict[str, Any]],
        vectors: list[list[float]],
        *,
        corpus_fingerprint: str,
        embedding_fingerprint: str,
        dimension: int,
    ) -> dict[str, Any]:
        """新建版本化索引、批量写入文档并切换稳定别名。"""
        if len(rows) != len(vectors):
            raise ValueError("OpenSearch 索引文本和向量数量不一致")
        concrete_index = f"{self.alias}-{corpus_fingerprint[:16]}"
        exists, _ = self._request("HEAD", f"/{concrete_index}", allowed=(200, 404))
        if exists == 404:
            mapping = {
                "settings": {
                    "index": {
                        "knn": True,
                        "number_of_shards": int(os.getenv("RAG_OPENSEARCH_SHARDS", "1")),
                        "number_of_replicas": int(os.getenv("RAG_OPENSEARCH_REPLICAS", "0")),
                    }
                },
                "mappings": {
                    "dynamic": False,
                    "_meta": {
                        "corpus_fingerprint": corpus_fingerprint,
                        "embedding_fingerprint": embedding_fingerprint,
                        "created_at": int(time.time()),
                    },
                    "properties": {
                        "chunk_id": {"type": "keyword"},
                        "parent_id": {"type": "keyword"},
                        "document_path": {"type": "keyword"},
                        "title": {"type": "text"},
                        "section_text": {"type": "text"},
                        "search_text": {"type": "text"},
                        "filter_tokens": {"type": "keyword"},
                        "status": {"type": "keyword"},
                        "authority_priority": {"type": "float"},
                        "embedding": {
                            "type": "knn_vector",
                            "dimension": dimension,
                            "data_type": "float",
                            "method": {
                                "name": "hnsw",
                                "engine": "lucene",
                                "space_type": "cosinesimil",
                                "parameters": {
                                    "m": int(os.getenv("RAG_HNSW_M", "16")),
                                    "ef_construction": int(os.getenv("RAG_HNSW_EF_CONSTRUCTION", "100")),
                                },
                            },
                        },
                    },
                },
            }
            self._request("PUT", f"/{concrete_index}", mapping, allowed=(200, 201))

        batch_size = max(1, int(os.getenv("RAG_OPENSEARCH_BULK_SIZE", "200")))
        indexed = 0
        for start in range(0, len(rows), batch_size):
            lines: list[str] = []
            for row, vector in zip(rows[start:start + batch_size], vectors[start:start + batch_size]):
                metadata = dict(row["metadata"])
                tokens: list[str] = []
                for key, value in metadata.items():
                    tokens.extend(filter_token(key, value))
                document = {
                    "chunk_id": row["chunk_id"],
                    "parent_id": row["parent_id"],
                    "document_path": row["document_path"],
                    "title": row["title"],
                    "section_text": " > ".join(row["section_path"]),
                    "search_text": row["search_text"],
                    "filter_tokens": sorted(set(tokens)),
                    "status": str(metadata.get("status", "published")).casefold(),
                    "authority_priority": float(metadata.get("authority_priority", 50)),
                    "embedding": [float(value) for value in vector],
                }
                lines.append(json.dumps({"index": {"_id": row["chunk_id"]}}, separators=(",", ":")))
                lines.append(json.dumps(document, ensure_ascii=False, separators=(",", ":")))
            body = ("\n".join(lines) + "\n").encode("utf-8")
            _, response = self._request(
                "POST", f"/{concrete_index}/_bulk?refresh=wait_for", body,
                content_type="application/x-ndjson", allowed=(200,),
            )
            if response.get("errors"):
                failed = [
                    item for item in response.get("items", [])
                    if int(next(iter(item.values())).get("status", 500)) >= 300
                ]
                raise OpenSearchError(f"OpenSearch Bulk 写入失败: {failed[:3]}")
            indexed += len(lines) // 2

        alias_status, aliases = self._request("GET", f"/_alias/{self.alias}", allowed=(200, 404))
        old_indexes = set(aliases) if alias_status == 200 else set()
        actions = [
            {"remove": {"index": index, "alias": self.alias}}
            for index in sorted(old_indexes - {concrete_index})
        ]
        if concrete_index not in old_indexes:
            actions.append({"add": {"index": concrete_index, "alias": self.alias, "is_write_index": True}})
        if actions:
            self._request("POST", "/_aliases", {"actions": actions})
        return {
            "backend": "opensearch",
            "index": concrete_index,
            "alias": self.alias,
            "documents": indexed,
            "vector_index": "hnsw-lucene-cosine",
        }

    def search(
        self,
        query: str,
        vector: list[float],
        *,
        metadata_filters: dict[str, Any] | None = None,
        include_superseded: bool = False,
        limit: int = 100,
    ) -> tuple[list[SearchHit], list[SearchHit]]:
        size = max(1, min(int(limit), 1000))
        filters = [
            {"term": {"filter_tokens": token}}
            for key, value in (metadata_filters or {}).items()
            if value not in (None, "")
            for token in filter_token(key, value)
        ]
        must_not = [] if include_superseded else [{"term": {"status": "superseded"}}]
        keyword_query = {
            "size": size,
            "track_total_hits": False,
            "_source": False,
            "query": {
                "bool": {
                    "must": [{
                        "multi_match": {
                            "query": query,
                            "fields": ["search_text^3", "title^2", "section_text^2"],
                            "type": "best_fields",
                        }
                    }],
                    "filter": filters,
                    "must_not": must_not,
                }
            },
        }
        vector_filter: dict[str, Any] | None = None
        if filters or must_not:
            vector_filter = {"bool": {"filter": filters, "must_not": must_not}}
        vector_clause: dict[str, Any] = {"vector": vector, "k": size}
        if vector_filter is not None:
            vector_clause["filter"] = vector_filter
        vector_query = {
            "size": size,
            "track_total_hits": False,
            "_source": False,
            "query": {"knn": {"embedding": vector_clause}},
        }
        _, keyword_response = self._request("POST", f"/{self.alias}/_search", keyword_query)
        _, vector_response = self._request("POST", f"/{self.alias}/_search", vector_query)

        def hits(payload: dict[str, Any]) -> list[SearchHit]:
            return [
                SearchHit(str(item["_id"]), float(item.get("_score") or 0.0))
                for item in payload.get("hits", {}).get("hits", [])
            ]

        return hits(keyword_response), hits(vector_response)
