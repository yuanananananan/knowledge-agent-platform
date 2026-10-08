"""文件用途：编排 OpenSearch 混合召回、重排、图谱扩展、缓存和 SQLite 来源存储。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
import time
import yaml
from array import array
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.rag import graph as knowledge_graph
from app.rag.embedding import EmbeddingProvider, build_embedding_provider, lexical_tokens
from app.rag.loaders import load_sections, split_section
from app.rag.opensearch import OpenSearchError, OpenSearchHybridRetriever
from app.rag.query_expansion import QueryExpander, build_query_expander
from app.rag.query_planning import QueryPlanner, build_query_plan, build_query_planner, route_query
from app.rag.knowledge_schema import evidence_role, extract_claims, extract_entities
from app.rag.reranker import Reranker, build_reranker
from app.rag.answerability import EvidenceAssessment, assess_evidence, build_evidence_assessor
from app.utils.redis_cache import RedisJSONCache


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_ROOT = PROJECT_ROOT / "rag_knowledge_base"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "instance" / "rag_index.sqlite"
INDEX_SCHEMA_VERSION = "generic-entity-claim-opensearch-v5"
SEARCH_CACHE_ENV_KEYS = (
    "RAG_SEARCH_BACKEND",
    "RAG_OPENSEARCH_INDEX",
    "RAG_RETRIEVAL_CANDIDATES",
    "RAG_HNSW_M",
    "RAG_HNSW_EF_CONSTRUCTION",
    "RAG_QUERY_EXPANSION_ENABLED",
    "RAG_FUSION_QUERY_COUNT",
    "RAG_TOTAL_QUERY_LIMIT",
    "RAG_FUSION_ORIGINAL_QUERY_WEIGHT",
    "RAG_COMPLEX_QUERY_MIN_CHARS",
    "RAG_RERANKER_MODEL",
    "RAG_RERANK_CANDIDATES",
    "RAG_RERANK_WEIGHT",
    "RAG_MAX_CHUNKS_PER_DOCUMENT",
    "RAG_GRAPH_ENABLED",
    "RAG_GRAPH_CANDIDATES",
    "RAG_MMR_LAMBDA",
    "RAG_CLAIM_NOVELTY_WEIGHT",
    "RAG_ENTITY_COVERAGE_WEIGHT",
    "RAG_REQUIRED_CLAIM_WEIGHT",
    "RAG_ROLE_COVERAGE_WEIGHT",
    "RAG_SUBQUERY_RESERVE_POOL",
    "RAG_ENTITY_LINK_THRESHOLD",
)


@dataclass
class SearchResult:
    """一次检索命中的完整结果，包括分数、来源、实体和图谱路径。"""
    chunk_id: str
    parent_id: str
    document_path: str
    title: str
    section_path: list[str]
    content: str
    matched_content: str
    score: float
    keyword_score: float
    vector_score: float
    metadata: dict[str, Any]
    entity_ids: list[str]
    claim_ids: list[str]
    authority: float
    graph_paths: list[dict[str, str]] = field(default_factory=list)
    query_route: str = ""
    query_expansion_used: bool = False
    query_planner_used: str = ""
    page_start: int | None = None
    page_end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pack(vector: list[float]) -> bytes:
    return array("f", vector).tobytes()


def _unpack(blob: bytes) -> list[float]:
    values = array("f")
    values.frombytes(blob)
    return list(values)


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


class HybridKnowledgeStore:
    """统一管理知识索引构建、混合检索、重排、图谱扩展和证据判定。"""
    def __init__(
        self,
        corpus_root: Path | str | None = None,
        index_path: Path | str | None = None,
        embedding: EmbeddingProvider | None = None,
        query_expander: QueryExpander | None = None,
        reranker: Reranker | None = None,
        query_planner: QueryPlanner | None = None,
        search_retriever: OpenSearchHybridRetriever | None = None,
    ):
        configured_corpus = Path(corpus_root or os.getenv("RAG_CORPUS_PATH", DEFAULT_CORPUS_ROOT))
        configured_index = Path(index_path or os.getenv("RAG_INDEX_PATH", DEFAULT_INDEX_PATH))
        self.corpus_root = (configured_corpus if configured_corpus.is_absolute() else PROJECT_ROOT / configured_corpus).resolve()
        self.index_path = (configured_index if configured_index.is_absolute() else PROJECT_ROOT / configured_index).resolve()
        self.embedding = embedding or build_embedding_provider()
        self.query_expander = query_expander or build_query_expander()
        self.reranker = reranker or build_reranker()
        self.query_planner = query_planner or build_query_planner()
        self.evidence_assessor = build_evidence_assessor()
        self.search_cache = RedisJSONCache("rag-search")
        self.search_backend = os.getenv("RAG_SEARCH_BACKEND", "opensearch").strip().casefold()
        if self.search_backend not in {"opensearch", "local"}:
            raise ValueError("RAG_SEARCH_BACKEND 只支持 opensearch 或 local")
        self.search_retriever = (
            search_retriever or OpenSearchHybridRetriever()
            if self.search_backend == "opensearch" else None
        )
        self._lock = threading.RLock()

    def _connect(self) -> sqlite3.Connection:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.index_path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _schema(self, connection: sqlite3.Connection) -> None:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS rag_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rag_chunks (
                chunk_id TEXT PRIMARY KEY,
                parent_id TEXT,
                document_path TEXT NOT NULL,
                title TEXT NOT NULL,
                section_path TEXT NOT NULL,
                content TEXT NOT NULL,
                search_text TEXT NOT NULL,
                vector BLOB NOT NULL,
                metadata TEXT NOT NULL,
                page_start INTEGER,
                page_end INTEGER
            );
            CREATE TABLE IF NOT EXISTS rag_parents (
                parent_id TEXT PRIMARY KEY,
                document_path TEXT NOT NULL,
                title TEXT NOT NULL,
                section_path TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL,
                page_start INTEGER,
                page_end INTEGER
            );
            CREATE TABLE IF NOT EXISTS rag_entities (
                entity_id TEXT PRIMARY KEY,
                aliases TEXT NOT NULL,
                profile TEXT NOT NULL,
                vector BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rag_claims (
                claim_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                claim_text TEXT NOT NULL,
                PRIMARY KEY (claim_id, chunk_id)
            );
            CREATE INDEX IF NOT EXISTS ix_rag_document_path ON rag_chunks(document_path);
        """)
        knowledge_graph.schema(connection)
        columns = {row[1] for row in connection.execute("PRAGMA table_info(rag_chunks)")}
        if "parent_id" not in columns:
            connection.execute("ALTER TABLE rag_chunks ADD COLUMN parent_id TEXT")
        connection.execute("CREATE INDEX IF NOT EXISTS ix_rag_parent_id ON rag_chunks(parent_id)")

    def _manifest(self) -> dict[str, Any]:
        path = self.corpus_root / "corpus_manifest.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def _claim_registry(self) -> dict[str, Any]:
        path = self.corpus_root / "claim_registry.yaml"
        if not path.exists():
            return {"documents": {}}
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {"documents": {}}

    def corpus_fingerprint(self) -> str:
        digest = hashlib.sha256()
        manifest = self._manifest()
        claim_registry = self.corpus_root / "claim_registry.yaml"
        if claim_registry.exists():
            digest.update(claim_registry.read_bytes())
        for item in manifest["documents"]:
            if not item.get("ingest"):
                continue
            path = (self.corpus_root / item["path"]).resolve()
            if self.corpus_root not in path.parents:
                raise ValueError(f"知识文件越出语料目录: {item['path']}")
            digest.update(item["path"].encode("utf-8"))
            digest.update(path.read_bytes())
            digest.update(json.dumps(item, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(self.embedding.fingerprint.encode("utf-8"))
        digest.update(INDEX_SCHEMA_VERSION.encode("utf-8"))
        digest.update(json.dumps({
            "search_backend": self.search_backend,
            "hnsw_m": os.getenv("RAG_HNSW_M", "16"),
            "hnsw_ef_construction": os.getenv("RAG_HNSW_EF_CONSTRUCTION", "100"),
        }, sort_keys=True).encode("utf-8"))
        return digest.hexdigest()

    def status(self) -> dict[str, Any]:
        """检查本地来源库和搜索后端是否与当前语料保持一致。"""
        if not self.index_path.exists():
            return {"ready": False, "reason": "index_missing", "chunks": 0}
        with self._connect() as connection:
            self._schema(connection)
            meta = dict(connection.execute("SELECT key, value FROM rag_meta").fetchall())
            if meta.get("graph_version") != "1":
                knowledge_graph.rebuild(connection)
                connection.commit()
            graph_nodes = connection.execute("SELECT COUNT(*) FROM rag_graph_nodes").fetchone()[0]
            graph_edges = connection.execute("SELECT COUNT(*) FROM rag_graph_edges").fetchone()[0]
            chunks = connection.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
            parents = connection.execute("SELECT COUNT(*) FROM rag_parents").fetchone()[0]
            entities = connection.execute("SELECT COUNT(*) FROM rag_entities").fetchone()[0]
            claims = connection.execute("SELECT COUNT(DISTINCT claim_id) FROM rag_claims").fetchone()[0]
        current = self.corpus_fingerprint()
        local_ready = bool(chunks) and meta.get("corpus_fingerprint") == current
        backend_status: dict[str, Any] = {
            "ready": local_ready,
            "reason": "ready" if local_ready else "stale",
            "backend": self.search_backend,
        }
        if local_ready and self.search_retriever is not None:
            try:
                backend_status.update(self.search_retriever.status(current))
            except OpenSearchError as exc:
                backend_status.update({
                    "ready": False,
                    "reason": "opensearch_unavailable",
                    "error": str(exc),
                })
        ready = local_ready and bool(backend_status["ready"])
        return {
            "ready": ready,
            "reason": "ready" if ready else backend_status["reason"],
            "chunks": chunks,
            "parents": parents,
            "graph": {"nodes": graph_nodes, "edges": graph_edges, "version": "1"},
            "entities": entities,
            "claims": claims,
            "documents": int(meta.get("document_count", "0")),
            "embedding": meta.get("embedding"),
            "retrieval": {
                "backend": self.search_backend,
                "keyword": "OpenSearch BM25" if self.search_retriever else "application BM25",
                "vector": "OpenSearch Lucene HNSW" if self.search_retriever else "application cosine scan",
                "fusion": "RRF",
                "reranker": self.reranker.name,
                "query_expander": self.query_expander.name,
                "query_planner": self.query_planner.name,
                "entity_linking": "indexed-neural-profiles",
                "diversification": "MMR+claim-novelty",
                "metadata_filtering": True,
                "redis_cache": self.search_cache.enabled,
            },
            "search_backend": backend_status,
            "built_at": meta.get("built_at"),
            "index_path": str(self.index_path),
        }

    def ensure_ready(self) -> None:
        """确保索引可用；发现未初始化或过期时触发重建。"""
        with self._lock:
            if not self.status()["ready"]:
                self.rebuild()

    def rebuild(self) -> dict[str, Any]:
        """重新读取语料、生成向量并原子更新本地及 OpenSearch 索引。"""
        with self._lock:
            manifest = self._manifest()
            claim_registry = self._claim_registry().get("documents", {})
            rows: list[dict[str, Any]] = []
            parents: dict[str, dict[str, Any]] = {}
            entity_profiles: dict[str, dict[str, Any]] = {}
            document_count = 0
            errors: list[dict[str, str]] = []
            max_chars = int(os.getenv("RAG_CHUNK_MAX_CHARS", "1600"))
            overlap = int(os.getenv("RAG_CHUNK_OVERLAP_CHARS", "180"))
            for document in manifest["documents"]:
                if not document.get("ingest"):
                    continue
                relative = str(document["path"])
                registered_document = claim_registry.get(relative, {})
                path = (self.corpus_root / relative).resolve()
                if self.corpus_root not in path.parents or not path.is_file():
                    errors.append({"path": relative, "error": "文件不存在或越出语料目录"})
                    continue
                try:
                    sections = load_sections(path)
                except Exception as exc:
                    errors.append({"path": relative, "error": str(exc)})
                    continue
                document_count += 1
                title = path.stem
                parent_depth = 2 if document.get("document_type") in {"modality_guide", "operator_catalog"} else 1
                grouped_sections: dict[tuple[str, ...], list[Any]] = {}
                for section in sections:
                    key = tuple((section.section_path or [title])[:parent_depth])
                    grouped_sections.setdefault(key, []).append(section)
                section_parents: dict[int, str] = {}
                for parent_path, members in grouped_sections.items():
                    parent_text = "\n\n".join(
                        f"{' > '.join(member.section_path)}\n{member.text}".strip()
                        for member in members
                    )
                    parent_raw = f"{relative}:{parent_path}:{parent_text}"
                    parent_id = hashlib.sha256(parent_raw.encode("utf-8")).hexdigest()[:24]
                    parent_metadata = dict(document)
                    parents[parent_id] = {
                        "parent_id": parent_id,
                        "document_path": relative,
                        "title": title,
                        "section_path": list(parent_path),
                        "content": parent_text,
                        "metadata": parent_metadata,
                        "page_start": min((item.page_start for item in members if item.page_start), default=None),
                        "page_end": max((item.page_end for item in members if item.page_end), default=None),
                    }
                    for member in members:
                        section_parents[id(member)] = parent_id
                for section_number, section in enumerate(sections):
                    for part_number, part in enumerate(split_section(section, max_chars, overlap)):
                        section_path = part.section_path or [title]
                        # Manifest fields are reviewed ingestion metadata and take
                        # precedence over untrusted document front matter.
                        metadata = {
                            **part.metadata,
                            **{key: value for key, value in registered_document.items() if key != "sections"},
                            **document,
                        }
                        section_claims: list[str] = []
                        for heading in section_path:
                            section_claims.extend(
                                registered_document.get("sections", {}).get(str(heading), [])
                            )
                        if section_claims:
                            metadata["supported_claims"] = list(dict.fromkeys(section_claims))
                        if metadata.get("name") and not metadata.get("operator"):
                            metadata["operator"] = metadata["name"]
                        prefix = " > ".join([title, *section_path])
                        search_text = f"{prefix}\n{part.text}".strip()
                        raw_id = f"{relative}:{section_number}:{part_number}:{search_text}"
                        chunk_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:24]
                        discovered = extract_entities(metadata, title, part.text)
                        entity_ids = [item["entity_id"] for item in discovered]
                        for item in discovered:
                            current = entity_profiles.setdefault(item["entity_id"], {
                                "aliases": set(), "profiles": [],
                            })
                            current["aliases"].update(part.strip() for part in item["aliases"].split("|") if part.strip())
                            current["profiles"].append(item["profile"])
                        claims = extract_claims({**metadata, "path": relative}, entity_ids, section_path, part.text)
                        metadata["entity_ids"] = entity_ids
                        metadata["claim_ids"] = [item["claim_id"] for item in claims]
                        metadata["evidence_role"] = evidence_role(metadata)
                        metadata["authority_priority"] = float(metadata.get("authority_priority", 50))
                        rows.append({
                            "chunk_id": chunk_id,
                            "parent_id": section_parents[id(section)],
                            "document_path": relative,
                            "title": title,
                            "section_path": section_path,
                            "content": part.text,
                            "search_text": search_text,
                            "metadata": metadata,
                            "page_start": part.page_start,
                            "page_end": part.page_end,
                            "claims": claims,
                        })
            if not rows:
                raise RuntimeError(f"没有生成知识片段: {errors}")
            parent_claims: dict[str, set[str]] = {}
            parent_entities: dict[str, set[str]] = {}
            for row in rows:
                parent_claims.setdefault(row["parent_id"], set()).update(row["metadata"].get("claim_ids", []))
                parent_entities.setdefault(row["parent_id"], set()).update(row["metadata"].get("entity_ids", []))
            for row in rows:
                row["metadata"]["parent_claim_ids"] = sorted(parent_claims.get(row["parent_id"], set()))
                row["metadata"]["parent_entity_ids"] = sorted(parent_entities.get(row["parent_id"], set()))
            for parent_id, parent in parents.items():
                parent["metadata"]["claim_ids"] = sorted(parent_claims.get(parent_id, set()))
                parent["metadata"]["entity_ids"] = sorted(parent_entities.get(parent_id, set()))
            batch_size = max(1, int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "32")))
            vectors: list[list[float]] = []
            for start in range(0, len(rows), batch_size):
                vectors.extend(self.embedding.embed([row["search_text"] for row in rows[start:start + batch_size]]))
            if len(vectors) != len(rows):
                raise RuntimeError("Embedding 返回数量与片段数量不一致")
            entity_rows = []
            entity_items = sorted(entity_profiles.items())
            entity_texts = ["\n".join(dict.fromkeys(item[1]["profiles"]))[:4000] for item in entity_items]
            entity_vectors = self.embedding.embed(entity_texts) if entity_texts else []
            for (entity_id, item), profile, vector in zip(entity_items, entity_texts, entity_vectors):
                entity_rows.append((entity_id, " | ".join(sorted(item["aliases"])), profile, _pack(vector)))
            fingerprint = self.corpus_fingerprint()
            with self._connect() as connection:
                self._schema(connection)
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("DELETE FROM rag_chunks")
                connection.execute("DELETE FROM rag_parents")
                connection.execute("DELETE FROM rag_entities")
                connection.execute("DELETE FROM rag_claims")
                connection.executemany(
                    """INSERT INTO rag_parents (
                        parent_id, document_path, title, section_path, content,
                        metadata, page_start, page_end
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    [(
                        parent["parent_id"], parent["document_path"], parent["title"],
                        json.dumps(parent["section_path"], ensure_ascii=False), parent["content"],
                        json.dumps(parent["metadata"], ensure_ascii=False),
                        parent["page_start"], parent["page_end"],
                    ) for parent in parents.values()],
                )
                connection.executemany(
                    """INSERT INTO rag_chunks (
                        chunk_id, parent_id, document_path, title, section_path, content,
                        search_text, vector, metadata, page_start, page_end
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [(
                        row["chunk_id"], row["parent_id"], row["document_path"], row["title"],
                        json.dumps(row["section_path"], ensure_ascii=False), row["content"],
                        row["search_text"], _pack(vector),
                        json.dumps(row["metadata"], ensure_ascii=False),
                        row["page_start"], row["page_end"],
                    ) for row, vector in zip(rows, vectors)],
                )
                connection.executemany(
                    "INSERT INTO rag_entities(entity_id, aliases, profile, vector) VALUES (?, ?, ?, ?)",
                    entity_rows,
                )
                connection.executemany(
                    "INSERT INTO rag_claims(claim_id, chunk_id, claim_text) VALUES (?, ?, ?)",
                    [(claim["claim_id"], row["chunk_id"], claim["claim_text"])
                     for row in rows for claim in row["claims"]],
                )
                knowledge_graph.rebuild(connection)
                meta = {
                    "corpus_fingerprint": fingerprint,
                    "embedding": self.embedding.fingerprint,
                    "document_count": str(document_count),
                    "chunk_count": str(len(rows)),
                    "evidence_roles": json.dumps(sorted({
                        str(row["metadata"].get("evidence_role", "reference")).casefold()
                        for row in rows
                    }), ensure_ascii=False),
                    "built_at": str(int(time.time())),
                    "errors": json.dumps(errors, ensure_ascii=False),
                }
                connection.executemany(
                    "INSERT OR REPLACE INTO rag_meta(key, value) VALUES (?, ?)",
                    list(meta.items()),
                )
                connection.commit()
            search_backend_result: dict[str, Any] = {"backend": "local"}
            if self.search_retriever is not None:
                search_backend_result = self.search_retriever.rebuild(
                    rows,
                    vectors,
                    corpus_fingerprint=fingerprint,
                    embedding_fingerprint=self.embedding.fingerprint,
                    dimension=len(vectors[0]),
                )
            return {
                "ready": True,
                "documents": document_count,
                "chunks": len(rows),
                "parents": len(parents),
                "entities": len(entity_rows),
                "claims": len({claim["claim_id"] for row in rows for claim in row["claims"]}),
                "embedding": self.embedding.fingerprint,
                "search_backend": search_backend_result,
                "errors": errors,
                "index_path": str(self.index_path),
            }

    def _rows(self) -> list[sqlite3.Row]:
        self.ensure_ready()
        with self._connect() as connection:
            return connection.execute("""
                SELECT c.*, p.content AS parent_content,
                       p.section_path AS parent_section_path,
                       p.page_start AS parent_page_start,
                       p.page_end AS parent_page_end
                FROM rag_chunks c
                LEFT JOIN rag_parents p ON p.parent_id = c.parent_id
            """).fetchall()

    def _rows_by_ids(self, chunk_ids: list[str]) -> list[tuple[sqlite3.Row, dict[str, Any]]]:
        """Load only candidates returned by OpenSearch, preserving rank order."""
        if not chunk_ids:
            return []
        found: dict[str, sqlite3.Row] = {}
        with self._connect() as connection:
            for start in range(0, len(chunk_ids), 500):
                batch = chunk_ids[start:start + 500]
                placeholders = ",".join("?" for _ in batch)
                for row in connection.execute(f"""
                    SELECT c.*, p.content AS parent_content,
                           p.section_path AS parent_section_path,
                           p.page_start AS parent_page_start,
                           p.page_end AS parent_page_end
                    FROM rag_chunks c
                    LEFT JOIN rag_parents p ON p.parent_id = c.parent_id
                    WHERE c.chunk_id IN ({placeholders})
                """, batch):
                    found[str(row["chunk_id"])] = row
        return [
            (found[chunk_id], json.loads(found[chunk_id]["metadata"]))
            for chunk_id in chunk_ids if chunk_id in found
        ]

    def _opensearch_candidates(
        self,
        expanded_queries: list[str],
        query_vectors: list[list[float]],
        metadata_filters: dict[str, Any],
        include_superseded: bool,
        top_k: int,
    ) -> tuple[list[tuple[sqlite3.Row, dict[str, Any]]], list[float], list[float], list[float], list[list[int]]]:
        if self.search_retriever is None:
            raise RuntimeError("OpenSearch 检索器未配置")
        candidate_limit = max(
            top_k * 8,
            int(os.getenv("RAG_RERANK_CANDIDATES", "15")) * 4,
            int(os.getenv("RAG_RETRIEVAL_CANDIDATES", "100")),
        )
        candidate_limit = min(candidate_limit, 1000)
        fused_by_id: dict[str, float] = defaultdict(float)
        first_keyword_scores: dict[str, float] = {}
        first_vector_scores: dict[str, float] = {}
        per_query_ids: list[list[str]] = []
        for query_number, (expanded_query, query_vector) in enumerate(zip(expanded_queries, query_vectors)):
            keyword_hits, vector_hits = self.search_retriever.search(
                expanded_query,
                query_vector,
                metadata_filters=metadata_filters,
                include_superseded=include_superseded,
                limit=candidate_limit,
            )
            query_weight = (
                float(os.getenv("RAG_FUSION_ORIGINAL_QUERY_WEIGHT", "2.0"))
                if query_number == 0 else 1.0
            )
            current: dict[str, float] = defaultdict(float)
            for rank, hit in enumerate(keyword_hits, start=1):
                contribution = query_weight / (60 + rank)
                fused_by_id[hit.chunk_id] += contribution
                current[hit.chunk_id] += contribution
                if query_number == 0:
                    first_keyword_scores[hit.chunk_id] = hit.score
            for rank, hit in enumerate(vector_hits, start=1):
                contribution = query_weight / (60 + rank)
                fused_by_id[hit.chunk_id] += contribution
                current[hit.chunk_id] += contribution
                if query_number == 0:
                    first_vector_scores[hit.chunk_id] = hit.score
            per_query_ids.append(sorted(current, key=current.get, reverse=True))

        ordered_ids = sorted(fused_by_id, key=fused_by_id.get, reverse=True)
        rows = self._rows_by_ids(ordered_ids)
        positions = {str(row["chunk_id"]): index for index, (row, _) in enumerate(rows)}
        keyword_scores = [first_keyword_scores.get(str(row["chunk_id"]), 0.0) for row, _ in rows]
        vector_scores = [first_vector_scores.get(str(row["chunk_id"]), 0.0) for row, _ in rows]
        fused = [fused_by_id[str(row["chunk_id"])] for row, _ in rows]
        per_query_rankings = [
            [positions[chunk_id] for chunk_id in ranking if chunk_id in positions]
            for ranking in per_query_ids
        ]
        return rows, keyword_scores, vector_scores, fused, per_query_rankings

    def _link_entities(self, query: str) -> list[dict[str, Any]]:
        self.ensure_ready()
        with self._connect() as connection:
            entities = connection.execute("SELECT * FROM rag_entities").fetchall()
        if not entities:
            return []
        normalized = query.casefold()
        query_terms = set(lexical_tokens(query))
        exact = []
        for row in entities:
            names = [row["entity_id"], *row["aliases"].split("|")]
            substring_match = any(
                name.strip().casefold() in normalized for name in names if len(name.strip()) >= 3
            )
            # Metadata aliases often combine an identifier and a display label
            # (for example "ABC display label"). Matching their distinctive
            # ASCII tokens links such entities without a domain vocabulary.
            alias_terms = {
                term for name in names for term in lexical_tokens(name)
                if len(term) >= 2 and any(character.isascii() and character.isalnum() for character in term)
            }
            if substring_match or bool(query_terms & alias_terms):
                exact.append({"entity_id": row["entity_id"], "score": 1.0, "match": "exact"})
        if exact:
            return exact[:4]
        query_vector = self.embedding.embed([query])[0]
        ranked = sorted(
            ((float(_cosine(query_vector, _unpack(row["vector"]))), row) for row in entities),
            reverse=True, key=lambda item: item[0],
        )
        threshold = float(os.getenv("RAG_ENTITY_LINK_THRESHOLD", "0.58"))
        return [
            {"entity_id": row["entity_id"], "score": round(score, 6), "match": "semantic"}
            for score, row in ranked[:2] if score >= threshold
        ]

    def _planning_context(self, linked_entities: list[dict[str, Any]]) -> dict[str, Any]:
        """Expose only IDs already present in the index to the model planner."""
        self.ensure_ready()
        linked_ids = {str(item["entity_id"]) for item in linked_entities}
        with self._connect() as connection:
            entities = [
                {"entity_id": row["entity_id"], "aliases": row["aliases"]}
                for row in connection.execute("SELECT entity_id, aliases FROM rag_entities ORDER BY entity_id")
            ]
            all_claims = [row[0] for row in connection.execute(
                "SELECT DISTINCT claim_id FROM rag_claims ORDER BY claim_id"
            )]
            raw_roles = connection.execute(
                "SELECT value FROM rag_meta WHERE key='evidence_roles'"
            ).fetchone()
            roles = json.loads(raw_roles[0]) if raw_roles else ["reference"]
        structured_claims = [
            claim for claim in all_claims
            if claim.startswith(("entity.", "experiment.", "version.", "policy."))
        ]
        claims = [
            claim for claim in structured_claims
            if not linked_ids or any(claim.startswith(f"entity.{entity}.") for entity in linked_ids)
        ]
        # General experiment, policy and version claims remain available even
        # when an entity was linked.
        claims.extend(claim for claim in structured_claims
                      if claim.startswith(("experiment.", "version.", "policy.")))
        return {"entities": entities, "claims": list(dict.fromkeys(claims))[:160], "roles": roles}

    def search(
        self,
        query: str,
        top_k: int = 6,
        modality: str = "",
        operator: str = "",
        metadata_filters: dict[str, Any] | None = None,
        include_superseded: bool = False,
        use_query_expansion: bool | None = None,
        use_decomposition: bool = True,
        use_reranker: bool = True,
        use_parent: bool = True,
        use_graph: bool | None = None,
    ) -> list[SearchResult]:
        """执行带缓存的混合检索，并返回带来源信息的排序结果。"""
        query = str(query).strip()
        if not query:
            return []
        if not self.search_cache.enabled:
            return self._search_uncached(
                query, top_k, modality, operator, metadata_filters,
                include_superseded, use_query_expansion, use_decomposition,
                use_reranker, use_parent, use_graph,
            )

        # The corpus fingerprint makes rebuild invalidation automatic: a new
        # knowledge version produces a different key without Redis SCAN/DEL.
        cache_key = self.search_cache.key({
            "cache_version": os.getenv("RAG_SEARCH_CACHE_VERSION", "v1"),
            "schema": INDEX_SCHEMA_VERSION,
            "corpus": self.corpus_fingerprint(),
            "embedding": self.embedding.fingerprint,
            "reranker": self.reranker.name,
            "query_expander": self.query_expander.name,
            "query_planner": self.query_planner.name,
            "retrieval_config": {name: os.getenv(name) for name in SEARCH_CACHE_ENV_KEYS},
            "query": query,
            "top_k": max(1, min(int(top_k), 20)),
            "modality": modality,
            "operator": operator,
            "metadata_filters": dict(metadata_filters or {}),
            "include_superseded": bool(include_superseded),
            "use_query_expansion": use_query_expansion,
            "use_decomposition": bool(use_decomposition),
            "use_reranker": bool(use_reranker),
            "use_parent": bool(use_parent),
            "use_graph": use_graph,
        })

        def load() -> list[dict[str, Any]]:
            return [item.to_dict() for item in self._search_uncached(
                query, top_k, modality, operator, metadata_filters,
                include_superseded, use_query_expansion, use_decomposition,
                use_reranker, use_parent, use_graph,
            )]

        payload = self.search_cache.get_or_compute(
            cache_key,
            max(1, int(os.getenv("RAG_SEARCH_CACHE_TTL_SECONDS", "300"))),
            load,
            lock_seconds=max(5, int(os.getenv("RAG_SEARCH_CACHE_LOCK_SECONDS", "60"))),
            wait_seconds=max(0.0, float(os.getenv("RAG_SEARCH_CACHE_WAIT_SECONDS", "1"))),
        )
        return [SearchResult(**item) for item in payload]

    def _search_uncached(
        self,
        query: str,
        top_k: int = 6,
        modality: str = "",
        operator: str = "",
        metadata_filters: dict[str, Any] | None = None,
        include_superseded: bool = False,
        use_query_expansion: bool | None = None,
        use_decomposition: bool = True,
        use_reranker: bool = True,
        use_parent: bool = True,
        use_graph: bool | None = None,
    ) -> list[SearchResult]:
        query = str(query).strip()
        if not query:
            return []
        linked_entities = self._link_entities(query)
        route = route_query(query, linked_entities)
        plan = (
            self.query_planner.plan(query, linked_entities, self._planning_context(linked_entities))
            if route.use_model_planner else build_query_plan(query, linked_entities)
        )
        query = plan.normalized
        top_k = max(1, min(int(top_k), 20))
        metadata_filters = dict(metadata_filters or {})
        if modality:
            metadata_filters["modality"] = modality
        if operator:
            metadata_filters["operator"] = operator
        if use_query_expansion is None:
            globally_enabled = os.getenv("RAG_QUERY_EXPANSION_ENABLED", "true").casefold() in {"1", "true", "yes", "on"}
            use_query_expansion = globally_enabled and route.use_query_expansion
        expanded_queries = (
            self.query_expander.expand(query, max(1, min(int(os.getenv("RAG_FUSION_QUERY_COUNT", "4")), 6)))
            if use_query_expansion else [query]
        )
        expanded_queries = list(dict.fromkeys(expanded_queries))
        subquery_start = len(expanded_queries)
        if use_decomposition and plan.complex_query:
            planned_subqueries = [item for item in plan.subqueries if item not in expanded_queries]
            expanded_queries = [*expanded_queries, *planned_subqueries][
                :max(1, min(int(os.getenv("RAG_TOTAL_QUERY_LIMIT", "8")), 12))
            ]
        query_vectors = self.embedding.embed(expanded_queries)
        if self.search_retriever is not None:
            rows, keyword_scores, vector_scores, fused, per_query_rankings = self._opensearch_candidates(
                expanded_queries,
                query_vectors,
                metadata_filters,
                include_superseded,
                top_k,
            )
        else:
            rows = []
            for row in self._rows():
                metadata = json.loads(row["metadata"])
                normalized_metadata = dict(metadata)
                normalized_metadata.setdefault("operator", metadata.get("name", ""))
                normalized_metadata.setdefault("status", "published")
                if any(
                    str(normalized_metadata.get(key, "")).casefold() != str(value).casefold()
                    for key, value in metadata_filters.items()
                    if value not in (None, "")
                ):
                    continue
                if not include_superseded and str(metadata.get("status", "published")).casefold() == "superseded":
                    continue
                rows.append((row, metadata))
            documents = [lexical_tokens(row["search_text"]) for row, _ in rows]
            frequencies = [Counter(tokens) for tokens in documents]
            doc_freq = Counter(term for counts in frequencies for term in counts)
            avg_len = sum(len(tokens) for tokens in documents) / len(documents) or 1.0
            keyword_scores = [0.0] * len(rows)
            vector_scores = [0.0] * len(rows)
            fused = [0.0] * len(rows)
            per_query_rankings = []
            for query_number, (expanded_query, query_vector) in enumerate(zip(expanded_queries, query_vectors)):
                query_weight = float(os.getenv("RAG_FUSION_ORIGINAL_QUERY_WEIGHT", "2.0")) if query_number == 0 else 1.0
                query_terms = lexical_tokens(expanded_query)
                current_keyword_scores: list[float] = []
                current_vector_scores: list[float] = []
                for (row, _), counts, tokens in zip(rows, frequencies, documents):
                    score = 0.0
                    for term in set(query_terms):
                        tf = counts.get(term, 0)
                        if not tf:
                            continue
                        idf = math.log(1 + (len(rows) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
                        denominator = tf + 1.5 * (1 - 0.75 + 0.75 * len(tokens) / avg_len)
                        score += idf * tf * 2.5 / denominator
                    current_keyword_scores.append(score)
                    current_vector_scores.append(_cosine(query_vector, _unpack(row["vector"])))
                if query_number == 0:
                    keyword_scores = current_keyword_scores
                    vector_scores = current_vector_scores
                for rank, index in enumerate(sorted(range(len(rows)), key=lambda i: current_keyword_scores[i], reverse=True), start=1):
                    fused[index] += query_weight / (60 + rank)
                for rank, index in enumerate(sorted(range(len(rows)), key=lambda i: current_vector_scores[i], reverse=True), start=1):
                    fused[index] += query_weight / (60 + rank)
                per_query_rankings.append(sorted(
                    range(len(rows)),
                    key=lambda i: current_keyword_scores[i] + max(0.0, current_vector_scores[i]),
                    reverse=True,
                ))
        if not rows:
            return []
        for index, (row, metadata) in enumerate(rows):
            authority = max(0.0, min(float(metadata.get("authority_priority", 50)), 100.0))
            fused[index] += authority / 100000.0
        if use_graph is None:
            use_graph = os.getenv("RAG_GRAPH_ENABLED", "true").casefold() in {"1", "true", "yes", "on"}
        # Graph bridges help relational questions; a single-entity fact should
        # not lose its evidence slot to a merely associated experiment.
        graph_intent = any(term in query.casefold() for term in (
            "比较", "对比", "对照", "消融", "互补", "组合", "拼接", "案例", "依赖", "关联",
            "compare", "comparison", "versus", "ablation", "dependency", "relationship",
        ))
        use_graph = bool(use_graph and graph_intent)
        graph_candidates = {}
        if use_graph:
            # Anchor expansion in both linked entities and highly ranked operator
            # cards, so a mistaken entity link does not monopolize graph seeds.
            seeds = list(plan.entities)
            cards = sorted((i for i, (_, m) in enumerate(rows)
                            if m.get("document_type") == "operator_card"),
                           key=lambda i: fused[i], reverse=True)
            card_documents = set()
            for i in cards:
                path = rows[i][0]["document_path"]
                if path in card_documents:
                    continue
                card_documents.add(path)
                seeds.extend(rows[i][1].get("entity_ids", []))
                if len(card_documents) >= 3:
                    break
            with self._connect() as connection:
                graph_candidates = knowledge_graph.candidates(
                    connection, rows, list(dict.fromkeys(seeds))[:8], fused,
                    limit=max(0, min(int(os.getenv("RAG_GRAPH_CANDIDATES", "6")), 20)),
                )
        if use_reranker:
            rerank_limit = max(top_k, min(int(os.getenv("RAG_RERANK_CANDIDATES", "15")), len(rows)))
            rerank_candidates = sorted(range(len(rows)), key=lambda i: fused[i], reverse=True)[:rerank_limit]
            rerank_candidates = list(dict.fromkeys([*rerank_candidates, *graph_candidates]))
            rerank_scores = self.reranker.score(query, [rows[index][0]["search_text"] for index in rerank_candidates])
            rerank_weight = float(os.getenv("RAG_RERANK_WEIGHT", "1.5"))
            for rank, candidate_position in enumerate(
                sorted(range(len(rerank_candidates)), key=lambda i: rerank_scores[i], reverse=True), start=1
            ):
                fused[rerank_candidates[candidate_position]] += rerank_weight / (60 + rank)

        # A long source may contain several independent facts. Allow more than
        # one chunk while applying a configurable cap so adjacent fragments from
        # the same source cannot consume the entire context window.
        selected: list[int] = []
        document_counts: Counter[str] = Counter()
        max_chunks_per_document = max(
            1, min(int(os.getenv("RAG_MAX_CHUNKS_PER_DOCUMENT", "2")), top_k)
        )
        ordered = sorted(range(len(rows)), key=lambda i: fused[i], reverse=True)

        def candidate_claims(index: int) -> set[str]:
            key = "parent_claim_ids" if use_parent else "claim_ids"
            return set(rows[index][1].get(key, rows[index][1].get("claim_ids", [])))

        def candidate_entities(index: int) -> set[str]:
            key = "parent_entity_ids" if use_parent else "entity_ids"
            return set(rows[index][1].get(key, rows[index][1].get("entity_ids", [])))

        def select_best(predicate, prefer_explicit_claims: bool = False,
                        candidate_whitelist: set[int] | None = None) -> None:
            candidates = [
                candidate for candidate in ordered
                if candidate not in selected
                and (candidate_whitelist is None or candidate in candidate_whitelist)
                and document_counts[str(rows[candidate][0]["document_path"])] < max_chunks_per_document
                and predicate(rows[candidate][1])
            ]
            if not candidates:
                return
            def priority(candidate: int) -> tuple[float, float, float]:
                metadata = rows[candidate][1]
                explicit = float(bool(metadata.get("supported_claims") or metadata.get("claims")))
                authority = float(metadata.get("authority_priority", 50))
                return (explicit if prefer_explicit_claims else 0.0, authority, fused[candidate])
            candidate = max(candidates, key=priority)
            document_counts[str(rows[candidate][0]["document_path"])] += 1
            selected.append(candidate)

        if use_graph:
            # Preserve the strongest textual evidence before coverage heuristics.
            select_best(lambda _metadata: True, candidate_whitelist={ordered[0]})
            bridges = {i for i, evidence in graph_candidates.items() if evidence["bridge"]}
            if bridges and len(selected) < top_k:
                best_bridge = max(bridges, key=lambda i: fused[i])
                select_best(lambda _metadata: True, candidate_whitelist={best_bridge})
                # A case mentioning both operators does not replace the operator
                # explanations. Retrieve their cards through the same graph path.
                bridge_entities = {path["entity"] for path in graph_candidates[best_bridge]["paths"]}
                for entity in sorted(bridge_entities):
                    if len(selected) >= top_k:
                        break
                    if any(rows[i][1].get("document_type") == "operator_card"
                           and entity in candidate_entities(i) for i in selected):
                        continue
                    cards = {i for i, (_, metadata) in enumerate(rows)
                             if metadata.get("document_type") == "operator_card"
                             and entity in candidate_entities(i)}
                    if cards:
                        select_best(lambda _metadata: True,
                                    candidate_whitelist={max(cards, key=lambda i: fused[i])})

        # Complex questions need coverage, not five near-duplicate chunks. Reserve
        # one slot per named operator and per required evidence class, then fill
        # the remaining slots by fused relevance.
        for entity in plan.entities:
            if len(selected) >= top_k:
                break
            if any(entity in candidate_entities(index) for index in selected):
                continue
            select_best(
                lambda metadata, entity=entity: entity in [str(item).casefold() for item in
                    metadata.get("parent_entity_ids" if use_parent else "entity_ids", [])],
                prefer_explicit_claims=True,
            )
        while len(selected) < top_k:
            covered = {claim for index in selected for claim in candidate_claims(index)}
            missing = set(plan.required_claims) - covered
            candidates = [index for index in ordered if index not in selected and candidate_claims(index) & missing]
            if not candidates:
                break
            best = max(candidates, key=lambda index: (len(candidate_claims(index) & missing), fused[index]))
            select_best(lambda _metadata: True, prefer_explicit_claims=True, candidate_whitelist={best})
        for role in plan.required_roles:
            if len(selected) >= top_k:
                break
            select_best(lambda metadata, role=role: str(metadata.get("evidence_role", "reference")).casefold() == role)

        # Each model-generated subquestion contributes at least one candidate.
        # This prevents a globally popular catalog from hiding one side of a
        # comparison or the procedure evidence of an experiment question.
        subquery_pool = max(1, int(os.getenv("RAG_SUBQUERY_RESERVE_POOL", "5")))
        for ranking in per_query_rankings[subquery_start:]:
            if len(selected) >= top_k:
                break
            candidate_set = set(ranking[:subquery_pool])
            select_best(lambda _metadata: True, candidate_whitelist=candidate_set)

        # Maximal Marginal Relevance prevents duplicate handbooks and cards from
        # consuming the full context while remaining independent of the domain.
        mmr_lambda = float(os.getenv("RAG_MMR_LAMBDA", "0.75"))
        remaining = [index for index in ordered if index not in selected]
        while remaining and len(selected) < top_k:
            def mmr(candidate: int) -> float:
                relevance = fused[candidate]
                duplication = max(
                    (_cosine(_unpack(rows[candidate][0]["vector"]), _unpack(rows[item][0]["vector"])) for item in selected),
                    default=0.0,
                )
                selected_claims = {claim for item in selected for claim in candidate_claims(item)}
                current_claims = candidate_claims(candidate)
                novelty = len(current_claims - selected_claims) / max(1, len(current_claims))
                selected_entities = {entity for item in selected for entity in candidate_entities(item)}
                current_entities = candidate_entities(candidate)
                entity_coverage = len((set(plan.entities) - selected_entities) & current_entities)
                claim_coverage = len((set(plan.required_claims) - selected_claims) & current_claims)
                selected_roles = {str(rows[item][1].get("evidence_role", "reference")).casefold()
                                  for item in selected}
                candidate_role = str(rows[candidate][1].get("evidence_role", "reference")).casefold()
                role_coverage = float(candidate_role in set(plan.required_roles) - selected_roles)
                return (mmr_lambda * relevance
                        - (1.0 - mmr_lambda) * max(0.0, duplication) / 10.0
                        + float(os.getenv("RAG_CLAIM_NOVELTY_WEIGHT", "0.002")) * novelty
                        + float(os.getenv("RAG_ENTITY_COVERAGE_WEIGHT", "0.008")) * entity_coverage
                        + float(os.getenv("RAG_REQUIRED_CLAIM_WEIGHT", "0.01")) * claim_coverage
                        + float(os.getenv("RAG_ROLE_COVERAGE_WEIGHT", "0.006")) * role_coverage)
            index = max(remaining, key=mmr)
            document_path = str(rows[index][0]["document_path"])
            remaining.remove(index)
            if document_counts[document_path] >= max_chunks_per_document:
                continue
            document_counts[document_path] += 1
            selected.append(index)
        return [
            SearchResult(
                chunk_id=rows[index][0]["chunk_id"],
                parent_id=rows[index][0]["parent_id"],
                document_path=rows[index][0]["document_path"],
                title=rows[index][0]["title"],
                section_path=json.loads(
                    rows[index][0]["parent_section_path"]
                    if use_parent and rows[index][0]["parent_section_path"]
                    else rows[index][0]["section_path"]
                ),
                content=(rows[index][0]["parent_content"] if use_parent else None) or rows[index][0]["content"],
                matched_content=rows[index][0]["content"],
                score=round(fused[index], 6),
                keyword_score=round(keyword_scores[index], 6),
                vector_score=round(vector_scores[index], 6),
                metadata=rows[index][1],
                entity_ids=sorted(candidate_entities(index)),
                claim_ids=sorted(candidate_claims(index)),
                authority=float(rows[index][1].get("authority_priority", 50)),
                graph_paths=graph_candidates.get(index, {}).get("paths", []),
                query_route=route.kind,
                query_expansion_used=bool(use_query_expansion),
                query_planner_used=plan.planner,
                page_start=(rows[index][0]["parent_page_start"] if use_parent else None) or rows[index][0]["page_start"],
                page_end=(rows[index][0]["parent_page_end"] if use_parent else None) or rows[index][0]["page_end"],
            )
            for index in selected
        ]

    def assess(self, query: str, results: list[SearchResult]) -> EvidenceAssessment:
        """判断当前检索结果是否足以支撑回答用户问题。"""
        if self.evidence_assessor:
            try:
                return self.evidence_assessor.assess(query, results)
            except Exception:
                pass
        return assess_evidence(query, results)


@lru_cache(maxsize=1)
def get_knowledge_store() -> HybridKnowledgeStore:
    """返回进程内共享的知识库实例。"""
    return HybridKnowledgeStore()
