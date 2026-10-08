import json

from app.rag.opensearch import OpenSearchHybridRetriever


class RecordingOpenSearch(OpenSearchHybridRetriever):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.search_responses = []

    def _request(self, method, path, payload=None, **kwargs):
        self.calls.append((method, path, payload, kwargs))
        if method == "HEAD":
            return 404, {}
        if path.endswith("/_bulk?refresh=wait_for"):
            return 200, {"errors": False}
        if path.startswith("/_alias/"):
            return 404, {}
        if path.endswith("/_search"):
            return 200, self.search_responses.pop(0)
        return 200, {"acknowledged": True}


def test_rebuild_creates_lucene_hnsw_index_and_bulk_documents(monkeypatch):
    monkeypatch.setenv("RAG_OPENSEARCH_INDEX", "test-rag")
    client = RecordingOpenSearch()
    rows = [{
        "chunk_id": "chunk-1",
        "parent_id": "parent-1",
        "document_path": "guide.md",
        "title": "Guide",
        "section_path": ["Features"],
        "search_text": "Guide Features texture",
        "metadata": {"modality": "SAR", "status": "published"},
    }]

    result = client.rebuild(
        rows,
        [[0.1, 0.2, 0.3]],
        corpus_fingerprint="a" * 64,
        embedding_fingerprint="test:3",
        dimension=3,
    )

    create = next(call for call in client.calls if call[0] == "PUT")
    vector_mapping = create[2]["mappings"]["properties"]["embedding"]
    assert vector_mapping["type"] == "knn_vector"
    assert vector_mapping["method"] == {
        "name": "hnsw",
        "engine": "lucene",
        "space_type": "cosinesimil",
        "parameters": {"m": 16, "ef_construction": 100},
    }
    bulk = next(call for call in client.calls if "_bulk" in call[1])
    bulk_lines = bulk[2].decode().splitlines()
    document = json.loads(bulk_lines[1])
    assert document["embedding"] == [0.1, 0.2, 0.3]
    assert "modality=sar" in document["filter_tokens"]
    assert result["vector_index"] == "hnsw-lucene-cosine"


def test_search_runs_bm25_and_filtered_hnsw_queries(monkeypatch):
    monkeypatch.setenv("RAG_OPENSEARCH_INDEX", "test-rag")
    client = RecordingOpenSearch()
    client.search_responses = [
        {"hits": {"hits": [{"_id": "keyword", "_score": 4.2}]}},
        {"hits": {"hits": [{"_id": "vector", "_score": 0.91}]}},
    ]

    keyword, vector = client.search(
        "纹理特征",
        [0.1, 0.2, 0.3],
        metadata_filters={"modality": "SAR"},
        limit=20,
    )

    searches = [call[2] for call in client.calls if call[1].endswith("/_search")]
    keyword_bool = searches[0]["query"]["bool"]
    assert keyword_bool["must"][0]["multi_match"]["query"] == "纹理特征"
    assert {"term": {"filter_tokens": "modality=sar"}} in keyword_bool["filter"]
    vector_clause = searches[1]["query"]["knn"]["embedding"]
    assert vector_clause["k"] == 20
    assert {"term": {"filter_tokens": "modality=sar"}} in vector_clause["filter"]["bool"]["filter"]
    assert keyword[0].chunk_id == "keyword"
    assert vector[0].score == 0.91
