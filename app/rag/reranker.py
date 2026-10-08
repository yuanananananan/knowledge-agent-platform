"""文件用途：提供独立的第二阶段规则、远程或本地交叉编码重排器。"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.request
from typing import Any

from app.rag.embedding import lexical_tokens


class Reranker:
    """候选文档第二阶段相关性打分器的统一接口。"""
    name = "base"

    def score(self, query: str, documents: list[str]) -> list[float]:
        raise NotImplementedError


class LocalFeatureReranker(Reranker):
    """用覆盖率、短语和标识符特征实现快速离线重排。"""

    name = "local-feature-reranker-v1"

    def score(self, query: str, documents: list[str]) -> list[float]:
        query_tokens = set(lexical_tokens(query))
        identifiers = set(re.findall(r"[a-z][a-z0-9_]{2,}", query.casefold()))
        scores = []
        for document in documents:
            normalized = document.casefold()
            document_tokens = set(lexical_tokens(document))
            coverage = len(query_tokens & document_tokens) / max(1, len(query_tokens))
            identifier_hits = sum(identifier in normalized for identifier in identifiers)
            phrase_bonus = 1.0 if query.casefold() in normalized else 0.0
            scores.append(coverage + 0.25 * identifier_hits + 0.2 * phrase_bonus)
        return scores


class HTTPNeuralReranker(Reranker):
    """调用通用 /rerank 接口获取候选下标和相关性分数。"""

    name = "http-neural-reranker"

    def __init__(self):
        self.api_key = os.getenv("RAG_RERANKER_API_KEY")
        self.url = os.getenv("RAG_RERANKER_URL", "").strip()
        self.model = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        self.timeout = float(os.getenv("RAG_RERANKER_TIMEOUT_SECONDS", "30"))
        if not self.url:
            raise RuntimeError("神经 Reranker 缺少 RAG_RERANKER_URL")

    def score(self, query: str, documents: list[str]) -> list[float]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.url,
            data=json.dumps({"model": self.model, "query": query, "documents": documents}).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        results = payload.get("results") or payload.get("data") or []
        scores = [0.0] * len(documents)
        for result in results:
            index = int(result["index"])
            scores[index] = float(result.get("relevance_score", result.get("score", 0.0)))
        return scores


class HuggingFaceNeuralReranker(Reranker):
    """使用本地交叉编码器联合计算问题与每个文档的相关性。"""

    name = "huggingface-cross-encoder"

    def __init__(self):
        self.model_name = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-base")
        self.device = os.getenv("RAG_RERANKER_DEVICE", "cpu")
        self.max_length = int(os.getenv("RAG_RERANKER_MAX_LENGTH", "512"))
        self.batch_size = max(1, int(os.getenv("RAG_RERANKER_BATCH_SIZE", "8")))
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("本地神经 Reranker 需要 transformers；请安装 requirements.txt") from exc
        local_only = os.getenv("RAG_MODEL_LOCAL_FILES_ONLY", "true").casefold() in {"1", "true", "yes", "on"}
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=local_only)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, local_files_only=local_only
        ).to(self.device)
        self._model.eval()

    def score(self, query: str, documents: list[str]) -> list[float]:
        self._load()
        import torch

        scores: list[float] = []
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start:start + self.batch_size]
            encoded = self._tokenizer(
                [[query, document] for document in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = self._model(**encoded).logits.view(-1)
            scores.extend(logits.cpu().float().tolist())
        return scores


class FallbackReranker(Reranker):
    """神经重排失败时自动使用本地特征重排。"""
    def __init__(self, primary: Reranker, fallback: Reranker):
        self.primary = primary
        self.fallback = fallback
        self.name = f"{primary.name}|fallback:{fallback.name}"

    def score(self, query: str, documents: list[str]) -> list[float]:
        try:
            return self.primary.score(query, documents)
        except Exception:
            return self.fallback.score(query, documents)


def build_reranker() -> Reranker:
    """根据配置选择本地或远程重排器。"""
    fallback = LocalFeatureReranker()
    provider = os.getenv("RAG_RERANKER_PROVIDER", "huggingface").casefold()
    if provider in {"huggingface", "hf", "local-neural"}:
        return HuggingFaceNeuralReranker()
    if provider in {"http", "neural", "api"}:
        return FallbackReranker(HTTPNeuralReranker(), fallback)
    return fallback
