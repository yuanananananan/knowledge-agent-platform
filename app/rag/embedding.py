"""文件用途：提供 RAG 索引使用的远程和本地文本向量编码器。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Any

from app.utils.redis_cache import RedisJSONCache


ASCII_WORD = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
CJK = re.compile(r"[\u3400-\u9fff]")


def lexical_tokens(text: str) -> list[str]:
    """把中英文文本拆成词和中文一元/二元组，供词法检索和规则打分使用。"""
    normalized = text.casefold()
    words = ASCII_WORD.findall(normalized)
    chars = CJK.findall(normalized)
    grams = chars + ["".join(chars[i:i + 2]) for i in range(len(chars) - 1)]
    return words + grams


class EmbeddingProvider:
    """所有文本向量提供器都必须实现的统一接口。"""
    name = "base"
    dimension = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @property
    def fingerprint(self) -> str:
        return f"{self.name}:{self.dimension}"


class CachedEmbeddingProvider(EmbeddingProvider):
    """为任意向量提供器增加按文本内容寻址的 Redis 缓存。"""

    def __init__(self, inner: EmbeddingProvider):
        self.inner = inner
        self.name = inner.name
        self.cache_identity = inner.fingerprint
        self.cache = RedisJSONCache("embedding")

    @property
    def dimension(self) -> int:
        return self.inner.dimension

    @property
    def fingerprint(self) -> str:
        return self.inner.fingerprint

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts or not self.cache.enabled:
            return self.inner.embed(texts)
        ttl = max(60, int(os.getenv("RAG_EMBEDDING_CACHE_TTL_SECONDS", "2592000")))
        output: list[list[float] | None] = [None] * len(texts)
        missing: dict[str, list[int]] = {}
        keys: dict[str, str] = {}
        for index, text in enumerate(texts):
            key = self.cache.key({"provider": self.cache_identity, "text": text})
            keys[text] = key
            cached = self.cache.get(key)
            if isinstance(cached, list) and cached:
                output[index] = [float(value) for value in cached]
            else:
                missing.setdefault(text, []).append(index)
        if missing:
            missing_texts = list(missing)
            computed = self.inner.embed(missing_texts)
            if len(computed) != len(missing_texts):
                raise RuntimeError("Embedding 返回数量与输入不一致")
            for text, vector in zip(missing_texts, computed):
                normalized = [float(value) for value in vector]
                for index in missing[text]:
                    output[index] = normalized
                self.cache.set(keys[text], normalized, ttl)
        if any(vector is None for vector in output):
            raise RuntimeError("Embedding 缓存未能填充全部结果")
        return [vector for vector in output if vector is not None]


@dataclass
class LocalHashEmbedding(EmbeddingProvider):
    """无需模型的本地哈希向量实现，主要作为轻量离线方案。"""
    dimension: int = 768
    name: str = "local-hash-char-ngram-v1"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in lexical_tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "little")
            index = raw % self.dimension
            sign = 1.0 if raw & (1 << 63) else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector


class OpenAICompatibleEmbedding(EmbeddingProvider):
    """通过 OpenAI 兼容的 embeddings HTTP 接口生成文本向量。"""
    name = "openai-compatible"

    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 60.0):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.dimension = int(os.getenv("RAG_EMBEDDING_DIMENSION", "0"))

    @property
    def fingerprint(self) -> str:
        return f"{self.name}:{self.model}:{self.dimension or 'auto'}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model": self.model, "input": texts}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        ordered = sorted(payload["data"], key=lambda item: item["index"])
        vectors = [list(map(float, item["embedding"])) for item in ordered]
        if vectors:
            self.dimension = len(vectors[0])
        return vectors


class HuggingFaceNeuralEmbedding(EmbeddingProvider):
    """使用本地 Transformer、掩码平均池化和 L2 归一化生成向量。"""

    name = "huggingface-neural"

    def __init__(self, model: str, device: str = "cpu", max_length: int = 512):
        self.model_name = model
        self.device = device
        self.max_length = max_length
        self.dimension = int(os.getenv("RAG_EMBEDDING_DIMENSION", "512"))
        self._tokenizer: Any = None
        self._model: Any = None

    @property
    def fingerprint(self) -> str:
        return f"{self.name}:{self.model_name}:{self.dimension}:max-{self.max_length}"

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "本地神经 Embedding 需要 transformers；请安装 requirements.txt"
            ) from exc
        local_only = os.getenv("RAG_MODEL_LOCAL_FILES_ONLY", "true").casefold() in {"1", "true", "yes", "on"}
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=local_only)
        self._model = AutoModel.from_pretrained(self.model_name, local_files_only=local_only).to(self.device)
        self._model.eval()
        self.dimension = int(self._model.config.hidden_size)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        import torch

        if not texts:
            return []
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            hidden = self._model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().float().tolist()


def build_embedding_provider() -> EmbeddingProvider:
    """按照配置选择向量模型，并在 Redis 可用时自动套上缓存。"""
    provider = os.getenv("RAG_EMBEDDING_PROVIDER", "huggingface").strip().casefold()
    if provider in {"local", "hash", "local-hash"}:
        result: EmbeddingProvider = LocalHashEmbedding(
            int(os.getenv("RAG_LOCAL_EMBEDDING_DIMENSION", "768"))
        )
    elif provider in {"openai", "openai-compatible"}:
        api_key = os.getenv("RAG_EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY")
        if not api_key:
            raise RuntimeError("RAG_EMBEDDING_PROVIDER=openai 时必须配置 RAG_EMBEDDING_API_KEY")
        result = OpenAICompatibleEmbedding(
            api_key=api_key,
            base_url=os.getenv("RAG_EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
            timeout=float(os.getenv("RAG_EMBEDDING_TIMEOUT_SECONDS", "60")),
        )
    elif provider in {"huggingface", "hf", "neural", "local-neural"}:
        result = HuggingFaceNeuralEmbedding(
            model=os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            device=os.getenv("RAG_EMBEDDING_DEVICE", "cpu"),
            max_length=int(os.getenv("RAG_EMBEDDING_MAX_LENGTH", "512")),
        )
    else:
        raise ValueError(f"不支持的 RAG_EMBEDDING_PROVIDER: {provider}")
    return CachedEmbeddingProvider(result) if RedisJSONCache("probe").enabled else result
