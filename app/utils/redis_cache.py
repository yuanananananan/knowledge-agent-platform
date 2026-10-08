"""文件用途：提供失败开放、JSON 序列化并带缓存击穿保护的轻量 Redis 缓存。"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
import uuid
from typing import Any, Callable

import redis


logger = logging.getLogger(__name__)

RELEASE_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


class RedisJSONCache:
    """可选 JSON 缓存；Redis 故障时自动绕过并回到权威数据源。"""

    def __init__(self, namespace: str, client=None):
        self.namespace = namespace.strip(":")
        self._injected_client = client
        self._client_instance = None
        self._disabled_until = 0.0

    @property
    def enabled(self) -> bool:
        return self._injected_client is not None or bool(
            os.getenv("RAG_CACHE_REDIS_URL", "").strip()
        )

    def _client(self):
        if self._injected_client is not None:
            return self._injected_client
        if self._client_instance is None:
            url = os.getenv("RAG_CACHE_REDIS_URL", "").strip()
            if not url.startswith(("redis://", "rediss://")):
                raise ValueError("RAG_CACHE_REDIS_URL 必须使用 redis:// 或 rediss://")
            self._client_instance = redis.Redis.from_url(
                url,
                socket_connect_timeout=1,
                socket_timeout=1,
                health_check_interval=30,
                decode_responses=True,
            )
        return self._client_instance

    def key(self, value: Any) -> str:
        """将任意可序列化输入转换为稳定、定长的缓存键。"""
        canonical = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"knowledge-agent:cache:{self.namespace}:{digest}"

    def _available(self) -> bool:
        return self.enabled and time.monotonic() >= self._disabled_until

    def _mark_failure(self, exc: Exception) -> None:
        now = time.monotonic()
        if now >= self._disabled_until:
            logger.warning("Redis cache unavailable; temporarily bypassing it: %s", exc)
        backoff = max(1.0, float(os.getenv("RAG_CACHE_FAILURE_BACKOFF_SECONDS", "5")))
        self._disabled_until = now + backoff

    def get(self, key: str) -> Any | None:
        if not self._available():
            return None
        try:
            raw = self._client().get(key)
            return json.loads(raw) if raw is not None else None
        except (redis.RedisError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self._mark_failure(exc)
            return None

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        if not self._available():
            return
        jitter = max(0, int(os.getenv("RAG_CACHE_TTL_JITTER_SECONDS", "60")))
        ttl = max(1, int(ttl_seconds)) + random.randint(0, jitter)
        try:
            self._client().setex(
                key,
                ttl,
                json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str),
            )
        except (redis.RedisError, ValueError, TypeError) as exc:
            self._mark_failure(exc)

    def get_or_compute(
        self,
        key: str,
        ttl_seconds: int,
        loader: Callable[[], Any],
        lock_seconds: int = 30,
        wait_seconds: float = 1.0,
    ) -> Any:
        """读取缓存；未命中时用短锁合并并发计算，避免缓存击穿。"""
        cached = self.get(key)
        if cached is not None:
            return cached
        if not self._available():
            return loader()

        token = uuid.uuid4().hex
        lock_key = f"{key}:lock"
        try:
            acquired = bool(self._client().set(lock_key, token, nx=True, ex=max(1, lock_seconds)))
        except (redis.RedisError, ValueError) as exc:
            self._mark_failure(exc)
            return loader()

        if acquired:
            try:
                # Double-check after acquiring the lock: another request may
                # have filled the cache between our first read and SET NX.
                cached = self.get(key)
                if cached is not None:
                    return cached
                value = loader()
                self.set(key, value, ttl_seconds)
                return value
            finally:
                try:
                    self._client().eval(RELEASE_SCRIPT, 1, lock_key, token)
                except redis.RedisError:
                    pass

        deadline = time.monotonic() + max(0.0, wait_seconds)
        while time.monotonic() < deadline:
            time.sleep(0.05)
            cached = self.get(key)
            if cached is not None:
                return cached
        # Do not let a slow or crashed cache builder block the request. The
        # authoritative computation remains safe and its result may be cached.
        value = loader()
        self.set(key, value, ttl_seconds)
        return value
