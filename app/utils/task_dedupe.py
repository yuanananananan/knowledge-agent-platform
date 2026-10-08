"""文件用途：使用 MySQL 执行记录和 Redis 锁实现后台任务幂等、租约与结果防覆盖。"""

import json
import os
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from typing import Any, Callable

import redis
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError

from app import db
from app.models.task_execution import CeleryTaskExecution


RENEW_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('PEXPIRE', KEYS[1], ARGV[2])
end
return 0
"""

RELEASE_SCRIPT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""


class TaskDedupeUnavailable(RuntimeError):
    """去重存储或租约不可用，无法保证任务只执行一次。"""
    pass


def _redis_url() -> str:
    value = (
        os.getenv("CELERY_DEDUP_REDIS_URL")
        or os.getenv("REDIS_URL")
        or os.getenv("CELERY_BROKER_URL")
        or ""
    )
    if not value.startswith(("redis://", "rediss://")):
        raise TaskDedupeUnavailable("Celery 去重要求 redis:// 或 rediss:// 地址")
    return value


def _lock_ttl_ms() -> int:
    try:
        seconds = int(os.getenv("CELERY_DEDUP_LOCK_TTL_SECONDS", "60"))
    except ValueError as exc:
        raise TaskDedupeUnavailable("CELERY_DEDUP_LOCK_TTL_SECONDS 必须是整数") from exc
    if seconds < 15:
        raise TaskDedupeUnavailable("CELERY_DEDUP_LOCK_TTL_SECONDS 不能小于 15")
    return seconds * 1000


def _redis_client():
    return redis.Redis.from_url(
        _redis_url(),
        socket_connect_timeout=3,
        socket_timeout=3,
        health_check_interval=30,
        decode_responses=True,
    )


def _json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _duplicate_result(row: CeleryTaskExecution | None, execution_key: str) -> dict:
    if row is None:
        status, result, error = "in_progress", None, None
    else:
        status, result, error = row.status, row.result, row.error_message
    return {
        "deduplicated": True,
        "execution_key": execution_key,
        "execution_status": status,
        "result": result,
        "error": error,
    }


def _existing(execution_key: str) -> CeleryTaskExecution | None:
    db.session.rollback()
    return CeleryTaskExecution.query.filter_by(execution_key=execution_key).first()


def _claim(execution_key: str, task_name: str, celery_task_id: str | None, token: str):
    existing = CeleryTaskExecution.query.filter_by(execution_key=execution_key).first()
    if existing is not None:
        return None, existing
    row = CeleryTaskExecution(
        execution_key=execution_key,
        task_name=task_name,
        celery_task_id=celery_task_id,
        status="running",
        owner_token=token,
        fencing_token=1,
        heartbeat_at=datetime.utcnow(),
        started_at=datetime.utcnow(),
    )
    db.session.add(row)
    try:
        db.session.commit()
        return row, None
    except IntegrityError:
        db.session.rollback()
        return None, _existing(execution_key)


class _Heartbeat:
    """后台续期 Redis 租约，并同步刷新数据库心跳。"""
    def __init__(self, redis_client, redis_key: str, execution_key: str, token: str, ttl_ms: int):
        self.redis = redis_client
        self.redis_key = redis_key
        self.execution_key = execution_key
        self.token = token
        self.ttl_ms = ttl_ms
        self.stop = threading.Event()
        self.lost = threading.Event()
        self.engine = db.engine
        self.thread = threading.Thread(target=self._run, name=f"dedupe-{execution_key[-24:]}", daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=5)

    def _run(self):
        interval = max(1.0, self.ttl_ms / 3000.0)
        while not self.stop.wait(interval):
            try:
                renewed = self.redis.eval(RENEW_SCRIPT, 1, self.redis_key, self.token, self.ttl_ms)
                if renewed != 1:
                    self.lost.set()
                    return
                now = datetime.utcnow()
                with self.engine.begin() as connection:
                    result = connection.execute(
                        update(CeleryTaskExecution)
                        .where(
                            CeleryTaskExecution.execution_key == self.execution_key,
                            CeleryTaskExecution.owner_token == self.token,
                            CeleryTaskExecution.status == "running",
                        )
                        .values(heartbeat_at=now, updated_at=now)
                    )
                if result.rowcount != 1:
                    self.lost.set()
                    return
            except Exception:
                self.lost.set()
                return


@contextmanager
def _execution_lock(execution_key: str):
    client = _redis_client()
    ttl_ms = _lock_ttl_ms()
    token = uuid.uuid4().hex
    redis_key = f"knowledge-agent:celery:lock:{execution_key}"
    try:
        acquired = client.set(redis_key, token, nx=True, px=ttl_ms)
    except redis.RedisError as exc:
        raise TaskDedupeUnavailable("Redis 去重锁不可用，任务已拒绝执行") from exc
    if not acquired:
        try:
            yield None, _existing(execution_key)
        finally:
            client.close()
        return
    heartbeat = _Heartbeat(client, redis_key, execution_key, token, ttl_ms)
    try:
        yield (token, heartbeat), None
    finally:
        heartbeat.close()
        try:
            client.eval(RELEASE_SCRIPT, 1, redis_key, token)
        except redis.RedisError:
            pass
        client.close()


def execute_once(
    task_name: str,
    celery_task_id: str | None,
    execution_key: str,
    operation: Callable[[], Any],
):
    """在 Redis 租约和数据库永久执行键的共同保护下执行一次操作。"""
    if not execution_key or len(execution_key) > 191:
        raise ValueError("Celery execution_key 不能为空且不能超过 191 字符")
    with _execution_lock(execution_key) as (lock, duplicate):
        if lock is None:
            return _duplicate_result(duplicate, execution_key)
        token, heartbeat = lock
        claimed, duplicate = _claim(execution_key, task_name, celery_task_id, token)
        if claimed is None:
            return _duplicate_result(duplicate, execution_key)
        heartbeat.start()
        try:
            result = operation()
        except Exception as exc:
            db.session.rollback()
            CeleryTaskExecution.query.filter_by(
                execution_key=execution_key,
                owner_token=token,
                status="running",
            ).update({
                "status": "failed",
                "error_message": str(exc)[:4000],
                "heartbeat_at": datetime.utcnow(),
                "finished_at": datetime.utcnow(),
            })
            db.session.commit()
            raise
        serialized = _json_value(result)
        updated = CeleryTaskExecution.query.filter_by(
            execution_key=execution_key,
            owner_token=token,
            status="running",
        ).update({
            "status": "succeeded",
            "result": serialized,
            "error_message": None,
            "heartbeat_at": datetime.utcnow(),
            "finished_at": datetime.utcnow(),
        })
        db.session.commit()
        if updated != 1:
            raise TaskDedupeUnavailable("任务执行权已丢失，拒绝提交执行结果")
        return result


def deduplicated(key_builder: Callable[..., str]):
    """给 Celery 任务增加基于业务执行键的幂等装饰器。"""
    def decorator(function):
        @wraps(function)
        def wrapped(task, *args, **kwargs):
            execution_key = str(key_builder(*args, **kwargs))
            task_id = getattr(getattr(task, "request", None), "id", None)
            task_name = getattr(task, "name", function.__name__)
            outcome = execute_once(
                task_name,
                task_id,
                execution_key,
                lambda: function(task, *args, **kwargs),
            )
            if isinstance(outcome, dict) and outcome.get("deduplicated") is True:
                if outcome.get("execution_status") == "succeeded":
                    return outcome.get("result")
                # Do not let a duplicate delivery overwrite the original
                # task_id's Redis result state with a false SUCCESS/FAILURE.
                from celery.exceptions import Ignore

                raise Ignore()
            return outcome

        return wrapped

    return decorator
