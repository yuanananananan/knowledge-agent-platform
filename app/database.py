"""文件用途：提供 gRPC、Celery、MCP 共用的独立 SQLAlchemy 初始化和会话生命周期。"""

from __future__ import annotations

import os
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, relationship, scoped_session, sessionmaker
from dotenv import load_dotenv


load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"), override=False)


_session_scope_id: ContextVar[str | None] = ContextVar(
    "knowledge_agent_sqlalchemy_scope", default=None
)


# 为线程或异步 Context 生成 SQLAlchemy scoped_session 的隔离键。
def _scope_key() -> str:
    return _session_scope_id.get() or f"thread:{threading.get_ident()}"


class Database:
    """管理 Engine、scoped_session 和所有模型共享的声明基类。"""
    """Small legacy-ORM-compatible facade backed directly by SQLAlchemy."""

    Model = declarative_base()
    Column = Column
    Integer = Integer
    BigInteger = BigInteger
    Float = Float
    String = String
    Text = Text
    Boolean = Boolean
    DateTime = DateTime
    JSON = JSON
    ForeignKey = ForeignKey
    UniqueConstraint = UniqueConstraint
    relationship = staticmethod(relationship)
    func = func

    def __init__(self) -> None:
        self.session = scoped_session(
            sessionmaker(autoflush=True, expire_on_commit=False),
            scopefunc=_scope_key,
        )
        self.Model.query = self.session.query_property()
        self._engine: Engine | None = None
        self._url: str | None = None
        self._lock = threading.Lock()

    def init(self, database_url: str | None = None) -> Engine:
        url = (database_url or os.getenv("DATABASE_URL") or "").strip()
        if not url:
            raise RuntimeError(
                "未配置 DATABASE_URL，请设置为 mysql+pymysql://用户名:密码@主机:3306/数据库名"
            )
        with self._lock:
            if self._engine is not None and self._url == url:
                return self._engine
            if self._engine is not None:
                self.session.remove()
                self._engine.dispose()
            options: dict = {"pool_pre_ping": True, "pool_recycle": 280}
            if url.startswith("sqlite:"):
                options.pop("pool_recycle", None)
                options["connect_args"] = {"check_same_thread": False}
            self._engine = create_engine(url, **options)
            self._url = url
            self.session.configure(bind=self._engine)
            self.Model.metadata.bind = self._engine
            return self._engine

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            return self.init()
        return self._engine

    def create_all(self) -> None:
        self.Model.metadata.create_all(bind=self.engine)

    def drop_all(self) -> None:
        self.Model.metadata.drop_all(bind=self.engine)


db = Database()


# 使用参数或环境变量中的数据库 URI 初始化全局运行时。
def init_database(database_url: str | None = None) -> Engine:
    return db.init(database_url)


@contextmanager
def session_scope(database_url: str | None = None) -> Iterator[Session]:
    """Create an isolated session scope, including for nested tool calls."""
    init_database(database_url)
    token = _session_scope_id.set(uuid.uuid4().hex)
    try:
        yield db.session
    except Exception:
        db.session.rollback()
        raise
    finally:
        db.session.remove()
        _session_scope_id.reset(token)
