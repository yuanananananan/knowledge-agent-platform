"""文件用途：创建缺失的数据表、初始化生产管理员并写入版本化评测数据。"""

import hashlib
import os
from pathlib import Path
import secrets

from app import db, init_database, session_scope
from app.agent.evaluation import seed_golden_cases
from app.models.agent_action import AgentActionProposal  # noqa: F401
from app.models.agent_runtime import (  # noqa: F401
    AgentCheckpoint,
    AgentConversation,
    AgentEvalResult,
    AgentEvalRun,
    AgentGoldenCase,
    AgentRun,
    AgentTraceEvent,
)
from app.models.audit_log import AuditLog  # noqa: F401
from app.models.backbone import Backbone  # noqa: F401
from app.models.classifier import Classifier  # noqa: F401
from app.models.dataset import Dataset  # noqa: F401
from app.models.experiment_outbox import ExperimentOutbox  # noqa: F401
from app.models.feature_extractor import (  # noqa: F401
    FeatureExtractor,
    FeatureGroup,
    FeatureGroupItem,
)
from app.models.image import Image  # noqa: F401
from app.models.nas_search_job import NasSearchJob  # noqa: F401
from app.models.task_execution import CeleryTaskExecution  # noqa: F401
from app.models.test_record import TestRecord  # noqa: F401
from app.models.train_job import TrainJob  # noqa: F401
from app.models.user import User  # noqa: F401
from sqlalchemy import inspect, text


def ensure_bootstrap_admin() -> None:
    """根据服务器环境变量创建首个管理员；重复启动时不会重复创建。"""
    account = os.getenv("BOOTSTRAP_ADMIN_ACCOUNT", "").strip()
    password = os.getenv("BOOTSTRAP_ADMIN_PASSWORD", "")
    name = os.getenv("BOOTSTRAP_ADMIN_NAME", "系统管理员").strip() or "系统管理员"
    if not account and not password:
        return
    if not account or len(password) < 12:
        raise RuntimeError("BOOTSTRAP_ADMIN_ACCOUNT 和至少 12 位的 BOOTSTRAP_ADMIN_PASSWORD 必须同时配置")

    existing = User.query.filter_by(userAccount=account, isDelete=False).first()
    if existing is not None:
        print(f"Bootstrap administrator already exists: {account}")
        return

    # 使用 Python 标准库生成 Go 登录服务兼容的 PBKDF2 密码编码。
    salt = secrets.token_hex(16)
    rounds = 600_000
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        rounds,
        dklen=32,
    )
    encoded = f"pbkdf2:sha256:{rounds}${salt}${derived.hex()}"
    db.session.add(User(
        userName=name,
        userAccount=account,
        userPassword=encoded,
        userRole="admin",
        isDelete=False,
    ))
    db.session.flush()
    print(f"Bootstrap administrator created: {account}")


def ensure_task_identity_indexes() -> None:
    """Add uniqueness to existing installations created before Alembic ran."""
    inspector = inspect(db.engine)
    required = {
        "train_jobs": ("uq_train_jobs_job_id", "job_id"),
        "test_records": ("uq_test_records_val_job_id", "val_job_id"),
    }
    for table, (index_name, column) in required.items():
        if not inspector.has_table(table):
            continue
        indexes = inspector.get_indexes(table)
        constraints = inspector.get_unique_constraints(table)
        already_unique = any(
            item.get("unique") and item.get("column_names") == [column]
            for item in indexes
        ) or any(
            item.get("column_names") == [column]
            for item in constraints
        )
        if already_unique:
            continue
        with db.engine.begin() as connection:
            connection.execute(text(
                f"CREATE UNIQUE INDEX {index_name} ON {table} ({column})"
            ))


def ensure_additive_agent_schema() -> None:
    """Apply additive changes required by older databases."""
    with db.engine.begin() as connection:
        connection.execute(text("""CREATE TABLE IF NOT EXISTS audit_logs (
            id BIGINT AUTO_INCREMENT PRIMARY KEY, user_id BIGINT NULL,
            action VARCHAR(128) NOT NULL, resource_type VARCHAR(64) NULL,
            resource_id VARCHAR(128) NULL, detail JSON NULL,
            success BOOLEAN NOT NULL DEFAULT TRUE,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            INDEX ix_audit_user (user_id), INDEX ix_audit_action (action)
        )"""))
    inspector = inspect(db.engine)
    if inspector.has_table("agent_runs"):
        columns = {column["name"] for column in inspector.get_columns("agent_runs")}
        with db.engine.begin() as connection:
            if "request_hash" not in columns:
                connection.execute(text("ALTER TABLE agent_runs ADD COLUMN request_hash VARCHAR(64) NULL"))
            if "answer" not in columns:
                connection.execute(text("ALTER TABLE agent_runs ADD COLUMN answer LONGTEXT NULL"))
    if inspector.has_table("agent_action_proposals"):
        columns = {column["name"] for column in inspector.get_columns("agent_action_proposals")}
        if "conversation_id" not in columns:
            with db.engine.begin() as connection:
                connection.execute(text("ALTER TABLE agent_action_proposals ADD COLUMN conversation_id VARCHAR(64) NULL"))
        indexes = {item.get("name") for item in inspect(db.engine).get_indexes("agent_action_proposals")}
        if "ix_agent_action_conversation_id" not in indexes:
            with db.engine.begin() as connection:
                connection.execute(text("CREATE INDEX ix_agent_action_conversation_id ON agent_action_proposals (conversation_id)"))


# 初始化数据库结构并同步版本化 Golden Case。
def main() -> None:
    init_database()
    dataset_file = Path(__file__).resolve().parents[1] / "evals" / "golden_dataset.json"
    with session_scope():
        db.create_all()
        ensure_additive_agent_schema()
        ensure_task_identity_indexes()
        ensure_bootstrap_admin()
        count = seed_golden_cases(dataset_file)
        print(f"Database ready; seeded {count} Golden Dataset cases")


if __name__ == "__main__":
    main()
