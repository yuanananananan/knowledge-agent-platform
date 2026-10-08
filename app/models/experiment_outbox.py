"""文件用途：定义实验任务命令与状态事件共用的事务发件箱。"""

from datetime import datetime

from sqlalchemy import Index

from app import db


class ExperimentOutbox(db.Model):
    """先随业务状态写入 MySQL，再由 Go 后台程序投递 Redis 或 Kafka。"""

    __tablename__ = "experiment_outbox"
    __table_args__ = (
        db.UniqueConstraint(
            "message_kind", "aggregate_id", "event_type",
            name="uq_experiment_outbox_event",
        ),
        Index(
            "ix_experiment_outbox_pending",
            "published_at", "available_at", "lease_expires_at",
        ),
    )

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    message_kind = db.Column(db.String(16), nullable=False)
    aggregate_id = db.Column(db.String(64), nullable=False)
    task_type = db.Column(db.String(32), nullable=True)
    event_type = db.Column(db.String(96), nullable=False)
    payload = db.Column(db.JSON, nullable=False)
    attempts = db.Column(db.Integer, nullable=False, default=0)
    available_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    worker_id = db.Column(db.String(96), nullable=True)
    lease_expires_at = db.Column(db.DateTime, nullable=True)
    published_at = db.Column(db.DateTime, nullable=True)
    last_error = db.Column(db.String(1000), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
