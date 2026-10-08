"""文件用途：在实验状态事务中登记待发布的 Kafka 事件。"""

from typing import Any

from app import db
from app.models.experiment_outbox import ExperimentOutbox


def add_experiment_event(
    aggregate_id: str,
    task_type: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
) -> None:
    """登记一次关键状态变化；调用方随后提交即可与业务状态一起成功或失败。"""
    exists = ExperimentOutbox.query.filter_by(
        message_kind="event",
        aggregate_id=str(aggregate_id),
        event_type=event_type,
    ).first()
    if exists is not None:
        return
    body = {"task_type": task_type, **(payload or {})}
    db.session.add(ExperimentOutbox(
        message_kind="event",
        aggregate_id=str(aggregate_id),
        task_type=task_type,
        event_type=event_type,
        payload=body,
    ))


def add_compute_dispatch(
    aggregate_id: str,
    task_type: str,
    record_id: int,
    payload: dict[str, Any] | None = None,
) -> None:
    """把实验计算命令写入当前事务，交给 Go 后台程序可靠投递到 Redis 队列。"""
    db.session.add(ExperimentOutbox(
        message_kind="dispatch",
        aggregate_id=str(aggregate_id),
        task_type=task_type,
        event_type="compute.requested",
        payload={"record_id": int(record_id), "payload": payload or {}},
    ))
