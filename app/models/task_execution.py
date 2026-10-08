"""文件用途：定义后台任务幂等执行、租约、重试和结果提交记录。"""

from datetime import datetime

from app import db


class CeleryTaskExecution(db.Model):
    """一个逻辑 Celery 任务的永久幂等记录。

    执行键不会被自动回收。若 Worker 领取任务后异常退出，运维人员需要先检查
    目标资源，再明确创建重试任务；这里优先防止训练或验证被重复执行。
    """

    __tablename__ = "celery_task_executions"

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    execution_key = db.Column(db.String(191), nullable=False, unique=True, index=True)
    task_name = db.Column(db.String(191), nullable=False, index=True)
    celery_task_id = db.Column(db.String(128), nullable=True, index=True)
    status = db.Column(db.String(24), nullable=False, default="running", index=True)
    owner_token = db.Column(db.String(64), nullable=False)
    fencing_token = db.Column(db.BigInteger, nullable=False, default=1)
    result = db.Column(db.JSON, nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    heartbeat_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    started_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    finished_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    def to_dict(self):
        """返回幂等执行记录及心跳、结果等状态。"""
        return {
            "execution_key": self.execution_key,
            "task_name": self.task_name,
            "celery_task_id": self.celery_task_id,
            "status": self.status,
            "fencing_token": self.fencing_token,
            "result": self.result,
            "error_message": self.error_message,
            "heartbeat_at": self.heartbeat_at.isoformat() if self.heartbeat_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }
