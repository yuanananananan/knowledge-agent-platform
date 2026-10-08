"""文件用途：创建 Celery 应用并配置 Redis、晚确认、低预取和数据库会话隔离。"""

from celery import Celery
from kombu import Queue
import sys
import os

from app import db, init_database

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

celery_app = Celery(
    "trainer",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
)

# Redis transport is at-least-once. Late acknowledgement prevents losing a
# task before execution starts; the MySQL/Redis execution guard in tasks.py
# makes redelivery safe by refusing to execute an existing logical task key.
celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
    broker_connection_retry_on_startup=True,
    broker_transport_options={
        "visibility_timeout": int(os.getenv("CELERY_VISIBILITY_TIMEOUT", "86400")),
    },
    result_expires=int(os.getenv("CELERY_RESULT_EXPIRES", "86400")),
    # 不同实验类型使用独立队列，后续可按 CPU、GPU 数量分别扩容和限流。
    task_queues=(
        Queue("training"),
        Queue("validation"),
        Queue("nas"),
        Queue("evaluation"),
    ),
    task_routes={
        "app.utils.tasks.run_training_task": {"queue": "training"},
        "app.utils.tasks.run_valuating_task": {"queue": "validation"},
        "app.utils.tasks.run_nas_search_task": {"queue": "nas"},
        "agent.run_golden_evaluation": {"queue": "evaluation"},
    },
)

TaskBase = celery_app.Task


class DatabaseTask(TaskBase):
    """每次执行前初始化数据库，并在结束后释放当前线程的会话。"""
    abstract = True

    def __call__(self, *args, **kwargs):
        init_database()
        try:
            return TaskBase.__call__(self, *args, **kwargs)
        finally:
            db.session.remove()


# Every task receives an isolated SQLAlchemy scoped session.
celery_app.Task = DatabaseTask

def make_celery():
    """初始化数据库并返回已配置好的 Celery 应用。"""
    init_database()
    return celery_app
