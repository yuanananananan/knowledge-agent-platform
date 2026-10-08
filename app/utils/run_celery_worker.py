"""文件用途：初始化数据库并导出 Celery Worker 启动入口。"""

from app import init_database
from app.utils.celery_worker import make_celery

init_database()
celery_app = make_celery()

# Import task modules after the database has initialized. Celery only
# registers decorated tasks when their defining module is imported.
from app.utils import tasks as _registered_tasks  # noqa: E402,F401
