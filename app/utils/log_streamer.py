"""文件用途：按时间戳增量读取任务日志，供轮询或流式展示使用。"""

from pathlib import Path
from datetime import datetime
import os

# 项目根目录，例如 knowledge_emb/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"

def append_log(job_id, message):
    """为指定任务追加一条带本地时间戳的日志。"""
    message = (message or "").strip()
    if not message:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{job_id}.log"

    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%H:%M:%S}] {message}\n")
