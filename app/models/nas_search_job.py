"""文件用途：定义神经架构搜索任务的配置、进度、日志和结果模型。"""

from datetime import datetime

from app import db


class NasSearchJob(db.Model):
    """保存一次 NAS 搜索任务从提交到产出骨干网络的完整状态。"""
    __tablename__ = "nas_search_jobs"

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)

    # Celery 异步任务 id
    job_id = db.Column(db.String(128), nullable=True, unique=True)

    user_id = db.Column(db.BigInteger, nullable=False)
    dataset_id = db.Column(db.BigInteger, nullable=False)

    modality = db.Column(db.String(32), nullable=False)
    description = db.Column(db.String(256), nullable=True)

    config = db.Column(db.JSON, nullable=False)
    status = db.Column(db.String(32), default="pending", nullable=False)

    progress = db.Column(db.BigInteger, default=0)
    genotype = db.Column(db.JSON, nullable=True)
    result = db.Column(db.JSON, nullable=True)

    backbone_id = db.Column(db.BigInteger, nullable=True)
    checkpoint_dir = db.Column(db.String(256), nullable=True)
    log_path = db.Column(db.String(256), nullable=True)

    createTime = db.Column(db.DateTime, default=datetime.now)
    updateTime = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    isDelete = db.Column(db.Boolean, default=False, nullable=False)
