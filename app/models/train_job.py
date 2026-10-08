"""文件用途：定义训练任务的资源引用、配置、状态、进度和结果记录。"""

from app import db
from datetime import datetime

class TrainJob(db.Model):
    """保存一次模型训练任务的输入配置、运行进度和产物位置。"""
    __tablename__ = 'train_jobs'

    id = db.Column(db.BigInteger, primary_key=True)
    job_id = db.Column(db.String(128), nullable=True, unique=True, index=True)
    description = db.Column(db.String(256), nullable=True)
    user_id = db.Column(db.BigInteger, nullable=False)
    dataset_id = db.Column(db.BigInteger, nullable=False)
    nodes = db.Column(db.JSON, nullable=False)
    backbone_id = db.Column(db.BigInteger, nullable=False)
    classifier_id = db.Column(db.BigInteger, nullable=False)
    config = db.Column(db.JSON, nullable=False)
    feature_ids = db.Column(db.JSON, nullable=False)
    adaptive = db.Column(db.Boolean, nullable=False)
    status = db.Column(db.String(32), default="pending", nullable=False)
    progress = db.Column(db.Integer, default=0)
    result = db.Column(db.JSON)
    model_path = db.Column(db.String(128))
    log_path = db.Column(db.String(128))
    createTime = db.Column(db.DateTime, default=datetime.now)
    updateTime = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    isDelete = db.Column(db.Boolean, default=False)
