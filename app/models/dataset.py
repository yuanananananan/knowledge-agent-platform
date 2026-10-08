"""文件用途：定义用户数据集、类别和存储路径等元数据。"""

from app import db
from datetime import datetime

class Dataset(db.Model):
    """保存一个训练数据集的名称、模态、路径和标签信息。"""
    __tablename__ = 'datasets'

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    name = db.Column(db.String(128), nullable=False)
    modality = db.Column(db.String(128), nullable=False)
    path = db.Column(db.String(256), nullable=False)
    labels = db.Column(db.String(128), nullable=False)
    createTime = db.Column(db.DateTime, default=datetime.now)
    updateTime = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    isDelete = db.Column(db.Boolean, default=False)
