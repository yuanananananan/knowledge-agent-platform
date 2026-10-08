"""文件用途：定义数据集图片路径、标签和数据划分信息。"""

from app import db
from datetime import datetime

# ===== 数据表模型 =====
class Image(db.Model):
    """记录数据集中的单个样本文件及其标签和训练划分。"""
    __tablename__ = 'images'

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True, comment='id')
    dataset_id = db.Column(db.BigInteger, nullable=False, comment='所属数据集ID')
    name = db.Column(db.String(128), nullable=False, comment='所属数据集名称')
    modality = db.Column(db.String(128), nullable=False, comment='模态类型(SAR, RD, 1D)')
    path = db.Column(db.String(256), nullable=False, comment='数据文件路径（绝对路径）')
    label = db.Column(db.String(128), nullable=False, comment='标签')
    createTime = db.Column(db.DateTime, default=datetime.now)
    updateTime = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    isDelete = db.Column(db.Integer, default=0, nullable=False, comment='是否删除')
    split = db.Column(db.String(16), nullable=True)
