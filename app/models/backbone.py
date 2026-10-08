"""文件用途：定义可用于训练任务的骨干网络模型元数据。"""

from app import db
from datetime import datetime

class Backbone(db.Model):
    """描述平台可选择的骨干网络及其参数约束和模型产物。"""
    __tablename__ = 'backbones'

    id = db.Column(db.BigInteger, primary_key=True)
    name = db.Column(db.String(128), nullable=False)

    modality = db.Column(db.String(32), nullable=True)
    backbone_type = db.Column(db.String(64), default="normal")
    
    params_schema = db.Column(db.JSON, nullable=False)
    artifact_path = db.Column(db.String(256), nullable=True
                              )
    description = db.Column(db.Text)
    createTime = db.Column(db.DateTime, default=datetime.now)
    updateTime = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    isDelete = db.Column(db.Boolean, default=False)
