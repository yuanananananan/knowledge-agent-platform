"""文件用途：定义分类器组件及其参数 Schema 的数据模型。"""

from app import db
from datetime import datetime

class Classifier(db.Model):
    """描述平台可选择的分类器及其参数约束。"""
    __tablename__ = 'classifiers'

    id = db.Column(db.BigInteger, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    params_schema = db.Column(db.JSON, nullable=False)
    description = db.Column(db.Text)
    createTime = db.Column(db.DateTime, default=datetime.now)
    updateTime = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    isDelete = db.Column(db.Boolean, default=False)
