"""文件用途：定义记录用户、安全和业务操作的审计日志模型。"""

from datetime import datetime
from app import db


class AuditLog(db.Model):
    """记录谁在何时对什么资源执行了何种操作以及是否成功。"""
    __tablename__ = "audit_logs"
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    user_id = db.Column(db.BigInteger, nullable=True, index=True)
    action = db.Column(db.String(128), nullable=False, index=True)
    resource_type = db.Column(db.String(64), nullable=True)
    resource_id = db.Column(db.String(128), nullable=True)
    detail = db.Column(db.JSON, nullable=True)
    success = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
