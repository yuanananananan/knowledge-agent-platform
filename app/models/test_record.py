"""文件用途：定义模型验证/测试任务的参数、状态、日志和结果记录。"""

from app import db
from datetime import datetime

# This is an ORM model, not a pytest module, despite the filename prefix.
__test__ = False


class TestRecord(db.Model):
    """保存一次模型验证任务的配置、状态和最终指标。"""
    __tablename__ = 'test_records'
    # pytest may import this model once as ``test_record`` (filename matches
    # its discovery pattern) and once as ``app.models.test_record``. Reuse the
    # existing SQLAlchemy table in that case instead of failing collection.
    __table_args__ = {'extend_existing': True}

    id = db.Column(db.BigInteger, primary_key=True)
    val_job_id = db.Column(db.String(64), nullable=False, unique=True, index=True)
    train_job_id = db.Column(db.String(64), nullable=False)
    user_id = db.Column(db.BigInteger, nullable=False)
    config = db.Column(db.JSON, nullable=False)
    result = db.Column(db.JSON)
    status = db.Column(db.String(32), default='pending', nullable=False)
    createTime = db.Column(db.DateTime, default=datetime.now)
    updateTime = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    isDelete = db.Column(db.Boolean, default=False)
