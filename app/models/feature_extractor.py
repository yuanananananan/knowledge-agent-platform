"""文件用途：定义特征提取器及用户自定义特征组合的数据模型。"""

from app import db
from datetime import datetime

class FeatureExtractor(db.Model):
    """描述一个可用于训练流水线的特征提取器。"""
    __tablename__ = 'feature_extractors'

    id = db.Column(db.BigInteger, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    modality = db.Column(db.String(128), nullable=False)
    params_schema = db.Column(db.JSON)
    description = db.Column(db.Text)
    createTime = db.Column(db.DateTime, default=datetime.now)
    updateTime = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    isDelete = db.Column(db.Boolean, default=False)

class FeatureGroup(db.Model):
    """保存用户定义的特征提取器组合。"""
    __tablename__ = "feature_groups"

    id = db.Column(db.BigInteger, primary_key=True)
    name = db.Column(db.String(128), nullable=False, comment="组合名称")
    description = db.Column(db.Text, comment="组合描述")
    createTime = db.Column(db.DateTime, server_default=db.func.now())
    updateTime = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now())
    isDelete = db.Column(db.Boolean, default=False)

    items = db.relationship("FeatureGroupItem", backref="group", cascade="all, delete-orphan")


class FeatureGroupItem(db.Model):
    """建立特征组合与具体特征提取器之间的关联。"""
    __tablename__ = "feature_group_items"

    id = db.Column(db.BigInteger, primary_key=True)
    group_id = db.Column(db.BigInteger, db.ForeignKey("feature_groups.id", ondelete="CASCADE"))
    feature_extractor_id = db.Column(db.BigInteger, db.ForeignKey("feature_extractors.id", ondelete="CASCADE"))

    feature_extractor = db.relationship("FeatureExtractor", backref="group_links")
