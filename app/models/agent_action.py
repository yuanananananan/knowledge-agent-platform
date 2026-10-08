"""文件用途：定义 Agent 动作提案和动作审计事件的 SQLAlchemy 数据模型。"""

from datetime import datetime

from app import db


class AgentActionProposal(db.Model):
    """持久化 Agent 的动作提案，确保有副作用的操作先经过人工审批。"""

    __tablename__ = "agent_action_proposals"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    proposal_id = db.Column(db.String(64), nullable=False, unique=True, index=True)
    user_id = db.Column(db.BigInteger, nullable=False, index=True)
    conversation_id = db.Column(db.String(64), nullable=True, index=True)
    action_type = db.Column(db.String(64), nullable=False)
    payload = db.Column(db.JSON, nullable=False)
    reason = db.Column(db.Text, nullable=True)
    impact = db.Column(db.Text, nullable=False)
    idempotency_key = db.Column(db.String(64), nullable=False, unique=True, index=True)
    status = db.Column(
        db.String(32),
        nullable=False,
        default="pending_approval",
        index=True,
    )
    result = db.Column(db.JSON, nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    expires_at = db.Column(db.DateTime, nullable=False)
    approved_at = db.Column(db.DateTime, nullable=True)
    executed_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    def to_dict(self):
        """把数据库对象转换为可直接返回给 API 调用方的字典。"""
        return {
            "proposal_id": self.proposal_id,
            "action_type": self.action_type,
            "conversation_id": self.conversation_id,
            "payload": self.payload,
            "reason": self.reason,
            "impact": self.impact,
            "status": self.status,
            "result": self.result,
            "error_message": self.error_message,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
