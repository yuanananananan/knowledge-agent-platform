"""文件用途：定义 Agent 会话、检查点、运行轨迹和 Golden Eval 相关数据模型。"""

from datetime import datetime

from app import db


class AgentConversation(db.Model):
    """保存一次可持续恢复的 Agent 对话及其最新检查点版本。"""
    __tablename__ = "agent_conversations"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    conversation_id = db.Column(db.String(64), nullable=False, unique=True, index=True)
    user_id = db.Column(db.BigInteger, nullable=False, index=True)
    title = db.Column(db.String(160), nullable=False, default="新对话")
    checkpoint_version = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        """返回对外展示的会话摘要。"""
        return {
            "conversation_id": self.conversation_id,
            "title": self.title,
            "checkpoint_version": self.checkpoint_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class AgentCheckpoint(db.Model):
    """保存不可变的会话快照，用于进程重启后的断点恢复。"""

    __tablename__ = "agent_checkpoints"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    checkpoint_id = db.Column(db.String(64), nullable=False, unique=True, index=True)
    conversation_id = db.Column(db.String(64), nullable=False, index=True)
    user_id = db.Column(db.BigInteger, nullable=False, index=True)
    version = db.Column(db.Integer, nullable=False)
    state = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("conversation_id", "version", name="uq_agent_checkpoint_version"),
    )


class AgentRun(db.Model):
    """记录一次 Agent 运行的状态、答案、模型和起止时间。"""
    __tablename__ = "agent_runs"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    run_id = db.Column(db.String(64), nullable=False, unique=True, index=True)
    conversation_id = db.Column(db.String(64), nullable=False, index=True)
    user_id = db.Column(db.BigInteger, nullable=False, index=True)
    status = db.Column(db.String(24), nullable=False, default="running", index=True)
    model = db.Column(db.String(128), nullable=True)
    request_hash = db.Column(db.String(64), nullable=True)
    answer = db.Column(db.Text, nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    started_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)

    def to_dict(self):
        """返回一次运行的 API 表示。"""
        return {
            "run_id": self.run_id,
            "conversation_id": self.conversation_id,
            "status": self.status,
            "model": self.model,
            "request_hash": self.request_hash,
            "answer": self.answer,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class AgentTraceEvent(db.Model):
    """按严格递增序号保存运行事件，供流式回放和故障排查使用。"""
    __tablename__ = "agent_trace_events"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    run_id = db.Column(db.String(64), nullable=False, index=True)
    user_id = db.Column(db.BigInteger, nullable=False, index=True)
    sequence = db.Column(db.Integer, nullable=False)
    event_type = db.Column(db.String(48), nullable=False, index=True)
    payload = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (db.UniqueConstraint("run_id", "sequence", name="uq_agent_trace_sequence"),)

    def to_dict(self):
        """返回前端流式协议使用的事件结构。"""
        return {
            "sequence": self.sequence,
            "type": self.event_type,
            "payload": self.payload,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AgentGoldenCase(db.Model):
    """保存用于回归评测的标准问题和预期行为。"""
    __tablename__ = "agent_golden_cases"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    case_id = db.Column(db.String(64), nullable=False, unique=True, index=True)
    name = db.Column(db.String(160), nullable=False)
    category = db.Column(db.String(64), nullable=False, index=True)
    prompt = db.Column(db.Text, nullable=False)
    required_keywords = db.Column(db.JSON, nullable=False, default=list)
    forbidden_keywords = db.Column(db.JSON, nullable=False, default=list)
    expected_intent = db.Column(db.String(32), nullable=True)
    expected_tools = db.Column(db.JSON, nullable=False, default=list)
    enabled = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def to_dict(self):
        """返回标准评测用例的可序列化结构。"""
        return {
            "case_id": self.case_id,
            "name": self.name,
            "category": self.category,
            "prompt": self.prompt,
            "required_keywords": self.required_keywords or [],
            "forbidden_keywords": self.forbidden_keywords or [],
            "expected_intent": self.expected_intent,
            "expected_tools": self.expected_tools or [],
            "enabled": self.enabled,
        }


class AgentEvalRun(db.Model):
    """汇总一批 Agent 标准用例的评测状态和指标。"""
    __tablename__ = "agent_eval_runs"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    eval_run_id = db.Column(db.String(64), nullable=False, unique=True, index=True)
    user_id = db.Column(db.BigInteger, nullable=False, index=True)
    status = db.Column(db.String(24), nullable=False, default="running")
    total = db.Column(db.Integer, nullable=False, default=0)
    passed = db.Column(db.Integer, nullable=False, default=0)
    score = db.Column(db.Float, nullable=False, default=0.0)
    metrics = db.Column(db.JSON, nullable=False, default=dict)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)

    def to_dict(self):
        """返回整批评测的统计摘要。"""
        return {
            "eval_run_id": self.eval_run_id,
            "status": self.status,
            "total": self.total,
            "passed": self.passed,
            "score": self.score,
            "metrics": self.metrics or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class AgentEvalResult(db.Model):
    """记录某个标准用例在一次评测中的详细结果。"""
    __tablename__ = "agent_eval_results"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    eval_run_id = db.Column(db.String(64), nullable=False, index=True)
    case_id = db.Column(db.String(64), nullable=False, index=True)
    passed = db.Column(db.Boolean, nullable=False)
    score = db.Column(db.Float, nullable=False)
    answer = db.Column(db.Text, nullable=True)
    details = db.Column(db.JSON, nullable=False)
    latency_ms = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def to_dict(self):
        """返回单条评测结果。"""
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "score": self.score,
            "answer": self.answer,
            "details": self.details,
            "latency_ms": self.latency_ms,
        }
