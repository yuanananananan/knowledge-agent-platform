"""文件用途：以不可变版本快照持久化 Agent 会话消息和结构化上下文。"""

import uuid
import re
from copy import deepcopy
from datetime import datetime

from app import db
from app.models.agent_runtime import AgentCheckpoint, AgentConversation


# 读取用户会话；不存在时创建并生成初始标题。
def get_or_create_conversation(user_id: int, conversation_id: str, first_message: str = ""):
    conversation = AgentConversation.query.filter_by(
        user_id=int(user_id), conversation_id=conversation_id
    ).first()
    if conversation:
        return conversation

    # A globally unique public ID prevents two users from aliasing the same checkpoint row.
    collision = AgentConversation.query.filter_by(conversation_id=conversation_id).first()
    if collision:
        raise ValueError("conversation_id 已被占用")
    title = first_message.strip().replace("\n", " ")[:80] or "新对话"
    conversation = AgentConversation(
        user_id=int(user_id), conversation_id=conversation_id, title=title
    )
    db.session.add(conversation)
    db.session.commit()
    return conversation


JOB_ID_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_-])(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{8}-[0-9a-fA-F-]{27}|(?=[0-9a-fA-F]{8}(?![A-Za-z0-9_-]))(?=[0-9a-fA-F]*[a-fA-F])[0-9a-fA-F]{8}|(?:train|val|nas)[A-Za-z0-9_-]+)(?![A-Za-z0-9_-])",
    re.IGNORECASE,
)
EPOCH_PATTERN = re.compile(r"(?:epoch(?:s)?|轮数|训练轮数)\s*(?:改为|设为|设置为|=|:|：)?\s*(\d{1,3})", re.IGNORECASE)


# 读取会话最新版本的不可变状态快照。
def load_state(user_id: int, conversation_id: str) -> dict:
    checkpoint = (
        AgentCheckpoint.query.filter_by(user_id=int(user_id), conversation_id=conversation_id)
        .order_by(AgentCheckpoint.version.desc())
        .first()
    )
    if not checkpoint:
        return {"messages": [], "agent_context": {}}
    state = deepcopy(checkpoint.state or {})
    state.setdefault("messages", [])
    state.setdefault("agent_context", {})
    return state


# 从最新检查点中返回消息历史。
def load_messages(user_id: int, conversation_id: str) -> list[dict]:
    return list(load_state(user_id, conversation_id).get("messages", []))


# 从最新检查点中返回结构化 Agent 上下文。
def load_agent_context(user_id: int, conversation_id: str) -> dict:
    return dict(load_state(user_id, conversation_id).get("agent_context", {}))


# 从新增消息中提取基线任务、恢复任务、训练轮数和确认信息。
def _updated_context(context: dict, role: str, content: str) -> dict:
    result = deepcopy(context or {})
    job_ids = JOB_ID_PATTERN.findall(content or "")
    normalized = (content or "").casefold()
    baseline_ids = []
    for line in (content or "").splitlines():
        if any(word in line.casefold() for word in ("基线", "当前最佳", "最优任务")):
            baseline_ids.extend(JOB_ID_PATTERN.findall(line))
    if baseline_ids:
        result["baseline_job_id"] = baseline_ids[0]
    elif job_ids and not result.get("baseline_job_id"):
        result["baseline_job_id"] = job_ids[0]
    if job_ids and role == "user" and any(
        word in normalized for word in ("失败", "重跑", "恢复")
    ):
        result["recovery_job_id"] = job_ids[-1]
    epoch = EPOCH_PATTERN.search(content or "")
    if epoch:
        result.setdefault("config_overrides", {})["training.epochs"] = int(epoch.group(1))
    if role == "user":
        if any(word in normalized for word in ("开始训练", "创建训练", "确认创建", "就按这个")):
            result.setdefault("confirmations", {})["create_train_job"] = True
    result["updated_at"] = datetime.utcnow().isoformat()
    return result


# 以递增版本写入新的不可变会话快照。
def save_state(user_id: int, conversation_id: str, state: dict):
    conversation = get_or_create_conversation(user_id, conversation_id)
    next_version = conversation.checkpoint_version + 1
    checkpoint = AgentCheckpoint(
        checkpoint_id=f"cp_{uuid.uuid4().hex}",
        conversation_id=conversation_id,
        user_id=int(user_id),
        version=next_version,
        state=deepcopy(state),
    )
    conversation.checkpoint_version = next_version
    conversation.updated_at = datetime.utcnow()
    db.session.add(checkpoint)
    db.session.commit()
    return checkpoint


# 保存完整消息列表，并增量更新结构化上下文。
def save_messages(user_id: int, conversation_id: str, messages: list[dict]):
    previous = load_state(user_id, conversation_id)
    context = previous.get("agent_context", {})
    for message in messages[len(previous.get("messages", [])):]:
        context = _updated_context(context, str(message.get("role", "")), str(message.get("content", "")))
    return save_state(user_id, conversation_id, {
        "messages": messages,
        "agent_context": context,
    })


def append_message(user_id: int, conversation_id: str, role: str, content: str):
    """Append one durable turn without replacing structured conversation state."""
    get_or_create_conversation(user_id, conversation_id, content if role == "user" else "")
    state = load_state(user_id, conversation_id)
    messages = list(state.get("messages", []))
    if not messages or messages[-1] != {"role": role, "content": content}:
        messages.append({"role": role, "content": content})
    state["messages"] = messages
    state["agent_context"] = _updated_context(
        state.get("agent_context", {}), role, content
    )
    return save_state(user_id, conversation_id, state)


# 按更新时间倒序查询用户的普通会话。
def list_conversations(user_id: int, limit: int = 50):
    return (
        AgentConversation.query.filter_by(user_id=int(user_id))
        .filter(AgentConversation.conversation_id.notlike("eval\\_%", escape="\\"))
        .order_by(AgentConversation.updated_at.desc())
        .limit(max(1, min(limit, 100)))
        .all()
    )


# 校验会话归属并返回会话及其最新消息。
def conversation_messages(user_id: int, conversation_id: str):
    conversation = AgentConversation.query.filter_by(
        user_id=int(user_id), conversation_id=conversation_id
    ).first()
    if not conversation:
        raise ValueError("对话不存在")
    return conversation, load_messages(user_id, conversation_id)
