"""文件用途：构建 LangGraph ReAct Agent，并实现非流式/流式执行、事件记录和会话持久化。"""

import hashlib
import json
import os
import re
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from sqlalchemy.exc import IntegrityError

from app import db
from app.agent.actions import ActionValidationError
from app.agent.checkpoints import append_message, get_or_create_conversation, load_agent_context, load_messages
from app.agent.context import bind_agent_user, reset_agent_user
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import ALL_AGENT_TOOLS
from app.agent.tools import draft_agent_action
from app.agent.workflow import route_request
from app.models.train_job import TrainJob
from app.models.agent_runtime import AgentRun, AgentTraceEvent


load_dotenv()

CONVERSATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
TRAIN_JOB_TOKEN_PATTERN = re.compile(
    r"(?<![0-9a-fA-F])(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{8}-[0-9a-fA-F-]{27}|(?=[0-9a-fA-F]{8}(?![0-9a-fA-F]))(?=[0-9a-fA-F]*[a-fA-F])[0-9a-fA-F]{8})(?![0-9a-fA-F])"
)


# 表示缺少模型密钥等无法启动 Agent 的配置错误。
class AgentConfigurationError(RuntimeError):
    pass


class AgentRunConflictError(RuntimeError):
    """The stable run id is already running or has failed and cannot be replayed."""

    pass


# 保存一次运行的稳定标识、用户、会话和当前事件序号。
@dataclass
class AgentRunHandle:
    run_id: str
    conversation_id: str
    user_id: int
    sequence: int = 0


# 校验调用方提供的会话 ID；为空时生成新 ID。
def normalize_conversation_id(conversation_id: str | None) -> str:
    if not conversation_id:
        return f"conv_{uuid.uuid4().hex}"
    if not CONVERSATION_ID_PATTERN.fullmatch(conversation_id):
        raise ValueError("conversation_id 只能包含字母、数字、下划线和连字符，且最长 64 位")
    return conversation_id


# 根据环境变量构建 OpenAI 兼容聊天模型客户端。
def _build_model() -> ChatOpenAI:
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise AgentConfigurationError("未配置 LLM_API_KEY，无法调用 DeepSeek API")
    return ChatOpenAI(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        model=os.getenv("LLM_MODEL_NAME", "deepseek-chat"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
    )


@lru_cache(maxsize=1)
def build_agent():
    """Return the tool-using Agent with access to the complete tool registry."""
    return create_react_agent(
        _build_model(),
        ALL_AGENT_TOOLS,
        prompt=SYSTEM_PROMPT,
        name="experiment_agent",
    )


# 将字符串或多段消息内容统一提取成纯文本。
def _message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(text for text in texts if text)
    return str(content or "")


def _confirmed_training_draft(message: str, user_id: int, conversation_id: str) -> str | None:
    """Create a reviewable training draft for an explicit confirmation.

    This deterministic path prevents a model wording variation from dropping a
    confirmed create_train_job request. It still uses the same validation and
    approval boundary as every other write action.
    """
    normalized = message.casefold()
    if not any(word in normalized for word in (
        "开始训练", "创建训练", "确认创建", "就按这个", "生成草案",
        "create_train_job", "clone_train_job",
    )):
        return None
    context = load_agent_context(user_id, conversation_id)
    candidates = TRAIN_JOB_TOKEN_PATTERN.findall(message)
    requested_id = candidates[-1] if candidates else ""
    source = None
    if requested_id:
        matches = TrainJob.query.filter(
            TrainJob.user_id == int(user_id),
            TrainJob.isDelete.is_(False),
            TrainJob.job_id.like(f"{requested_id}%"),
        ).limit(2).all()
        if len(matches) == 1:
            source = matches[0]
    recovery_request = any(word in normalized for word in ("重跑", "恢复", "失败任务"))
    if source is None and recovery_request:
        recovery_id = str(context.get("recovery_job_id") or "").strip()
        if recovery_id:
            recovery_matches = TrainJob.query.filter(
                TrainJob.user_id == int(user_id),
                TrainJob.isDelete.is_(False),
                TrainJob.job_id.like(f"{recovery_id}%"),
            ).limit(2).all()
            if len(recovery_matches) == 1:
                source = recovery_matches[0]
        if source is None:
            source = TrainJob.query.filter_by(
                user_id=int(user_id), status="failed", isDelete=False
            ).order_by(TrainJob.id.desc()).first()
    baseline_id = str(context.get("baseline_job_id") or "").strip()
    if source is None and baseline_id:
        source = TrainJob.query.filter_by(
            user_id=int(user_id), job_id=baseline_id, isDelete=False
        ).first()
    if not source:
        return None
    baseline_id = str(source.job_id)
    config = deepcopy(source.config or {})
    overrides = context.get("config_overrides") or {}
    epochs = overrides.get("training.epochs")
    if epochs is not None:
        config.setdefault("training", {})["epochs"] = int(epochs)
    result = json.loads(draft_agent_action.invoke({
        "action_type": "create_train_job",
        "payload": {
            "dataset_id": source.dataset_id,
            "backbone_id": source.backbone_id,
            "classifier_id": source.classifier_id,
            "feature_ids": list(source.feature_ids or []),
            "nodes": list(source.nodes or []),
            "config": config,
            "description": f"基于 {baseline_id} 的 Agent 优化训练",
        },
        "reason": f"用户确认以 {baseline_id} 为基线创建训练，等待人工审批。",
    }))
    if not result.get("ok"):
        error = result.get("error") or {}
        return (
            "训练草案未创建："
            + str(error.get("message") or "参数校验未通过")
            + "。未执行任何写操作。"
        )
    proposal = result.get("data") or {}
    proposal_id = proposal.get("proposal_id") or proposal.get("id") or ""
    epoch_text = f"，epochs={epochs}" if epochs is not None else ""
    return (
        f"已基于任务 `{baseline_id}`{epoch_text} 生成训练草案"
        f"（proposal_id: `{proposal_id}`）。尚未执行训练，请到页面右上角「待审批」中批准并执行。"
    )


def run_agent(
    message: str,
    user_id: int,
    conversation_id: str | None = None,
    run_id: str | None = None,
) -> tuple[str, str]:
    """Execute one turn using a durable, database-backed conversation checkpoint."""
    normalized_id = normalize_conversation_id(conversation_id)
    request_hash = hashlib.sha256(json.dumps(
        {
            "user_id": int(user_id),
            "conversation_id": normalized_id,
            "message": message,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    if run_id:
        existing = AgentRun.query.filter_by(run_id=run_id, user_id=int(user_id)).first()
        if existing:
            if existing.request_hash != request_hash:
                raise AgentRunConflictError(f"run {run_id} was created for another request")
            if existing.status == "completed" and existing.answer:
                return existing.answer, existing.conversation_id
            raise AgentRunConflictError(
                f"run {run_id} already exists with status {existing.status}"
            )
    get_or_create_conversation(user_id, normalized_id, message)
    history = load_messages(user_id, normalized_id)
    already_persisted = bool(history and history[-1] == {"role": "user", "content": message})
    if not already_persisted:
        append_message(user_id, normalized_id, "user", message)
        history = load_messages(user_id, normalized_id)
    input_messages = history
    run = _start_run(user_id, normalized_id, run_id, request_hash=request_hash)
    _record_event(run, "run.started", {"conversation_id": normalized_id})
    token = bind_agent_user(user_id, normalized_id)
    direct_answer = None
    try:
        direct_answer = _confirmed_training_draft(message, user_id, normalized_id)
        result = ({"answer": direct_answer} if direct_answer else build_agent().invoke(
            {"messages": input_messages},
            config={"recursion_limit": int(os.getenv("AGENT_MAX_STEPS", "20"))},
        ))
    except Exception as exc:
        _recover_db_session()
        _best_effort_failure(run, exc)
        raise
    finally:
        reset_agent_user(token)

    _record_workflow_result(
        run,
        result,
        message,
        direct_tools=["draft_agent_action"] if direct_answer else None,
    )
    answer = str(result.get("answer", "")).strip()
    if not answer:
        messages = result.get("messages", [])
        answer = _message_text(messages[-1].content) if messages else "Agent 没有返回结果。"
    append_message(user_id, normalized_id, "assistant", answer)
    _record_event(run, "run.completed", {"answer": answer})
    _finish_run(run, "completed", answer=answer)
    return answer, normalized_id


# 创建 Python 独立运行路径使用的 AgentRun 记录。
def _start_run(
    user_id: int,
    conversation_id: str,
    run_id: str | None = None,
    request_hash: str | None = None,
) -> AgentRunHandle:
    run_id = run_id or f"run_{uuid.uuid4().hex}"
    run = AgentRun(
        run_id=run_id,
        conversation_id=conversation_id,
        user_id=int(user_id),
        model=os.getenv("LLM_MODEL_NAME", "deepseek-chat"),
        request_hash=request_hash,
    )
    db.session.add(run)
    try:
        db.session.commit()
    except IntegrityError as exc:
        db.session.rollback()
        raise AgentRunConflictError(f"run {run_id} already exists") from exc
    return AgentRunHandle(
        run_id=run_id,
        conversation_id=conversation_id,
        user_id=int(user_id),
    )


# 分配递增序号、提交轨迹事件并返回流式事件信封。
def _record_event(run: AgentRunHandle, event_type: str, payload: dict) -> dict:
    run.sequence += 1
    event = AgentTraceEvent(
        run_id=run.run_id,
        user_id=run.user_id,
        sequence=run.sequence,
        event_type=event_type,
        payload=payload,
    )
    db.session.add(event)
    db.session.commit()
    return {
        "run_id": run.run_id,
        "sequence": run.sequence,
        "type": event_type,
        "payload": payload,
    }


def _react_tool_events(result: dict) -> list[dict]:
    """Extract this turn's completed tools from a ReAct agent result."""
    messages = result.get("messages") or []
    args_by_call_id: dict[str, dict] = {}
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls or []:
            call_id = str(call.get("id") or "")
            if call_id:
                args_by_call_id[call_id] = call.get("args") or {}

    events: list[dict] = []
    seen: set[str] = set()
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        call_id = str(message.tool_call_id or "")
        name = str(message.name or "tool")
        key = call_id or f"{name}:{len(events)}"
        if key in seen:
            continue
        seen.add(key)
        events.append({
            "name": name,
            "tool_call_id": call_id or None,
            "args": args_by_call_id.get(call_id, {}),
            "output": _message_text(message.content)[:8000],
        })
    return events


def _record_workflow_result(
    run: AgentRunHandle,
    result: dict,
    message: str,
    *,
    direct_tools: list[str] | None = None,
) -> None:
    """Persist intent and tool evidence for non-streaming/evaluation runs."""
    intent, route_reason = route_request(message)
    tool_events = _react_tool_events(result)
    tool_events.extend({"name": name, "args": {}, "output": ""} for name in direct_tools or [])
    _record_event(run, "workflow.node.completed", {
        "node": "route_intent",
        "intent": intent,
        "route_reason": route_reason,
        "tool_count": len(tool_events),
    })
    for event in tool_events:
        _record_event(run, "tool.completed", event)


# 将 Python 自管运行更新为终态并保存最终答案或错误。
def _finish_run(
    run: AgentRunHandle,
    status: str,
    error_message: str | None = None,
    answer: str | None = None,
):
    AgentRun.query.filter_by(run_id=run.run_id, user_id=run.user_id).update({
        "status": status,
        "answer": answer,
        "error_message": error_message[:2000] if error_message else None,
        "completed_at": datetime.utcnow(),
    })
    db.session.commit()


def _recover_db_session():
    """Discard a failed DBAPI connection before issuing any follow-up write."""
    try:
        db.session.rollback()
    except Exception:
        try:
            db.session.invalidate()
        except Exception:
            pass
    finally:
        try:
            db.session.remove()
        except Exception:
            pass


# 数据库可能异常时尽最大努力记录安全的失败事件。
def _best_effort_failure(run: AgentRunHandle, exc: Exception) -> dict:
    payload = {
        "error": type(exc).__name__,
        "message": "Agent 调用失败，请稍后重试",
    }
    try:
        _finish_run(run, "failed", str(exc))
        return _record_event(run, "run.failed", payload)
    except Exception:
        _recover_db_session()
        run.sequence += 1
        return {
            "run_id": run.run_id,
            "sequence": run.sequence,
            "type": "run.failed",
            "payload": payload,
        }


def run_agent_stream(
    message: str,
    user_id: int,
    conversation_id: str | None = None,
    run_id: str | None = None,
    *,
    persist_runtime: bool = True,
    user_message_persisted: bool = False,
):
    """Yield token/tool events. Go may own persistence while Python only executes."""
    normalized_id = normalize_conversation_id(conversation_id)
    get_or_create_conversation(user_id, normalized_id, message)
    history = load_messages(user_id, normalized_id)
    if not user_message_persisted and not (history and history[-1] == {"role": "user", "content": message}):
        append_message(user_id, normalized_id, "user", message)
        history = load_messages(user_id, normalized_id)
    input_messages = history
    run = _start_run(user_id, normalized_id, run_id) if persist_runtime else AgentRunHandle(
        run_id=run_id or f"run_{uuid.uuid4().hex}", conversation_id=normalized_id, user_id=user_id
    )

    def emit(event_type: str, event_payload: dict) -> dict:
        if persist_runtime:
            return _record_event(run, event_type, event_payload)
        run.sequence += 1
        return {"run_id": run.run_id, "sequence": run.sequence, "type": event_type, "payload": event_payload}

    yield emit("run.started", {
        "conversation_id": normalized_id,
        "agent_context": load_agent_context(user_id, normalized_id),
    })
    intent, route_reason = route_request(message)
    yield emit("workflow.node.completed", {
        "node": "route_intent",
        "intent": intent,
        "route_reason": route_reason,
    })

    answer_parts: list[str] = []
    last_partial_save = 0.0
    seen_tools: set[str] = set()
    token = bind_agent_user(user_id, normalized_id)
    try:
        direct_answer = _confirmed_training_draft(message, user_id, normalized_id)
        if direct_answer:
            yield emit("workflow.node.completed", {"node": "prepare_approval"})
            yield emit("tool.completed", {
                "name": "draft_agent_action",
                "args": {},
                "output": "",
            })
            yield emit("model.delta", {"content": direct_answer})
            append_message(user_id, normalized_id, "assistant", direct_answer)
            if persist_runtime:
                _finish_run(run, "completed", answer=direct_answer)
            yield emit("run.completed", {"answer": direct_answer, "conversation_id": normalized_id})
            return
        updates = build_agent().stream(
            {"messages": input_messages},
            config={
                "recursion_limit": int(os.getenv("AGENT_MAX_STEPS", "20")),
            },
            stream_mode=["messages", "updates"],
        )
        final_message = ""
        for mode, update in updates:
            if mode == "messages":
                chunk, metadata = update
                if isinstance(chunk, (AIMessage, AIMessageChunk)):
                    content = _message_text(chunk.content)
                    if content:
                        answer_parts.append(content)
                        yield emit("model.delta", {"content": content})
                continue
            if mode != "updates" or not isinstance(update, dict):
                continue
            for node_name, patch in update.items():
                patch = patch if isinstance(patch, dict) else {}
                yield emit("workflow.node.completed", {"node": node_name})
                for item in patch.get("messages", []):
                    if isinstance(item, ToolMessage):
                        key = f"{item.name}:{item.tool_call_id}"
                        if key not in seen_tools:
                            seen_tools.add(key)
                            yield emit("tool.completed", {
                                "name": item.name or "tool",
                                "tool_call_id": item.tool_call_id,
                                "output": _message_text(item.content)[:8000],
                            })
                    elif isinstance(item, AIMessage):
                        final_message = _message_text(item.content).strip() or final_message

        answer = "".join(answer_parts).strip() or final_message or "Agent 没有返回结果。"
        append_message(user_id, normalized_id, "assistant", answer)
        if persist_runtime:
            _finish_run(run, "completed", answer=answer)
        yield emit("run.completed", {
            "answer": answer,
            "conversation_id": normalized_id,
        })
    except GeneratorExit:
        _recover_db_session()
        try:
            if persist_runtime:
                _finish_run(run, "cancelled", "运行已取消")
        except Exception:
            _recover_db_session()
        raise
    except Exception as exc:
        _recover_db_session()
        detail = str(exc) if isinstance(exc, ActionValidationError) else "Agent 调用失败，请稍后重试"
        partial = "".join(answer_parts).strip()
        failure_answer = (
            f"{partial}\n\n草案未创建：{detail}"
            if partial else f"本次 Agent 运行失败：{detail}"
        )
        try:
            append_message(user_id, normalized_id, "assistant", failure_answer)
        except Exception:
            _recover_db_session()
        if persist_runtime:
            event = _best_effort_failure(run, exc)
            event["payload"]["message"] = detail
            event["payload"]["partial_answer"] = failure_answer
            yield event
        else:
            yield emit("run.failed", {
                "error": type(exc).__name__,
                "message": detail,
                "partial_answer": failure_answer,
            })
    finally:
        reset_agent_user(token)
