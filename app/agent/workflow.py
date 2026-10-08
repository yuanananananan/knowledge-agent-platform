"""文件用途：定义显式、可审计的实验 Agent LangGraph 工作流。

The old implementation delegated both routing and tool selection to a generic
``create_agent`` loop.  This module keeps model generation, but makes the
business-critical control flow deterministic and inspectable:

    route intent -> collect scoped evidence -> diagnose/plan/prepare approval
                 -> compose answer

Only the approval node may create an action *proposal*.  It can never execute
an operation; execution remains behind the authenticated REST approval API.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import (
    analyze_job_log,
    compare_val_jobs,
    diff_train_configs,
    draft_agent_action,
    get_experiment_lineage,
    get_train_job_detail,
    list_recent_train_jobs,
    list_recent_nas_jobs,
    observe_datasets,
    observe_model_resources,
    rank_validation_jobs,
    recommend_next_experiment,
    search_knowledge_base,
)


Intent = Literal["diagnose", "plan", "compare", "observe", "knowledge", "action", "general"]
WorkflowNode = Literal[
    "route_intent",
    "collect_diagnostic_evidence",
    "collect_planning_evidence",
    "diagnose_evidence",
    "plan_experiment",
    "collect_comparison_evidence",
    "collect_observation_evidence",
    "collect_knowledge_evidence",
    "prepare_approval",
    "compose_answer",
]


class AgentWorkflowState(TypedDict, total=False):
    """The durable logical state passed between explicitly named graph nodes."""

    messages: list[dict[str, str]]
    intent: Intent
    route_reason: str
    evidence: list[dict[str, Any]]
    analysis_guidance: str
    approval: dict[str, Any] | None
    answer: str


JOB_ID_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_-])(?:[0-9a-fA-F]{32}|[0-9a-fA-F]{8}-[0-9a-fA-F-]{27}|(?:train|val|nas)[A-Za-z0-9_-]+)(?![A-Za-z0-9_-])",
    re.IGNORECASE,
)
MODEL_ROUTABLE_INTENTS: set[Intent] = {"diagnose", "plan", "compare", "observe", "knowledge", "general"}


# 从工作流消息中提取最近一条用户输入。
def latest_user_message(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


def extract_job_ids(text: str) -> list[str]:
    """Extract known job-id-shaped tokens without accepting arbitrary paths."""
    return list(dict.fromkeys(match.group(0) for match in JOB_ID_PATTERN.finditer(text)))[:2]


def route_request(text: str) -> tuple[Intent, str]:
    """Route high-risk/structured requests deterministically before model use."""
    normalized = text.casefold()
    if any(word in normalized for word in ("停止", "终止", "取消训练", "重试", "恢复训练", "克隆", "创建训练", "开始训练", "启动验证")):
        return "action", "检测到训练或验证的副作用请求，进入审批准备分支"
    if any(word in normalized for word in ("平台可用", "当前平台", "模型资源")) and any(
        word in normalized for word in ("数据集", "模型", "骨干", "分类器", "特征算子")
    ):
        return "observe", "检测到平台数据集或模型资源查询请求"
    if any(word in normalized for word in (
        "算子", "特征", "glcm", "lbp", "hog", "fourier", "小波", "多普勒",
        "方位", "纹理", "知识库", "rag", "切分", "召回", "引用",
    )) and not any(word in normalized for word in (
        "任务失败", "报错", "日志", "oom", "nan", "inf",
        "下一轮", "方案", "规划", "建议", "提升", "过拟合", "资源估算",
    )):
        return "knowledge", "检测到算子知识、选型依据或 RAG 资料问题"
    if any(word in normalized for word in (
        "诊断", "失败", "报错", "错误", "异常", "oom", "显存", "日志",
        "为什么", "nan", "inf", "非法数值",
    )):
        return "diagnose", "检测到故障诊断或日志分析请求"
    if any(word in normalized for word in ("下一轮", "方案", "规划", "建议", "提升", "过拟合", "资源估算")):
        return "plan", "检测到实验规划或优化建议请求"
    if any(word in normalized for word in ("对比", "比较", "排名", "显著性", "指标")):
        return "compare", "检测到实验指标或配置比较请求"
    if any(word in normalized for word in ("数据集", "模型资源", "骨干", "分类器", "训练任务", "nas")):
        return "observe", "检测到平台资源或任务查询请求"
    if normalized.strip(" ！!。,.？?") in {"你好", "您好", "hello", "hi", "谢谢", "感谢"}:
        return "general", "检测到普通寒暄，无需调用分类模型"
    return "general", "未匹配到专用工作流，保留对话上下文直接回答"


# 将工具响应包装成带名称、状态和元数据的证据对象。
def _artifact(name: str, result: str, **metadata: Any) -> dict[str, Any]:
    # Tool responses may include a full log.  Keep a useful, bounded audit
    # record and avoid sending an unbounded prompt to the final model node.
    return {"tool": name, "result": str(result)[:16_000], **metadata}


# 安全调用证据工具，并把异常转换为可审计结果。
def _call(tool_instance, name: str, **kwargs: Any) -> dict[str, Any]:
    try:
        return _artifact(name, tool_instance.invoke(kwargs), args=kwargs)
    except Exception as exc:
        # A failed evidence source should be visible to the final answer rather
        # than silently causing the model to fill the gap with a hallucination.
        return _artifact(name, json.dumps({
            "ok": False,
            "error": {"code": "tool_failure", "message": type(exc).__name__},
        }, ensure_ascii=False), args=kwargs)


# 从模型返回的字符串或内容块中提取纯文本。
def _model_message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content or "")


# 解析模型生成的结构化路由，并拒绝未知意图。
def _parse_model_route(content: str) -> tuple[Intent, str] | None:
    match = re.search(r"\{[\s\S]*\}", content)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    intent = str(payload.get("intent", "")).casefold()
    if intent not in MODEL_ROUTABLE_INTENTS:
        return None
    reason = str(payload.get("reason", "模型补充分类")).strip()[:200]
    return intent, reason or "模型补充分类"


def route_intent_with_model(model: Any) -> Callable[[AgentWorkflowState], dict[str, Any]]:
    """Use deterministic rules first and a constrained model classifier only as fallback."""
    def node(state: AgentWorkflowState) -> dict[str, Any]:
        text = latest_user_message(state.get("messages", []))
        intent, reason = route_request(text)
        if intent != "general" or "普通寒暄" in reason:
            return {"intent": intent, "route_reason": f"规则路由：{reason}", "evidence": []}
        classifier_messages = [
            {
                "role": "system",
                "content": (
                    "你是实验平台的意图分类器。只输出一个 JSON 对象，不要回答用户问题。"
                    "intent 只能是 diagnose、plan、compare、observe、knowledge、general 之一；"
                    "不得输出 action，所有有副作用操作只能由代码规则识别。"
                    "diagnose=任务故障或日志分析；plan=实验方案规划；compare=配置或指标比较；"
                    "observe=查询平台实时资源；knowledge=算子原理或RAG知识；general=其他对话。"
                    "格式：{\"intent\":\"general\",\"reason\":\"简短理由\"}"
                ),
            },
            {"role": "user", "content": text[:4000]},
        ]
        try:
            parsed = _parse_model_route(_model_message_text(model.invoke(classifier_messages)))
        except Exception:
            parsed = None
        if parsed is None:
            return {"intent": "general", "route_reason": "模型补充路由失败，安全回退到 general", "evidence": []}
        model_intent, model_reason = parsed
        return {
            "intent": model_intent,
            "route_reason": f"模型补充路由：{model_reason}",
            "evidence": [],
        }

    return node


# 为任务失败、日志异常等诊断请求收集证据。
def collect_diagnostic_evidence(state: AgentWorkflowState) -> dict[str, Any]:
    text = latest_user_message(state.get("messages", []))
    job_ids = extract_job_ids(text)
    evidence: list[dict[str, Any]] = []
    if not job_ids:
        evidence.append(_call(list_recent_train_jobs, "list_recent_train_jobs", limit=5, status=""))
        return {"evidence": evidence}
    for job_id in job_ids:
        evidence.extend([
            _call(get_train_job_detail, "get_train_job_detail", job_id=job_id),
            _call(analyze_job_log, "analyze_job_log", job_id=job_id, max_lines=300),
            _call(get_experiment_lineage, "get_experiment_lineage", job_id=job_id),
        ])
    return {"evidence": evidence}


# 为下一步实验规划收集基线、资源和配置证据。
def collect_planning_evidence(state: AgentWorkflowState) -> dict[str, Any]:
    text = latest_user_message(state.get("messages", []))
    job_ids = extract_job_ids(text)
    evidence: list[dict[str, Any]] = []
    if job_ids:
        objective = "recover_failure" if any(word in text.casefold() for word in ("失败", "恢复")) else "improve_accuracy"
        evidence.extend([
            _call(get_train_job_detail, "get_train_job_detail", job_id=job_ids[0]),
            _call(recommend_next_experiment, "recommend_next_experiment", source_job_id=job_ids[0], objective=objective),
        ])
    else:
        evidence.extend([
            _call(list_recent_train_jobs, "list_recent_train_jobs", limit=5, status="completed"),
            _call(observe_model_resources, "observe_model_resources", modality=""),
        ])
    evidence.append(_call(search_knowledge_base, "search_knowledge_base", query=text, modality="", operator="", top_k=6))
    return {"evidence": evidence}


# 从 RAG 知识库收集带来源的回答证据。
def collect_knowledge_evidence(state: AgentWorkflowState) -> dict[str, Any]:
    text = latest_user_message(state.get("messages", []))
    return {
        "evidence": [
            _call(search_knowledge_base, "search_knowledge_base", query=text, modality="", operator="", top_k=6),
        ]
    }


def diagnose_evidence(state: AgentWorkflowState) -> dict[str, Any]:
    """Make the diagnostic contract explicit before the model writes prose."""
    return {
        "analysis_guidance": (
            "诊断节点已完成：先陈述日志或任务状态中的确定证据；再将未证实的"
            "推断标为可能原因；最后给出一个最小改动的验证方案。"
        )
    }


def plan_experiment(state: AgentWorkflowState) -> dict[str, Any]:
    """Make the experimental-design constraint a visible graph transition."""
    return {
        "analysis_guidance": (
            "规划节点已完成：从证据中的候选方案选择一个最适合当前目标的实验，"
            "每次只改变一个核心变量，并说明基线、预期观察指标和停止条件。"
        )
    }


# 收集多个实验或验证任务之间的比较证据。
def collect_comparison_evidence(state: AgentWorkflowState) -> dict[str, Any]:
    text = latest_user_message(state.get("messages", []))
    job_ids = extract_job_ids(text)
    evidence: list[dict[str, Any]] = []
    if len(job_ids) >= 2 and all(job_id.casefold().startswith("val") for job_id in job_ids):
        evidence.append(_call(compare_val_jobs, "compare_val_jobs", job_id_1=job_ids[0], job_id_2=job_ids[1]))
    elif len(job_ids) >= 2:
        evidence.append(_call(diff_train_configs, "diff_train_configs", job_id_1=job_ids[0], job_id_2=job_ids[1]))
    else:
        evidence.append(_call(rank_validation_jobs, "rank_validation_jobs", metric="oa", limit=10))
    return {"evidence": evidence}


# 为普通平台查询收集数据集、模型和任务概况。
def collect_observation_evidence(state: AgentWorkflowState) -> dict[str, Any]:
    text = latest_user_message(state.get("messages", []))
    normalized = text.casefold()
    job_ids = extract_job_ids(text)
    evidence: list[dict[str, Any]] = []
    if job_ids and job_ids[0].casefold().startswith("train"):
        evidence.append(_call(get_train_job_detail, "get_train_job_detail", job_id=job_ids[0]))
    if "数据集" in text or "dataset" in normalized:
        evidence.append(_call(observe_datasets, "observe_datasets", modality="", limit=20))
    if any(word in normalized for word in ("模型", "骨干", "分类器", "特征")):
        evidence.append(_call(observe_model_resources, "observe_model_resources", modality=""))
    if any(word in normalized for word in ("训练任务", "train")):
        if not job_ids:
            evidence.append(_call(list_recent_train_jobs, "list_recent_train_jobs", limit=10, status=""))
    if "nas" in normalized:
        evidence.append(_call(list_recent_nas_jobs, "list_recent_nas_jobs", limit=10, status=""))
    if not evidence:
        evidence.append(_call(list_recent_train_jobs, "list_recent_train_jobs", limit=5, status=""))
    return {"evidence": evidence}


# 从用户消息识别明确的动作草案类型和目标任务。
def _action_request(text: str) -> tuple[str | None, str | None]:
    normalized = text.casefold()
    job_ids = extract_job_ids(text)
    job_id = job_ids[0] if job_ids else None
    if any(word in normalized for word in ("停止", "终止", "取消训练")):
        return "stop_train_job", job_id
    if any(word in normalized for word in ("重试", "恢复训练")):
        return "retry_train_job", job_id
    if "克隆" in text:
        return "clone_train_job", job_id
    if any(word in normalized for word in ("启动验证", "创建验证")):
        return "start_validation", job_id
    return None, job_id


def prepare_approval(state: AgentWorkflowState) -> dict[str, Any]:
    """Create only a reviewable proposal for unambiguous low-parameter actions."""
    text = latest_user_message(state.get("messages", []))
    action_type, job_id = _action_request(text)
    evidence: list[dict[str, Any]] = []
    if job_id:
        evidence.append(_call(get_train_job_detail, "get_train_job_detail", job_id=job_id))
    if not action_type or not job_id:
        return {
            "evidence": evidence,
            "approval": {
                "status": "needs_input",
                "message": "创建训练或验证草案前需要完整、可验证的参数；未执行任何操作。",
            },
        }
    # start_validation expects a training job id and is intentionally included
    # only after the source job has been resolved by the scoped tool above.
    payload = {"train_job_id": job_id} if action_type == "start_validation" else {"job_id": job_id}
    proposal = _call(
        draft_agent_action,
        "draft_agent_action",
        action_type=action_type,
        payload=payload,
        reason=f"用户在 Agent 对话中请求 {action_type}，等待人工审批。",
    )
    evidence.append(proposal)
    try:
        proposal_ok = bool(json.loads(str(proposal["result"])).get("ok"))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        proposal_ok = False
    return {
        "evidence": evidence,
        "approval": {
            "status": "proposal_created" if proposal_ok else "proposal_not_created",
            "action_type": action_type,
            "job_id": job_id,
            "message": (
                "草案已创建；必须由用户在审批界面确认后才会执行。"
                if proposal_ok
                else "草案未创建，请根据工具返回的校验错误补充或修正参数；未执行任何操作。"
            ),
        },
    }


# 将结构化证据整理为供最终模型阅读的提示文本。
def _evidence_prompt(evidence: list[dict[str, Any]]) -> str:
    return json.dumps(evidence, ensure_ascii=False, indent=2, default=str)


# 创建根据问题、路由和证据生成最终回答的工作流节点。
def compose_answer(model: Any) -> Callable[[AgentWorkflowState], dict[str, Any]]:
    def node(state: AgentWorkflowState) -> dict[str, Any]:
        intent = state.get("intent", "general")
        approval = state.get("approval")
        workflow_rules = f"""
你正在执行显式 LangGraph 工作流。当前意图：{intent}。
路由理由：{state.get('route_reason', '')}。
分析节点约束：{state.get('analysis_guidance', '根据证据回答；证据不足时说明缺口。')}。
下面的 JSON 是已完成的、受当前用户权限约束的工具证据。只能依据这些证据下结论；若工具返回失败或证据不足，明确说明。知识库证据包含 source_id、document、section 和页码；使用知识结论时在对应句末标注 `[来源：document > section]`，PDF 同时标注页码。不得用算子说明虚构实验指标。

输出格式：先给结论，再列出关键证据和下一步。诊断时区分“确定证据”和“可能原因”。规划时一次只推荐改变一个核心变量。不要复述原始 JSON。
"""
        if approval:
            workflow_rules += "\n审批状态：" + json.dumps(approval, ensure_ascii=False)
            workflow_rules += "\n严禁声称操作已执行；只有审批 API 能执行草案。"
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": workflow_rules},
            {"role": "system", "content": "工具证据：\n" + _evidence_prompt(state.get("evidence", []))},
            *state.get("messages", [])[-12:],
        ]
        response = model.invoke(prompt_messages)
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "\n".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        return {"answer": str(content).strip() or "Agent 没有返回结果。"}

    return node


# 根据已识别意图选择下一条证据收集分支。
def route_after_intent(state: AgentWorkflowState) -> WorkflowNode:
    intent = state.get("intent", "general")
    return {
        "diagnose": "collect_diagnostic_evidence",
        "plan": "collect_planning_evidence",
        "compare": "collect_comparison_evidence",
        "observe": "collect_observation_evidence",
        "knowledge": "collect_knowledge_evidence",
        "action": "prepare_approval",
        "general": "compose_answer",
    }[intent]


def build_experiment_workflow(model: Any):
    """Compile the explicit graph; callers may cache the compiled workflow."""
    graph = StateGraph(AgentWorkflowState)
    graph.add_node("route_intent", route_intent_with_model(model))
    graph.add_node("collect_diagnostic_evidence", collect_diagnostic_evidence)
    graph.add_node("collect_planning_evidence", collect_planning_evidence)
    graph.add_node("diagnose_evidence", diagnose_evidence)
    graph.add_node("plan_experiment", plan_experiment)
    graph.add_node("collect_comparison_evidence", collect_comparison_evidence)
    graph.add_node("collect_observation_evidence", collect_observation_evidence)
    graph.add_node("collect_knowledge_evidence", collect_knowledge_evidence)
    graph.add_node("prepare_approval", prepare_approval)
    graph.add_node("compose_answer", compose_answer(model))
    graph.add_edge(START, "route_intent")
    graph.add_conditional_edges("route_intent", route_after_intent)
    graph.add_edge("collect_diagnostic_evidence", "diagnose_evidence")
    graph.add_edge("diagnose_evidence", "compose_answer")
    graph.add_edge("collect_planning_evidence", "plan_experiment")
    graph.add_edge("plan_experiment", "compose_answer")
    for node in (
        "collect_comparison_evidence",
        "collect_observation_evidence",
        "collect_knowledge_evidence",
        "prepare_approval",
    ):
        graph.add_edge(node, "compose_answer")
    graph.add_edge("compose_answer", END)
    return graph.compile()
