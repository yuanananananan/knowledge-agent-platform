"""文件用途：运行 Agent Golden Case 评测，计算答案、意图和工具调用指标并持久化结果。"""

import math
import json
import os
import statistics
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from app import db
from app.utils.experiment_events import add_experiment_event
from app.agent.service import run_agent
from app.models.agent_runtime import (
    AgentEvalResult,
    AgentEvalRun,
    AgentGoldenCase,
    AgentRun,
    AgentTraceEvent,
)
from app.models.test_record import TestRecord
from app.models.train_job import TrainJob


VERSIONED_DATASET_FILE = Path(__file__).resolve().parents[2] / "evals" / "golden_dataset.json"


def versioned_golden_case_ids(dataset_file: Path = VERSIONED_DATASET_FILE) -> list[str]:
    """Return the canonical case IDs in their version-controlled order."""
    cases = json.loads(dataset_file.read_text(encoding="utf-8"))
    return [str(item["case_id"]) for item in cases]


# 将版本控制的数据集同步到 Golden Case 数据表。
def seed_golden_cases(dataset_file: Path) -> int:
    cases = json.loads(dataset_file.read_text(encoding="utf-8"))
    for item in cases:
        case = AgentGoldenCase.query.filter_by(case_id=item["case_id"]).first()
        if not case:
            case = AgentGoldenCase(case_id=item["case_id"])
            db.session.add(case)
        case.name = item["name"]
        case.category = item["category"]
        case.prompt = item["prompt"]
        case.required_keywords = item.get("required_keywords", [])
        case.forbidden_keywords = item.get("forbidden_keywords", [])
        case.expected_intent = item.get("expected_intent")
        case.expected_tools = item.get("expected_tools", [])
        case.enabled = item.get("enabled", True)
    db.session.commit()
    return len(cases)


# 根据必含词和禁用词计算答案得分及失败原因。
def score_answer(answer: str, required: list[str], forbidden: list[str]):
    normalized = answer.casefold()
    required_hits = [item for item in required if item.casefold() in normalized]
    forbidden_hits = [item for item in forbidden if item.casefold() in normalized]
    checks = len(required) + len(forbidden)
    passed_checks = len(required_hits) + len(forbidden) - len(forbidden_hits)
    score = passed_checks / checks if checks else 1.0
    return score, {
        "required_hits": required_hits,
        "missing_required": [item for item in required if item not in required_hits],
        "forbidden_hits": forbidden_hits,
    }


def score_tool_selection(expected: list[str], actual: list[str]):
    """Return Jaccard accuracy so missing and unnecessary calls both count."""
    expected_set = set(expected)
    actual_set = set(actual)
    union = expected_set | actual_set
    score = len(expected_set & actual_set) / len(union) if union else 1.0
    return score, {
        "expected_tools": sorted(expected_set),
        "actual_tools": sorted(actual_set),
        "missing_tools": sorted(expected_set - actual_set),
        "unexpected_tools": sorted(actual_set - expected_set),
    }


# 按序读取指定用户和会话最近一次运行的轨迹事件。
def _run_trace(user_id: int, conversation_id: str):
    run = AgentRun.query.filter_by(
        user_id=int(user_id), conversation_id=conversation_id
    ).order_by(AgentRun.id.desc()).first()
    if not run:
        return None, [], None
    events = AgentTraceEvent.query.filter_by(
        user_id=int(user_id), run_id=run.run_id
    ).order_by(AgentTraceEvent.sequence.asc()).all()
    tools = [
        str(event.payload.get("name"))
        for event in events
        if event.event_type == "tool.completed" and event.payload.get("name")
    ]
    intent = next((
        str(event.payload.get("intent"))
        for event in events
        if event.event_type == "workflow.node.completed" and event.payload.get("intent")
    ), None)
    return run, tools, intent


# 计算整数延迟序列的 P95。
def _p95(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def resolve_case_prompt(prompt: str, user_id: int) -> str:
    """Resolve reusable Golden Dataset placeholders from the current user's data."""
    trains = TrainJob.query.filter_by(
        user_id=int(user_id), isDelete=False
    ).filter(TrainJob.job_id.isnot(None)).order_by(TrainJob.id.desc()).all()
    validations = TestRecord.query.filter_by(
        user_id=int(user_id), isDelete=False
    ).order_by(TestRecord.id.desc()).all()

    def train_with_status(status: str, fallback: str) -> str:
        return next((str(job.job_id) for job in trains if job.status == status), fallback)

    train_ids = [str(job.job_id) for job in trains]
    val_ids = [str(job.val_job_id) for job in validations if job.val_job_id]
    values = {
        "<TRAIN_ANY>": train_ids[0] if train_ids else "train-eval-missing",
        "<TRAIN_RUNNING>": train_with_status("running", "train-eval-running-missing"),
        "<TRAIN_COMPLETED>": train_with_status("completed", "train-eval-completed-missing"),
        "<TRAIN_FAILED>": train_with_status("failed", "train-eval-failed-missing"),
        "<TRAIN_A>": train_ids[0] if train_ids else "train-eval-a-missing",
        "<TRAIN_B>": train_ids[1] if len(train_ids) > 1 else "train-eval-b-missing",
        "<VAL_A>": val_ids[0] if val_ids else "val-eval-a-missing",
        "<VAL_B>": val_ids[1] if len(val_ids) > 1 else "val-eval-b-missing",
    }
    resolved = prompt
    for placeholder, value in values.items():
        resolved = resolved.replace(placeholder, value)
    return resolved


# 按可选 ID 筛选启用的正式 Golden Case。
def _selected_cases(case_ids: list[str] | None = None):
    query = AgentGoldenCase.query.filter_by(enabled=True)
    if case_ids:
        query = query.filter(AgentGoldenCase.case_id.in_(case_ids))
    else:
        # A default Golden Eval is the reproducible, version-controlled suite.
        # Custom and legacy rows remain available through explicit case_ids.
        query = query.filter(AgentGoldenCase.case_id.in_(versioned_golden_case_ids()))
    cases = query.order_by(AgentGoldenCase.case_id.asc()).all()
    if not cases:
        raise ValueError("没有可运行的 Golden Dataset 样例")
    return cases


def create_golden_evaluation(user_id: int, case_ids: list[str] | None = None):
    """Persist an evaluation before handing execution to a background worker."""
    cases = _selected_cases(case_ids)

    evaluation = AgentEvalRun(
        eval_run_id=f"eval_{uuid.uuid4().hex}",
        user_id=int(user_id),
        total=len(cases),
        metrics={"execution": "celery", "completed_cases": 0, "progress": 0.0},
    )
    db.session.add(evaluation)
    db.session.commit()
    return evaluation


# 汇总通过数、平均得分和延迟指标并更新评测记录。
def _update_evaluation_summary(evaluation: AgentEvalRun, *, completed: bool = False, error: str | None = None):
    results = AgentEvalResult.query.filter_by(eval_run_id=evaluation.eval_run_id).all()
    completed_count = len(results)
    passed = sum(1 for result in results if result.passed)
    failures = sum(1 for result in results if (result.details or {}).get("error"))
    latencies = [int(result.latency_ms or 0) for result in results]
    tool_scores: list[float] = []
    intent_scores: list[float] = []
    for result in results:
        details = result.details or {}
        tools = details.get("tools") or {}
        if tools:
            tool_scores.append(score_tool_selection(
                tools.get("expected_tools") or [], tools.get("actual_tools") or []
            )[0])
        intent = details.get("intent") or {}
        if intent:
            intent_scores.append(float(bool(intent.get("matched"))))

    evaluation.passed = passed
    evaluation.score = passed / completed_count if completed_count else 0.0
    evaluation.metrics = {
        "execution": "celery",
        "completed_cases": completed_count,
        "progress": completed_count / evaluation.total if evaluation.total else 0.0,
        "task_success_rate": passed / completed_count if completed_count else 0.0,
        "tool_accuracy": statistics.mean(tool_scores) if tool_scores else 0.0,
        "intent_accuracy": statistics.mean(intent_scores) if intent_scores else 0.0,
        "average_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": _p95(latencies),
        "failure_rate": failures / completed_count if completed_count else 0.0,
        "failed_cases": failures,
        **({"error": error} if error else {}),
    }
    if completed:
        evaluation.status = "completed"
        evaluation.completed_at = datetime.utcnow()
        add_experiment_event(
            evaluation.eval_run_id,
            "agent_evaluation",
            "experiment.agent_evaluation.completed",
            {"score": evaluation.score, "passed": evaluation.passed, "total": evaluation.total},
        )
    elif error:
        evaluation.status = "failed"
        evaluation.completed_at = datetime.utcnow()
        add_experiment_event(
            evaluation.eval_run_id,
            "agent_evaluation",
            "experiment.agent_evaluation.failed",
            {"error": str(error)[:1000]},
        )
    else:
        evaluation.status = "running"
    db.session.commit()


def execute_golden_evaluation(eval_run_id: str, user_id: int, case_ids: list[str] | None = None):
    """Run or resume one durable Golden Eval from its missing cases."""
    evaluation = AgentEvalRun.query.filter_by(
        eval_run_id=eval_run_id, user_id=int(user_id)
    ).first()
    if not evaluation:
        raise ValueError("评测记录不存在")
    if evaluation.status == "completed":
        return evaluation
    cases = _selected_cases(case_ids)
    evaluation.total = len(cases)
    evaluation.status = "running"
    evaluation.completed_at = None
    add_experiment_event(
        evaluation.eval_run_id,
        "agent_evaluation",
        "experiment.agent_evaluation.started",
        {"total": evaluation.total},
    )
    db.session.commit()

    existing_case_ids = {
        row[0] for row in db.session.query(AgentEvalResult.case_id).filter_by(
            eval_run_id=evaluation.eval_run_id
        ).all()
    }

    for case in cases:
        if case.case_id in existing_case_ids:
            continue
        started = time.perf_counter()
        answer = ""
        conversation_id = f"eval_{uuid.uuid4().hex[:24]}"
        try:
            resolved_prompt = resolve_case_prompt(case.prompt, user_id)
            answer, conversation_id = run_agent(
                resolved_prompt,
                user_id,
                conversation_id=conversation_id,
            )
            answer_score, answer_details = score_answer(
                answer, case.required_keywords or [], case.forbidden_keywords or []
            )
            run, actual_tools, actual_intent = _run_trace(user_id, conversation_id)
            tool_score, tool_details = score_tool_selection(
                case.expected_tools or [], actual_tools
            )
            expected_intent = case.expected_intent or None
            intent_score = float(not expected_intent or actual_intent == expected_intent)
            score = statistics.mean([answer_score, tool_score, intent_score])
            case_passed = answer_score == 1.0 and tool_score == 1.0 and intent_score == 1.0
            details = {
                "answer": answer_details,
                "tools": tool_details,
                "intent": {
                    "expected": expected_intent,
                    "actual": actual_intent,
                    "matched": bool(intent_score),
                },
                "run_id": run.run_id if run else None,
                "resolved_prompt": resolved_prompt,
            }
        except Exception as exc:
            # A model/tool/database error in one case must not leave the
            # SQLAlchemy session unusable for all remaining cases.
            db.session.rollback()
            score, case_passed = 0.0, False
            details = {"error": type(exc).__name__, "message": str(exc)}
        latency_ms = round((time.perf_counter() - started) * 1000)
        db.session.add(AgentEvalResult(
            eval_run_id=evaluation.eval_run_id,
            case_id=case.case_id,
            passed=case_passed,
            score=score,
            answer=answer,
            details=details,
            latency_ms=latency_ms,
        ))
        db.session.commit()
        _update_evaluation_summary(evaluation)

    _update_evaluation_summary(evaluation, completed=True)
    return evaluation


# 将无法继续的评测任务收敛为 failed 并保存错误摘要。
def fail_golden_evaluation(eval_run_id: str, user_id: int, error: str):
    evaluation = AgentEvalRun.query.filter_by(
        eval_run_id=eval_run_id, user_id=int(user_id)
    ).first()
    if evaluation and evaluation.status != "completed":
        _update_evaluation_summary(evaluation, error=error)
    return evaluation


def reconcile_stale_evaluations(user_id: int):
    """Mark abandoned synchronous evaluations as interrupted instead of 0/total forever."""
    try:
        stale_seconds = max(300, int(os.getenv("AGENT_EVAL_STALE_SECONDS", "3600")))
    except ValueError:
        stale_seconds = 3600
    cutoff = datetime.utcnow() - timedelta(seconds=stale_seconds)
    runs = AgentEvalRun.query.filter_by(user_id=int(user_id), status="running").all()
    for evaluation in runs:
        if (evaluation.metrics or {}).get("execution") != "celery":
            _update_evaluation_summary(
                evaluation,
                error="旧版同步评测已随 HTTP 请求中断，请重新运行",
            )
            continue
        last_result = AgentEvalResult.query.filter_by(
            eval_run_id=evaluation.eval_run_id
        ).order_by(AgentEvalResult.created_at.desc()).first()
        last_activity = last_result.created_at if last_result else evaluation.created_at
        if last_activity and last_activity < cutoff:
            _update_evaluation_summary(
                evaluation,
                error="评测执行进程已中断，请重新运行",
            )


def run_golden_evaluation(user_id: int, case_ids: list[str] | None = None):
    """Synchronous compatibility wrapper used by the CLI and tests."""
    evaluation = create_golden_evaluation(user_id, case_ids)
    return execute_golden_evaluation(evaluation.eval_run_id, user_id, case_ids)
