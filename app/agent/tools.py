"""文件用途：注册 Agent 可调用的只读查询、RAG、实验分析和动作草案工具。"""

import json
import math
import re
import statistics
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any, Literal

from langchain.tools import tool

from app import db, session_scope
from app.agent.actions import (
    ActionValidationError,
    create_action_proposal,
    describe_action_impact,
    get_action_proposal,
    validate_action_payload,
    verify_action_result,
)
from app.agent.context import get_agent_conversation_id, get_agent_user_id
from app.control_plane import client as control_plane
from app.models.backbone import Backbone
from app.models.classifier import Classifier
from app.models.dataset import Dataset
from app.models.feature_extractor import FeatureExtractor
from app.models.nas_search_job import NasSearchJob
from app.models.test_record import TestRecord
from app.models.train_job import TrainJob
from app.rag import get_knowledge_store
from app.utils.log_streamer import LOG_DIR


# 将工具结果编码为稳定的中文 JSON 字符串。
def _to_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=str)


# 构造统一的成功工具响应。
def _success(data: Any, **metadata: Any) -> str:
    return _to_json({"ok": True, "data": data, **metadata})


# 构造包含错误码和消息的统一失败响应。
def _error(message: str, code: str = "not_found") -> str:
    return _to_json({"ok": False, "error": {"code": code, "message": message}})


def _isolate_agent_tool(tool_instance):
    """Give every LangGraph tool call an independent SQLAlchemy session."""
    original = tool_instance.func

    @wraps(original)
    def isolated(*args, **kwargs):
        with session_scope():
            try:
                return original(*args, **kwargs)
            except ActionValidationError as exc:
                db.session.rollback()
                return _error(str(exc), "invalid_action")
            except Exception as exc:
                db.session.rollback()
                return _error(
                    f"{tool_instance.name} 调用失败：{type(exc).__name__}",
                    "tool_failure",
                )

    tool_instance.func = isolated
    return tool_instance


# 将工具查询条数限制在安全范围内。
def _bounded_limit(limit: int, maximum: int = 50) -> int:
    return max(1, min(int(limit), maximum))


@tool
def search_knowledge_base(
    query: str,
    modality: str = "",
    operator: str = "",
    top_k: int = 6,
) -> str:
    """检索算子说明、适用限制、版本记录和实验设计资料，返回可引用来源。"""
    store = get_knowledge_store()
    results = store.search(
        query=query,
        modality=modality,
        operator=operator,
        top_k=_bounded_limit(top_k, 10),
        include_superseded=any(word in query.casefold() for word in ("旧版", "旧版本", "冲突", "version", "v1")),
    )
    assessment = store.assess(query, results)
    return _success([
        {
            "source_id": item.chunk_id,
            "parent_id": item.parent_id,
            "document": item.document_path,
            "section": " > ".join(item.section_path),
            "page_start": item.page_start,
            "page_end": item.page_end,
            "content": item.content,
            "matched_content": item.matched_content,
            "score": item.score,
            "entities": item.entity_ids,
            "graph_paths": item.graph_paths,
            "claims": item.claim_ids,
            "authority": item.authority,
            "metadata": {
                key: value for key, value in item.metadata.items()
                if key in {"document_type", "modality", "operator", "version", "status"}
            },
        }
        for item in results
    ], retrieval="normalized_decomposed_bm25_vector_rrf_rerank_evidence_completion_parent", query=query,
       evidence_assessment=assessment.to_dict())


# 查询当前用户拥有的训练任务。
def _owned_train_job(job_id: str) -> TrainJob | None:
    return TrainJob.query.filter_by(
        job_id=job_id,
        user_id=get_agent_user_id(),
        isDelete=False,
    ).first()


# 从验证记录中提取可比较的指标字典。
def _metrics(record: TestRecord) -> dict[str, Any]:
    result = record.result if isinstance(record.result, dict) else {}
    raw = result.get("Metrics") or result.get("metrics") or {}
    return {str(key).lower(): value for key, value in raw.items()}


# 将日志路径约束在允许目录中，拒绝路径穿越。
def _safe_log_path(job_id: str) -> Path | None:
    user_id = get_agent_user_id()
    exists = (
        TrainJob.query.filter_by(job_id=job_id, user_id=user_id, isDelete=False).first()
        or TestRecord.query.filter_by(
            val_job_id=job_id,
            user_id=user_id,
            isDelete=False,
        ).first()
        or NasSearchJob.query.filter_by(
            job_id=job_id,
            user_id=user_id,
            isDelete=False,
        ).first()
    )
    if not exists or Path(job_id).name != job_id:
        return None
    root = LOG_DIR.resolve()
    candidate = (root / f"{job_id}.log").resolve()
    if root not in candidate.parents:
        return None
    return candidate


# 将训练任务转换为适合模型阅读的精简摘要。
def _job_summary(job: TrainJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "description": job.description,
        "dataset_id": job.dataset_id,
        "backbone_id": job.backbone_id,
        "classifier_id": job.classifier_id,
        "status": job.status,
        "progress": job.progress,
        "result": job.result,
        "created_at": job.createTime.isoformat() if job.createTime else None,
        "updated_at": job.updateTime.isoformat() if job.updateTime else None,
    }


@tool
def observe_datasets(modality: str = "", limit: int = 20) -> str:
    """查询可用数据集及其模态和标签，不返回服务器文件路径。"""
    query = Dataset.query.filter_by(isDelete=False)
    if modality:
        query = query.filter(Dataset.modality.ilike(modality))
    datasets = query.order_by(Dataset.id.desc()).limit(_bounded_limit(limit)).all()
    return _success([
        {
            "id": item.id,
            "name": item.name,
            "modality": item.modality,
            "labels": item.labels,
        }
        for item in datasets
    ])


@tool
def inspect_dataset_quality(dataset_id: int) -> str:
    """检查数据集目录、类别一致性、样本数量和常见文件问题。"""
    dataset = Dataset.query.filter_by(id=dataset_id, isDelete=False).first()
    if not dataset:
        return _error("数据集不存在或已删除")
    root = Path(dataset.path)
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[2] / root
    report: dict[str, Any] = {"dataset_id": dataset.id, "name": dataset.name, "path": str(dataset.path)}
    if not root.is_dir():
        report.update({"healthy": False, "issues": ["数据集目录不存在"], "splits": {}})
        return _success(report)
    splits = {}
    issues = []
    for split in ("train", "test"):
        split_root = root / split
        classes = {}
        if split_root.is_dir():
            for class_dir in sorted(item for item in split_root.iterdir() if item.is_dir()):
                classes[class_dir.name] = sum(1 for item in class_dir.rglob("*") if item.is_file())
        else:
            issues.append(f"缺少 {split} 目录")
        splits[split] = {"classes": classes, "total": sum(classes.values())}
    if splits["train"]["classes"] and splits["test"]["classes"] and set(splits["train"]["classes"]) != set(splits["test"]["classes"]):
        issues.append("train/test 类别目录不一致")
    for split, info in splits.items():
        if info["total"] == 0:
            issues.append(f"{split} 分区没有样本")
    report.update({"healthy": not issues, "issues": issues, "splits": splits})
    return _success(report)


@tool
def observe_model_resources(modality: str = "") -> str:
    """查询可用于实验规划的骨干网络、分类器和特征算子。"""
    backbones_query = Backbone.query.filter_by(isDelete=False)
    features_query = FeatureExtractor.query.filter_by(isDelete=False)
    if modality:
        backbones_query = backbones_query.filter(Backbone.modality.ilike(modality))
        features_query = features_query.filter(FeatureExtractor.modality.ilike(modality))
    backbones = backbones_query.order_by(Backbone.id).all()
    classifiers = Classifier.query.filter_by(isDelete=False).order_by(Classifier.id).all()
    features = features_query.order_by(FeatureExtractor.id).all()
    return _success({
        "backbones": [
            {
                "id": item.id,
                "name": item.name,
                "modality": item.modality,
                "type": item.backbone_type,
                "description": item.description,
            }
            for item in backbones
        ],
        "classifiers": [
            {"id": item.id, "name": item.name, "description": item.description}
            for item in classifiers
        ],
        "feature_extractors": [
            {
                "id": item.id,
                "name": item.name,
                "modality": item.modality,
                "description": item.description,
            }
            for item in features
        ],
    })


@tool
def list_recent_train_jobs(limit: int = 5, status: str = "") -> str:
    """查询当前登录用户最近的训练任务，可按状态过滤。"""
    query = TrainJob.query.filter_by(user_id=get_agent_user_id(), isDelete=False)
    if status:
        query = query.filter_by(status=status)
    jobs = query.order_by(TrainJob.id.desc()).limit(_bounded_limit(limit)).all()
    return _success([_job_summary(job) for job in jobs])


@tool
def get_train_job_detail(job_id: str) -> str:
    """根据 job_id 查询当前用户的训练配置、状态和结果。"""
    job = _owned_train_job(job_id)
    if not job:
        return _error("训练任务不存在或不属于当前用户")
    data = _job_summary(job)
    data.update({
        "nodes": job.nodes,
        "config": job.config,
        "feature_ids": job.feature_ids,
        "adaptive": job.adaptive,
    })
    return _success(data)


@tool
def get_experiment_lineage(job_id: str) -> str:
    """查询训练任务及其验证任务，形成简化实验血缘。"""
    job = _owned_train_job(job_id)
    if not job:
        return _error("训练任务不存在或不属于当前用户")
    validations = TestRecord.query.filter_by(
        train_job_id=job_id,
        user_id=get_agent_user_id(),
        isDelete=False,
    ).order_by(TestRecord.id).all()
    return _success({
        "training": _job_summary(job),
        "validations": [
            {
                "val_job_id": item.val_job_id,
                "status": item.status,
                "metrics": _metrics(item),
                "created_at": item.createTime.isoformat() if item.createTime else None,
            }
            for item in validations
        ],
    })


@tool
def list_recent_nas_jobs(limit: int = 5, status: str = "") -> str:
    """查询当前登录用户最近的 NAS 搜索任务，可按状态过滤。"""
    query = NasSearchJob.query.filter_by(user_id=get_agent_user_id(), isDelete=False)
    if status:
        query = query.filter_by(status=status)
    jobs = query.order_by(NasSearchJob.id.desc()).limit(_bounded_limit(limit)).all()
    return _success([
        {
            "job_id": job.job_id,
            "description": job.description,
            "dataset_id": job.dataset_id,
            "modality": job.modality,
            "config": job.config,
            "status": job.status,
            "progress": job.progress,
            "genotype": job.genotype,
            "result": job.result,
            "backbone_id": job.backbone_id,
            "created_at": job.createTime.isoformat() if job.createTime else None,
        }
        for job in jobs
    ])


@tool
def read_job_log(job_id: str, max_lines: int = 150) -> str:
    """读取当前用户训练、验证或 NAS 任务的日志尾部。"""
    log_path = _safe_log_path(job_id)
    if not log_path:
        return _error("任务不存在、不属于当前用户或任务 ID 非法", "forbidden")
    if not log_path.exists():
        return _error("该任务尚未生成日志")
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    tail = lines[-_bounded_limit(max_lines, maximum=500):]
    return _success({"job_id": job_id, "line_count": len(tail), "lines": tail})


ERROR_PATTERNS = (
    ("out_of_memory", re.compile(r"out of memory|cuda.*oom", re.I)),
    ("missing_file", re.compile(r"no such file|file not found|找不到.*文件", re.I)),
    ("shape_mismatch", re.compile(r"shape.*mismatch|size mismatch|维度.*不匹配", re.I)),
    ("invalid_value", re.compile(r"valueerror|invalid value|nan|inf", re.I)),
    ("permission", re.compile(r"permission denied|unauthorized|forbidden", re.I)),
    ("dependency", re.compile(r"modulenotfounderror|importerror|no module named", re.I)),
    ("traceback", re.compile(r"traceback|exception|error", re.I)),
)


@tool
def analyze_job_log(job_id: str, max_lines: int = 300) -> str:
    """提取当前用户任务日志中的错误类别、关键证据行和确定性修复建议。"""
    log_path = _safe_log_path(job_id)
    if not log_path:
        return _error("任务不存在、不属于当前用户或任务 ID 非法", "forbidden")
    if not log_path.exists():
        return _error("该任务尚未生成日志")
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    inspected = lines[-_bounded_limit(max_lines, maximum=1000):]
    matches: list[dict[str, Any]] = []
    seen = set()
    for line_number, line in enumerate(inspected, start=max(1, len(lines) - len(inspected) + 1)):
        for category, pattern in ERROR_PATTERNS:
            if pattern.search(line):
                key = (category, line.strip())
                if key not in seen:
                    matches.append({
                        "line_number": line_number,
                        "category": category,
                        "evidence": line.strip()[:1000],
                    })
                    seen.add(key)
                break
    suggestions = {
        "out_of_memory": "降低 batch_size、输入尺寸或模型规模，并检查显存占用。",
        "missing_file": "核验数据集、Checkpoint 和配置中的路径是否存在。",
        "shape_mismatch": "比较数据维度、类别数、骨干输出和分类器输入配置。",
        "invalid_value": "检查学习率、归一化、损失函数和输入数据中的 NaN/Inf。",
        "permission": "检查文件权限、鉴权状态和任务归属。",
        "dependency": "补齐并锁定缺失依赖，重新构建运行环境。",
        "traceback": "结合完整 Traceback 定位首个业务代码异常。",
    }
    categories = list(dict.fromkeys(item["category"] for item in matches))
    return _success({
        "job_id": job_id,
        "inspected_lines": len(inspected),
        "categories": categories,
        "evidence": matches[-20:],
        "suggestions": [suggestions[item] for item in categories],
        "conclusion": "找到明确错误证据" if matches else "未找到明确错误模式，需要更多日志或配置证据",
    })


# 将嵌套训练配置展开为点分路径，便于比较变更。
def _flatten_config(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_config(item, path))
        return flattened
    return {prefix: value}


@tool
def diff_train_configs(job_id_1: str, job_id_2: str) -> str:
    """比较当前用户两个训练任务的嵌套配置差异。"""
    job1 = _owned_train_job(job_id_1)
    job2 = _owned_train_job(job_id_2)
    if not job1 or not job2:
        return _error("至少一个训练任务不存在或不属于当前用户")
    left = _flatten_config(job1.config or {})
    right = _flatten_config(job2.config or {})
    changes = []
    for key in sorted(set(left) | set(right)):
        if left.get(key) != right.get(key):
            changes.append({"field": key, "job_1": left.get(key), "job_2": right.get(key)})
    return _success({
        "job_id_1": job_id_1,
        "job_id_2": job_id_2,
        "change_count": len(changes),
        "changes": changes,
    })


@tool
def compare_val_jobs(job_id_1: str, job_id_2: str) -> str:
    """对比当前用户两个验证任务的指标，指标名称不区分大小写。"""
    user_id = get_agent_user_id()
    job1 = TestRecord.query.filter_by(
        val_job_id=job_id_1,
        user_id=user_id,
        isDelete=False,
    ).first()
    job2 = TestRecord.query.filter_by(
        val_job_id=job_id_2,
        user_id=user_id,
        isDelete=False,
    ).first()
    if not job1 or not job2:
        return _error("至少一个验证任务不存在或不属于当前用户")
    metrics1 = _metrics(job1)
    metrics2 = _metrics(job2)
    comparison = {}
    for key in sorted(set(metrics1) | set(metrics2)):
        value1 = metrics1.get(key)
        value2 = metrics2.get(key)
        comparison[key] = {
            "job_1": value1,
            "job_2": value2,
            "diff": value2 - value1
            if isinstance(value1, (int, float)) and isinstance(value2, (int, float))
            else None,
        }
    return _success({
        "job_1": {"val_job_id": job1.val_job_id, "metrics": metrics1},
        "job_2": {"val_job_id": job2.val_job_id, "metrics": metrics2},
        "comparison": comparison,
    })


@tool
def rank_validation_jobs(metric: str = "oa", limit: int = 10) -> str:
    """按指定指标对当前用户已完成的验证任务排名。Loss 指标按升序，其他指标按降序。"""
    records = TestRecord.query.filter_by(
        user_id=get_agent_user_id(),
        status="completed",
        isDelete=False,
    ).order_by(TestRecord.id.desc()).limit(200).all()
    metric_key = metric.lower()
    ranked = []
    for record in records:
        value = _metrics(record).get(metric_key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            ranked.append({
                "val_job_id": record.val_job_id,
                "train_job_id": record.train_job_id,
                "value": value,
            })
    ranked.sort(key=lambda item: item["value"], reverse=metric_key != "loss")
    return _success(ranked[:_bounded_limit(limit)], metric=metric_key)


@tool
def analyze_metric_significance(
    baseline_job_ids: list[str],
    candidate_job_ids: list[str],
    metric: str = "oa",
) -> str:
    """使用多次验证结果近似评估候选实验相对基线的统计显著性；每组至少需要两个任务。"""
    if len(baseline_job_ids) < 2 or len(candidate_job_ids) < 2:
        return _error("显著性分析要求基线组和候选组各至少两个验证任务", "insufficient_data")
    user_id = get_agent_user_id()
    metric_key = metric.lower()

    def values(job_ids: list[str]) -> list[float]:
        records = TestRecord.query.filter(
            TestRecord.val_job_id.in_(job_ids),
            TestRecord.user_id == user_id,
            TestRecord.isDelete.is_(False),
        ).all()
        by_id = {item.val_job_id: item for item in records}
        output = []
        for job_id in job_ids:
            record = by_id.get(job_id)
            value = _metrics(record).get(metric_key) if record else None
            if isinstance(value, (int, float)) and math.isfinite(value):
                output.append(float(value))
        return output

    baseline = values(baseline_job_ids)
    candidate = values(candidate_job_ids)
    if len(baseline) < 2 or len(candidate) < 2:
        return _error("部分任务不存在、越权或缺少数值指标", "insufficient_data")
    mean_a = statistics.mean(baseline)
    mean_b = statistics.mean(candidate)
    var_a = statistics.variance(baseline)
    var_b = statistics.variance(candidate)
    standard_error = math.sqrt(var_a / len(baseline) + var_b / len(candidate))
    if standard_error == 0:
        statistic = math.inf if mean_a != mean_b else 0.0
    else:
        statistic = (mean_b - mean_a) / standard_error
    approximate_p = 0.0 if math.isinf(statistic) else math.erfc(abs(statistic) / math.sqrt(2))
    return _success({
        "metric": metric_key,
        "baseline": {"count": len(baseline), "mean": mean_a, "values": baseline},
        "candidate": {"count": len(candidate), "mean": mean_b, "values": candidate},
        "mean_diff": mean_b - mean_a,
        "welch_statistic": statistic,
        "approximate_two_sided_p": approximate_p,
        "significant_at_0_05": approximate_p < 0.05,
        "caveat": "p 值使用正态近似；小样本结论应结合更多重复实验或采用严格的 Welch t 分布计算。",
    })


@tool
def estimate_experiment_resources(
    dataset_id: int,
    backbone_id: int,
    limit: int = 20,
) -> str:
    """根据当前用户相同数据集和骨干网络的历史任务估算训练耗时。"""
    jobs = TrainJob.query.filter_by(
        user_id=get_agent_user_id(),
        dataset_id=dataset_id,
        backbone_id=backbone_id,
        status="completed",
        isDelete=False,
    ).order_by(TrainJob.id.desc()).limit(_bounded_limit(limit)).all()
    durations = [
        (job.updateTime - job.createTime).total_seconds()
        for job in jobs
        if job.createTime and job.updateTime and job.updateTime >= job.createTime
    ]
    if not durations:
        return _success({
            "sample_count": 0,
            "estimated_duration_seconds": None,
            "confidence": "low",
            "caveat": "没有匹配的历史任务，无法估算 GPU 显存或训练耗时。",
        })
    return _success({
        "sample_count": len(durations),
        "estimated_duration_seconds": statistics.median(durations),
        "mean_duration_seconds": statistics.mean(durations),
        "min_duration_seconds": min(durations),
        "max_duration_seconds": max(durations),
        "confidence": "medium" if len(durations) >= 5 else "low",
        "caveat": "估算包含排队时间；当前任务记录没有 GPU 型号和显存字段。",
    })


# 兼容不同配置结构并返回训练参数分区。
def _training_section(config: dict[str, Any]) -> dict[str, Any]:
    section = config.setdefault("training", {})
    return section if isinstance(section, dict) else {}


@tool
def recommend_next_experiment(
    source_job_id: str,
    objective: Literal["improve_accuracy", "reduce_overfit", "recover_failure"] = "improve_accuracy",
) -> str:
    """基于已有训练配置生成下一轮实验方案，只返回配置草案，不创建任务。"""
    source = _owned_train_job(source_job_id)
    if not source:
        return _error("训练任务不存在或不属于当前用户")
    base = deepcopy(source.config or {})
    variants = []

    def add_variant(name: str, field: str, transform) -> None:
        config = deepcopy(base)
        training = _training_section(config)
        old_value = training.get(field)
        if not isinstance(old_value, (int, float)):
            return
        new_value = transform(old_value)
        training[field] = new_value
        variants.append({
            "name": name,
            "changed_field": f"training.{field}",
            "old_value": old_value,
            "new_value": new_value,
            "config": config,
        })

    if objective == "improve_accuracy":
        add_variant("降低学习率", "learning_rate", lambda value: value * 0.5)
        add_variant("延长训练", "epochs", lambda value: max(int(value * 1.25), int(value) + 1))
        add_variant("增强正则化", "weight_decay", lambda value: value * 2)
    elif objective == "reduce_overfit":
        add_variant("增强正则化", "weight_decay", lambda value: value * 2)
        add_variant("提前停止候选", "epochs", lambda value: max(1, int(value * 0.8)))
    else:
        add_variant("减小批量", "batch_size", lambda value: max(1, int(value) // 2))
        add_variant("降低学习率", "learning_rate", lambda value: value * 0.5)

    return _success({
        "source_job_id": source_job_id,
        "objective": objective,
        "variants": variants,
        "next_step": "选择一个变量进行单因素对照；确认后再创建 create_train_job 操作草案。",
    })


@tool
def draft_agent_action(
    action_type: Literal[
        "create_train_job",
        "stop_train_job",
        "retry_train_job",
        "clone_train_job",
        "start_validation",
    ],
    payload: dict[str, Any],
    reason: str,
) -> str:
    """创建有副作用操作的待审批草案；此工具不会执行操作。"""
    if control_plane.enabled():
        user_id = get_agent_user_id()
        clean_payload = validate_action_payload(user_id, action_type, payload)
        response = control_plane.create_action(user_id, {
            "conversation_id": get_agent_conversation_id(),
            "action_type": action_type,
            "payload": clean_payload,
            "reason": reason,
            "impact": describe_action_impact(action_type, clean_payload),
        })
        return _success(response.get("proposal") or {}, created=bool(response.get("created")))
    proposal, created = create_action_proposal(
        user_id=get_agent_user_id(),
        action_type=action_type,
        payload=payload,
        reason=reason,
        conversation_id=get_agent_conversation_id(),
    )
    return _success(proposal.to_dict(), created=created)


@tool
def get_agent_action(proposal_id: str) -> str:
    """查询当前用户某个 Agent 操作草案的审批和执行状态。"""
    if control_plane.enabled():
        response = control_plane.get_action(get_agent_user_id(), proposal_id)
        return _success({"proposal": response.get("proposal") or {}})
    proposal = get_action_proposal(get_agent_user_id(), proposal_id)
    return _success({
        "proposal": proposal.to_dict(),
        "verification": verify_action_result(proposal),
    })


ALL_AGENT_TOOLS = [_isolate_agent_tool(item) for item in [
    search_knowledge_base,
    observe_datasets,
    inspect_dataset_quality,
    observe_model_resources,
    list_recent_train_jobs,
    get_train_job_detail,
    get_experiment_lineage,
    list_recent_nas_jobs,
    read_job_log,
    analyze_job_log,
    diff_train_configs,
    compare_val_jobs,
    rank_validation_jobs,
    analyze_metric_significance,
    estimate_experiment_resources,
    recommend_next_experiment,
    draft_agent_action,
    get_agent_action,
]]
