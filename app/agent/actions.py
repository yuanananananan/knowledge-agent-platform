"""文件用途：定义 Agent 高风险动作提案的校验、幂等创建、审批执行和审计记录。"""

import hashlib
import json
import uuid
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from celery.result import AsyncResult

from app import db
from app.models.agent_action import AgentActionProposal
from app.models.backbone import Backbone
from app.models.classifier import Classifier
from app.models.dataset import Dataset
from app.models.feature_extractor import FeatureExtractor
from app.models.test_record import TestRecord
from app.models.train_job import TrainJob
from app.models.audit_log import AuditLog
from app.utils.celery_worker import celery_app
from app.utils.experiment_events import add_compute_dispatch, add_experiment_event


SUPPORTED_ACTIONS = {
    "create_train_job",
    "stop_train_job",
    "retry_train_job",
    "clone_train_job",
    "start_validation",
}

TERMINAL_PROPOSAL_STATUSES = {"succeeded", "failed", "rejected", "expired"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_STANDARD_BACKBONES = {
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "vgg8_bn",
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".npy"}


# 表示动作参数、资源归属或审批条件不满足。
class ActionValidationError(ValueError):
    pass


# 表示动作正在执行、已经终态或幂等状态冲突。
class ActionConflictError(RuntimeError):
    pass


# 写入一条动作相关审计日志。
def _audit(user_id, action, resource_id=None, detail=None, success=True):
    try:
        db.session.add(AuditLog(user_id=user_id, action=action, resource_type="agent_action", resource_id=str(resource_id) if resource_id else None, detail=detail, success=success))
        db.session.commit()
    except Exception:
        db.session.rollback()


# 检查动作 payload 是否包含全部必填字段。
def _require_fields(payload: dict[str, Any], fields: tuple[str, ...]) -> None:
    missing = [field for field in fields if payload.get(field) in (None, "")]
    if missing:
        raise ActionValidationError(f"缺少必要字段：{', '.join(missing)}")


# 读取当前用户拥有且未删除的训练任务。
def _owned_train_job(user_id: int, job_id: str) -> TrainJob:
    job = TrainJob.query.filter_by(
        job_id=job_id,
        user_id=user_id,
        isDelete=False,
    ).first()
    if not job:
        raise ActionValidationError("训练任务不存在或不属于当前用户")
    return job


# 递归合并嵌套配置，同时避免修改原对象。
def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _normalize_train_config(
    dataset: Dataset,
    backbone: Backbone,
    classifier: Classifier,
    config: dict[str, Any],
    feature_ids: list[Any],
) -> tuple[dict[str, Any], list[Any]]:
    """Bind Agent-provided config to trusted dataset and feature records."""
    modality = str(dataset.modality or "").upper()
    if modality not in {"SAR", "RD", "MON"}:
        raise ActionValidationError(f"不支持的数据集模态：{dataset.modality}")
    template_path = PROJECT_ROOT / "configs" / f"{modality.lower()}_config.yaml"
    try:
        default_config = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ActionValidationError(f"无法加载 {modality} 默认训练配置") from exc
    if not isinstance(default_config, dict):
        raise ActionValidationError(f"{modality} 默认训练配置格式错误")

    requested_model = config.get("model") if isinstance(config.get("model"), dict) else {}
    requested_gnn_config = (
        requested_model.get("GNN") if isinstance(requested_model.get("GNN"), dict) else {}
    )
    requested_gnn = requested_gnn_config.get("is_GNN")
    normalized = _deep_merge(default_config, config)
    data_config = normalized.setdefault("data", {})
    model_config = normalized.setdefault("model", {})
    if not isinstance(data_config, dict):
        raise ActionValidationError("config.data 必须是 JSON 对象")
    if not isinstance(model_config, dict):
        raise ActionValidationError("config.model 必须是 JSON 对象")

    dataset_path = Path(str(dataset.path or "").strip())
    if not str(dataset_path) or str(dataset_path) == ".":
        raise ActionValidationError("数据集未配置存储路径")
    resolved_root = dataset_path if dataset_path.is_absolute() else PROJECT_ROOT / dataset_path
    missing_splits = [name for name in ("train", "test") if not (resolved_root / name).is_dir()]
    if missing_splits:
        raise ActionValidationError(
            f"数据集目录缺少 {', '.join(missing_splits)}：{dataset.path}，请重新上传或修正数据集路径"
        )
    split_classes = {
        split: sorted(item.name for item in (resolved_root / split).iterdir() if item.is_dir())
        for split in ("train", "test")
    }
    if not split_classes["train"] or not split_classes["test"]:
        raise ActionValidationError("数据集的 train 和 test 分区都必须包含类别目录")
    if split_classes["train"] != split_classes["test"]:
        raise ActionValidationError("数据集 train 与 test 的类别目录不一致")
    for split in ("train", "test"):
        has_samples = any(
            item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
            for item in (resolved_root / split).rglob("*")
        )
        if not has_samples:
            raise ActionValidationError(f"数据集 {split} 分区没有可读取的样本")

    data_config["root"] = str(dataset_path)
    data_config["name"] = dataset.name
    data_config["mode"] = modality
    data_config["num_classes"] = len(split_classes["train"])

    try:
        data_config["batch_size"] = int(data_config["batch_size"])
        data_config["num_workers"] = int(data_config["num_workers"])
        data_config["train_ratio"] = float(data_config["train_ratio"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ActionValidationError("batch_size、num_workers 和 train_ratio 必须是有效数字") from exc
    if data_config["batch_size"] < 1:
        raise ActionValidationError("data.batch_size 必须大于 0")
    if data_config["num_workers"] < 0:
        raise ActionValidationError("data.num_workers 不能小于 0")
    if not 0 < data_config["train_ratio"] <= 1:
        raise ActionValidationError("data.train_ratio 必须大于 0 且不超过 1")
    img_size = data_config.get("img_size")
    if isinstance(img_size, str):
        try:
            img_size = json.loads(img_size)
        except json.JSONDecodeError as exc:
            raise ActionValidationError("data.img_size 必须是包含两个正整数的数组") from exc
    if not isinstance(img_size, (list, tuple)) or len(img_size) != 2:
        raise ActionValidationError("data.img_size 必须是包含两个正整数的数组")
    try:
        data_config["img_size"] = [int(value) for value in img_size]
    except (TypeError, ValueError) as exc:
        raise ActionValidationError("data.img_size 必须是包含两个正整数的数组") from exc
    if any(value < 1 for value in data_config["img_size"]):
        raise ActionValidationError("data.img_size 必须是包含两个正整数的数组")

    unique_feature_ids = list(dict.fromkeys(feature_ids))
    feature_rows = []
    if unique_feature_ids:
        feature_rows = FeatureExtractor.query.filter(
            FeatureExtractor.id.in_(unique_feature_ids),
            FeatureExtractor.isDelete.is_(False),
        ).all()
        if len(feature_rows) != len(unique_feature_ids):
            raise ActionValidationError("包含不存在的特征算子")
    rows_by_id = {str(item.id): item for item in feature_rows}
    feature_names = [rows_by_id[str(item_id)].name for item_id in unique_feature_ids]
    mismatched_features = [
        item.name
        for item in feature_rows
        if item.modality and str(item.modality).upper() != modality
    ]
    if mismatched_features:
        raise ActionValidationError(f"特征算子模态不匹配：{', '.join(mismatched_features)}")

    gnn_config = model_config.setdefault("GNN", {})
    attention_config = model_config.setdefault("Cross_attention", {})
    if not isinstance(gnn_config, dict):
        raise ActionValidationError("config.model.GNN 必须是 JSON 对象")
    if not isinstance(attention_config, dict):
        raise ActionValidationError("config.model.Cross_attention 必须是 JSON 对象")

    is_gnn = bool(feature_names) if requested_gnn is None else bool(requested_gnn)
    is_nas_backbone = backbone.backbone_type == "nas_search"
    if not is_nas_backbone and backbone.name not in SUPPORTED_STANDARD_BACKBONES:
        raise ActionValidationError(f"普通训练暂不支持骨干：{backbone.name}")
    if backbone.modality and str(backbone.modality).upper() != modality:
        raise ActionValidationError("骨干网络与数据集模态不匹配")
    if is_nas_backbone and is_gnn:
        raise ActionValidationError("NAS 搜索骨干不支持同时启用 GNN")
    if is_gnn and not feature_names:
        raise ActionValidationError("启用 GNN 时至少需要选择一个特征算子")
    if not is_gnn:
        unique_feature_ids = []
        feature_names = []
        attention_config["is_attention"] = False
    model_config["backbone"] = backbone.name
    model_config["classifier"] = classifier.name
    gnn_config["is_GNN"] = is_gnn
    normalized["feature_combinations"] = feature_names
    normalized["is_nas"] = is_nas_backbone
    training_config = normalized.get("training")
    if not isinstance(training_config, dict):
        raise ActionValidationError("config.training 必须是 JSON 对象")
    try:
        training_config["epochs"] = int(training_config["epochs"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ActionValidationError("training.epochs 必须是正整数") from exc
    if training_config["epochs"] < 1:
        raise ActionValidationError("training.epochs 必须是正整数")
    if training_config["epochs"] > 200:
        raise ActionValidationError("training.epochs 不能超过 200，高资源任务请拆分实验")
    training_config["checkpoint_dir"] = "model_ckp"
    optimizer_config = training_config.get("optimizer")
    if not isinstance(optimizer_config, dict):
        raise ActionValidationError("config.training.optimizer 必须是 JSON 对象")
    optimizer_type = str(optimizer_config.get("type", "Adam"))
    if optimizer_type.lower() not in {"adam", "sgd"}:
        raise ActionValidationError("training.optimizer.type 仅支持 Adam 或 SGD")
    optimizer_config["type"] = optimizer_type
    for key in ("lr", "weight_decay"):
        try:
            optimizer_config[key] = float(optimizer_config[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ActionValidationError(f"training.optimizer.{key} 必须是有效数字") from exc
    if optimizer_config["lr"] <= 0 or optimizer_config["weight_decay"] < 0:
        raise ActionValidationError("优化器学习率必须大于 0，权重衰减不能小于 0")
    if int(data_config.get("batch_size", 1)) > 512:
        raise ActionValidationError("batch_size 不能超过 512")
    return normalized, unique_feature_ids


# 按动作类型校验资源归属、参数范围并返回规范化 payload。
def validate_action_payload(
    user_id: int,
    action_type: str,
    payload: dict[str, Any] | None,
) -> dict[str, Any]:
    if action_type not in SUPPORTED_ACTIONS:
        raise ActionValidationError(f"不支持的操作类型：{action_type}")
    if not isinstance(payload, dict):
        raise ActionValidationError("payload 必须是 JSON 对象")

    clean = deepcopy(payload)
    clean.pop("user_id", None)

    if action_type == "create_train_job":
        _require_fields(clean, ("dataset_id", "backbone_id", "classifier_id", "config"))
        if not isinstance(clean["config"], dict):
            raise ActionValidationError("config 必须是 JSON 对象")
        dataset = Dataset.query.filter_by(id=clean["dataset_id"], isDelete=False).first()
        if not dataset:
            raise ActionValidationError("数据集不存在")
        backbone = Backbone.query.filter_by(id=clean["backbone_id"], isDelete=False).first()
        if not backbone:
            raise ActionValidationError("骨干网络不存在")
        classifier = Classifier.query.filter_by(id=clean["classifier_id"], isDelete=False).first()
        if not classifier:
            raise ActionValidationError("分类器不存在")

        feature_ids = clean.get("feature_ids") or []
        if not isinstance(feature_ids, list):
            raise ActionValidationError("feature_ids 必须是数组")
        clean["config"], clean["feature_ids"] = _normalize_train_config(
            dataset,
            backbone,
            classifier,
            clean["config"],
            feature_ids,
        )
        clean["nodes"] = clean.get("nodes") or []
        clean["adaptive"] = bool(clean["config"].get("is_nas", False))

    elif action_type in {"stop_train_job", "retry_train_job", "clone_train_job"}:
        _require_fields(clean, ("job_id",))
        job = _owned_train_job(user_id, str(clean["job_id"]))
        if action_type == "stop_train_job" and job.status not in {
            "pending",
            "running",
            "retry",
        }:
            raise ActionValidationError(f"状态为 {job.status} 的任务不能停止")
        if action_type == "retry_train_job" and job.status not in {
            "failed",
            "cancelled",
            "deleted",
        }:
            raise ActionValidationError(f"状态为 {job.status} 的任务不能重试")

    elif action_type == "start_validation":
        _require_fields(clean, ("train_job_id",))
        job = _owned_train_job(user_id, str(clean["train_job_id"]))
        if job.status != "completed":
            raise ActionValidationError("只有已完成的训练任务才能启动验证")

    return clean


# 生成供审批人阅读的动作影响说明。
def describe_action_impact(action_type: str, payload: dict[str, Any]) -> str:
    descriptions = {
        "create_train_job": "将创建训练记录并向 Celery 提交新的训练任务，消耗计算资源。",
        "stop_train_job": "将撤销正在运行的 Celery 任务，并把训练任务标记为已取消。",
        "retry_train_job": "将复制失败任务的配置并提交新的训练任务，原任务保持不变。",
        "clone_train_job": "将复制指定任务的配置并提交新的训练任务，原任务保持不变。",
        "start_validation": "将基于已完成的训练任务创建验证记录并向 Celery 提交验证任务。",
    }
    return descriptions[action_type]


# 根据用户、动作和参数生成稳定幂等键。
def _idempotency_key(user_id: int, action_type: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(
        {"user_id": user_id, "action_type": action_type, "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# 幂等创建待审批动作，并在允许时重开可重试的终态提案。
def create_action_proposal(
    user_id: int,
    action_type: str,
    payload: dict[str, Any],
    reason: str = "",
    expires_in_minutes: int = 30,
    conversation_id: str | None = None,
    idempotency_key: str | None = None,
    reopen_terminal: bool = True,
) -> tuple[AgentActionProposal, bool]:
    clean_payload = validate_action_payload(user_id, action_type, payload)
    if idempotency_key:
        key = hashlib.sha256(
            f"control-plane:{int(user_id)}:{idempotency_key}".encode("utf-8")
        ).hexdigest()
    else:
        key = _idempotency_key(user_id, action_type, clean_payload)
    existing = AgentActionProposal.query.filter_by(idempotency_key=key).first()
    if existing:
        if reopen_terminal and existing.status in {"rejected", "expired", "failed"}:
            try:
                ttl = max(1, min(int(expires_in_minutes), 120))
            except (TypeError, ValueError) as exc:
                raise ActionValidationError("expires_in_minutes 必须是整数") from exc
            existing.status = "pending_approval"
            existing.reason = (reason or "").strip()[:2000]
            existing.result = None
            existing.error_message = None
            existing.approved_at = None
            existing.executed_at = None
            existing.expires_at = datetime.utcnow() + timedelta(minutes=ttl)
            db.session.commit()
        return existing, False

    try:
        ttl = max(1, min(int(expires_in_minutes), 120))
    except (TypeError, ValueError) as exc:
        raise ActionValidationError("expires_in_minutes 必须是整数") from exc

    proposal = AgentActionProposal(
        proposal_id=f"act_{uuid.uuid4().hex}",
        user_id=user_id,
        conversation_id=conversation_id,
        action_type=action_type,
        payload=clean_payload,
        reason=(reason or "").strip()[:2000],
        impact=describe_action_impact(action_type, clean_payload),
        idempotency_key=key,
        status="pending_approval",
        expires_at=datetime.utcnow() + timedelta(minutes=ttl),
    )
    db.session.add(proposal)
    db.session.commit()
    _audit(user_id, "action.drafted", proposal.proposal_id, {"action_type": action_type})
    return proposal, True


# 拒绝当前用户拥有的待审批提案。
def reject_action_proposal(user_id: int, proposal_id: str) -> AgentActionProposal:
    proposal = _get_owned_proposal(user_id, proposal_id, lock=True)
    if proposal.status != "pending_approval":
        raise ActionConflictError(f"当前状态 {proposal.status} 不能拒绝")
    proposal.status = "rejected"
    db.session.commit()
    _audit(user_id, "action.rejected", proposal_id)
    return proposal


# 读取当前用户拥有的动作提案，不存在时抛出校验错误。
def _get_owned_proposal(
    user_id: int,
    proposal_id: str,
    lock: bool = False,
) -> AgentActionProposal:
    query = AgentActionProposal.query.filter_by(
        proposal_id=proposal_id,
        user_id=user_id,
    )
    if lock:
        query = query.with_for_update()
    proposal = query.first()
    if not proposal:
        raise ActionValidationError("操作草案不存在或不属于当前用户")
    return proposal


# 复制已有训练任务并生成新的待执行记录。
def _clone_train_job(source: TrainJob, user_id: int, description: str) -> TrainJob:
    dataset = Dataset.query.filter_by(id=source.dataset_id, isDelete=False).first()
    if not dataset:
        raise ActionValidationError("源任务的数据集不存在")
    backbone = Backbone.query.filter_by(id=source.backbone_id, isDelete=False).first()
    if not backbone:
        raise ActionValidationError("源任务的骨干网络不存在")
    classifier = Classifier.query.filter_by(id=source.classifier_id, isDelete=False).first()
    if not classifier:
        raise ActionValidationError("源任务的分类器不存在")
    config, feature_ids = _normalize_train_config(
        dataset,
        backbone,
        classifier,
        source.config or {},
        source.feature_ids or [],
    )
    job = TrainJob(
        job_id=str(uuid.uuid4()),
        user_id=user_id,
        description=description,
        dataset_id=source.dataset_id,
        nodes=deepcopy(source.nodes or []),
        backbone_id=source.backbone_id,
        classifier_id=source.classifier_id,
        config=config,
        feature_ids=feature_ids,
        adaptive=bool(source.adaptive),
        status="pending",
        progress=0,
        isDelete=False,
    )
    db.session.add(job)
    db.session.flush()
    add_compute_dispatch(job.job_id, "training", job.id)
    db.session.commit()
    return job


# 请求 Celery 撤销指定任务。
def _revoke_task(task_id: str) -> None:
    AsyncResult(task_id, app=celery_app).revoke(terminate=True)


# 根据提案类型执行经过批准的实际业务副作用。
def _execute_action(proposal: AgentActionProposal) -> dict[str, Any]:
    user_id = proposal.user_id
    payload = validate_action_payload(user_id, proposal.action_type, proposal.payload)

    if proposal.action_type == "create_train_job":
        job = TrainJob(
            job_id=str(uuid.uuid4()),
            user_id=user_id,
            description=payload.get("description") or "Agent 创建的训练任务",
            dataset_id=payload["dataset_id"],
            nodes=payload["nodes"],
            backbone_id=payload["backbone_id"],
            classifier_id=payload["classifier_id"],
            config=payload["config"],
            feature_ids=payload["feature_ids"],
            adaptive=payload["adaptive"],
            status="pending",
            progress=0,
            isDelete=False,
        )
        db.session.add(job)
        db.session.flush()
        add_compute_dispatch(job.job_id, "training", job.id)
        db.session.commit()
        return {"job_id": job.job_id, "database_id": job.id, "status": job.status}

    if proposal.action_type == "stop_train_job":
        job = _owned_train_job(user_id, payload["job_id"])
        _revoke_task(job.job_id)
        job.status = "cancelled"
        job.updateTime = datetime.now()
        add_experiment_event(
            job.job_id, "training", "experiment.training.cancelled"
        )
        db.session.commit()
        return {"job_id": job.job_id, "status": job.status}

    if proposal.action_type in {"retry_train_job", "clone_train_job"}:
        source = _owned_train_job(user_id, payload["job_id"])
        prefix = "重试" if proposal.action_type == "retry_train_job" else "克隆"
        description = payload.get("description") or f"{prefix}_{source.description or source.job_id}"
        job = _clone_train_job(source, user_id, description)
        return {
            "source_job_id": source.job_id,
            "job_id": job.job_id,
            "database_id": job.id,
            "status": job.status,
        }

    if proposal.action_type == "start_validation":
        source = _owned_train_job(user_id, payload["train_job_id"])
        job = TestRecord(
            val_job_id=str(uuid.uuid4()),
            train_job_id=source.job_id,
            user_id=user_id,
            config=deepcopy(source.config or {}),
            status="pending",
            isDelete=False,
        )
        db.session.add(job)
        db.session.flush()
        add_compute_dispatch(job.val_job_id, "validation", job.id)
        db.session.commit()
        return {
            "train_job_id": source.job_id,
            "val_job_id": job.val_job_id,
            "database_id": job.id,
            "status": job.status,
        }

    raise ActionValidationError(f"不支持的操作类型：{proposal.action_type}")


# 原子认领提案、执行动作，并持久化成功或失败终态。
def approve_and_execute_action(user_id: int, proposal_id: str) -> AgentActionProposal:
    proposal = _get_owned_proposal(user_id, proposal_id, lock=True)

    if proposal.status == "succeeded":
        return proposal
    if proposal.status != "pending_approval":
        raise ActionConflictError(f"当前状态 {proposal.status} 不能审批")
    if proposal.expires_at <= datetime.utcnow():
        proposal.status = "expired"
        db.session.commit()
        raise ActionConflictError("操作草案已过期，请重新生成")

    proposal.status = "executing"
    proposal.approved_at = datetime.utcnow()
    db.session.commit()

    try:
        proposal.result = _execute_action(proposal)
        proposal.status = "succeeded"
        proposal.executed_at = datetime.utcnow()
        proposal.error_message = None
        db.session.commit()
        _audit(user_id, "action.approved", proposal_id, {"action_type": proposal.action_type})
    except Exception as exc:
        db.session.rollback()
        proposal = _get_owned_proposal(user_id, proposal_id)
        proposal.status = "failed"
        proposal.error_message = str(exc)[:2000]
        proposal.executed_at = datetime.utcnow()
        db.session.commit()
        _audit(user_id, "action.failed", proposal_id, {"error": str(exc)[:500]}, success=False)
        raise

    return proposal


# 返回当前用户可访问的单个动作提案。
def get_action_proposal(user_id: int, proposal_id: str) -> AgentActionProposal:
    return _get_owned_proposal(user_id, proposal_id)


def verify_action_result(proposal: AgentActionProposal) -> dict[str, Any]:
    """Read the current target state without producing additional side effects."""
    result = proposal.result if isinstance(proposal.result, dict) else {}
    action_type = proposal.action_type

    if action_type in {
        "create_train_job",
        "retry_train_job",
        "clone_train_job",
        "stop_train_job",
    }:
        job_id = result.get("job_id") or proposal.payload.get("job_id")
        job = TrainJob.query.filter_by(
            job_id=job_id,
            user_id=proposal.user_id,
            isDelete=False,
        ).first()
        if not job:
            return {"verified": False, "target": "train_job", "job_id": job_id}
        return {
            "verified": True,
            "target": "train_job",
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "result": job.result if job.status in {"completed", "failed"} else None,
        }

    if action_type == "start_validation":
        val_job_id = result.get("val_job_id")
        job = TestRecord.query.filter_by(
            val_job_id=val_job_id,
            user_id=proposal.user_id,
            isDelete=False,
        ).first()
        if not job:
            return {"verified": False, "target": "validation", "val_job_id": val_job_id}
        return {
            "verified": True,
            "target": "validation",
            "val_job_id": job.val_job_id,
            "status": job.status,
            "result": job.result if job.status in {"completed", "failed"} else None,
        }

    return {"verified": False, "target": "unknown"}


# 按用户、状态和会话过滤动作提案列表。
def list_action_proposals(
    user_id: int,
    status: str | None = None,
    limit: int = 20,
    conversation_id: str | None = None,
) -> list[AgentActionProposal]:
    query = AgentActionProposal.query.filter_by(user_id=user_id)
    if status:
        query = query.filter_by(status=status)
    if conversation_id:
        query = query.filter_by(conversation_id=conversation_id)
    return query.order_by(AgentActionProposal.id.desc()).limit(max(1, min(limit, 100))).all()
