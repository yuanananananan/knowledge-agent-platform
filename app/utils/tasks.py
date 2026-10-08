"""文件用途：定义训练、验证、NAS 和 Agent 评测的 Celery 后台任务。"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path


# 把项目根目录加入 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = PROJECT_ROOT / "ml"

for source_root in (ML_ROOT, PROJECT_ROOT):
    if str(source_root) not in sys.path:
        sys.path.append(str(source_root))

from app.utils.celery_worker import celery_app
from app import db
from app.models.train_job import TrainJob
from app.models.backbone import Backbone
from app.models.test_record import TestRecord
from app.models.backbone import Backbone
from app.models.nas_search_job import NasSearchJob
from app.utils.utils import json_to_config
from app.utils.log_streamer import append_log, LOG_DIR
from app.utils.nas_log_parser import parse_best_genotype_from_lines
from app.utils.nas_task_utils import build_train_nas_cmd, run_nas_backbone_training, run_subprocess_and_stream, \
    register_nas_backbone, build_train_search_cmd
from app.utils.task_dedupe import deduplicated
from app.utils.experiment_events import add_experiment_event


@celery_app.task(bind=True, name="agent.run_golden_evaluation")
def run_agent_evaluation_task(self, eval_run_id, user_id, case_ids=None):
    """脱离浏览器请求，在后台可靠执行一轮 Golden Eval。"""
    from app.agent.evaluation import execute_golden_evaluation, fail_golden_evaluation

    try:
        evaluation = execute_golden_evaluation(
            str(eval_run_id), int(user_id), list(case_ids) if case_ids else None
        )
        return evaluation.to_dict()
    except Exception as exc:
        db.session.rollback()
        fail_golden_evaluation(str(eval_run_id), int(user_id), str(exc))
        raise


@celery_app.task(bind=True)
@deduplicated(lambda job_id: f"training:{job_id}")
def run_training_task(self, job_id):
    """执行普通或 NAS Backbone 训练，并持久化进度、结果和失败原因。"""

    from train import train
    from utils.utils import get_model, get_dataloader

    job = None
    try:
        job = db.session.get(TrainJob, job_id)
        if not job:
            raise ValueError(f"训练任务不存在：{job_id}")

        real_job_id = job.job_id or str(job_id)

        job.status = "running"
        job.progress = 0
        job.updateTime = datetime.now()
        add_experiment_event(real_job_id, "training", "experiment.training.started")
        db.session.commit()

        config = job.config
        config_dict = json_to_config(config)

        # Older/manual jobs may have stored the selected backbone only in
        # ``backbone_id`` and therefore lack config.model.backbone. Recover it
        # from the trusted database record before constructing the model.
        model_config = config_dict.setdefault("model", {})
        if not isinstance(model_config, dict):
            raise ValueError("训练配置中的 model 必须是对象")
        if not str(model_config.get("backbone", "")).strip():
            backbone = db.session.get(Backbone, job.backbone_id)
            if backbone and backbone.name:
                model_config["backbone"] = backbone.name
            else:
                raise ValueError("训练任务未配置有效的 backbone，请重新创建或重试任务")
        model_config.setdefault("GNN", {"is_GNN": bool(job.adaptive)})
        if not isinstance(model_config["GNN"], dict):
            model_config["GNN"] = {"is_GNN": bool(job.adaptive)}

        append_log(real_job_id, "开始训练")
        append_log(real_job_id, f"任务数据库ID: {job_id}")
        append_log(real_job_id, f"任务运行ID: {real_job_id}")

        model_cfg = config_dict.get("model", {})
        gnn_cfg = model_cfg.get("GNN", {})

        is_gnn = gnn_cfg.get("is_GNN",config_dict.get("is_GNN", True))
        is_nas_from_config = config_dict.get("is_nas",model_cfg.get("is_nas", False))
        is_nas_from_job = bool(getattr(job, "adaptive", False))

        #查询当前选择的 backbone
        backbone = None
        if getattr(job, "backbone_id", None):
            backbone = Backbone.query.get(job.backbone_id)

        is_nas_backbone = False
        if backbone is not None:
            backbone_type = getattr(backbone, "backbone_type", "")
            is_nas_backbone = backbone_type == "nas_search"

        is_nas = is_nas_from_config or is_nas_from_job or is_nas_backbone

        append_log(real_job_id, f"is_GNN={is_gnn}")
        append_log(real_job_id, f"is_nas={is_nas}")

        if backbone is not None:
            append_log(real_job_id, f"backbone_id={backbone.id}")
            append_log(real_job_id, f"backbone_name={backbone.name}")
            append_log(real_job_id, f"backbone_type={getattr(backbone, 'backbone_type', None)}")
        else:
            append_log(real_job_id, "未查询到 backbone，按普通训练流程处理")

        if is_nas:
            if is_gnn:
                raise ValueError(
                    "当前选择的是 NAS 搜索主干，但 is_GNN=True。"
                    "ml/NAS/train_nas.py 目前用于训练识别主干，不支持嵌入知识/GNN。"
                    "请在创建任务时关闭 is_GNN。"
                )

            if backbone is None:
                raise ValueError(
                    "当前任务被标记为 NAS 训练，但没有找到对应的 backbone。"
                    "请确认 train_job.backbone_id 是否正确。"
                )

            if not is_nas_backbone:
                raise ValueError(
                    f"当前任务 is_nas=True，但选择的 backbone_type 不是 nas_search，"
                    f"实际为: {getattr(backbone, 'backbone_type', None)}。"
                )

            append_log(real_job_id, "检测到 NAS 搜索主干，开始调用 ml/NAS/train_nas.py 训练主干模型")

            job.progress = 1
            job.updateTime = datetime.now()
            db.session.commit()

            result, model_path, log_path = run_nas_backbone_training(
                job=job,
                config_dict=config_dict,
                backbone=backbone,
            )

            job.status = "completed"
            job.progress = 100
            job.result = result
            job.model_path = model_path
            job.log_path = log_path
            job.updateTime = datetime.now()
            add_experiment_event(
                real_job_id, "training", "experiment.training.completed",
                {"progress": 100, "model_path": model_path},
            )
            db.session.commit()

            append_log(real_job_id, "NAS 主干训练结束")

            return {
                "message": "nas backbone training completed",
                "job_id": real_job_id,
                "result": result,
            }


        append_log(real_job_id, "检测到普通训练任务，开始加载数据集")
        epoches = config_dict["training"]["epochs"]
        train_loader, test_loader = get_dataloader(config_dict)
        append_log(real_job_id, "数据集加载完成，开始加载模型")

        model = get_model(config_dict)
        append_log(real_job_id, "模型加载完成，开始训练")

        def update_progress(epoch):
            job.progress = int((epoch + 1) / epoches * 100)
            job.updateTime = datetime.now()
            db.session.commit()

        _, result, model_path, log_path = train(
            config_dict,
            model,
            train_loader,
            test_loader,
            real_job_id,
            update_fn=update_progress,
        )

        append_log(real_job_id, "训练结束")

        job.status = "completed"
        job.progress = 100
        job.result = result
        job.model_path = model_path
        job.log_path = log_path
        job.updateTime = datetime.now()
        add_experiment_event(
            real_job_id, "training", "experiment.training.completed",
            {"progress": 100, "model_path": model_path},
        )
        db.session.commit()

        return {
            "message": "training completed",
            "job_id": real_job_id,
            "result": result,
        }

    except Exception as e:
        if job is not None:
            db.session.rollback()
            real_job_id = getattr(job, "job_id", None) or str(job_id)

            append_log(real_job_id, f"训练失败: {str(e)}")

            job.status = "failed"
            job.result = {"error": str(e)}
            job.updateTime = datetime.now()
            add_experiment_event(
                real_job_id, "training", "experiment.training.failed",
                {"error": str(e)[:1000]},
            )
            db.session.commit()

        raise

@celery_app.task(bind=True)
@deduplicated(lambda job_id: f"validation:{job_id}")
def run_valuating_task(self, job_id):
    """加载训练产物执行验证，并持久化指标和任务状态。"""
    from utils.utils import get_model, get_dataloader

    from val import val

    job = None
    try:
        job = db.session.get(TestRecord, job_id)
        if not job:
            raise ValueError(f"验证任务不存在：{job_id}")

        job.status = "running"
        job.updateTime = datetime.now()
        add_experiment_event(job.val_job_id, "validation", "experiment.validation.started")
        db.session.commit()

        # 数据加载
        _, test_loader = get_dataloader(config_dict)
        # 模型加载
        model = get_model(config_dict)

        real_job_id = job.val_job_id
        append_log(real_job_id, "开始测试")

        source_job = TrainJob.query.filter_by(
            job_id=job.train_job_id,
            user_id=job.user_id,
            isDelete=False,
        ).first()
        if not source_job:
            raise ValueError("验证任务对应的训练任务不存在")
        if source_job.status != "completed":
            raise ValueError(f"训练任务状态为 {source_job.status}，不能执行验证")
        model_path = source_job.model_path
        if not model_path:
            raise ValueError("训练任务没有可用的模型文件")

        _, result = val(config_dict, model, test_loader, model_path, real_job_id)

        append_log(real_job_id, "测试结束")

        job.status = "completed"
        job.result = result
        job.updateTime = datetime.now()
        add_experiment_event(
            real_job_id, "validation", "experiment.validation.completed"
        )
        db.session.commit()

        return {
            "message": "valuation completed",
            "job_id": real_job_id,
            "result": result,
        }

    except Exception as e:
        if job is not None:
            db.session.rollback()
            job.status = "failed"
            job.result = {"error": str(e)}
            job.updateTime = datetime.now()
            add_experiment_event(
                job.val_job_id, "validation", "experiment.validation.failed",
                {"error": str(e)[:1000]},
            )
            db.session.commit()
        raise

@celery_app.task(bind=True)
@deduplicated(lambda search_job_id: f"nas-search:{search_job_id}")
def run_nas_search_task(self, search_job_id):
    """执行 NAS 搜索、解析最优 genotype，并自动注册为 Backbone。"""
    job = None
    try:
        job = NasSearchJob.query.get(search_job_id)

        if not job:
            return {"error": "Invalid nas_search_job_id"}

        job.status = "running"
        job.progress = 0
        job.updateTime = datetime.now()
        add_experiment_event(job.job_id, "nas", "experiment.nas.started")
        db.session.commit()

        real_job_id = job.job_id
        cfg = job.config or {}

        modality = str(job.modality or cfg.get("modality", "sar")).lower()
        epochs = cfg.get("epochs")
        total_epochs = int(epochs) if epochs is not None else None

        append_log(real_job_id, "开始 NAS 主干搜索")
        append_log(real_job_id, f"modality: {modality}")
        if cfg:
            append_log(
                real_job_id,
                "NAS搜索配置: " + json.dumps(cfg, ensure_ascii=False)
            )

            # 3. 构造并执行 train_search.py 命令
        cmd = build_train_search_cmd(modality, cfg)

        lines, checkpoint_dir = run_subprocess_and_stream(
            cmd,
            real_job_id,
            job_obj=job,
            total_epochs=total_epochs,
        )

        # 4. 从日志中解析最优 genotype
        parsed = parse_best_genotype_from_lines(lines)
        genotype = parsed.get("best_genotype")

        # 5. 注册为新的 Backbone
        backbone = register_nas_backbone(
            job=job,
            cfg=cfg,
            modality=modality,
            parsed=parsed,
            checkpoint_dir=checkpoint_dir,
        )

        # 6. 回写 NAS 搜索任务结果
        job.status = "completed"
        job.progress = 100
        job.genotype = genotype
        job.result = {
            "best_valid_acc": parsed.get("best_valid_acc"),
            "best_genotype": genotype,
            "backbone_id": backbone.id,
            "checkpoint_dir": checkpoint_dir,
        }
        job.backbone_id = backbone.id
        job.checkpoint_dir = checkpoint_dir
        job.log_path = str(LOG_DIR / f"{real_job_id}.log")
        job.updateTime = datetime.now()

        add_experiment_event(
            real_job_id, "nas", "experiment.nas.completed",
            {"progress": 100, "backbone_id": backbone.id},
        )

        db.session.commit()

        append_log(
            real_job_id,
            "最优NAS结构字典: " + json.dumps(genotype, ensure_ascii=False),
        )
        append_log(
            real_job_id,
            f"已注册为骨干网络 backbone_id={backbone.id}, name={backbone.name}",
        )

        return {
            "message": "nas search completed",
            "job_id": real_job_id,
            "backbone_id": backbone.id,
            "result": job.result,
        }

    except Exception as e:
        if job is not None:
            job.status = "failed"
            job.result = {"error": str(e)}
            job.updateTime = datetime.now()
            add_experiment_event(
                job.job_id or str(search_job_id), "nas", "experiment.nas.failed",
                {"error": str(e)[:1000]},
            )
            db.session.commit()

            if job.job_id:
                append_log(job.job_id, f"NAS 搜索失败: {e}")

        raise
