"""文件用途：准备 NAS 任务目录、生成配置、启动子进程并处理搜索产物。"""

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import json

from app import db
from app.models.backbone import Backbone
from app.utils.log_streamer import append_log, LOG_DIR
from app.utils.nas_log_parser import extract_save_dir, parse_best_genotype_from_lines


# 项目根目录
# app/utils/nas_task_utils.py -> app/utils -> app -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = PROJECT_ROOT / "ml"

def extract_epoch(line):
    """从训练日志中提取 epoch 序号，用来估算任务进度。"""
    m = re.search(r'epoch\s+(\d+)\s+lr', line)
    if not m:
        return None

    return int(m.group(1))

def extract_best_acc_from_lines(lines):
    """扫描日志并返回最后记录的最佳验证准确率。"""
    best_acc = None

    for line in lines:
        m = re.search(r"New Best Validation Accuracy:\s*([0-9]+(?:\.\d+)?)", line)
        if m:
            best_acc = float(m.group(1))

    return best_acc

def run_subprocess_and_stream(cmd, real_job_id, job_obj=None, total_epochs=None):
    """运行训练子进程，把输出实时追加到平台日志并同步任务进度。"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ML_ROOT), str(PROJECT_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    append_log(real_job_id,"执行命令：" + " " .join(cmd))

    process = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)

    all_lines = []
    checkpoint_dir = None

    for raw_line in process.stdout:
        line = raw_line.rstrip("\n")
        all_lines.append(line)

        append_log(real_job_id, line)

        maybe_dir = extract_save_dir(line)
        if maybe_dir:
            checkpoint_dir = maybe_dir

        # 粗略根据 epoch 更新进度。
        if job_obj is not None and total_epochs:
            epoch = extract_epoch(line)
            if epoch is not None:
                job_obj.progress = min(99, int((epoch + 1) / int(total_epochs) * 100),)
                job_obj.updateTime = datetime.now()
                db.session.commit()

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"子进程执行失败，return_code={return_code}")

    return all_lines, checkpoint_dir

def build_train_search_cmd(modality, cfg):
    """根据模态和配置构造 NAS 架构搜索命令。"""
    args_keys = [
        "data",
        "batch_size",
        "learning_rate",
        "learning_rate_min",
        "momentum",
        "weight_decay",
        "report_freq",
        "gpu",
        "epochs",
        "init_channels",
        "layers",
        "save",
        "seed",
        "grad_clip",
        "train_portion",
        "arch_learning_rate",
        "arch_weight_decay",
    ]
    cmd = [sys.executable, "-m", "NAS.train_search", "--modality", modality]

    for key in args_keys:
        val = cfg.get(key)
        if val is None:
            continue
        cmd.extend([f"--{key}", str(val)])

    return cmd

def get_nas_args_from_config(config_dict, backbone_schema=None):
    """合并训练配置与 Backbone Schema，提取 train_nas 所需参数。"""
    backbone_schema = backbone_schema or {}
    train_args = backbone_schema.get("train_args", {}) or {}

    training = config_dict.get("training", {}) or {}
    data_cfg = config_dict.get("data", {}) or {}
    optimizer = training.get("optimizer", {}) or {}

    nas_args = {}
    modality = (
        data_cfg.get("mode")
        or backbone_schema.get("modality")
        or train_args.get("modality")
    )
    if modality is not None:
        nas_args["modality"] = str(modality).lower()

    if "epochs" in training:
        nas_args["epochs"] = training["epochs"]

    if "batch_size" in data_cfg:
        nas_args["batch_size"] = data_cfg["batch_size"]

    if "lr" in optimizer:
        nas_args["learning_rate"] = optimizer["lr"]

    if "momentum" in optimizer:
        nas_args["momentum"] = optimizer["momentum"]

    if "weight_decay" in optimizer:
        nas_args["weight_decay"] = optimizer["weight_decay"]

    # NAS 主干结构相关参数，优先来自搜索后注册的 backbone
    for key in [
        "init_channels",
        "layers",
        "data",
        "report_freq",
        "gpu",
        "model_path",
        "auxiliary_weight",
        "cutout",
        "cutout_length",
        "drop_path_prob",
        "save",
        "seed",
        "arch",
        "grad_clip",
    ]:
        if key in train_args:
            nas_args[key] = train_args[key]

    return nas_args


def build_train_nas_cmd(nas_args):
    """把 NAS 训练参数转换为安全的命令参数列表。"""
    arg_keys = [
        "modality",
        "data",
        "batch_size",
        "learning_rate",
        "momentum",
        "weight_decay",
        "report_freq",
        "gpu",
        "epochs",
        "init_channels",
        "layers",
        "model_path",
        "auxiliary_weight",
        "cutout_length",
        "drop_path_prob",
        "save",
        "seed",
        "arch",
        "grad_clip",
        "genotype_json_path",
    ]
    cmd = [sys.executable, "-m", "NAS.train_nas"]

    for key in arg_keys:
        val = nas_args.get(key)

        if val is None:
            continue

        cmd.extend([f"--{key}", str(val)])

    if nas_args.get("cutout", False):
        cmd.extend(["--cutout"])

    return cmd

def run_nas_backbone_training(job, config_dict, backbone):
    """任务选择 NAS Backbone 且未启用 GNN 时，执行搜索结构的正式训练。"""
    real_job_id = job.job_id

    schema = backbone.params_schema or {}
    nas_args = get_nas_args_from_config(config_dict, schema)
    genotype = schema.get("genotype")
    if not genotype:
        raise ValueError(
            "当前 NAS Backbone 缺少 genotype，无法训练搜索得到的结构。"
            "请确认 NAS 搜索任务已成功解析并注册 genotype。"
        )

    genotype_path = LOG_DIR / f"{real_job_id}_genotype.json"
    with genotype_path.open("w", encoding="utf-8") as f:
        json.dump(genotype, f, ensure_ascii=False, indent=2)

    nas_args["genotype_json_path"] = str(genotype_path)

    append_log(real_job_id, "开始 NAS 主干训练")
    append_log(real_job_id, f"NAS训练参数: {nas_args}")
    append_log(real_job_id, f"使用搜索得到的 genotype 文件: {genotype_path}")

    cmd = build_train_nas_cmd(nas_args)

    lines, checkpoint_dir = run_subprocess_and_stream(
        cmd,
        real_job_id,
        job_obj=job,
        total_epochs=nas_args.get("epochs"),
    )

    best_acc = extract_best_acc_from_lines(lines)

    model_path = None
    if checkpoint_dir:
        candidate = PROJECT_ROOT / checkpoint_dir / "weights.pt"
        model_path = str(candidate)

    result = {
        "type": "nas_backbone_train",
        "best_valid_acc": best_acc,
        "checkpoint_dir": checkpoint_dir,
        "backbone_id": job.backbone_id,
        "backbone_name": backbone.name,
        "genotype_json_path": str(genotype_path),
        "nas_args": nas_args,
    }

    append_log(real_job_id, "NAS 主干训练结束")

    return result, model_path, str(LOG_DIR / f"{real_job_id}.log")

def register_nas_backbone(job, cfg, modality, parsed, checkpoint_dir):
    """把搜索得到的最优 genotype 注册为新的 Backbone 记录。"""
    genotype = parsed.get("best_genotype")

    if not genotype:
        raise ValueError(
            "NAS 搜索完成，但没有从日志中解析到 genotype。"
            "请检查 train_search.py 是否输出 'genotype = ...' 和 'valid_acc ...'"
        )

    backbone_name = cfg.get("backbone_name") or (
        f"NAS_{modality}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    )
    train_arg_keys = [
        "data",
        "batch_size",
        "learning_rate",
        "learning_rate_min",
        "momentum",
        "weight_decay",
        "report_freq",
        "gpu",
        "epochs",
        "init_channels",
        "layers",
        "save",
        "seed",
        "grad_clip",
        "train_portion",
        "arch_learning_rate",
        "arch_weight_decay",
    ]

    train_args = {
        key: cfg[key]
        for key in train_arg_keys
        if key in cfg and cfg[key] is not None
    }

    backbone = Backbone(
        name=backbone_name,
        modality=modality,
        backbone_type="nas_search",
        params_schema={
            "type": "nas_search",
            "modality": modality,
            "source_search_job_id": job.id,
            "genotype": genotype,
            "genotype_text": parsed.get("best_genotype_text"),
            "best_valid_acc": parsed.get("best_valid_acc"),
            "train_args": train_args,
        },
        artifact_path=checkpoint_dir,
        description=f"由 NAS 搜索任务 {job.id} 自动生成的识别主干",
    )

    db.session.add(backbone)
    db.session.flush()

    return backbone
