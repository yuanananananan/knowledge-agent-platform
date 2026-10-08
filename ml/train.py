import os
from pathlib import Path
import scipy.io
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from dataset.sar import get_sar_dataset_full
from model import resnet18, resnet50, KED
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import load_config, get_dataloader, get_model, save_model, load_model
from app.utils.log_streamer import append_log


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def train_step(model, im, label, feature, criterion, is_ked, device):
    model.train()
    im = im.to(device)
    label = label.to(device)
    feature = feature.to(device)

    if is_ked:
        result, feats = model(im, feature)
        loss = criterion(result, label)
    else:
        result, feats = model(im)
        loss = criterion(result, label)

    return loss


def val_step(model, im, label, feature, criterion, is_ked, device):
    with torch.no_grad():
        model.eval()
        im = im.to(device)
        label = label.to(device)
        feature = feature.to(device)

        if is_ked:
            result, feats = model(im, feature)
        else:
            result, feats = model(im)
        loss = criterion(result, label)

    return loss, result


def train(cfg, model, train_loader, test_loader, job_id, update_fn=None):
    epochs = int(cfg['training']['epochs'])
    checkpoint_root = Path(str(cfg['training']['checkpoint_dir']))
    if not checkpoint_root.is_absolute():
        checkpoint_root = PROJECT_ROOT / checkpoint_root
    checkpoint_root = checkpoint_root.resolve()
    allowed_checkpoint_root = (PROJECT_ROOT / 'model_ckp').resolve()
    if checkpoint_root != allowed_checkpoint_root and allowed_checkpoint_root not in checkpoint_root.parents:
        raise ValueError('training.checkpoint_dir 必须位于项目 model_ckp 目录内')
    model_path = checkpoint_root / str(job_id) / 'model_best.pth'
    is_ked = bool(cfg['model']['GNN']['is_GNN'])
    num_classes = int(cfg['data']['num_classes'])

    optimizer_cfg = cfg['training']['optimizer']
    optimizer_type = str(optimizer_cfg.get('type', 'Adam')).lower()
    optimizer_kwargs = {
        'lr': float(optimizer_cfg['lr']),
        'weight_decay': float(optimizer_cfg.get('weight_decay', 0)),
    }
    if optimizer_type == 'adam':
        optim = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    elif optimizer_type == 'sgd':
        optim = torch.optim.SGD(
            model.parameters(),
            momentum=float(optimizer_cfg.get('momentum', 0)),
            **optimizer_kwargs,
        )
    else:
        raise ValueError(f"不支持的优化器：{optimizer_cfg.get('type')}")
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    OA = -1.0
    F1 = -1
    LOG = {}
    train_loss_global = []
    test_loss_global = []
    metrics_per_epoch = []  # 保存每轮指标

    if len(train_loader) == 0 or len(test_loader) == 0:
        raise ValueError('训练集和测试集都必须至少产生一个批次')

    for ep in range(epochs):
        # ======== Train ========
        losses_list = []
        for im, label, feature in tqdm(train_loader):
            loss = train_step(model, im, label, feature, criterion, is_ked, device)
            loss.backward()
            optim.step()
            optim.zero_grad()
            losses_list.append(loss.mean().item())  # 转成 float
        train_loss = float(sum(losses_list) / len(losses_list))
        train_loss_global.append(train_loss)

        # ======== Validation ========
        correct = 0.0
        total = 0
        losses_list = []
        class_total = np.zeros(num_classes)
        class_correct = np.zeros(num_classes)

        all_preds = []
        all_labels = []
        for im, label, feature in tqdm(test_loader):
            loss, result = val_step(model, im, label, feature, criterion, is_ked, device)
            result = result.max(-1)[1].float()
            result_bin = torch.where(result == label.to(device), 1., 0.)
            correct += float(result_bin.sum().item())
            total += len(result_bin)
            losses_list.append(loss.mean().item())

            for i in range(label.size(0)):
                l = int(label[i].item())
                class_total[l] += 1
                class_correct[l] += float(result_bin[i].item())

            all_preds.extend([int(x) for x in result.cpu().numpy()])
            all_labels.extend([int(x) for x in label.cpu().numpy()])

        precision = float(precision_score(all_labels, all_preds, average='macro', zero_division=0))
        recall = float(recall_score(all_labels, all_preds, average='macro', zero_division=0))
        f1 = float(f1_score(all_labels, all_preds, average='macro', zero_division=0))
        val_loss = float(sum(losses_list) / len(losses_list))
        test_loss_global.append(val_loss)

        class_accuracies = [
            round(100.0 * class_correct[i] / class_total[i], 2) if class_total[i] > 0 else 0.0
            for i in range(num_classes)
        ]

        oa = float(100.0 * correct / total)
        kappa = float(cohen_kappa_score(all_labels, all_preds))

        # 保存每轮指标
        metrics_per_epoch.append({
            "epoch": int(ep),
            "OA": round(oa, 4),
            "Kappa": round(kappa, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "Train_Loss": round(train_loss, 4),
            "Test_Loss": round(val_loss, 4),
            "Class_Accuracy": [float(x) for x in class_accuracies]
        })

        print(f"epoch:{ep}, OA: {oa:.2f}%, Kappa: {kappa:.4f}, "
              f"Cls Acc: {class_accuracies}, P:{precision:.4f}, R:{recall:.4f}, "
              f"F1:{f1:.4f}, loss:{val_loss:.2f}")
        append_log(
            job_id, f"epoch:{ep}, OA: {oa:.2f}%, Kappa: {kappa:.4f}, "
            f"Cls Acc: {class_accuracies}, P:{precision:.4f}, "
            f"R:{recall:.4f}, F1:{f1:.4f}, loss:{val_loss:.2f}")
        if oa > OA:
            OA = oa
            LOG.update({"OA": round(OA, 4), "kappa": round(kappa, 4), "ep": int(ep)})
            save_model(model, str(model_path))
        if update_fn:
            update_fn(ep)

    # 保存 loss 曲线
    save_dir = PROJECT_ROOT / "img" / str(job_id)
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(train_loss_global)), train_loss_global, marker='o', color='b', label='train')
    plt.plot(range(len(test_loss_global)), test_loss_global, marker='x', color='r', label='test')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    loss_png_path = save_dir / "loss.png"
    plt.savefig(loss_png_path)
    plt.close()

    # 汇总结果
    summary = {
        "Best": LOG,
        "Metrics": metrics_per_epoch,
        "Train_Loss": [float(x) for x in train_loss_global],
        "Test_Loss": [float(x) for x in test_loss_global]
    }
    log_path = PROJECT_ROOT / "logs" / f"{job_id}.log"
    abs_model = str(model_path)
    abs_log = str(log_path.resolve())
    # 返回模型和纯 Python 数据
    return model, summary, abs_model, abs_log


if __name__ == '__main__':
    # seed = 10001
    # # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    cfg = load_config('configs/sar_config.yaml')
    train_loader, test_loader = get_dataloader(cfg)
    
    # model = resnet18(pretrained=True, num_classes=num_classes)
    # model = KED(cfg)
    model = get_model(cfg)

    train(cfg, model, train_loader, test_loader, job_id="2026_3_3")
