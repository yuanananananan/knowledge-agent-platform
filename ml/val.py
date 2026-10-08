import os
import scipy.io
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from dataset.sar import get_sar_dataset_full
from model import resnet18, resnet50, KED
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import load_config, get_dataloader, get_model, save_model, load_model
from app.utils.log_streamer import append_log


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


def val(cfg, model, test_loader, model_path, job_id):
    model_cfg = cfg.get('model', {}) if isinstance(cfg, dict) else {}
    gnn_cfg = model_cfg.get('GNN', {}) if isinstance(model_cfg, dict) else {}
    data_cfg = cfg.get('data', {}) if isinstance(cfg, dict) else {}
    is_ked = bool(gnn_cfg.get('is_GNN', False))
    try:
        num_classes = int(data_cfg['num_classes'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('验证配置缺少有效的 data.num_classes') from exc

    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model, path=model_path, device=device)
    model.to(device)

    correct = 0
    total = 0
    losses_list = []
    class_total = np.zeros(num_classes)
    class_correct = np.zeros(num_classes)

    all_preds = []  # 用来保存所有预测值
    all_labels = []  # 用来保存所有真实标签

    if len(test_loader) == 0:
        raise ValueError('测试集必须至少产生一个批次')

    for im, label, feature in tqdm(test_loader):
        loss, result = val_step(model, im, label, feature, criterion, is_ked, device)
        result = result.max(-1)[1].float()  # 选择最大值索引作为预测结果
        result_bin = torch.where(result == label.to(device), 1., 0.)  # 计算是否预测正确
        correct += result_bin.sum()
        total += len(result_bin)
        losses_list.append(loss.mean())

        # 计算每个类别的正确数和总数
        for i in range(label.size(0)):
            l = int(label[i].item())
            class_total[l] += 1
            class_correct[l] += result_bin[i].item()

        # 收集所有的预测和标签用于计算Kappa
        all_preds.extend(int(value) for value in result.cpu().numpy())
        all_labels.extend(int(value) for value in label.cpu().numpy())
    # 计算整体的Precision, Recall, F1
    precision = float(precision_score(all_labels, all_preds, average='macro', zero_division=0))
    recall = float(recall_score(all_labels, all_preds, average='macro', zero_division=0))
    f1 = float(f1_score(all_labels, all_preds, average='macro', zero_division=0))

    # 计算损失
    losses = float(sum(item.item() for item in losses_list) / len(losses_list))

    # 计算每个类别的准确率
    class_accuracies = [
        round(100 * class_correct[i] / class_total[i], 2) if class_total[i] > 0 else 0 for i in range(num_classes)
    ]

    if total == 0:
        raise ValueError('测试集没有可评估样本')
    oa = float(100.0 * correct.item() / total)
    kappa = float(cohen_kappa_score(all_labels, all_preds))
    print(
        f"OA: {oa:.2f}%, Kappa: {kappa:.4f}, Cls Acc: {class_accuracies}, P:{precision:.4f},R:{recall:.4f},F1:{f1:.4f},loss:{losses:.2f}"
    )
    append_log(
        job_id, f"OA: {oa:.2f}%, Kappa: {kappa:.4f}, "
        f"Cls Acc: {class_accuracies}, P:{precision:.4f}, "
        f"R:{recall:.4f}, F1:{f1:.4f}, loss:{losses:.2f}")
    metrics = ({
        "OA": round(oa, 4),
        "Kappa": round(kappa, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "Class_Accuracy": [float(x) for x in class_accuracies]
    })
    summary = {"Metrics": metrics, "Loss": losses}
    return model, summary


if __name__ == '__main__':

    cfg = load_config('configs/sar_config.yaml')
    model_path = 'model_ckp\\2026_3_3\\model_best.pth'
    train_loader, test_loader = get_dataloader(cfg)
    model = get_model(cfg)
    val(cfg, model, test_loader, model_path, job_id="2026_3_3")
