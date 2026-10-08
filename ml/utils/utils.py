import json
import yaml
import os
from pathlib import Path
import torch
from dataset import get_sar_dataset_full, get_rd_dataset, get_MON_dataset
from torch.utils.data import DataLoader
from model import (
    KED,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    vgg8_bn,
    vgg11_bn,
    vgg13_bn,
    vgg16_bn,
    vgg19_bn,
)


MODEL_FACTORIES = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'vgg8_bn': vgg8_bn,
    'vgg11_bn': vgg11_bn,
    'vgg13_bn': vgg13_bn,
    'vgg16_bn': vgg16_bn,
    'vgg19_bn': vgg19_bn,
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _normalize_image_size(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("data.img_size 必须是正整数或包含两个正整数的数组") from exc
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("data.img_size 必须大于 0")
        return value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            normalized = [int(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise ValueError("data.img_size 必须包含两个正整数") from exc
        if all(item > 0 for item in normalized):
            return normalized
    raise ValueError("data.img_size 必须是正整数或包含两个正整数的数组")


def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:  # 明确指定utf-8编码
        config = yaml.safe_load(f)
    return config


def get_dataloader(cfg):
    data_cfg = cfg.get('data')
    if not isinstance(data_cfg, dict):
        raise ValueError('训练配置缺少 data 对象')
    root_value = str(data_cfg.get('root', '')).strip()
    if not root_value:
        raise ValueError('训练配置缺少 data.root')
    root = Path(root_value)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()
    missing_splits = [name for name in ('train', 'test') if not (root / name).is_dir()]
    if missing_splits:
        raise FileNotFoundError(
            f"数据集目录 {root} 缺少分区：{', '.join(missing_splits)}"
        )
    data_cfg['root'] = str(root)
    data_cfg['img_size'] = _normalize_image_size(data_cfg.get('img_size'))

    mode = str(data_cfg.get('mode', '')).upper()
    if mode not in {'RD', 'SAR', 'MON'}:
        raise ValueError(f'不支持的数据集模态：{mode or "未设置"}')
    data_cfg['mode'] = mode
    if mode == 'SAR':
        train_dataset, test_dataset = get_sar_dataset_full(cfg)
    elif mode == 'RD':
        train_dataset, test_dataset = get_rd_dataset(cfg)
    elif mode == 'MON':
        train_dataset, test_dataset = get_MON_dataset(cfg)
    if len(train_dataset) == 0:
        raise ValueError('训练集没有可读取的样本')
    if len(test_dataset) == 0:
        raise ValueError('测试集没有可读取的样本')
    print(f"Total samples: {len(train_dataset)}")
    # Create data loader
    batch_size = int(data_cfg['batch_size'])
    shuffle = bool(data_cfg['shuffle'])
    num_workers = int(data_cfg['num_workers'])
    if batch_size < 1:
        raise ValueError('data.batch_size 必须大于 0')
    if num_workers < 0:
        raise ValueError('data.num_workers 不能小于 0')
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_loader, test_loader


def get_model(cfg):
    model_cfg = cfg.get('model')
    if not isinstance(model_cfg, dict):
        raise ValueError('训练配置缺少 model 对象')
    gnn_cfg = model_cfg.get('GNN')
    if not isinstance(gnn_cfg, dict):
        raise ValueError('训练配置缺少 model.GNN 对象')
    if bool(gnn_cfg.get('is_GNN', False)):
        return KED(cfg=cfg)

    backbone_name = str(model_cfg.get('backbone', '')).strip()
    model_factory = MODEL_FACTORIES.get(backbone_name)
    if model_factory is None:
        supported = ', '.join(sorted(MODEL_FACTORIES))
        raise ValueError(f'不支持的普通训练骨干：{backbone_name or "未设置"}；可选：{supported}')
    num_classes = int(cfg['data']['num_classes'])
    pretrained = bool(model_cfg.get('pretrained', False))
    if backbone_name.startswith('resnet'):
        return model_factory(pretrained=pretrained, num_classes=num_classes)
    if pretrained:
        raise ValueError(f'{backbone_name} 当前不支持 pretrained=true')
    return model_factory(num_classes=num_classes)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path, device='cuda'):
    model_path = Path(path)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")
    state_dict = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model
