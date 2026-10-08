"""文件用途：提供配置读取、数据转换、训练/验证任务和图像处理等通用算法工具。"""

import json
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from app.models.image import Image
import io
import os
from collections import defaultdict
from ruamel.yaml import YAML

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.npy')


def load_config(config_path):
    """读取 YAML 配置，且不加载模型或训练模块。"""
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)

def json_to_config(json_obj):
    """将 JSON（字符串或字典）转换为 Python 字典，效果等同于 load_config"""
    if isinstance(json_obj, str):
        # 如果是 JSON 字符串，先解析
        data_dict = json.loads(json_obj)
    elif isinstance(json_obj, dict):
        data_dict = json_obj
    else:
        raise TypeError("必须是 JSON 字符串 或 Python 字典")

    # 转成 YAML 再解析一次，保证和 load_config 一致
    yaml_str = yaml.dump(data_dict, allow_unicode=True, sort_keys=False)
    config = yaml.safe_load(yaml_str)  # 这里的 config 就是 Python 字典
    return config


def config_to_json(config):
    """将 Python 字典（配置）转换为 JSON 字符串"""
    if not isinstance(config, dict):
        raise TypeError("必须是 Python 字典")

    # 转成 YAML 再解析一次，保证和 json_to_config 一致
    yaml_str = yaml.dump(config, allow_unicode=True, sort_keys=False)
    config_dict = yaml.safe_load(yaml_str)

    # 将字典转换为 JSON 字符串
    json_str = json.dumps(config_dict, ensure_ascii=False, indent=4)
    return json_str

# def rd_to_image(data):
#     # 对数据取绝对值并取对数
#     data = np.log10(np.abs(data))
#     # 创建图形和坐标轴
#     fig, ax = plt.subplots(figsize=(8, 8))
#     # 绘制图像
#     im = ax.imshow(data, cmap='jet', aspect='auto')
#     # 添加颜色条
#     plt.colorbar(im, ax=ax)
#     # 保存图像到内存缓冲区
#     buf = BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     # 从缓冲区创建PIL图像对象
#     image = Image.open(buf)
#     plt.show()
#     # 关闭图形以释放内存
#     plt.close(fig)
#     return image

def rd_to_image(data) -> io.BytesIO:
    """将RD矩阵转换为伪彩色图像并返回 PNG 字节流"""
    data = np.log10(np.abs(data) + 1e-8)  # 避免 log(0)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(data, cmap="jet", aspect="auto")
    plt.colorbar(im, ax=ax)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf

def yaml_to_schema(cfg):
    """
    将 YAML 配置转换为可用于前端动态表单的 schema
    """
    def parse_section(section):
        schema = []
        for key, value in section.items():
            field = {"name": key}
            if isinstance(value, dict):
                # 嵌套结构
                field["type"] = "group"
                field["children"] = parse_section(value)
            elif isinstance(value, list):
                field["type"] = "list"
                field["default"] = value
            elif isinstance(value, bool):
                field["type"] = "boolean"
                field["default"] = value
            elif isinstance(value, (int, float)):
                field["type"] = "number"
                field["default"] = value
            elif isinstance(value, str) or value is None:
                field["type"] = "string"
                field["default"] = value
            else:
                field["type"] = "unknown"
                field["default"] = str(value)
            schema.append(field)
        return schema

    return parse_section(cfg)

# ============================================================
# 1. 读取 YAML（保留注释）
# ============================================================
def load_yaml_with_comments(path):
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.load(f)
    return data


# 提取 YAML 注释（字段右侧的 # xxx）
def extract_comments(node, parent_key=""):
    """
    遍历 YAML，将注释提取为：
    {
        "data.mode": "数据场景",
        "model.GNN.k": "图邻居数",
        ...
    }
    """
    comments = {}

    if not isinstance(node, dict):
        return comments

    for key, value in node.items():
        full_key = f"{parent_key}.{key}" if parent_key else key

        # ruamel.yaml 注释结构：ca.items[key][2] 是 key 后面的注释
        key_ca = node.ca.items.get(key)
        if key_ca and key_ca[2]:
            comment_text = key_ca[2].value.strip().lstrip("#").strip()
            comments[full_key] = comment_text

        # 递归
        if isinstance(value, dict):
            comments.update(extract_comments(value, full_key))

    return comments


# 扁平化 + 自动生成 schema
def yaml_to_schema_flatten(cfg, comments):
    """
    将 YAML 配置扁平化，并生成 schema，
    每个字段包含：
    - name
    - type
    - default
    - description (来自注释或自动生成)
    """
    flat_schema = []

    # 需要过滤掉的字段名
    EXCLUDE_FIELDS = {"feature_combinations"}

    def get_comment(full_key):
        return comments.get(full_key, f"{full_key} 配置项。")

    def flatten(section, parent_key=""):
        for key, value in section.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            comment = get_comment(full_key)
            full_key = key
            # ----------------------------
            # 过滤掉 feature_combinations
            # ----------------------------
            if key in EXCLUDE_FIELDS:
                continue

            description = get_comment(key)

            # ----------------------------
            # 两个特殊规则
            # ----------------------------
            if key == "img_size":
                flat_schema.append({
                    "name": key,
                    "type": "string",
                    "default": str(value),  # 直接变成 "[128,128]"
                    "value": str(value),
                    "description": description
                })
                continue

            if isinstance(value, dict):
                flatten(value, full_key)

            elif isinstance(value, list):
                flat_schema.append({
                    "name": full_key,
                    "type": "list",
                    "value": value,
                    "description": comment
                })

            elif isinstance(value, bool):
                flat_schema.append({
                    "name": full_key,
                    "type": "boolean",
                    "value": value,
                    "description": comment
                })

            elif isinstance(value, (int, float)):
                flat_schema.append({
                    "name": full_key,
                    "type": "number",
                    "value": value,
                    "description": comment
                })

            elif isinstance(value, str) or value is None:
                flat_schema.append({
                    "name": full_key,
                    "type": "string",
                    "value": value,
                    "description": comment
                })

            else:
                flat_schema.append({
                    "name": full_key,
                    "type": "unknown",
                    "value": str(value),
                    "description": comment
                })

    flatten(cfg)
    return flat_schema

# 生成扁平config模板
def generate_schema(yaml_path):
    """读取带注释的 YAML，并生成前端使用的扁平表单 Schema。"""
    yaml_data = load_yaml_with_comments(yaml_path)
    comments = extract_comments(yaml_data)
    schema = yaml_to_schema_flatten(yaml_data, comments)
    return schema

def collect_labels_from_split_dir(modality_dir):
    """
    modality_dir 结构:
    modality_dir/
        train/
            class1/
            class2/
        test/
            class1/
            class2/
    """
    labels = set()

    for split in ["train", "test"]:
        split_dir = os.path.join(modality_dir, split)
        if not os.path.isdir(split_dir):
            continue

        for entry in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, entry)
            if os.path.isdir(class_dir):
                labels.add(entry)

    return sorted(labels)

def insert_images_per_class(db, base_dir, dataset_id, dataset_name, modality, per_class_limit=10):
    """
    每个类别取固定数量的图片插入数据库
    适配结构：
    base_dir/
        train/
            class_xxx/
        test/
            class_xxx/
    """
    inserted_count = defaultdict(int)
    total_inserted = 0

    for split in ["train", "test"]:
        split_dir = os.path.join(base_dir, split)
        if not os.path.isdir(split_dir):
            continue

        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            count = 0
            for fname in os.listdir(class_dir):
                if count >= per_class_limit:
                    break

                if not fname.lower().endswith(IMAGE_EXTENSIONS):
                    continue

                abs_path = os.path.abspath(os.path.join(class_dir, fname))

                image = Image(
                    dataset_id=dataset_id,
                    name=dataset_name,
                    modality=modality,
                    path=abs_path,
                    label=class_name,
                    split=split
                )

                db.session.add(image)

                inserted_count[f"{split}_{class_name}"] += 1
                total_inserted += 1
                count += 1
    db.session.commit()

    print(f"完成！共插入 {total_inserted} 条记录，每类最多 {per_class_limit} 条")
    print(dict(inserted_count))

    # ✅ 建议加返回值
    return {
        "total_inserted": total_inserted,
        "detail": dict(inserted_count)
    }
