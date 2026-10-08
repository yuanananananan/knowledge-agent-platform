import os
import random
import shutil

root_dir = "/media/4TB/GJ/TS_V1/dataset/dataset_0420/image"
output_dir = "/media/4TB/GJ/TS/splitedData/RD_data_0815"
os.makedirs(output_dir, exist_ok=True)

channels = ["Shh", "Shv", "Svh", "Svv"]

# 收集所有样本
samples = []  # (class_name, filename, {channel: filepath})
for cls in ["ship", "jf"]:
    cls_path = os.path.join(root_dir, cls)
    for sub_cls in os.listdir(cls_path):
        sub_cls_path = os.path.join(cls_path, sub_cls)
        if not os.path.isdir(sub_cls_path):
            continue

        sample_files = os.listdir(os.path.join(sub_cls_path, "Shh"))
        for f in sample_files:
            sample_info = {}
            has_all = True
            for ch in channels:
                ch_path = os.path.join(sub_cls_path, ch, f)
                if os.path.exists(ch_path):
                    sample_info[ch] = ch_path
                else:
                    has_all = False
                    break
            if has_all:
                samples.append((cls, f, sample_info))

print(f"总样本数: {len(samples)}")

# 划分
random.seed(42)
test_samples = random.sample(samples, 100)
train_samples = [s for s in samples if s not in test_samples]


# 复制
def copy_samples(sample_list, split_name):
    for cls, fname, ch_dict in sample_list:
        for ch, src_path in ch_dict.items():
            dst_dir = os.path.join(output_dir, split_name, "image", cls, ch)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(src_path, os.path.join(dst_dir, src_path.split('/')[9] + '_' + fname))


copy_samples(train_samples, "train")
copy_samples(test_samples, "test")

print(f"训练集: {len(train_samples)}，测试集: {len(test_samples)}")

