import os
import torch
import cv2
import copy
import numpy as np
import albumentations as A
from torch.utils.data import Dataset, random_split
from PIL import Image, ImageOps
from torchvision import transforms
from skimage.segmentation import slic
from utils.FeatureExtractor_2D import FeatureExtractor
import torch.nn.functional as F
from utils.get_feature import get_outra_feature


def resize_and_pad_image(img, target_size=(128, 128)):
    original_width, original_height = img.size

    aspect_ratio = original_width / original_height
    if aspect_ratio > 1:  # 宽度大于高度
        new_width = target_size[0]
        new_height = int(target_size[0] / aspect_ratio)
    else:  # 高度大于宽度
        new_height = target_size[1]
        new_width = int(target_size[1] * aspect_ratio)

    img_resized = img.resize((new_width, new_height))

    padding_top = (target_size[1] - new_height) // 2
    padding_bottom = target_size[1] - new_height - padding_top
    padding_left = (target_size[0] - new_width) // 2
    padding_right = target_size[0] - new_width - padding_left

    img_padded = ImageOps.expand(img_resized, (padding_left, padding_top, padding_right, padding_bottom), fill=0)

    return img_padded


class MON_Dataset(Dataset):

    def __init__(self, root_dir, transform=None, img_size=(128, 128), is_aug=True, cfg=None, is_train=True):
        self.cfg = cfg
        self.root_dir = root_dir
        self.transform = transform

        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.img_size = tuple(img_size)
        self.is_aug = is_aug

        # 获取所有类别的文件夹名
        classes = sorted(os.listdir(root_dir))
        self.classes = [cls for cls in classes if not '.' in cls]
        print(f"Classes found: {self.classes}")
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # 获取所有图像文件路径
        self.img_paths = []
        self.labels = []

        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            if os.path.isdir(class_dir):
                # 使用os.walk遍历所有子目录
                for root, _, files in os.walk(class_dir):
                    for img_name in files:
                        img_path = os.path.join(root, img_name)
                        if img_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):  # 确保是图像文件
                            self.img_paths.append((img_path, self.class_to_idx[cls]))
                            # self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        """返回数据集的大小"""
        return len(self.img_paths)

    def __getitem__(self, idx):
        """返回指定索引的图像和标签"""
        img_path, label = self.img_paths[idx]

        # 打开图像
        img_source = Image.open(img_path).convert('L')
        aug_transforms = A.Compose([
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            # A.RandomRotate90(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.03,
                               scale_limit=0.05,
                               rotate_limit=10,
                               border_mode=cv2.BORDER_REFLECT,
                               p=0.3),
            # A.OneOf([A.RandomGamma(gamma_limit=(90, 110), p=0.5),
            #          A.GaussNoise(var_limit=20.0, p=0.5)], p=0.3),
            # A.RandomResizedCrop(128, 128, scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=cv2.INTER_LANCZOS4),
        ])
        if self.transform:
            if self.is_train and self.is_aug:
                img = aug_transforms(image=np.array(img_source))["image"]
            else:
                img = img_source

            img = np.array(img)
            img = cv2.resize(img, self.img_size)
            data = img.astype(np.float32) / 255.
            feature = get_outra_feature(data, self.cfg)
            img = self.transform(img)
            img = img.repeat(3, 1, 1).float()
        return img, label, feature


def get_MON_dataset(cfg):
    img_size = cfg['data']['img_size']
    root_dir = cfg['data']['root']
    train_ratio = cfg['data']['train_ratio']
    is_aug = cfg['data']['is_aug']
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Resize(img_size, antialias=True),  # 调整大小
    ])

    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'test')
    train_dataset = MON_Dataset(
        root_dir=train_dir,
        transform=transform,
        img_size=img_size,
        is_aug=is_aug,
        cfg=cfg,
        is_train=True,
    )
    train_dataset, val_dataset = random_split(
        train_dataset,
        [int(train_ratio * len(train_dataset)),
         len(train_dataset) - int(train_ratio * len(train_dataset))],
        generator=torch.Generator().manual_seed(42))
    test_dataset = MON_Dataset(
        root_dir=test_dir,
        transform=transform,
        img_size=img_size,
        is_aug=False,
        cfg=cfg,
        is_train=False,
    )
    return train_dataset, test_dataset
