import cv2
import numpy as np
from joblib import Parallel, delayed
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize


def _pad_or_truncate(arr, dim):
    """将数组扩展或截断到固定长度 dim."""
    if arr.size < dim:
        return np.pad(arr, (0, dim - arr.size), 'constant')
    else:
        return arr[:dim]


# ============ 各种局部函数搬到全局 ============

def _process_sift(img, descriptor_dim=64):
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.astype(np.uint8)
    _, descriptors = sift.detectAndCompute(gray, None)
    if descriptors is None:
        return np.zeros(descriptor_dim, dtype=np.float32)
    return _pad_or_truncate(descriptors.mean(axis=0), descriptor_dim).astype(np.float32)


def _process_orb(img, descriptor_dim=64):
    orb = cv2.ORB_create()
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.astype(np.uint8)
    _, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None:
        return np.zeros(descriptor_dim, dtype=np.float32)
    return _pad_or_truncate(descriptors.mean(axis=0), descriptor_dim).astype(np.float32)


def _process_akaze(img, descriptor_dim=64):
    akaze = cv2.AKAZE_create()
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.astype(np.uint8)
    _, descriptors = akaze.detectAndCompute(gray, None)
    if descriptors is None:
        return np.zeros(descriptor_dim, dtype=np.float32)
    return _pad_or_truncate(descriptors.mean(axis=0), descriptor_dim).astype(np.float32)


def _process_brisk(img, descriptor_dim=64):
    brisk = cv2.BRISK_create()
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.astype(np.uint8)
    _, descriptors = brisk.detectAndCompute(gray, None)
    if descriptors is None:
        return np.zeros(descriptor_dim, dtype=np.float32)
    return _pad_or_truncate(descriptors.mean(axis=0), descriptor_dim).astype(np.float32)


def _process_fast_brief(img, brief_descriptor_dim=64):
    try:
        fast = cv2.FastFeatureDetector_create()
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    except AttributeError:
        raise RuntimeError("需要安装 opencv-contrib-python 才能使用 BRIEF")

    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.astype(np.uint8)
    keypoints = fast.detect(gray, None)
    _, descriptors = brief.compute(gray, keypoints)
    if descriptors is None:
        return np.zeros(brief_descriptor_dim, dtype=np.float32)
    return _pad_or_truncate(descriptors.mean(axis=0), brief_descriptor_dim).astype(np.float32)


def _process_shitomasi(img, max_corners=50, descriptor_dim=64):
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.astype(np.uint8)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=10)
    if corners is None:
        return np.zeros(descriptor_dim, dtype=np.float32)
    coords = corners.reshape(-1, 2).flatten()
    return _pad_or_truncate(coords, descriptor_dim).astype(np.float32)


# ============ 提取器接口函数 ============

def extract_sift_features(images, descriptor_dim=64, n_jobs=-1):
    return np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(_process_sift)(im, descriptor_dim) for im in images
        )
    )


def extract_orb_features(images, descriptor_dim=64, n_jobs=-1):
    return np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(_process_orb)(im, descriptor_dim) for im in images
        )
    )


def extract_akaze_features(images, descriptor_dim=64, n_jobs=-1):
    return np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(_process_akaze)(im, descriptor_dim) for im in images
        )
    )


def extract_brisk_features(images, descriptor_dim=64, n_jobs=-1):
    return np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(_process_brisk)(im, descriptor_dim) for im in images
        )
    )


def extract_fast_brief_features(images, brief_descriptor_dim=64, n_jobs=-1):
    return np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(_process_fast_brief)(im, brief_descriptor_dim) for im in images
        )
    )


def extract_shitomasi_features(images, max_corners=50, descriptor_dim=64, n_jobs=-1):
    return np.array(
        Parallel(n_jobs=n_jobs)(
            delayed(_process_shitomasi)(im, max_corners, descriptor_dim) for im in images
        )
    )



# ----------------- 网络定义 -----------------
class RandomKeypointNet(nn.Module):
    def __init__(self, descriptor_dim=256):
        super().__init__()
        # 简单 CNN 特征提取器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),  # 240x320 -> 240x320
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        # 关键点热力图预测
        self.heatmap = nn.Conv2d(128, 1, 1)  # [B,1,H,W]
        # 特征描述子预测
        self.descriptor = nn.Conv2d(128, descriptor_dim, 1)  # [B,D,H,W]

    def forward(self, x):
        # x: [B,1,H,W]
        feat = self.encoder(x)
        heatmap = torch.sigmoid(self.heatmap(feat))  # 关键点概率
        desc = F.normalize(self.descriptor(feat), p=2, dim=1)  # L2 归一化
        return heatmap, desc

# ----------------- 封装接口 -----------------
class DeepKeypointExtractor:
    def __init__(self, device=None, max_keypoints=50, descriptor_dim=256, batch_size=4):
        self.device = "cuda:0" if (device is None and torch.cuda.is_available()) else "cpu"
        self.max_keypoints = max_keypoints
        self.batch_size = batch_size
        self.model = RandomKeypointNet(descriptor_dim=descriptor_dim).to(self.device)
        self.model.eval()

    def extract_batch(self, images):
        """
        images: List[np.ndarray], 每张图 [H,W] 或 [H,W,3]
        返回: np.ndarray [N, descriptor_dim]
        """
        N = len(images)
        feat_list = []

        for i in range(0, N, self.batch_size):
            mini_batch = images[i:i+self.batch_size]
            feat_mini = self._extract_mini_batch(mini_batch)
            feat_list.append(feat_mini)

            # 清理显存
            torch.cuda.empty_cache()

        return np.vstack(feat_list)

    def _extract_mini_batch(self, images):
        H, W = 224, 224
        D = self.model.descriptor.out_channels
        B = len(images)

        # 预处理
        tensor_imgs = []
        for img in images:
            if img.ndim == 3:
                gray = np.mean(img, axis=2).astype(np.float32) / 255.0
            else:
                gray = img.astype(np.float32) / 255.0
            gray_resized = resize(gray, (H, W), mode='reflect', anti_aliasing=True)
            tensor_imgs.append(torch.from_numpy(gray_resized))
        tensor_imgs = torch.stack(tensor_imgs).unsqueeze(1).to(self.device)  # [B,1,H,W]

        with torch.no_grad():
            heatmap, desc_map = self.model(tensor_imgs)  # [B,1,H,W], [B,D,H,W]

            # flatten H*W, 找 top-N
            heatmap_flat = heatmap.view(B, -1)  # [B, H*W]
            topk_vals, topk_idx = torch.topk(heatmap_flat, self.max_keypoints, dim=1)  # [B, max_keypoints]

            # 将 flat idx 转成 y,x
            W_ = desc_map.shape[3]
            topk_y = topk_idx // W_
            topk_x = topk_idx % W_

            # 批量索引取描述子
            batch_idx = torch.arange(B, device=self.device).view(-1,1).repeat(1, self.max_keypoints)
            topk_desc = desc_map[batch_idx, :, topk_y, topk_x]  # [B,D,max_keypoints]

            # 平均池化
            feat_vector = topk_desc.mean(dim=2)  # [B,D]

        return feat_vector.cpu().numpy()

# ----------------- 使用示例 -----------------
def deep_keypoint_features(images, max_keypoints=10):
    extractor = DeepKeypointExtractor(max_keypoints=max_keypoints)
    return extractor.extract_batch(images)