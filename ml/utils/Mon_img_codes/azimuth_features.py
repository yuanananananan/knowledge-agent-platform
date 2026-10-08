import numpy as np
from scipy.signal import find_peaks

def azimuth_energy_profile(images):
    """
    批量方位能量分布 (Azimuth Energy Profile)
    输入: images (N, H, W)
    输出: profiles (N, H) 每行是每个 range bin 的方位能量
    """
    return np.sum(images, axis=2)  # W 轴求和 -> (N, H)


def azimuth_centroid(images):
    """
    批量方位质心 (Azimuth Centroid)
    输入: images (N, H, W)
    输出: centroids (N, H) 每行是对应 range bin 的质心
    """
    N, H, W = images.shape
    x = np.arange(W)
    centroids = np.zeros((N, H), dtype=float)
    for n in range(N):
        for h in range(H):
            profile = images[n, h, :]
            centroids[n, h] = np.sum(profile * x) / (np.sum(profile) + 1e-12)
    return centroids


def azimuth_spread(images):
    """
    批量方位展宽度 (Azimuth Spread)
    输入: images (N, H, W)
    输出: spreads (N, H) 每行是对应 range bin 的展宽度
    """
    N, H, W = images.shape
    x = np.arange(W)
    spreads = np.zeros((N, H), dtype=float)
    centroids = azimuth_centroid(images)
    for n in range(N):
        for h in range(H):
            profile = images[n, h, :]
            c = centroids[n, h]
            spreads[n, h] = np.sqrt(np.sum(((x - c) ** 2) * profile) / (np.sum(profile) + 1e-12))
    return spreads


def azimuth_spectral_entropy(images):
    """
    批量方位谱熵 (Azimuth Spectral Entropy)
    输入: images (N, H, W)
    输出: entropies (N, H) 每行是对应 range bin 的谱熵
    """
    N, H, W = images.shape
    entropies = np.zeros((N, H), dtype=float)
    for n in range(N):
        for h in range(H):
            profile = images[n, h, :]
            profile = profile / (np.sum(profile) + 1e-12)
            entropies[n, h] = -np.sum(profile * np.log2(profile + 1e-12))
    return entropies


def azimuth_peak_count(images):
    """
    批量方位峰数 (Azimuth Peak Count)
    输入: images (N, H, W)
    输出: peak_counts (N, H) 每行是对应 range bin 的峰值数量
    """
    N, H, W = images.shape
    counts = np.zeros((N, H), dtype=int)
    for n in range(N):
        for h in range(H):
            profile = images[n, h, :]
            peaks, _ = find_peaks(profile)
            counts[n, h] = len(peaks)
    return counts


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.transform import resize


# -------------------- 简化方位特征网络 --------------------
class SimpleAzimuthNet(nn.Module):
    """
    简化版方位特征网络
    通过几个方向卷积核捕捉水平、垂直、对角线的方位信息
    """
    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()
        c_each = out_channels // 3

        # 基本方向卷积
        self.conv_h = nn.Conv2d(in_channels, c_each, kernel_size=(1, 5), padding=(0, 2))  # 水平
        self.conv_v = nn.Conv2d(in_channels, c_each, kernel_size=(5, 1), padding=(2, 0))  # 垂直
        self.conv_d = nn.Conv2d(in_channels, out_channels - 2 * c_each, kernel_size=3, padding=1)  # 对角线

        # 全局池化
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        fh = F.relu(self.conv_h(x))
        fv = F.relu(self.conv_v(x))
        fd = F.relu(self.conv_d(x))

        feat = torch.cat([fh, fv, fd], dim=1)  # [B, C, H, W]

        avg_feat = self.pool_avg(feat).squeeze(-1).squeeze(-1)
        max_feat = self.pool_max(feat).squeeze(-1).squeeze(-1)

        out = torch.cat([avg_feat, max_feat], dim=1)
        return F.normalize(out, p=2, dim=1)  # [B, 2C]


# -------------------- 方位特征提取器 --------------------
class DeepAzimuthExtractor:
    def __init__(self, device=None, out_channels=32, batch_size=4):
        self.device = "cuda:0" if (device is None and torch.cuda.is_available()) else "cpu"
        self.batch_size = batch_size
        self.model = SimpleAzimuthNet(in_channels=1, out_channels=out_channels).to(self.device)
        self.model.eval()

    def extract_batch(self, images):
        """
        images: List[np.ndarray], 每个为 2D 灰度图像
        """
        N = len(images)
        feat_list = []

        for i in range(0, N, self.batch_size):
            mini_batch = images[i:i + self.batch_size]
            feat_mini = self._extract_mini_batch(mini_batch)
            feat_list.append(feat_mini)
            torch.cuda.empty_cache()

        return np.vstack(feat_list)

    def _extract_mini_batch(self, images):
        H, W = 128, 128
        tensor_imgs = []
        for img in images:
            if img.ndim == 3:  # 转灰度
                img = np.mean(img, axis=2)
            img_norm = img.astype(np.float32) / 255.0
            img_resized = resize(img_norm, (H, W), mode='reflect', anti_aliasing=True)
            tensor_imgs.append(torch.from_numpy(img_resized))
        tensor_imgs = torch.stack(tensor_imgs).unsqueeze(1).to(self.device)  # [B,1,H,W]

        with torch.no_grad():
            feat_vector = self.model(tensor_imgs)
        return feat_vector.cpu().numpy()


# -------------------- 调用接口 --------------------
def deep_azimuth_features(images, out_channels=16, batch_size=4):
    extractor = DeepAzimuthExtractor(out_channels=out_channels, batch_size=batch_size)
    return extractor.extract_batch(images)
