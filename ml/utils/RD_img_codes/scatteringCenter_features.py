import numpy as np
from scipy.ndimage import maximum_filter
from joblib import Parallel, delayed
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 散射中心检测 ----------------
def detect_scattering_centers(image, threshold=0.5, neighborhood_size=3):
    """
    检测散射中心（局部极大值）
    """
    local_max = (image == maximum_filter(image, size=neighborhood_size))
    detected_peaks = np.where(local_max & (image > threshold))
    centers = list(zip(detected_peaks[0], detected_peaks[1]))
    amplitudes = image[detected_peaks]
    return centers, amplitudes


# ---------------- 散射中心数量 ----------------
def num_centers_worker(img):
    centers, _ = detect_scattering_centers(img)
    return len(centers)


def num_centers(images, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(delayed(num_centers_worker)(im) for im in images))


# ---------------- 散射中心位置 ----------------
def centers_positions_worker(img, max_centers=32):
    centers, _ = detect_scattering_centers(img)
    features = np.zeros(max_centers * 2, dtype=np.float32)
    num = min(len(centers), max_centers)
    if num > 0:
        centers_array = np.array(centers[:num]).flatten()
        features[:num * 2] = centers_array
    return features


def centers_positions(images, max_centers=32, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(centers_positions_worker)(im, max_centers) for im in images))


# ---------------- 散射中心幅度 ----------------
def centers_amplitudes_worker(img, max_centers=32):
    _, amplitudes = detect_scattering_centers(img)
    amps = np.array(amplitudes[:max_centers])
    if len(amps) < max_centers:
        amps = np.pad(amps, (0, max_centers - len(amps)), 'constant')
    return amps.astype(np.float32)


def centers_amplitudes(images, max_centers=32, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(centers_amplitudes_worker)(im, max_centers) for im in images))


# ---------------- 散射中心加权质心 ----------------
def scattering_centroid_worker(img):
    centers, amplitudes = detect_scattering_centers(img)
    if len(centers) == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    centers = np.array(centers)
    amplitudes = np.array(amplitudes)
    sum_amp = np.sum(amplitudes)
    x_centroid = np.sum(centers[:, 0] * amplitudes) / sum_amp
    y_centroid = np.sum(centers[:, 1] * amplitudes) / sum_amp
    return np.array([x_centroid, y_centroid], dtype=np.float32)


def scattering_centroid(images, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(scattering_centroid_worker)(im) for im in images))


# ---------------- 散射中心空间分布标准差 ----------------
def scattering_std_worker(img):
    centers, amplitudes = detect_scattering_centers(img)
    if len(centers) == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    centers = np.array(centers)
    amplitudes = np.array(amplitudes)
    sum_amp = np.sum(amplitudes)
    x_centroid = np.sum(centers[:, 0] * amplitudes) / sum_amp
    y_centroid = np.sum(centers[:, 1] * amplitudes) / sum_amp
    std_x = np.sqrt(np.sum(amplitudes * (centers[:, 0] - x_centroid) ** 2) / sum_amp)
    std_y = np.sqrt(np.sum(amplitudes * (centers[:, 1] - y_centroid) ** 2) / sum_amp)
    return np.array([std_x, std_y], dtype=np.float32)


def scattering_std(images, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(scattering_std_worker)(im) for im in images))


# ---------------- 最大幅度 ----------------
def max_amplitude_worker(img):
    _, amplitudes = detect_scattering_centers(img)
    return float(np.max(amplitudes)) if len(amplitudes) > 0 else 0.0


def max_amplitude(images, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(max_amplitude_worker)(im) for im in images), dtype=np.float32)


# ---------------- 最小幅度 ----------------
def min_amplitude_worker(img):
    _, amplitudes = detect_scattering_centers(img)
    return float(np.min(amplitudes)) if len(amplitudes) > 0 else 0.0


def min_amplitude(images, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(min_amplitude_worker)(im) for im in images), dtype=np.float32)



class ScatteringFeature2D(nn.Module):
    """
    散射中心特征提取网络 (多尺度卷积 + 全局池化)
    支持任意 out_channels
    """
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        # 按比例分配通道
        c1 = out_channels // 3
        c2 = out_channels // 3
        c3 = out_channels - c1 - c2  # 确保总和等于 out_channels

        # 多尺度卷积核 -> 模拟不同尺度的散射点
        self.conv3 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, c2, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, c3, kernel_size=7, padding=3)

        # 全局池化
        self.pool_avg = nn.AdaptiveAvgPool2d(1)  # 模拟质心、分布信息
        self.pool_max = nn.AdaptiveMaxPool2d(1)  # 模拟最强散射中心

    def forward(self, x):
        # 多尺度卷积响应
        f3 = F.relu(self.conv3(x))
        f5 = F.relu(self.conv5(x))
        f7 = F.relu(self.conv7(x))
        feat = torch.cat([f3, f5, f7], dim=1)  # [B, C, H, W]

        # 散射中心特征 (平均池化 + 最大池化)
        avg_feat = self.pool_avg(feat).squeeze(-1).squeeze(-1)  # [B, C]
        max_feat = self.pool_max(feat).squeeze(-1).squeeze(-1)  # [B, C]

        return torch.cat([avg_feat, max_feat], dim=1)  # [B, 2*C]


class DeepScatteringExtractor:
    """
    散射中心深度特征提取器 (随机初始化, 无需训练)
    """
    def __init__(self, device=None, out_channels=64, batch_size=4):
        self.device = "cuda:0" if (device is None and torch.cuda.is_available()) else "cpu"
        self.batch_size = batch_size
        self.model = ScatteringFeature2D(in_channels=1, out_channels=out_channels).to(self.device)
        self.model.eval()

    def extract_batch(self, images):
        N = len(images)
        feat_list = []

        for i in range(0, N, self.batch_size):
            mini_batch = images[i:i+self.batch_size]
            feat_mini = self._extract_mini_batch(mini_batch)
            feat_list.append(feat_mini)
            torch.cuda.empty_cache()

        return np.vstack(feat_list)

    def _extract_mini_batch(self, images):
        H, W = 240, 320
        tensor_imgs = []
        for img in images:
            if img.ndim == 3:
                gray = np.mean(img, axis=2).astype(np.float32) / 255.0
            else:
                gray = img.astype(np.float32) / 255.0
            gray_resized = resize(gray, (H, W), mode='reflect', anti_aliasing=True)
            tensor_imgs.append(torch.from_numpy(gray_resized))
        tensor_imgs = torch.stack(tensor_imgs).unsqueeze(1).to(self.device)
        with torch.no_grad():
            feat_vector = self.model(tensor_imgs)
        return feat_vector.cpu().numpy()


def deep_scattering_features(images, out_channels=16, batch_size=4):
    extractor = DeepScatteringExtractor(out_channels=out_channels, batch_size=batch_size)
    return extractor.extract_batch(images)