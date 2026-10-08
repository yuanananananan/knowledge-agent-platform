import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize

def fft2_image(imgs):
    """
    imgs: np.array (N,H,W)
    return: magnitude (N,H,W), phase (N,H,W)
    """
    mags = []
    phases = []
    for img in imgs:
        F = np.fft.fftshift(np.fft.fft2(img))
        mags.append(np.abs(F))
        phases.append(np.angle(F))
    return np.array(mags), np.array(phases)

def fft_energy(imgs):
    mags, phases = fft2_image(imgs)
    return np.sum(mags**2, axis=(1,2))

def fft_centroid(imgs):
    mags, phases = fft2_image(imgs)
    N, h, w = mags.shape
    Y, X = np.indices((h, w))
    total_energy = np.sum(mags, axis=(1,2), keepdims=True)
    cx = np.sum(X * mags, axis=(1,2)) / (total_energy.squeeze() + 1e-12)
    cy = np.sum(Y * mags, axis=(1,2)) / (total_energy.squeeze() + 1e-12)
    return np.stack((cx, cy), axis=1)

def fft_inertia(imgs):
    mags, phases = fft2_image(imgs)
    N, h, w = mags.shape
    Y, X = np.indices((h, w))
    total_energy = np.sum(mags, axis=(1, 2), keepdims=True)
    cx = np.sum(X * mags, axis=(1, 2)) / (total_energy.squeeze() + 1e-12)
    cy = np.sum(Y * mags, axis=(1, 2)) / (total_energy.squeeze() + 1e-12)

    cx = cx[:, None, None]
    cy = cy[:, None, None]
    inertia = np.sum(((X - cx)**2 + (Y - cy)**2) * mags, axis=(1,2)) / (np.sum(mags, axis=(1,2)) + 1e-12)
    return inertia

def fft_mean(imgs):
    mags, phases = fft2_image(imgs)
    m = mags.reshape(mags.shape[0], -1)
    mean = np.mean(m, axis=1)
    return mean

def fft_var(imgs):
    mags, phases = fft2_image(imgs)
    m = mags.reshape(mags.shape[0], -1)
    var = np.var(m, axis=1)
    return var

def fft_skew(imgs):
    mags, phases = fft2_image(imgs)
    m = mags.reshape(mags.shape[0], -1)
    mean = np.mean(m, axis=1)
    skew = np.mean((m - mean[:, None])**3, axis=1) / (np.std(m, axis=1)**3 + 1e-8)
    return skew

def fft_kurt(imgs):
    mags, phases = fft2_image(imgs)
    m = mags.reshape(mags.shape[0], -1)
    mean = np.mean(m, axis=1)
    kurt = np.mean((m - mean[:, None])**4, axis=1) / (np.var(m, axis=1)**2 + 1e-8)
    return kurt

def fft_entropy(imgs):
    mags, phases = fft2_image(imgs)
    m = mags.reshape(mags.shape[0], -1)
    p = m / (np.sum(m, axis=1, keepdims=True) + 1e-12)
    p = np.where(p > 0, p, 1)
    entropy = -np.sum(p * np.log2(p), axis=1)
    return entropy

def fft_directional_energy_ratio(imgs, angle_bins=8):
    mags, phases = fft2_image(imgs)
    N, h, w = mags.shape
    cy, cx = h // 2, w // 2
    Y, X = np.indices((h, w))
    angles = np.arctan2(Y - cy, X - cx)
    bins = np.linspace(-np.pi, np.pi, angle_bins + 1)
    features = []
    for i in range(N):
        energies = []
        for b in range(angle_bins):
            mask = (angles >= bins[b]) & (angles < bins[b+1])
            energies.append(np.sum(mags[i][mask]))
        energies = np.array(energies)
        energies /= np.sum(energies) + 1e-12
        features.append(energies)
    return np.array(features)


class FourierFeature2D(nn.Module):
    """
    傅里叶变换域特征提取网络
    (多尺度卷积 + 全局池化)
    """
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        # 通道划分
        c1 = out_channels // 3
        c2 = out_channels // 3
        c3 = out_channels - c1 - c2

        # 多尺度卷积
        self.conv3 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, c2, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, c3, kernel_size=7, padding=3)

        # 全局池化
        self.pool_avg = nn.AdaptiveAvgPool2d(1)  # 频谱整体分布
        self.pool_max = nn.AdaptiveMaxPool2d(1)  # 主频峰值

    def forward(self, x):
        # 多尺度卷积
        f3 = F.relu(self.conv3(x))
        f5 = F.relu(self.conv5(x))
        f7 = F.relu(self.conv7(x))
        feat = torch.cat([f3, f5, f7], dim=1)  # [B, C, H, W]

        # 全局特征 (平均 + 最大)
        avg_feat = self.pool_avg(feat).squeeze(-1).squeeze(-1)
        max_feat = self.pool_max(feat).squeeze(-1).squeeze(-1)

        return torch.cat([avg_feat, max_feat], dim=1)  # [B, 2*C]


class DeepFourierExtractor:
    """
    傅里叶变换域深度特征提取器
    """
    def __init__(self, device=None, out_channels=64, batch_size=4):
        self.device = "cuda:0" if (device is None and torch.cuda.is_available()) else "cpu"
        self.batch_size = batch_size
        self.model = FourierFeature2D(in_channels=1, out_channels=out_channels).to(self.device)
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

    def _extract_mini_batch(self, images, img_size = (224, 224)):
        H, W = img_size
        tensor_imgs = []
        for img in images:
            # 转灰度 & 归一化
            if img.ndim == 3:
                gray = np.mean(img, axis=2).astype(np.float32) / 255.0
            else:
                gray = img.astype(np.float32) / 255.0

            # Resize
            gray_resized = resize(gray, (H, W), mode='reflect', anti_aliasing=True)

            # 傅里叶变换 -> 幅度谱
            fft_img = np.fft.fftshift(np.fft.fft2(gray_resized))
            fft_mag = np.log1p(np.abs(fft_img))  # log 取幅度谱
            fft_mag = fft_mag.astype(np.float32)

            tensor_imgs.append(torch.from_numpy(fft_mag))

        tensor_imgs = torch.stack(tensor_imgs).unsqueeze(1).to(self.device)  # [B,1,H,W]
        with torch.no_grad():
            feat_vector = self.model(tensor_imgs)
        return feat_vector.cpu().numpy()


def deep_fourier_features(images, out_channels=8, batch_size=4):
    extractor = DeepFourierExtractor(out_channels=out_channels, batch_size=batch_size)
    return extractor.extract_batch(images)