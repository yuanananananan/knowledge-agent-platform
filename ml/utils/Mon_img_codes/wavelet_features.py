import numpy as np
import pywt

import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize

def wavelet_decompose(imgs, wavelet='db4', level=3):
    coeffs = [pywt.wavedec2(im, wavelet=wavelet, level=level) for im in imgs]
    return coeffs

def wavelet_subband_energy(imgs):
    coeffs = wavelet_decompose(imgs)
    energies = []
    for cset in coeffs:
        e = []
        for c in cset[1:]:
            for band in c:
                e.append(np.sum(np.square(band)))
        energies.append(e)
    return np.array(energies)

def wavelet_subband_mean(imgs, wavelet='db4', level=3):
    coeffs = wavelet_decompose(imgs, wavelet, level)
    means = []
    for cset in coeffs:
        m = []
        for c in cset[1:]:
            for band in c:
                m.append(np.mean(band))
        means.append(m)
    return np.array(means)

def wavelet_subband_var(imgs, wavelet='db4', level=3):
    coeffs = wavelet_decompose(imgs, wavelet, level)
    vars_ = []
    for cset in coeffs:
        v = []
        for c in cset[1:]:
            for band in c:
                v.append(np.var(band))
        vars_.append(v)
    return np.array(vars_)

def wavelet_subband_skew(imgs, wavelet='db4', level=3):
    coeffs = wavelet_decompose(imgs, wavelet, level)
    skews = []
    for cset in coeffs:
        s = []
        for c in cset[1:]:
            for band in c:
                b = band.flatten()
                mean = np.mean(b)
                s.append(np.mean((b - mean)**3) / (np.std(b)**3 + 1e-8))
        skews.append(s)
    return np.array(skews)

def wavelet_subband_kurt(imgs, wavelet='db4', level=3):
    coeffs = wavelet_decompose(imgs, wavelet, level)
    kurts = []
    for cset in coeffs:
        k = []
        for c in cset[1:]:
            for band in c:
                b = band.flatten()
                k.append(np.mean((b - np.mean(b))**4) / (np.var(b)**2 + 1e-8))
        kurts.append(k)
    return np.array(kurts)

def wavelet_energy_entropy(imgs):
    energies = wavelet_subband_energy(imgs)
    p = energies / (np.sum(energies, axis=1, keepdims=True) + 1e-12)
    p = np.where(p > 0, p, 1)
    entropy = -np.sum(p * np.log2(p), axis=1)
    return entropy

def wavelet_highfreq_lowfreq_ratio(imgs):
    coeffs = wavelet_decompose(imgs)
    ratios = []
    for cset in coeffs:
        low_energy = np.sum(np.square(cset[0]))
        high_energy = np.sum([np.sum(np.square(band)) for c in cset[1:] for band in c])
        ratios.append(high_energy / (low_energy + 1e-12))
    return np.array(ratios)

def wavelet_edge_density(imgs, threshold_ratio=0.2):
    coeffs = wavelet_decompose(imgs)
    densities = []
    for cset in coeffs:
        d = []
        for c in cset[1:]:
            for band in c:
                thresh = threshold_ratio * np.max(np.abs(band))
                d.append(np.sum(np.abs(band) > thresh) / band.size)
        densities.append(d)
    return np.array(densities)



class WaveletFeature2D(nn.Module):
    """
    小波变换域特征提取网络
    (多尺度卷积 + 全局池化)
    """
    def __init__(self, in_channels=4, out_channels=64):
        """
        in_channels = 4 (LL, LH, HL, HH 四个子带)
        """
        super().__init__()
        c1 = out_channels // 3
        c2 = out_channels // 3
        c3 = out_channels - c1 - c2

        self.conv3 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, c2, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, c3, kernel_size=7, padding=3)

        self.pool_avg = nn.AdaptiveAvgPool2d(1)  # 整体分布
        self.pool_max = nn.AdaptiveMaxPool2d(1)  # 主能量系数

    def forward(self, x):
        f3 = F.relu(self.conv3(x))
        f5 = F.relu(self.conv5(x))
        f7 = F.relu(self.conv7(x))
        feat = torch.cat([f3, f5, f7], dim=1)  # [B, C, H, W]

        avg_feat = self.pool_avg(feat).squeeze(-1).squeeze(-1)
        max_feat = self.pool_max(feat).squeeze(-1).squeeze(-1)

        return torch.cat([avg_feat, max_feat], dim=1)  # [B, 2*C]


class DeepWaveletExtractor:
    """
    小波变换域深度特征提取器
    """
    def __init__(self, device=None, out_channels=64, batch_size=4, wavelet="db1"):
        self.device = "cuda:0" if (device is None and torch.cuda.is_available()) else "cpu"
        self.batch_size = batch_size
        self.wavelet = wavelet
        self.model = WaveletFeature2D(in_channels=4, out_channels=out_channels).to(self.device)
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
        H, W = 224, 224
        tensor_imgs = []
        for img in images:
            # 转灰度 & 归一化
            if img.ndim == 3:
                gray = np.mean(img, axis=2).astype(np.float32) / 255.0
            else:
                gray = img.astype(np.float32) / 255.0

            # Resize
            gray_resized = resize(gray, (H, W), mode='reflect', anti_aliasing=True)

            # 小波变换 (单层分解)
            coeffs2 = pywt.dwt2(gray_resized, self.wavelet)
            LL, (LH, HL, HH) = coeffs2
            # 拼接 4 通道 (LL, LH, HL, HH)
            wavelet_img = np.stack([LL, LH, HL, HH], axis=0).astype(np.float32)

            tensor_imgs.append(torch.from_numpy(wavelet_img))

        tensor_imgs = torch.stack(tensor_imgs).to(self.device)  # [B, 4, H/2, W/2]
        with torch.no_grad():
            feat_vector = self.model(tensor_imgs)
        return feat_vector.cpu().numpy()


def deep_wavelet_features(images, out_channels=8, batch_size=4, wavelet="db1"):
    extractor = DeepWaveletExtractor(out_channels=out_channels, batch_size=batch_size, wavelet=wavelet)
    return extractor.extract_batch(images)