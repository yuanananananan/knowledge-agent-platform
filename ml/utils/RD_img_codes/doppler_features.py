import numpy as np
from scipy.signal import find_peaks
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.transform import resize

def doppler_spectrum_energy(rd_image):
    return np.sum(rd_image**2, axis=(1))

def doppler_frequency_shift(rd_image, center_frequency=10e9, sampling_rate=3072 / 20e-6):
    fft_freqs = np.fft.fftfreq(rd_image.shape[1], d=1/sampling_rate)
    peak_indices = np.argmax(rd_image, axis=1)
    peak_freqs = fft_freqs[peak_indices]
    return peak_freqs - center_frequency

def doppler_spectrum_centroid(rd_image, sampling_rate=3072 / 20e-6):
    fft_freqs = np.fft.fftfreq(rd_image.shape[1], d=1/sampling_rate)
    centroid = np.sum(rd_image * fft_freqs[:, None], axis=1) / np.sum(rd_image, axis=1)
    return centroid

def doppler_spectrum_entropy(rd_image):
    magnitude = rd_image / np.sum(rd_image, axis=1, keepdims=True)  # 归一化
    entropy = -np.sum(magnitude * np.log2(magnitude + np.finfo(float).eps), axis=1)
    return entropy

def doppler_spectrum_peak_count(rd_image):
    N, H, W = rd_image.shape
    counts = np.zeros((N, H), dtype=int)

    for n in range(N):
        for i in range(H):
            row = rd_image[n, i, :]       # 第 n 张图的第 i 行
            peaks, _ = find_peaks(row)    # 找峰
            counts[n, i] = len(peaks)     # 峰的数量
    return counts






class FrequencyAttention(nn.Module):
    """
    频率注意力模块
    根据多普勒频率方向 (W) 自适应加权
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1)

    def forward(self, x):
        # x: [B, C, H, W]
        freq_feat = torch.mean(x, dim=2)  # 沿距离方向池化 -> [B, C, W]
        attn = F.relu(self.fc1(freq_feat))
        attn = torch.sigmoid(self.fc2(attn))  # [B, C, W]
        attn = attn.unsqueeze(2)  # [B, C, 1, W]
        return x * attn


class DopplerFeature2D(nn.Module):
    """
    多普勒特征提取网络 (多尺度卷积 + 频率注意力 + 能量/峰值池化)
    """
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        # 按比例分配通道
        c1 = out_channels // 3
        c2 = out_channels // 3
        c3 = out_channels - c1 - c2

        # 多尺度卷积 (不同带宽的多普勒谱)
        self.conv3 = nn.Conv2d(in_channels, c1, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(in_channels, c2, kernel_size=(5, 5), padding=2)
        self.conv7 = nn.Conv2d(in_channels, c3, kernel_size=(7, 7), padding=3)

        # 频率方向注意力
        self.freq_attn = FrequencyAttention(out_channels)

        # 能量池化 + 峰值池化
        self.pool_energy = nn.AdaptiveAvgPool2d(1)  # 对应能量/质心
        self.pool_peak = nn.AdaptiveMaxPool2d(1)    # 对应主峰频率

    def forward(self, x):
        f3 = F.relu(self.conv3(x))
        f5 = F.relu(self.conv5(x))
        f7 = F.relu(self.conv7(x))
        feat = torch.cat([f3, f5, f7], dim=1)  # [B, C, H, W]

        # 频率注意力
        feat = self.freq_attn(feat)

        # 分别做能量池化 & 峰值池化
        energy_feat = self.pool_energy(feat).squeeze(-1).squeeze(-1)
        peak_feat = self.pool_peak(feat).squeeze(-1).squeeze(-1)

        return torch.cat([energy_feat, peak_feat], dim=1)  # [B, 2*C]


class DeepDopplerExtractor:
    """
    多普勒深度特征提取器
    """
    def __init__(self, device=None, out_channels=64, batch_size=4):
        self.device = "cuda:0" if (device is None and torch.cuda.is_available()) else "cpu"
        self.batch_size = batch_size
        self.model = DopplerFeature2D(in_channels=1, out_channels=out_channels).to(self.device)
        self.model.eval()

    def extract_batch(self, rd_images):
        """
        rd_images: List[np.ndarray], 每个为 [H, W] 的 RD 图
        """
        N = len(rd_images)
        feat_list = []

        for i in range(0, N, self.batch_size):
            mini_batch = rd_images[i:i+self.batch_size]
            feat_mini = self._extract_mini_batch(mini_batch)
            feat_list.append(feat_mini)
            torch.cuda.empty_cache()

        return np.vstack(feat_list)

    def _extract_mini_batch(self, rd_images):
        H, W = 128, 128  # 统一 RD 图大小
        tensor_imgs = []
        for img in rd_images:
            img_norm = img.astype(np.float32)
            img_norm /= (np.max(img_norm) + 1e-6)  # 归一化
            img_resized = resize(img_norm, (H, W), mode='reflect', anti_aliasing=True)
            tensor_imgs.append(torch.from_numpy(img_resized))

        tensor_imgs = torch.stack(tensor_imgs).unsqueeze(1).to(self.device)  # [B,1,H,W]
        with torch.no_grad():
            feat_vector = self.model(tensor_imgs)
        return feat_vector.cpu().numpy()


def deep_doppler_features(rd_images, out_channels=8, batch_size=4):
    extractor = DeepDopplerExtractor(out_channels=out_channels, batch_size=batch_size)
    return extractor.extract_batch(rd_images)
