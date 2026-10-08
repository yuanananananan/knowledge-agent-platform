import numpy as np
from joblib import Parallel, delayed
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize

# ======================
# 顶层工具函数
# ======================

def sample_entropy_1d(x, m=2, r=0.2):
    N = len(x)
    r *= np.std(x)

    def _phi(m):
        X = np.array([x[i:i + m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(X[:, None] - X[None, :]), axis=2) <= r, axis=0) - 1
        return np.sum(C) / (N - m + 1)

    return -np.log(_phi(m + 1) / _phi(m))


def permutation_entropy_1d(x, m=3, tau=1):
    n = len(x)
    patterns = {}
    for i in range(n - tau * (m - 1)):
        sorted_index_tuple = tuple(np.argsort(x[i:i + tau * m:tau]))
        patterns[sorted_index_tuple] = patterns.get(sorted_index_tuple, 0) + 1
    counts = np.array(list(patterns.values()))
    p = counts / counts.sum()
    return -np.sum(p * np.log(p))


def safe_sample_entropy_1d(x, m=2, r=0.15):
    N = len(x)
    r = max(r * np.std(x), 1e-8)

    def _phi(m):
        if N - m + 1 <= 0:
            return 1e-8
        X = np.array([x[i:i + m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(X[:, None] - X[None, :]), axis=2) <= r, axis=0) - 1
        return np.sum(C) / (N - m + 1)

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    if phi_m1 <= 0 or phi_m <= 0:
        return 0.0
    return -np.log(phi_m1 / phi_m)


def multiscale_entropy_1d(x, m=2, r=0.15, max_scale=10):
    length = len(x)
    safe_max_scale = min(max_scale, length // (m + 1))
    if safe_max_scale < 1:
        safe_max_scale = 1
    mse = []
    for scale in range(1, safe_max_scale + 1):
        coarse_len = length // scale * scale
        if coarse_len < m + 1:
            mse.append(0.0)
            continue
        cg = np.mean(x[:coarse_len].reshape(-1, scale), axis=1)
        value = safe_sample_entropy_1d(cg, m, r)
        if np.isnan(value):
            value = 0.0
        mse.append(value)
    while len(mse) < max_scale:
        mse.append(0.0)
    return np.array(mse)


def higuchi_fd_1d(x, kmax=10):
    N = len(x)
    L = []
    for k in range(1, kmax + 1):
        Lk = []
        for m in range(k):
            idxs = np.arange(m, N, k)
            Lm = np.sum(np.abs(np.diff(x[idxs]))) * (N - 1) / (len(idxs) * k)
            Lk.append(Lm)
        L.append(np.mean(Lk))
    lnL = np.log(L)
    lnk = np.log(1.0 / np.arange(1, kmax + 1))
    return np.polyfit(lnk, lnL, 1)[0]


def hurst_exponent_1d(x):
    N = len(x)
    T = np.arange(1, N + 1)
    Y = np.cumsum(x - np.mean(x))
    R = np.maximum.accumulate(Y) - np.minimum.accumulate(Y)
    S = np.array([np.std(x[:i + 1]) for i in range(N)])
    RS = R[1:] / (S[1:] + 1e-8)
    return np.polyfit(np.log(T[1:]), np.log(RS), 1)[0]


# ======================
# 包装函数（可并行）
# ======================

def extract_sample_entropy(images, m=2, r=0.2, n_jobs=-1, target_size=(16, 16)):
    images_resized = np.array([resize(im, target_size, anti_aliasing=True) for im in images])
    feats = Parallel(n_jobs=n_jobs)(
        delayed(sample_entropy_1d)(im.reshape(-1), m, r) for im in images_resized
    )
    return np.array(feats).reshape(len(images_resized), 1)


def extract_permutation_entropy(images, m=3, tau=1, n_jobs=-1, target_size=(32, 32)):
    images_resized = np.array([resize(im, target_size, anti_aliasing=True) for im in images])
    feats = Parallel(n_jobs=n_jobs)(
        delayed(permutation_entropy_1d)(im.reshape(-1), m, tau) for im in images_resized
    )
    return np.array(feats).reshape(len(images_resized), 1)


def extract_multiscale_entropy(images, m=2, r=0.15, max_scale=10, n_jobs=-1, target_size=(16, 16)):
    images_resized = np.array([resize(im, target_size, anti_aliasing=True) for im in images])
    feats = Parallel(n_jobs=n_jobs)(
        delayed(multiscale_entropy_1d)(im.reshape(-1), m, r, max_scale) for im in images_resized
    )
    return np.array(feats)


def extract_higuchi_fd(images, kmax=10, n_jobs=-1, target_size=(16, 16)):
    images_resized = np.array([resize(im, target_size, anti_aliasing=True) for im in images])
    feats = Parallel(n_jobs=n_jobs)(
        delayed(higuchi_fd_1d)(im.reshape(-1), kmax) for im in images_resized
    )
    return np.array(feats).reshape(len(images_resized), 1)


def extract_hurst_exponent(images, n_jobs=-1, target_size=(16, 16)):
    images_resized = np.array([resize(im, target_size, anti_aliasing=True) for im in images])
    feats = Parallel(n_jobs=n_jobs)(
        delayed(hurst_exponent_1d)(im.reshape(-1)) for im in images_resized
    )
    return np.array(feats).reshape(len(images_resized), 1)


class InceptionComplexity2D(nn.Module):
    """
    Inception 风格复杂度特征提取网络
    支持任意 out_channels
    """
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        # 分配各分支通道
        c1 = out_channels // 3
        c2 = out_channels // 3
        c3 = out_channels - c1 - c2  # 确保总和等于 out_channels

        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, c2, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(in_channels, c3, kernel_size=7, padding=3)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out3 = F.relu(self.conv3(x))
        out5 = F.relu(self.conv5(x))
        out = torch.cat([out1, out3, out5], dim=1)
        out = self.pool(out).squeeze(-1).squeeze(-1)
        return out


class DeepComplexityExtractorInception:
    def __init__(self, device=None, out_channels=64, batch_size=4):
        self.device = "cuda:0" if (device is None and torch.cuda.is_available()) else "cpu"
        self.batch_size = batch_size
        self.model = InceptionComplexity2D(in_channels=1, out_channels=out_channels).to(self.device)
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
        B = len(images)
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


def deep_complexity_features_inception(images, out_channels=8, batch_size=4):
    extractor = DeepComplexityExtractorInception(out_channels=out_channels, batch_size=batch_size)
    return extractor.extract_batch(images)