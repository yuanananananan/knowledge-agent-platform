import numpy as np
from joblib import Parallel, delayed
from skimage.transform import resize
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



# ----------------- 样本熵 -----------------
def _phi(x, m_, r):
    N = len(x)
    if N - m_ + 1 <= 0:
        return 1e-8
    X = np.array([x[i:i + m_] for i in range(N - m_ + 1)])
    C = np.sum(np.max(np.abs(X[:, None] - X[None, :]), axis=2) <= r, axis=0) - 1
    return np.sum(C) / (N - m_ + 1)

def sample_entropy_1d(x, m=2, r=0.2):
    N = len(x)
    r = max(r * np.std(x), 1e-8)  # 防止 r=0

    phi_m = _phi(x, m, r)
    phi_m1 = _phi(x, m + 1, r)
    if phi_m1 <= 0 or phi_m <= 0:
        return 0.0
    return -np.log(phi_m1 / phi_m)

def process_sample_entropy(img, m=2, r=0.2, target_size=(16,16)):
    img_resized = resize(img, target_size, anti_aliasing=True)
    return sample_entropy_1d(img_resized.flatten(), m, r)

def extract_sample_entropy(images, m=2, r=0.2, n_jobs=-1, target_size=(16,16)):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(process_sample_entropy)(im, m, r, target_size) for im in images
    )).reshape(len(images), 1)

# ----------------- 排列熵 -----------------
def permutation_entropy_1d(x, m=3, tau=1):
    n = len(x)
    patterns = {}
    for i in range(n - tau*(m-1)):
        sorted_index_tuple = tuple(np.argsort(x[i:i+tau*m:tau]))
        patterns[sorted_index_tuple] = patterns.get(sorted_index_tuple, 0) + 1
    counts = np.array(list(patterns.values()))
    p = counts / counts.sum()
    return -np.sum(p * np.log(p))

def process_permutation_entropy(img, m=3, tau=1, target_size=(16,16)):
    img_resized = resize(img, target_size, anti_aliasing=True)
    return permutation_entropy_1d(img_resized.flatten(), m, tau)

def extract_permutation_entropy(images, m=3, tau=1, n_jobs=-1, target_size=(16,16)):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(process_permutation_entropy)(im, m, tau, target_size) for im in images
    )).reshape(len(images), 1)

# ----------------- 多尺度熵 -----------------
def multiscale_entropy_1d(x, m=2, r=0.15, max_scale=5):
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
        value = sample_entropy_1d(cg, m, r)
        if np.isnan(value):
            value = 0.0
        mse.append(value)
    while len(mse) < max_scale:
        mse.append(0.0)
    return np.array(mse)

def process_multiscale_entropy(img, m=2, r=0.15, max_scale=5, target_size=(16,16)):
    img_resized = resize(img, target_size, anti_aliasing=True)
    return multiscale_entropy_1d(img_resized.flatten(), m, r, max_scale)

def extract_multiscale_entropy(images, m=2, r=0.15, max_scale=5, n_jobs=-1, target_size=(16,16)):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(process_multiscale_entropy)(im, m, r, max_scale, target_size) for im in images
    ))

# ----------------- 分形维数 -----------------
def higuchi_fd_1d(x, kmax=10):
    N = len(x)
    L = []
    for k in range(1, kmax+1):
        Lk = []
        for m in range(k):
            idxs = np.arange(m, N, k)
            Lm = np.sum(np.abs(np.diff(x[idxs]))) * (N-1)/(len(idxs)*k)
            Lk.append(Lm)
        L.append(np.mean(Lk))
    lnL = np.log(L)
    lnk = np.log(1.0/np.arange(1, kmax+1))
    return np.polyfit(lnk, lnL, 1)[0]

def process_higuchi_fd(img, kmax=10, target_size=(16,16)):
    img_resized = resize(img, target_size, anti_aliasing=True)
    return higuchi_fd_1d(img_resized.flatten(), kmax)

def extract_higuchi_fd(images, kmax=10, n_jobs=-1, target_size=(16,16)):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(process_higuchi_fd)(im, kmax, target_size) for im in images
    )).reshape(len(images), 1)

# ----------------- Hurst 指数 -----------------
def hurst_exponent_1d(x):
    N = len(x)
    T = np.arange(1, N+1)
    Y = np.cumsum(x - np.mean(x))
    R = np.maximum.accumulate(Y) - np.minimum.accumulate(Y)
    S = np.array([np.std(x[:i+1]) for i in range(N)])
    RS = R[1:] / (S[1:] + 1e-8)
    return np.polyfit(np.log(T[1:]), np.log(RS), 1)[0]

def process_hurst_exponent(img, target_size=(16,16)):
    img_resized = resize(img, target_size, anti_aliasing=True)
    return hurst_exponent_1d(img_resized.flatten())

def extract_hurst_exponent(images, n_jobs=-1, target_size=(16,16)):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(process_hurst_exponent)(im, target_size) for im in images
    )).reshape(len(images), 1)


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
        H, W = 240, 320
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


def deep_complexity_features_inception(images, out_channels=64, batch_size=4):
    extractor = DeepComplexityExtractorInception(out_channels=out_channels, batch_size=batch_size)
    return extractor.extract_batch(images)