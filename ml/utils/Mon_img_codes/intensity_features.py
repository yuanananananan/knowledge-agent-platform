import cv2
import numpy as np
from scipy.stats import skew
from skimage.measure import label
from joblib import Parallel, delayed
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize


# ======================
# 顶层处理函数
# ======================
def process_gray_histogram(img, bins):
    img = img.astype(np.uint8)
    hist = cv2.calcHist([img], [0], None, [bins], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def process_gray_moments(img):
    img = img.astype(np.uint8)
    pixels = img.flatten()
    mean = np.mean(pixels)
    std = np.std(pixels)
    skewness = skew(pixels)
    return np.array([mean, std, skewness])


def process_gray_ccv(img, bins, threshold):
    img = img.astype(np.uint8)
    quantized = (img // (256 // bins)).astype(np.uint8)
    total_pixels = img.size
    ccv = []
    for gray_val in range(bins):
        mask = (quantized == gray_val)
        labeled, num = label(mask, connectivity=1, return_num=True)
        coherent = 0
        incoherent = 0
        for i in range(1, num + 1):
            size = np.sum(labeled == i)
            if size >= threshold:
                coherent += size
            else:
                incoherent += size
        ccv.extend([coherent / total_pixels, incoherent / total_pixels])
    return np.array(ccv)


def process_gray_blocks_histogram(img, grid, bins):
    img = img.astype(np.uint8)
    h, w = img.shape
    bh, bw = h // grid[0], w // grid[1]
    feats = []
    for i in range(grid[0]):
        for j in range(grid[1]):
            block = img[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            hist = cv2.calcHist([block], [0], None, [bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            feats.extend(hist)
    return np.array(feats)


def process_gray_autocorrelogram(img, distance, bins):
    img = img.astype(np.uint8)
    quantized = (img // (256 // bins)).astype(np.uint8)
    h, w = quantized.shape
    correlogram = np.zeros(bins)
    offsets = [(0, distance), (distance, 0), (0, -distance), (-distance, 0)]
    for gray_val in range(bins):
        mask = (quantized == gray_val)
        ys, xs = np.where(mask)
        count = 0
        match = 0
        for (x, y) in zip(xs, ys):
            for dx, dy in offsets:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    count += 1
                    if quantized[ny, nx] == gray_val:
                        match += 1
        correlogram[gray_val] = match / count if count > 0 else 0
    return correlogram


# ======================
# 包装函数（可并行）
# ======================
def gray_histogram(images, bins=64, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(process_gray_histogram)(im, bins) for im in images))


def gray_moments(images, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(process_gray_moments)(im) for im in images))


def gray_ccv(images, bins=16, threshold=50, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(process_gray_ccv)(im, bins, threshold) for im in images))


def gray_blocks_histogram(images, grid=(4, 4), bins=8, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(process_gray_blocks_histogram)(im, grid, bins) for im in images))


def gray_autocorrelogram(images, distance=1, bins=16, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(process_gray_autocorrelogram)(im, distance, bins) for im in images))


# ----------------- 网络定义 -----------------
class RandomGrayFeatureNet(nn.Module):
    """
    灰度图特征提取网络
    输出全局描述子向量，可用于检索或匹配
    """
    def __init__(self, output_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 下采样 H/2,W/2
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 下采样 H/4,W/4
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 下采样 H/8,W/8
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        # x: [B,1,H,W]
        feat = self.encoder(x)
        pooled = self.pool(feat).view(x.size(0), -1)  # [B, 256]
        out = F.normalize(self.fc(pooled), p=2, dim=1)  # L2 归一化
        return out  # [B, output_dim]


# ----------------- 封装接口 -----------------
class DeepGrayFeatureExtractor:
    def __init__(self, device=None, output_dim=256, batch_size=4):
        self.device = "cuda:0" if (device is None and torch.cuda.is_available()) else "cpu"
        self.batch_size = batch_size
        self.model = RandomGrayFeatureNet(output_dim=output_dim).to(self.device)
        self.model.eval()

    def extract_batch(self, images):
        """
        images: List[np.ndarray], 每张图 [H,W] 或 [H,W,3]
        返回: np.ndarray [N, output_dim]
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
        H, W = 240, 320
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
            feat_vector = self.model(tensor_imgs)  # [B, output_dim]

        return feat_vector.cpu().numpy()


# ----------------- 使用示例 -----------------
def deep_gray_features(images, output_dim=16, batch_size=4):
    extractor = DeepGrayFeatureExtractor(output_dim=output_dim, batch_size=batch_size)
    return extractor.extract_batch(images)