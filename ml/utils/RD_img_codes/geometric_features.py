import cv2
import numpy as np
from joblib import Parallel, delayed
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize


# ======================
# 形状几何特征
# ======================
def _area(img):
    _, binary_image = cv2.threshold(img.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    return cv2.contourArea(max(contours, key=cv2.contourArea))

def extract_area(images, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(delayed(_area)(im) for im in images))


def _perimeter(img):
    _, binary_image = cv2.threshold(img.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    return cv2.arcLength(max(contours, key=cv2.contourArea), True)

def extract_perimeter(images, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(delayed(_perimeter)(im) for im in images))


def _aspect_ratio(img):
    _, binary_image = cv2.threshold(img.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return float(w) / h if h != 0 else 0

def extract_aspect_ratio(images, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(delayed(_aspect_ratio)(im) for im in images))


def _rectangularity(img):
    _, binary_image = cv2.threshold(img.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    return area / rect_area if rect_area != 0 else 0

def extract_rectangularity(images, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(delayed(_rectangularity)(im) for im in images))


def _sphericity(img):
    _, binary_image = cv2.threshold(img.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    return (4 * np.pi * area) / (perimeter ** 2) if perimeter != 0 else 0

def extract_sphericity(images, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(delayed(_sphericity)(im) for im in images))


def _hu_moments(img):
    _, binary_image = cv2.threshold(img.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(7)
    cnt = max(contours, key=cv2.contourArea)
    moments = cv2.moments(cnt)
    huMoments = cv2.HuMoments(moments).flatten()
    return huMoments

def extract_hu_moments(images, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(delayed(_hu_moments)(im) for im in images))


# ======================
# 边缘特征
# ======================
def _canny(img, down_size, threshold1, threshold2):
    edges = cv2.Canny(img.astype(np.uint8), threshold1, threshold2)
    edges = cv2.resize(edges, down_size, interpolation=cv2.INTER_AREA)
    return edges.flatten()

def extract_canny_features(images, down_size=(8,8), threshold1=100, threshold2=200, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(_canny)(im, down_size, threshold1, threshold2) for im in images
    ))


def _sobel(img, down_size, ksize):
    gray = img.astype(np.uint8) if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    sobel_x = cv2.resize(sobel_x, down_size, interpolation=cv2.INTER_AREA)
    sobel_y = cv2.resize(sobel_y, down_size, interpolation=cv2.INTER_AREA)
    return np.hstack([sobel_x.flatten(), sobel_y.flatten()])

def extract_sobel_features(images, down_size=(8,8), ksize=3, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(_sobel)(im, down_size, ksize) for im in images
    ))


def _log(img, down_size, ksize, sigma):
    gray = img.astype(np.uint8) if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 生成 LoG 核
    gaussian_kernel = cv2.getGaussianKernel(ksize, sigma)
    log_kernel = -gaussian_kernel @ gaussian_kernel.T + (1 / (np.pi * sigma ** 4))
    log_feature = cv2.filter2D(gray, cv2.CV_64F, log_kernel)
    log_feature = cv2.resize(log_feature, down_size, interpolation=cv2.INTER_AREA)
    return log_feature.flatten()

def extract_log_features(images, down_size=(8,8), ksize=5, sigma=1.0, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(_log)(im, down_size, ksize, sigma) for im in images
    ))


# -------------------- 结合特征网络 --------------------
class ShapeTextureFeatureNet(nn.Module):
    """
    结合几何 + 纹理特征的深度网络
    (多尺度卷积 + 方向卷积 + 全局池化)
    """
    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()
        c1 = out_channels // 3
        c2 = out_channels // 3
        c3 = out_channels - c1 - c2

        # 多尺度卷积（面积 + 局部模式）
        self.conv3 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, c2, kernel_size=5, padding=2)

        # 方向卷积核（水平 / 垂直 / 对角）
        self.dir_h = nn.Conv2d(in_channels, c3, kernel_size=(1, 7), padding=(0, 3))
        self.dir_v = nn.Conv2d(in_channels, c3, kernel_size=(7, 1), padding=(3, 0))

        # 全局池化（面积/方向/紧致度）
        self.pool_avg = nn.AdaptiveAvgPool2d(1)  # 面积感知
        self.pool_max = nn.AdaptiveMaxPool2d(1)  # 方向/边界主特征

    def forward(self, x):
        f3 = F.relu(self.conv3(x))
        f5 = F.relu(self.conv5(x))
        fh = F.relu(self.dir_h(x))
        fv = F.relu(self.dir_v(x))

        # 合并特征
        feat = torch.cat([f3, f5, fh, fv], dim=1)  # [B,C,H,W]

        # 全局池化
        avg_feat = self.pool_avg(feat).squeeze(-1).squeeze(-1)  # 面积相关
        max_feat = self.pool_max(feat).squeeze(-1).squeeze(-1)  # 方向/紧致度

        out = torch.cat([avg_feat, max_feat], dim=1)
        return F.normalize(out, p=2, dim=1)  # [B, 2C]


# -------------------- 结合特征提取器 --------------------
class DeepShapeTextureExtractor:
    def __init__(self, device=None, out_channels=32, batch_size=4):
        self.device = "cuda:0" if (device is None and torch.cuda.is_available()) else "cpu"
        self.batch_size = batch_size
        self.model = ShapeTextureFeatureNet(in_channels=1, out_channels=out_channels).to(self.device)
        self.model.eval()

    def extract_batch(self, images):
        """
        images: List[np.ndarray], 每个为灰度图 [H,W] 或 RGB 图
        """
        N = len(images)
        feat_list = []

        for i in range(0, N, self.batch_size):
            mini_batch = images[i:i+self.batch_size]
            feat_mini = self._extract_mini_batch(mini_batch)
            feat_list.append(feat_mini)
            torch.cuda.empty_cache()

        return np.vstack(feat_list)

    def _extract_mini_batch(self, images):
        H, W = 128, 128
        tensor_imgs = []
        for img in images:
            if img.ndim == 3:
                img = np.mean(img, axis=2)  # 转灰度
            img_norm = img.astype(np.float32) / 255.0
            img_resized = resize(img_norm, (H, W), mode='reflect', anti_aliasing=True)
            tensor_imgs.append(torch.from_numpy(img_resized))

        tensor_imgs = torch.stack(tensor_imgs).unsqueeze(1).to(self.device)  # [B,1,H,W]

        with torch.no_grad():
            feat_vector = self.model(tensor_imgs)
        return feat_vector.cpu().numpy()


# -------------------- 调用接口 --------------------
def deep_geometry(images, out_channels=32, batch_size=4):
    extractor = DeepShapeTextureExtractor(out_channels=out_channels, batch_size=batch_size)
    return extractor.extract_batch(images)