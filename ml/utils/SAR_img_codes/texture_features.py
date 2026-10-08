import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor
from joblib import Parallel, delayed
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.transform import resize

# ---------------- GLCM ----------------
def glcm_prop_single(img, prop, distances, angles, levels):
    glcm = graycomatrix(img.astype(np.uint8),
                        distances=distances,
                        angles=angles,
                        levels=levels,
                        symmetric=True,
                        normed=True)
    return graycoprops(glcm, prop).mean()

def glcm_contrast(images, distances=[1], angles=[0], levels=256, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(glcm_prop_single)(im, 'contrast', distances, angles, levels) for im in images
    ))

def glcm_dissimilarity(images, distances=[1], angles=[0], levels=256, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(glcm_prop_single)(im, 'dissimilarity', distances, angles, levels) for im in images
    ))

def glcm_homogeneity(images, distances=[1], angles=[0], levels=256, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(glcm_prop_single)(im, 'homogeneity', distances, angles, levels) for im in images
    ))

def glcm_correlation(images, distances=[1], angles=[0], levels=256, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(glcm_prop_single)(im, 'correlation', distances, angles, levels) for im in images
    ))

def glcm_energy(images, distances=[1], angles=[0], levels=256, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(glcm_prop_single)(im, 'energy', distances, angles, levels) for im in images
    ))


# ---------------- LBP ----------------
def lbp_histogram_single(img, P, R, method, normalize):
    n_bins = P + 2 if method == 'uniform' else 2 ** P
    hist, _ = np.histogram(
        local_binary_pattern(img.astype(np.uint8), P, R, method).ravel(),
        bins=np.arange(n_bins + 1),
        density=normalize
    )
    return hist

def lbp_histogram(images, P=4, R=1, method='default', normalize=True, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(lbp_histogram_single)(im, P, R, method, normalize) for im in images
    ))


# ---------------- Gabor ----------------
def gabor_response(img, frequency, theta):
    real, imag = gabor(img, frequency=frequency, theta=theta)
    magnitude = np.sqrt(real**2 + imag**2)
    return magnitude

def gabor_mean_single(img, frequency, theta):
    return gabor_response(img, frequency, theta).mean()

def gabor_variance_single(img, frequency, theta):
    return gabor_response(img, frequency, theta).var()

def gabor_energy_single(img, frequency, theta):
    mag = gabor_response(img, frequency, theta)
    return np.mean(mag**2)

def gabor_mean(images, frequency=0.2, theta=0, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(gabor_mean_single)(im, frequency, theta) for im in images
    ))

def gabor_variance(images, frequency=0.2, theta=0, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(gabor_variance_single)(im, frequency, theta) for im in images
    ))

def gabor_energy(images, frequency=0.2, theta=0, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(gabor_energy_single)(im, frequency, theta) for im in images
    ))



# -------------------- 纹理特征网络 --------------------
class TextureFeatureNet(nn.Module):
    """
    深度纹理特征网络
    (多尺度局部卷积 + 对比度增强 + 全局池化)
    """
    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()
        c1 = out_channels // 2
        c2 = out_channels - c1

        # 模拟 GLCM 的邻域关系 (3x3, 5x5 小卷积)
        self.conv3 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, c2, kernel_size=5, padding=2)

        # 局部对比度模块 (增强灰度差异)
        self.conv_contrast = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)

        # 全局池化 (能量 + 主纹理模式)
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        f3 = F.relu(self.conv3(x))
        f5 = F.relu(self.conv5(x))
        feat = torch.cat([f3, f5], dim=1)  # [B,C,H,W]

        # 对比度增强
        contrast_feat = F.relu(self.conv_contrast(feat))

        # 全局池化
        avg_feat = self.pool_avg(contrast_feat).squeeze(-1).squeeze(-1)
        max_feat = self.pool_max(contrast_feat).squeeze(-1).squeeze(-1)

        out = torch.cat([avg_feat, max_feat], dim=1)
        return F.normalize(out, p=2, dim=1)  # [B, 2C]


# -------------------- 纹理特征提取器 --------------------
class DeepTextureExtractor:
    def __init__(self, device=None, out_channels=32, batch_size=4):
        self.device = "cuda:0" if (device is None and torch.cuda.is_available()) else "cpu"
        self.batch_size = batch_size
        self.model = TextureFeatureNet(in_channels=1, out_channels=out_channels).to(self.device)
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
def deep_texture(images, out_channels=8, batch_size=4):
    extractor = DeepTextureExtractor(out_channels=out_channels, batch_size=batch_size)
    return extractor.extract_batch(images)