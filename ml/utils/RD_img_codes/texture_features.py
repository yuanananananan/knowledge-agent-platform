import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor
from joblib import Parallel, delayed
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.transform import resize

# ======================
# GLCM helpers
# ======================
def _glcm_feature(img, prop, distances, angles, levels):
    glcm = graycomatrix(img.astype(np.uint8),
                        distances=distances,
                        angles=angles,
                        levels=levels,
                        symmetric=True,
                        normed=True)
    # 转成 float32 后再计算平均值，最后输出 float16
    val = graycoprops(glcm, prop).astype(np.float32).mean()
    return np.float16(val)

def _glcm_parallel(images, prop, distances=[1], angles=[0], levels=256, n_jobs=-1):
    # 使用线程并行减少内存复制开销
    return Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_glcm_feature)(im, prop, distances, angles, levels) for im in images
    )

def glcm_contrast(images, **kwargs): return np.array(_glcm_parallel(images, 'contrast', **kwargs), dtype=np.float16)
def glcm_dissimilarity(images, **kwargs): return np.array(_glcm_parallel(images, 'dissimilarity', **kwargs), dtype=np.float16)
def glcm_homogeneity(images, **kwargs): return np.array(_glcm_parallel(images, 'homogeneity', **kwargs), dtype=np.float16)
def glcm_correlation(images, **kwargs): return np.array(_glcm_parallel(images, 'correlation', **kwargs), dtype=np.float16)
def glcm_energy(images, **kwargs): return np.array(_glcm_parallel(images, 'energy', **kwargs), dtype=np.float16)

# ======================
# LBP helpers
# ======================
def _lbp_hist(img, P, R, method, n_bins, normalize):
    lbp = local_binary_pattern(img.astype(np.uint8), P, R, method).astype(np.float32)
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(n_bins+1), density=normalize)
    return hist.astype(np.float16)

def lbp_histogram(images, P=4, R=1, method='default', normalize=True, n_jobs=-1):
    n_bins = P + 2 if method == 'uniform' else 2**P
    # 使用线程并行
    return np.array(Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_lbp_hist)(im, P, R, method, n_bins, normalize) for im in images
    ), dtype=np.float16)

# ======================
# Gabor helpers
# ======================
def _gabor_magnitude(img, frequency, theta):
    img = img.astype(np.float32)  # 输入 float32
    real, imag = gabor(img, frequency=frequency, theta=theta)
    # 中间计算都使用 float32
    mag = np.sqrt(real.astype(np.float32)**2 + imag.astype(np.float32)**2)
    return mag  # 返回 float32，减少中间内存开销

def _gabor_mean(img, frequency, theta):
    return np.float16(_gabor_magnitude(img, frequency, theta).mean())

def _gabor_variance(img, frequency, theta):
    return np.float16(_gabor_magnitude(img, frequency, theta).var())

def _gabor_energy(img, frequency, theta):
    mag = _gabor_magnitude(img, frequency, theta)
    return np.float16(np.mean(mag**2))

def gabor_mean(images, frequency=0.2, theta=0, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_gabor_mean)(im, frequency, theta) for im in images
    ), dtype=np.float16)

def gabor_variance(images, frequency=0.2, theta=0, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_gabor_variance)(im, frequency, theta) for im in images
    ), dtype=np.float16)

def gabor_energy(images, frequency=0.2, theta=0, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_gabor_energy)(im, frequency, theta) for im in images
    ), dtype=np.float16)

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
def deep_texture(images, out_channels=16, batch_size=4):
    extractor = DeepTextureExtractor(out_channels=out_channels, batch_size=batch_size)
    return extractor.extract_batch(images)