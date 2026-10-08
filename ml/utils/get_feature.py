import pywt
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
from scipy.stats import skew, kurtosis
import cv2
import pywt
import numpy as np
import skimage.feature as feature
from scipy.fftpack import dct
from skimage.feature import local_binary_pattern
from skimage.restoration import denoise_nl_means
from skimage.feature import graycomatrix, graycoprops
from skimage.transform import resize
import torch
import torch.nn.functional as F

import utils.MultiFeaturesExtractorSAR as sar_extractor
import utils.MultiFeaturesExtractorMon as mon_extractor
import utils.MultiFeaturesExtractorRD as rd_extractor

feature_name = [
    # 统计域 : rms
    # 时频域：wavelet
    # 空域：Gabor 纹理特征 、 HOG
]


def _prepare_feature_image(data, size=(32, 32)):
    """Convert one feature-operator input into a finite 2D float image."""
    image = np.asarray(data, dtype=np.float32)
    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError(f"特征算子要求二维单通道输入，实际形状为 {image.shape}")
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _quantize_gray_image(image, levels=32):
    """Quantize an image into [0, levels - 1] for texture descriptors."""
    low = float(np.min(image))
    high = float(np.max(image))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    normalized = (image - low) / (high - low)
    return np.floor(normalized * (levels - 1)).astype(np.uint8)


def get_glcm_feat(data, distances=(1, 2), angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4), levels=32):
    """Extract 40 GLCM texture statistics from one image."""
    image = _quantize_gray_image(_prepare_feature_image(data), levels=levels)
    matrix = graycomatrix(
        image,
        distances=list(distances),
        angles=list(angles),
        levels=levels,
        symmetric=True,
        normed=True,
    )
    properties = ("contrast", "dissimilarity", "homogeneity", "correlation", "energy")
    return np.concatenate([graycoprops(matrix, prop).ravel() for prop in properties]).astype(np.float32)


def get_lbp_feat(data, points=8, radius=1):
    """Extract a normalized uniform-LBP histogram from one image."""
    image = _quantize_gray_image(_prepare_feature_image(data), levels=256)
    codes = local_binary_pattern(image, points, radius, method="uniform")
    bins = points + 2
    histogram, _ = np.histogram(codes.ravel(), bins=np.arange(bins + 1), range=(0, bins))
    histogram = histogram.astype(np.float32)
    return histogram / max(float(histogram.sum()), 1.0)


def get_amplitude_feat(data):
    """Extract seven global amplitude and energy statistics from one image."""
    image = _prepare_feature_image(data)
    absolute = np.abs(image)
    rms = float(np.sqrt(np.mean(np.square(image, dtype=np.float64))))
    features = np.array(
        [
            np.max(image),
            np.min(image),
            np.mean(image),
            np.median(image),
            np.mean(np.square(image, dtype=np.float64)),
            rms,
            float(np.max(absolute)) / (rms + 1e-12),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def get_fourier_feat(data, radial_bins=4, angle_bins=8):
    """Extract fixed-length global frequency-distribution features."""
    image = _prepare_feature_image(data)
    centered = image - float(np.mean(image))
    magnitude = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(centered))))
    height, width = magnitude.shape
    yy, xx = np.indices((height, width))
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    angle = np.arctan2(yy - cy, xx - cx)
    total = float(np.sum(magnitude)) + 1e-12
    probability = magnitude.ravel() / total

    basic = [
        float(np.mean(magnitude)),
        float(np.std(magnitude)),
        float(-np.sum(probability * np.log2(probability + 1e-12))),
        float(np.sum(xx * magnitude) / total / max(width - 1, 1)),
        float(np.sum(yy * magnitude) / total / max(height - 1, 1)),
    ]

    radial_edges = np.linspace(0.0, float(radius.max()) + 1e-12, radial_bins + 1)
    radial = [float(np.sum(magnitude[(radius >= radial_edges[i]) & (radius < radial_edges[i + 1])]) / total) for i in range(radial_bins)]
    angle_edges = np.linspace(-np.pi, np.pi, angle_bins + 1)
    directional = [float(np.sum(magnitude[(angle >= angle_edges[i]) & (angle < angle_edges[i + 1])]) / total) for i in range(angle_bins)]
    return np.asarray(basic + radial + directional, dtype=np.float32)


def _normalized_profile_features(profile):
    """Describe a one-dimensional non-negative energy profile."""
    values = np.nan_to_num(np.asarray(profile, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    values = np.abs(values)
    total = float(values.sum())
    if total <= 1e-12:
        return np.zeros(5, dtype=np.float32)
    probability = values / total
    positions = np.linspace(0.0, 1.0, values.size)
    centroid = float(np.sum(positions * probability))
    spread = float(np.sqrt(np.sum(np.square(positions - centroid) * probability)))
    entropy = float(-np.sum(probability * np.log2(probability + 1e-12)) / np.log2(max(values.size, 2)))
    prominence = max(float(np.max(values)) * 0.05, 1e-12)
    peaks, _ = find_peaks(values, prominence=prominence)
    peak_density = float(len(peaks)) / max(values.size, 1)
    concentration = float(np.max(probability))
    return np.asarray([centroid, spread, entropy, peak_density, concentration], dtype=np.float32)


def get_rd_doppler_feat(data):
    """Extract normalized energy-distribution features along the Doppler axis."""
    image = _prepare_feature_image(data)
    doppler_profile = np.sum(np.square(np.abs(image), dtype=np.float64), axis=0)
    return _normalized_profile_features(doppler_profile)


def get_rd_micro_motion_feat(data, singular_values=5):
    """Extract low-rank and time-frequency structure from one RD image."""
    image = _prepare_feature_image(data)
    values = np.linalg.svd(image, compute_uv=False).astype(np.float64)
    total = float(values.sum())
    normalized = values / total if total > 1e-12 else np.zeros_like(values)
    selected = np.zeros(singular_values, dtype=np.float64)
    selected[: min(singular_values, normalized.size)] = normalized[:singular_values]
    entropy = float(-np.sum(normalized * np.log2(normalized + 1e-12)) / np.log2(max(normalized.size, 2)))
    top1_energy = float(np.square(values[:1]).sum() / (np.square(values).sum() + 1e-12))
    top3_energy = float(np.square(values[:3]).sum() / (np.square(values).sum() + 1e-12))
    return np.asarray([*selected, entropy, top1_energy, top3_energy], dtype=np.float32)


def get_rd_fluctuation_feat(data):
    """Extract robust and central-moment fluctuation statistics from an RD image."""
    values = _prepare_feature_image(data).ravel().astype(np.float64)
    mean = float(np.mean(values))
    centered = values - mean
    std = float(np.std(values))
    median = float(np.median(values))
    features = np.asarray(
        [
            np.ptp(values),
            np.var(values),
            std,
            np.percentile(values, 75) - np.percentile(values, 25),
            np.median(np.abs(values - median)),
            np.mean(np.power(centered, 3)),
            np.mean(np.power(centered, 4)),
            std / (abs(mean) + 1e-12),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def get_mon_azimuth_feat(data):
    """Extract normalized response-distribution features along the MON azimuth axis."""
    image = _prepare_feature_image(data)
    azimuth_profile = np.sum(np.square(np.abs(image), dtype=np.float64), axis=0)
    return _normalized_profile_features(azimuth_profile)


def get_mon_intensity_feat(data, bins=64):
    """Extract a normalized gray histogram and four distribution moments."""
    image = _prepare_feature_image(data)
    gray = _quantize_gray_image(image, levels=256)
    histogram, _ = np.histogram(gray.ravel(), bins=bins, range=(0, 256))
    histogram = histogram.astype(np.float32)
    histogram /= max(float(histogram.sum()), 1.0)
    values = gray.ravel().astype(np.float64) / 255.0
    moments = np.asarray(
        [np.mean(values), np.std(values), skew(values), kurtosis(values)],
        dtype=np.float32,
    )
    return np.concatenate([histogram, np.nan_to_num(moments, nan=0.0, posinf=0.0, neginf=0.0)])


def get_mon_fourier_feat(data):
    """Extract MON frequency statistics with four directional energy bins."""
    return get_fourier_feat(data, radial_bins=4, angle_bins=4)


def get_statis_feat(data, k=7, s=4, p=3):
    data = cv2.resize(data, (32, 32))
    h, w = data.shape

    data = np.pad(data, pad_width=p, mode='constant', constant_values=0)

    # 计算输出矩阵的形状
    out_h = (h - k + 2 * p) // s + 1
    out_w = (w - k + 2 * p) // s + 1

    rms_matrix = np.zeros((out_h, out_w))

    from numpy.lib.stride_tricks import as_strided

    new_shape = (out_h, out_w, k, k)
    new_strides = (data.strides[0] * s, data.strides[1] * s, data.strides[0], data.strides[1])

    windows = as_strided(data, shape=new_shape, strides=new_strides)
    rms_matrix = np.sqrt(np.mean(np.square(windows), axis=(2, 3))).flatten()
    std_matrix = np.std(windows, axis=(2, 3)).flatten()
    max_matrix = np.max(windows, axis=(2, 3)).flatten()
    mean_matrix = np.mean(windows, axis=(2, 3)).flatten()

    statis_feat = np.stack([rms_matrix, std_matrix, max_matrix, mean_matrix])
    return statis_feat


def get_time_freq_feat(data, level=2):
    data = cv2.resize(data, (32, 32))
    coeffs = pywt.wavedec2(data, 'haar', level=level)
    ll = coeffs[0]
    high_lv_bands = [c for c in coeffs[1]]  # 只获取最高级别频带
    high_lv_bands.extend([ll])

    time_freq_features = np.array(high_lv_bands).reshape(4, -1)
    return time_freq_features


def get_spatial_feat(data):
    '''
    每个cell的特征数 = nbins
    每个block的cell数 = (block_size/cell_size)^2
    每个block的特征数 = nbins * (block_size/cell_size)^2
    水平方向block数 = (win_width - block_width)/block_stride + 1
    垂直方向block数 = (win_height - block_height)/block_stride + 1
    总特征数 = 每个block的特征数 * 水平block数 * 垂直block数
    '''
    data = cv2.resize(data, (32, 32))

    # assert data.shape[0] == 64
    h, w = data.shape
    data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    data = data.astype(np.uint8)
    data = np.log1p(data).astype(np.uint8)
    nbins = 9
    hog = cv2.HOGDescriptor((h, w), (16, 16), (16, 16), (8, 8), nbins)

    # 总特征数 = 9 * (16/8)^2 * ((64-16)/16+1)^2 = 9*4*4*4 = 9 * 64
    hog_features = hog.compute(data)
    hog_features = hog_features.reshape(nbins, -1)

    return hog_features


def get_outra_feature(data, cfg):
    """
    安全地从配置中提取特征
    
    Args:
        data: 输入数据
        cfg: 配置字典，包含 'feature_combinations' 键
    
    Returns:
        特征列表
    """
    feat = []
    H, W = data.shape
    for fun_name in cfg['feature_combinations']:
        # 从当前模块的全局命名空间中查找函数
        func = globals().get(fun_name)

        # 检查函数是否存在
        if func is not None:
            data = data.reshape(H, W)
        else:
            # 调用外部优选特征
            mode = cfg['data']['mode']
            assert mode in ['RD', 'SAR', 'MON']
            if mode == 'SAR':
                func = getattr(sar_extractor, fun_name, None)
                data = data.reshape(1, H, W, 1)
            elif mode == 'RD':
                func = getattr(rd_extractor, fun_name, None)
                data = data.reshape(1, 1, H, W, 1)
            elif mode == 'MON':
                func = getattr(mon_extractor, fun_name, None)
                data = data.reshape(1, H, W, 1)
            if func is None:
                raise ValueError(f"未找到特征算子：{fun_name}")

        # 检查是否可调用
        if not callable(func):
            raise ValueError(f"特征算子不可调用：{fun_name}")

        try:
            result = func(data)
            if not isinstance(result, np.ndarray):
                result = np.array(result)

            result_flat = result.flatten()
            feat.extend(result_flat.tolist()[:256])  # 只取前256个特征

        except Exception as e:
            raise RuntimeError(f"特征算子 {fun_name} 执行失败：{e}") from e
    feat = np.array(feat).astype(np.float32)

    return feat


if __name__ == '__main__':

    data = np.random.rand(32, 32)  # 输入数据归一化0-1

    hog_features = get_spatial_feat(data)
    print(hog_features.shape)  # (9, 64)
    # print(hog_features)

    statis_feat = get_statis_feat(data)
    print(statis_feat.shape)  # (2, 64)
    # print(hog_features)

    time_freq_features = get_time_freq_feat(data)
    print(time_freq_features.shape)  # (4, 64)


def get_test_feat1(data):
    '''
    每个cell的特征数 = nbins
    每个block的cell数 = (block_size/cell_size)^2
    每个block的特征数 = nbins * (block_size/cell_size)^2
    水平方向block数 = (win_width - block_width)/block_stride + 1
    垂直方向block数 = (win_height - block_height)/block_stride + 1
    总特征数 = 每个block的特征数 * 水平block数 * 垂直block数
    '''

    # assert data.shape[0] == 64
    h, w = data.shape
    data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    data = data.astype(np.uint8)
    nbins = 9
    hog = cv2.HOGDescriptor((h, w), (16, 16), (16, 16), (8, 8), nbins)

    # 总特征数 = 9 * (16/8)^2 * ((64-16)/16+1)^2 = 9*4*4*4 = 9 * 64
    hog_features = hog.compute(data)
    hog_features = hog_features.reshape(nbins, -1)

    return hog_features


def get_test_feat2(data):
    '''
    每个cell的特征数 = nbins
    每个block的cell数 = (block_size/cell_size)^2
    每个block的特征数 = nbins * (block_size/cell_size)^2
    水平方向block数 = (win_width - block_width)/block_stride + 1
    垂直方向block数 = (win_height - block_height)/block_stride + 1
    总特征数 = 每个block的特征数 * 水平block数 * 垂直block数
    '''

    # assert data.shape[0] == 64
    h, w = data.shape
    data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    data = data.astype(np.uint8)
    nbins = 9
    hog = cv2.HOGDescriptor((h, w), (16, 16), (16, 16), (8, 8), nbins)

    # 总特征数 = 9 * (16/8)^2 * ((64-16)/16+1)^2 = 9*4*4*4 = 9 * 64
    hog_features = hog.compute(data)
    hog_features = hog_features.reshape(nbins, -1)

    return hog_features


def get_test_feat3(data):
    '''
    每个cell的特征数 = nbins
    每个block的cell数 = (block_size/cell_size)^2
    每个block的特征数 = nbins * (block_size/cell_size)^2
    水平方向block数 = (win_width - block_width)/block_stride + 1
    垂直方向block数 = (win_height - block_height)/block_stride + 1
    总特征数 = 每个block的特征数 * 水平block数 * 垂直block数
    '''

    # assert data.shape[0] == 64
    h, w = data.shape
    data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    data = data.astype(np.uint8)
    nbins = 9
    hog = cv2.HOGDescriptor((h, w), (16, 16), (16, 16), (8, 8), nbins)

    # 总特征数 = 9 * (16/8)^2 * ((64-16)/16+1)^2 = 9*4*4*4 = 9 * 64
    hog_features = hog.compute(data)
    hog_features = hog_features.reshape(nbins, -1)

    return hog_features


def get_test_feat4(data):
    '''
    每个cell的特征数 = nbins
    每个block的cell数 = (block_size/cell_size)^2
    每个block的特征数 = nbins * (block_size/cell_size)^2
    水平方向block数 = (win_width - block_width)/block_stride + 1
    垂直方向block数 = (win_height - block_height)/block_stride + 1
    总特征数 = 每个block的特征数 * 水平block数 * 垂直block数
    '''

    # assert data.shape[0] == 64
    h, w = data.shape
    data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    data = data.astype(np.uint8)
    nbins = 9
    hog = cv2.HOGDescriptor((h, w), (16, 16), (16, 16), (8, 8), nbins)

    # 总特征数 = 9 * (16/8)^2 * ((64-16)/16+1)^2 = 9*4*4*4 = 9 * 64
    hog_features = hog.compute(data)
    hog_features = hog_features.reshape(nbins, -1)

    return hog_features
