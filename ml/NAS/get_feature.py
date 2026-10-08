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

feature_name = [
    # 统计域 : rms
    # 时频域：wavelet
    # 空域：Gabor 纹理特征 、 HOG
]


def get_statis_feat(data, k=7, s=4, p=3):
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
    rms_matrix = np.sqrt(np.mean(np.square(windows), axis=(2, 3))).flatten().reshape(4, -1)
    std_matrix = np.std(windows, axis=(2, 3)).flatten().reshape(4, -1)
    max_matrix = np.max(windows, axis=(2, 3)).flatten().reshape(4, -1)
    mean_matrix = np.mean(windows, axis=(2, 3)).flatten().reshape(4, -1)

    statis_feat = np.concatenate([rms_matrix, std_matrix, max_matrix, mean_matrix])
    return statis_feat


def get_time_freq_feat(data, level=3):
    # Perform wavelet decomposition
    coeffs = pywt.wavedec2(data, 'haar', level=level)

    # Initialize list to store all frequency bands
    all_bands = []

    # Get the approximate coefficients (LL) from the highest level
    ll = coeffs[0]
    all_bands.append(ll)

    # Get all detail coefficients from each level
    for l in range(1, level + 1):
        bands = coeffs[l]
        all_bands.extend([bands[0], bands[1], bands[2]])  # LH, HL, HH for each level

    # Determine the target size (size of the highest level coefficients)
    target_size = ll.shape

    # Resize all bands to the target size
    resized_bands = []
    for band in all_bands:
        if band.shape != target_size:
            # Use skimage's resize function with anti-aliasing
            resized_band = resize(band, target_size, mode='reflect', anti_aliasing=True)
        else:
            resized_band = band
        resized_bands.append(resized_band)

    time_freq_features = np.array(resized_bands).reshape(10, -1)
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

    # assert data.shape[0] == 64
    h, w = data.shape
    data = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    data = data.astype(np.uint8)
    nbins = 12
    hog = cv2.HOGDescriptor((h, w), (16, 16), (16, 16), (8, 8), nbins)

    # 总特征数 = 9 * (16/8)^2 * ((64-16)/16+1)^2 = 9*4*4*4 = 9 * 64
    hog_features = hog.compute(data)
    hog_features = hog_features.reshape(nbins, -1)

    return hog_features


def get_structure_feature(data, K=16):

    def extract_patch(img, keypoints, patch_size):
        patch_arrys = []
        for (x, y) in keypoints:
            x, y = int(x), int(y)
            padded_img = np.pad(img, pad_width=patch_size // 2, mode='constant', constant_values=0)
            region = padded_img[y:y + patch_size, x:x + patch_size]
            patch = region.flatten()
            patch_arrys.append(patch)
        patch_arrys = np.array(patch_arrys)
        return patch_arrys

    img = np.uint8(data.copy() * 255)
    h, w = img.shape[:2]
    # 先缩放
    # h = h // 4
    # w = w // 4
    # img = cv2.resize(img, dsize=(h, w))
    dst = cv2.cornerHarris(img, 2, 3, 0.04)
    # threshold_set = np.sort(dst.flatten())[-100]
    nums = K
    indices = np.argpartition(dst.flatten(), -nums)[-nums:]
    indices = np.flip(indices)  #从大到小排序
    y, x = np.unravel_index(indices, dst.shape)
    keypoints = np.column_stack((x, y))
    keypoints_patch = extract_patch(data, keypoints, 8)
    structure_feature = np.array(keypoints_patch)
    return structure_feature


def get_outra_feature(data):
    # 输入数据归一化0-1 (64,64)
    # data = data / (data.max() + 1e-10)
    hog_features = get_spatial_feat(data)
    statis_feat = get_statis_feat(data)
    time_freq_features = get_time_freq_feat(data)
    # structure_features = get_structure_feature(data)
    # print(hog_features.shape)  # (9, 64)
    # print(statis_feat.shape)  # (2, 64)
    # print(time_freq_features.shape)  # (4, 64)

    # return np.concatenate([hog_features, statis_feat, time_freq_features], axis=0)
    return [hog_features, statis_feat, time_freq_features]


if __name__ == '__main__':
    data = np.random.rand(64, 64)  # 输入数据归一化0-1

    hog_features = get_spatial_feat(data)
    print(hog_features.shape)  # (9, 64)
    # print(hog_features)

    statis_feat = get_statis_feat(data)
    print(statis_feat.shape)  # (2, 64)
    # print(hog_features)

    time_freq_features = get_time_freq_feat(data)
    print(time_freq_features.shape)  # (4, 64)

    structure_features = get_structure_feature(data)
    print(structure_features.shape)  # (4, 64)
