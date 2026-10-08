import cv2
import numpy as np
from joblib import Parallel, delayed
from skimage.filters import gabor
from scipy.ndimage import maximum_filter


# ======================
# 顶层处理函数
# ======================

def process_shitomasi(img, max_corners, descriptor_dim):
    img = img.astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners,
                                      qualityLevel=0.01, minDistance=10)
    if corners is None or len(corners) == 0:
        return np.zeros(descriptor_dim)
    corners = corners.reshape(-1, 2).flatten()
    if corners.size < descriptor_dim:
        corners = np.pad(corners, (0, descriptor_dim - corners.size), 'constant')
    else:
        corners = corners[:descriptor_dim]
    return corners


def process_gabor_variance(img, frequency, theta):
    img = img.astype(np.float32)
    real, imag = gabor(img, frequency=frequency, theta=theta)
    magnitude = np.sqrt(real**2 + imag**2)
    return magnitude.var(dtype=np.float64)


def process_gray_blocks_histogram(img, grid, bins):
    img = img.astype(np.uint8)
    h, w = img.shape
    bh, bw = h // grid[0], w // grid[1]
    feats = []
    for i in range(grid[0]):
        for j in range(grid[1]):
            block = img[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            hist = cv2.calcHist([block], [0], None, [bins], [0,256])
            hist = cv2.normalize(hist, hist).flatten()
            feats.extend(hist)
    return np.array(feats)


def pac_fn(x, lag_h, lag_w):
    H, W = x.shape
    x = x - np.mean(x)
    def ac(lh, lw):
        return np.sum(
            x[:H-lh, :W-lw] * x[lh:, lw:]
        ) / np.sum(x**2)
    return ac(lag_h, lag_w) - ac(lag_h, 0) * ac(0, lag_w) / ac(0, 0)


def process_partial_autocorrelation(im, lag_h, lag_w):
    C, H, W = im.shape
    return [pac_fn(im[c], lag_h, lag_w) for c in range(C)]


def detect_scattering_centers(image, threshold=0.5, neighborhood_size=3):
    local_max = (image == maximum_filter(image, size=neighborhood_size))
    detected_peaks = np.where(local_max & (image > threshold))
    centers = list(zip(detected_peaks[0], detected_peaks[1]))
    amplitudes = image[detected_peaks]
    return centers, amplitudes


def process_scattering_std(img):
    centers, amplitudes = detect_scattering_centers(img)
    if len(centers) == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    centers = np.array(centers)
    amplitudes = np.array(amplitudes)
    sum_amp = np.sum(amplitudes)
    x_centroid = np.sum(centers[:, 0] * amplitudes) / sum_amp
    y_centroid = np.sum(centers[:, 1] * amplitudes) / sum_amp
    std_x = np.sqrt(np.sum(amplitudes * (centers[:, 0] - x_centroid) ** 2) / sum_amp)
    std_y = np.sqrt(np.sum(amplitudes * (centers[:, 1] - y_centroid) ** 2) / sum_amp)
    return np.array([std_x, std_y], dtype=np.float32)


# ======================
# 包装函数（可并行）
# ======================

# 方位能量分布
def azimuth_energy_profile(images):
    """
    images: np.ndarray, shape (B, H, W)
    return: np.ndarray, (B, H)
    """
    return np.sum(images, axis=2)


def extract_shitomasi_features(images, max_corners=50, descriptor_dim=64, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(process_shitomasi)(im, max_corners, descriptor_dim) for im in images))


def gabor_variance(images, frequency=0.2, theta=0, n_jobs=-1):
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_gabor_variance)(im, frequency, theta) for im in images)
    return np.array(results)[:, None]


def gray_blocks_histogram(images, grid=(4,4), bins=8, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(process_gray_blocks_histogram)(im, grid, bins) for im in images))


def partial_autocorrelation_2d(images, lag_h=1, lag_w=1, n_jobs=-1):
    if images.ndim == 3:  # (B, H, W)
        images = images[:, None, :, :]
    B, C, H, W = images.shape
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(process_partial_autocorrelation)(images[b], lag_h, lag_w) for b in range(B)))


def scattering_std(images, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(
        delayed(process_scattering_std)(im) for im in images))


# ========== 测试脚本 ==========
if __name__ == "__main__":
    B = 5
    H, W = 224, 224
    images = np.random.randint(0, 256, (B, H, W), dtype=np.uint8)

    print("=== 测试 azimuth_energy_profile ===")
    az_feats = azimuth_energy_profile(images)
    print("输出形状:", az_feats.shape)   # (B, H)

    print("=== 测试 extract_shitomasi_features ===")
    shi_feats = extract_shitomasi_features(images, descriptor_dim=64, n_jobs=-1)
    print("输出形状:", shi_feats.shape)   # (B, 64)

    print("=== 测试 gabor_variance ===")
    gabor_feats = gabor_variance(images, frequency=0.2, theta=0, n_jobs=-1)
    print("输出形状:", gabor_feats.shape)   # (B, 1)

    print("=== 测试 gray_blocks_histogram ===")
    gray_hist_feats = gray_blocks_histogram(images, grid=(4,4), bins=8, n_jobs=-1)
    print("输出形状:", gray_hist_feats.shape)   # (B, 4*4*8)

    print("=== 测试 partial_autocorrelation_2d ===")
    pac_feats = partial_autocorrelation_2d(images, lag_h=1, lag_w=1, n_jobs=-1)
    print("输出形状:", pac_feats.shape)   # (B, C)，这里 C=1

    print("=== 测试 scattering_std ===")
    scatter_feats = scattering_std(images, n_jobs=-1)
    print("输出形状:", scatter_feats.shape)   # (B, 2)
