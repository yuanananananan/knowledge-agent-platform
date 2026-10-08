import cv2
import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import maximum_filter


# 聚合自相关
def aggregated_autocorrelation_2d(images, lag_list=[0,1,2,3]):
    """
    对二维自相关按指定 lag_h, lag_w 取值聚合

    images: np.ndarray, shape (B, H, W)
    return: np.ndarray, (B, len(lag_list))
    """

    # 二维自相关
    def autocorrelation_2d(images, lag_h=1, lag_w=1, n_jobs=-1):
        if images.ndim == 3:
            images = images[:, None, :, :]  # (B,1,H,W)
        B, C, H, W = images.shape

        def process_one(b, c):
            x = images[b, c] - np.mean(images[b, c])
            lh = min(lag_h, H - 1)
            lw = min(lag_w, W - 1)
            ac = np.sum(x[:H - lh, :W - lw] * x[lh:, lw:]) / (np.sum(x ** 2) + 1e-12)
            return (b, c, ac)

        # 并行计算
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_one)(b, c) for b in range(B) for c in range(C)
        )

        # 整理结果为 (B,C)
        res = np.zeros((B, C))
        for b, c, ac in results:
            res[b, c] = ac
        return res

    results = []
    for lag in lag_list:
        ac = autocorrelation_2d(images, lag, lag)  # 或者单独指定 lh, lw
        results.append(ac)
    return np.stack(results, axis=-1).reshape(images.shape[0], -1)  # (B,C,L)


# 多普勒谱能量
def doppler_spectrum_energy(images):
    """
    images: np.ndarray, shape (B, H, W)
    return: np.ndarray, (B, W)
    """
    return np.sum(images**2, axis=(1))


# sift关键点特征
def extract_sift_features(images, descriptor_dim=64, n_jobs=-1):
    """
    images: np.ndarray, shape (B, H, W)
    return: np.ndarray, (B, descriptor_dim)
    """
    def process(img):
        img = img.astype(np.uint8)
        sift = cv2.SIFT_create()  # 在这里创建对象，避免pickle问题
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(descriptor_dim)
        desc_mean = descriptors.mean(axis=0)
        if desc_mean.size < descriptor_dim:
            desc_mean = np.pad(desc_mean, (0, descriptor_dim - desc_mean.size), 'constant')
        else:
            desc_mean = desc_mean[:descriptor_dim]
        return desc_mean

    return np.array(Parallel(n_jobs=n_jobs)(delayed(process)(im) for im in images))


# 灰度相关图
def gray_autocorrelogram(images, distance=1, bins=16, n_jobs=-1):
    """
    images: np.ndarray, shape (B, H, W)
    return: np.ndarray, (B, bins)
    """
    def autocorr_fn(img):
        img = img.astype(np.uint8)
        quantized = (img // (256//bins)).astype(np.uint8)
        h, w = quantized.shape
        correlogram = np.zeros(bins)
        offsets = [(0,distance),(distance,0),(0,-distance),(-distance,0)]
        for gray_val in range(bins):
            mask = (quantized == gray_val)
            ys, xs = np.where(mask)
            count = 0
            match = 0
            for (x,y) in zip(xs, ys):
                for dx, dy in offsets:
                    nx, ny = x+dx, y+dy
                    if 0<=nx<w and 0<=ny<h:
                        count += 1
                        if quantized[ny,nx] == gray_val:
                            match += 1
            correlogram[gray_val] = match/count if count > 0 else 0
        return correlogram
    return np.array(Parallel(n_jobs=n_jobs)(delayed(autocorr_fn)(im) for im in images))


# 散射中心加权质心坐标
def scattering_centroid(images, n_jobs=-1):
    """
    images: np.ndarray, shape (B, H, W)
    return: np.ndarray, (B, 2)
    """

    # 散射中心检测
    def detect_scattering_centers(image, threshold=0.5, neighborhood_size=3):
        # 局部极大值
        local_max = (image == maximum_filter(image, size=neighborhood_size))
        # 阈值筛选
        detected_peaks = np.where(local_max & (image > threshold))
        centers = list(zip(detected_peaks[0], detected_peaks[1]))
        amplitudes = image[detected_peaks]
        return centers, amplitudes

    def process(img):
        centers, amplitudes = detect_scattering_centers(img)
        if len(centers) == 0:
            return np.array([0.0, 0.0], dtype=np.float32)
        centers = np.array(centers)
        amplitudes = np.array(amplitudes)
        sum_amp = np.sum(amplitudes)
        x_centroid = np.sum(centers[:, 0] * amplitudes) / sum_amp
        y_centroid = np.sum(centers[:, 1] * amplitudes) / sum_amp
        return np.array([x_centroid, y_centroid], dtype=np.float32)

    return np.array(Parallel(n_jobs=n_jobs)(delayed(process)(im) for im in images))


# ========== 测试脚本 ==========
if __name__ == "__main__":
    B = 5
    H, W = 224, 224
    images = np.random.randint(0, 256, (B, H, W), dtype=np.uint8)

    print("=== 测试 aggregated_autocorrelation_2d ===")
    agg_feats = aggregated_autocorrelation_2d(images, lag_list=[0,1,2,3])
    print("输出形状:", agg_feats.shape)   # (B, len(lag_list))

    print("\n=== 测试 doppler_spectrum_energy ===")
    doppler_feats = doppler_spectrum_energy(images.astype(np.float32))
    print("输出形状:", doppler_feats.shape)  # (B, W)

    print("\n=== 测试 extract_sift_features ===")
    # SIFT 需要 3 通道图像
    rgb_images = np.random.randint(0, 256, (B, H, W, 3), dtype=np.uint8)
    sift_feats = extract_sift_features(rgb_images, descriptor_dim=64, n_jobs=-1)
    print("输出形状:", sift_feats.shape)  # (B, 64)

    print("\n=== 测试 gray_autocorrelogram ===")
    gray_feats = gray_autocorrelogram(images, distance=1, bins=16, n_jobs=-1)
    print("输出形状:", gray_feats.shape)  # (B, bins)

    print("\n=== 测试 scattering_centroid ===")
    scatter_feats = scattering_centroid(images.astype(np.float32), n_jobs=-1)
    print("输出形状:", scatter_feats.shape)  # (B, 2)

