import cv2
import numpy as np
from joblib import Parallel, delayed
from skimage.transform import resize


# 二维自相关
def autocorrelation_2d(images, lag_h=1, lag_w=1, n_jobs=-1):
    """
    计算二维自相关在指定滞后 (lag_h, lag_w) 下的值

    images: np.ndarray, shape (B, H, W)
    return: np.ndarray, shape (B, 1)
    """
    if images.ndim == 3:
        images = images[:, None, :, :]   # (B,1,H,W)
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


# Canny边缘特征
def extract_canny_features(images, down_size=(8,8), threshold1=100, threshold2=200, n_jobs=-1):
    """
    images: np.ndarray, shape (B, H, W)
    return: np.ndarray, shape (B, down_size[0]*down_size[1])
    """

    def process(img):
        edges = cv2.Canny(img.astype(np.uint8), threshold1, threshold2)
        edges = cv2.resize(edges, down_size, interpolation=cv2.INTER_AREA)
        return edges.flatten()
    return np.array(Parallel(n_jobs=n_jobs)(delayed(process)(im) for im in images))


# 多尺度熵
def extract_multiscale_entropy(images, m=2, r=0.15, max_scale=5, n_jobs=-1, target_size=(16, 16)):
    """
    images: np.ndarray, shape (B, H, W)
    return: np.ndarray, shape (B, max_scale)
    """

    def sample_entropy_1d(x, m=2, r=0.15):
        N = len(x)
        r = max(r * np.std(x), 1e-8)  # 防止 r=0

        def _phi(m):
            if N - m + 1 <= 0:
                return 1e-8  # 防止分母为0
            X = np.array([x[i:i + m] for i in range(N - m + 1)])
            C = np.sum(np.max(np.abs(X[:, None] - X[None, :]), axis=2) <= r, axis=0) - 1
            return np.sum(C) / (N - m + 1)

        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)
        if phi_m1 <= 0 or phi_m <= 0:
            return 0.0  # 防止 log(0)
        return -np.log(phi_m1 / phi_m)

    def multiscale_entropy_1d(x):
        length = len(x)
        safe_max_scale = min(max_scale, length // (m + 1))
        if safe_max_scale < 1:
            safe_max_scale = 1
        mse = []
        for scale in range(1, safe_max_scale + 1):
            # 粗粒化
            coarse_len = length // scale * scale
            if coarse_len < m + 1:
                mse.append(0.0)
                continue
            cg = np.mean(x[:coarse_len].reshape(-1, scale), axis=1)
            value = sample_entropy_1d(cg, m, r)
            if np.isnan(value):
                value = 0.0
            mse.append(value)
        # 如果 safe_max_scale < max_scale，填充0
        while len(mse) < max_scale:
            mse.append(0.0)
        return np.array(mse)

    # 缩放到 target_size
    images_resized = np.array([resize(im, target_size, anti_aliasing=True) for im in images])

    feats = Parallel(n_jobs=n_jobs)(
        delayed(lambda img: multiscale_entropy_1d(img.reshape(-1)))(im)
        for im in images_resized
    )
    return np.array(feats)  # shape (B, max_scale)


# Shi-Tomasi关键点特征
def extract_shitomasi_features(images, max_corners=50, descriptor_dim=64, n_jobs=-1):
    """
    images: np.ndarray, shape (B, H, W)
    return: np.ndarray, shape (B, descriptor_dim)
    """

    def process(img):
        img = img.astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=10)
        if corners is None or len(corners) == 0:
            return np.zeros(descriptor_dim)
        corners = corners.reshape(-1, 2).flatten()
        if corners.size < descriptor_dim:
            corners = np.pad(corners, (0, descriptor_dim - corners.size), 'constant')
        else:
            corners = corners[:descriptor_dim]
        return corners
    return np.array(Parallel(n_jobs=n_jobs)(delayed(process)(im) for im in images))


# 灰度块直方图
def gray_blocks_histogram(images, grid=(4,4), bins=8, n_jobs=-1):
    """
    images: np.ndarray, shape (B, H, W)
    return: np.ndarray, shape (B, grid[0]*grid[1]*bins)
    """

    def blocks_fn(img):
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
    return np.array(Parallel(n_jobs=n_jobs)(delayed(blocks_fn)(im) for im in images))


# ========== 测试脚本 ==========
if __name__ == "__main__":
    # 生成一些测试图像
    B = 5
    H, W = 224, 224
    images = np.random.randint(0, 256, (B, H, W), dtype=np.uint8)

    print("=== 测试 autocorrelation_2d ===")
    ac_feats = autocorrelation_2d(images, lag_h=2, lag_w=3, n_jobs=-1)
    print("输出形状:", ac_feats.shape)   # (B, C)，这里 C=1

    print("\n=== 测试 extract_canny_features ===")
    canny_feats = extract_canny_features(images, down_size=(8, 8))
    print("输出形状:", canny_feats.shape)  # (B, 64)

    print("\n=== 测试 extract_multiscale_entropy ===")
    mse_feats = extract_multiscale_entropy(images, max_scale=5, n_jobs=-1, target_size=(16,16))
    print("输出形状:", mse_feats.shape)  # (B, max_scale)

    print("\n=== 测试 extract_shitomasi_features ===")
    shitomasi_feats = extract_shitomasi_features(images, max_corners=50, descriptor_dim=64)
    print("输出形状:", shitomasi_feats.shape)  # (B, 64)

    print("\n=== 测试 gray_blocks_histogram ===")
    hist_feats = gray_blocks_histogram(images, grid=(4,4), bins=8)
    print("输出形状:", hist_feats.shape)  # (B, grid[0]*grid[1]*bins) = (B, 128)

