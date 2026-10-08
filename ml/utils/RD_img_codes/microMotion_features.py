import numpy as np
import cv2
from scipy.signal import correlate
from scipy.linalg import svd
from scipy.stats import entropy
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize


# ========== 1. 微动周期 ==========
def micro_motion_period(rd_images):
    rd_images = np.atleast_3d(rd_images)  # 保证 [N,H,W]
    periods = []
    for rd_image in rd_images:
        doppler_profile = np.sum(np.abs(rd_image), axis=0)
        ac = correlate(doppler_profile, doppler_profile, mode='full')
        ac = ac[ac.size//2:]
        spectrum = np.abs(np.fft.fft(ac))
        peak_idx = np.argmax(spectrum[1:]) + 1
        period = len(doppler_profile) / peak_idx if peak_idx > 0 else None
        periods.append(period)
    return np.array(periods)

# ========== 2. 瞬时频率 ==========
def instantaneous_frequency(rd_images, target_len=32):
    rd_images = np.atleast_3d(rd_images)
    inst_freqs = []

    for rd_image in rd_images:
        signal = np.sum(rd_image, axis=0)
        phase = np.unwrap(np.angle(np.fft.ifft(signal)))
        inst_freq = np.gradient(phase) / (2 * np.pi)

        # 插值到固定长度
        x_old = np.linspace(0, 1, len(inst_freq))
        x_new = np.linspace(0, 1, target_len)
        f = interp1d(x_old, inst_freq, kind="linear")
        inst_freq_resampled = f(x_new)

        inst_freqs.append(inst_freq_resampled)

    return np.stack(inst_freqs, axis=0)  # [N, target_len]

# ========== 3. 不变矩特征 ==========
def hu_moments(rd_images):
    rd_images = np.atleast_3d(rd_images)
    hu_all = []
    for rd_image in rd_images:
        rd_norm = cv2.normalize(np.abs(rd_image), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        moments = cv2.moments(rd_norm)
        hu = cv2.HuMoments(moments).flatten()
        hu_all.append(hu)
    return np.array(hu_all)

# ========== 4. 时频熵特征 ==========
def time_freq_entropy(rd_images):
    rd_images = np.atleast_3d(rd_images)
    ents = []
    for rd_image in rd_images:
        p = np.abs(rd_image)**2
        p = p / np.sum(p)
        tf_entropy = entropy(p.flatten())
        ents.append(tf_entropy)
    return np.array(ents)

# ========== 5. 奇异值分解 (SVD) 特征 ==========
def svd_features(rd_images, k=5):
    rd_images = np.atleast_3d(rd_images)
    feats = []
    for rd_image in rd_images:
        U, s, Vt = svd(rd_image, full_matrices=False)
        s_norm = s[:k] / np.sum(s)
        feats.append(s_norm)
    return np.array(feats)


# ========== 示例调用 ==========
if __name__ == "__main__":
    # 构造一批模拟 RD 图像 (N=3, H=128, W=128)
    N, H, W = 3, 128, 128
    x, y = np.meshgrid(np.linspace(-3,3,W), np.linspace(-3,3,H))
    rd_images = []
    for i in range(N):
        rd_images.append(np.exp(-(x**2 + y**2)) + 0.05*np.random.randn(H,W))
    rd_images = np.stack(rd_images, axis=0)  # [N,H,W]

    # 提取五类特征
    period = micro_motion_period(rd_images)
    inst_freq = instantaneous_frequency(rd_images)
    hu = hu_moments(rd_images)
    tf_ent = time_freq_entropy(rd_images)
    svd_feat = svd_features(rd_images)

    print("微动周期:", period)
    print("瞬时频率(第1个样本):", inst_freq[0][:10])  # 只看前10个
    print("Hu矩 shape:", hu.shape)
    print("时频熵:", tf_ent)
    print("SVD特征 shape:", svd_feat.shape)

class DeepMicroMotionNet(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=1)
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        # x: [B,1,H,W]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 全局平均池化 [B,64]
        x = F.adaptive_avg_pool2d(x, (1,1)).view(x.size(0), -1)
        x = F.normalize(self.fc(x), p=2, dim=1)  # 输出单位化向量 [B, output_dim]
        return x

# --------- 深度微动特征提取函数 ---------
def deep_micro_motion_features(rd_images, H=64, W=64, output_dim=128, device=None):
    """
    rd_images: numpy array [N,H,W] 或 [N,H,W,C] (灰度或RGB)
    返回: numpy array [N, output_dim]
    """
    device = "cuda:0" if (device is None and torch.cuda.is_available()) else "cpu"
    model = DeepMicroMotionNet(output_dim=output_dim).to(device)
    model.eval()

    tensor_imgs = []
    for img in rd_images:
        if img.ndim == 3:  # RGB 转灰度
            gray = np.mean(img, axis=2).astype(np.float32)
        else:
            gray = img.astype(np.float32)

        # resize 并归一化到 [0,1]
        gray_resized = resize(gray, (H, W), mode="reflect", anti_aliasing=True)
        gray_resized = (gray_resized - gray_resized.min()) / (gray_resized.max() - gray_resized.min() + 1e-12)
        tensor_imgs.append(torch.from_numpy(gray_resized))

    tensor_imgs = torch.stack(tensor_imgs).unsqueeze(1).to(device)  # [B,1,H,W]

    with torch.no_grad():
        feats = model(tensor_imgs)

    return feats.cpu().numpy()