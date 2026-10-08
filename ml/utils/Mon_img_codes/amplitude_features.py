import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize


# 1. 最大值
def get_max(image):
    return np.max(image, axis=(1, 2))

# 2. 最小值
def get_min(image):
    return np.min(image, axis=(1, 2))

# 3. 均值
def get_mean(image):
    return np.mean(image, axis=(1, 2))

# 4. 中位数
def get_median(image):
    return np.median(image, axis=(1, 2))

# 5. 能量
def get_energy(image):
    return np.sum(image**2, axis=(1, 2))

# 6. 均方根 (Root Mean Square)
def get_rms(image):
    return np.sqrt(np.mean(image**2, axis=(1, 2)))

# 7. 峰值因子
def get_crest_factor(image):
    peak = np.max(np.abs(image), axis=(1, 2))
    rms = get_rms(image)
    return peak / (rms + 1e-12)   # 防止除零

# 8. 深度幅度特征
# --------- 简易二维幅度特征提取网络 ---------
class Amplitude2DNet(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        # x: [B,1,H,W]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # 全局平均池化 [B,64]
        x = F.adaptive_avg_pool2d(x, (1,1)).view(x.size(0), -1)
        x = F.normalize(self.fc(x), p=2, dim=1)  # [B, output_dim]
        return x

# --------- 二维幅度特征提取函数 ---------
def deep_amplitude_features_2d(images, H=128, W=128, output_dim=128, device=None):
    """
    images: numpy array [N,H,W] 或 [N,H,W,C] (灰度或RGB)
    返回: numpy array [N, output_dim]
    """
    device = "cuda:0" if (device is None and torch.cuda.is_available()) else "cpu"
    model = Amplitude2DNet(output_dim=output_dim).to(device)
    model.eval()

    tensor_imgs = []
    for img in images:
        if img.ndim == 3:  # RGB 转灰度
            gray = np.mean(img, axis=2).astype(np.float32)
        else:
            gray = img.astype(np.float32)

        # resize 并归一化
        gray_resized = resize(gray, (H, W), mode="reflect", anti_aliasing=True)
        tensor_imgs.append(torch.from_numpy(gray_resized))

    tensor_imgs = torch.stack(tensor_imgs).unsqueeze(1).to(device)  # [B,1,H,W]

    with torch.no_grad():
        feats = model(tensor_imgs)

    return feats.cpu().numpy()


if __name__ == "__main__":
    # 生成随机图像数据 (N,H,W)
    N, H, W = 5, 64, 64
    images = np.random.rand(N, H, W).astype(np.float32) * 400

    print("===== 手工幅度特征提取 =====")
    print("最大值:", get_max(images))
    print("最小值:", get_min(images))
    print("均值:", get_mean(images))
    print("中位数:", get_median(images))
    print("能量:", get_energy(images))
    print("均方根:", get_rms(images))
    print("峰值因子:", get_crest_factor(images))

    print("\n===== 深度幅度特征提取 =====")
    features = deep_amplitude_features_2d(images, H=64, W=64, output_dim=32)
    print("深度幅度特征 shape:", features.shape)
    print("示例特征前10维:", features[0][:10])