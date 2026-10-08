import numpy as np
from scipy.stats import skew, kurtosis
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize


# 1. 极差
def get_range(image):
    return np.max(image, axis=(2)) - np.min(image, axis=(2))

# 2. 方差
def get_variance(image):
    return np.var(image, axis=(2))

# 0902 标准差
def get_std_dev(image):
    return np.std(image, axis=(2))

# 4. 四分位差
def get_iqr(image):
    q3 = np.percentile(image, 75, axis=(2))
    q1 = np.percentile(image, 25, axis=(2))
    return q3 - q1

# 5. 平均绝对偏差
def get_mad(image):
    mean_val = np.mean(image, axis=(2), keepdims=True)
    return np.mean(np.abs(image - mean_val), axis=(2))

# 6. 三阶中心矩
def get_third_central_moment(image):
    mean_val = np.mean(image, axis=(2), keepdims=True)
    return np.mean((image - mean_val)**3, axis=(2))

# 7. 四阶中心矩
def get_fourth_central_moment(image):
    mean_val = np.mean(image, axis=(2), keepdims=True)
    return np.mean((image - mean_val)**4, axis=(2))

# 8. 变异系数
def get_cv(image):
    mean_val = np.mean(image, axis=(2))
    std_val = np.std(image, axis=(2))
    return std_val / mean_val


# ----------------- 序列网络起伏特征 -----------------
class ReliefSeqNet(nn.Module):
    """
    使用 LSTM 提取图像起伏特征
    """
    def __init__(self, hidden_dim=64, output_dim=32, num_layers=1, bidirectional=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=1,      # 每个像素为一个时间步的输入
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim*factor, output_dim)

    def forward(self, x):
        # x: [B, 1, H, W] -> reshape成序列 [B, H*W, 1]
        B, C, H, W = x.shape
        seq = x.view(B, H*W, 1)
        lstm_out, _ = self.lstm(seq)                 # [B, H*W, hidden_dim*factor]
        pooled = torch.mean(lstm_out, dim=1)        # 全局平均池化 -> [B, hidden_dim*factor]
        out = F.normalize(self.fc(pooled), p=2, dim=1)  # [B, output_dim]
        return out

# ----------------- 封装接口 -----------------
class DeepReliefSeqExtractor:
    def __init__(self, device=None, output_dim=64, batch_size=4, hidden_dim=64):
        self.device = "cuda:0" if (device is None and torch.cuda.is_available()) else "cpu"
        self.batch_size = batch_size
        self.model = ReliefSeqNet(hidden_dim=hidden_dim, output_dim=output_dim).to(self.device)
        self.model.eval()

    def extract_batch(self, images):
        """
        使用 batch_size 循环提取特征
        images: List[np.ndarray]，每个元素为 HxW 或 HxWxC
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
        H, W = 224, 224
        tensor_imgs = []

        for img in images:
            # 灰度化 + 归一化
            if img.ndim == 3:
                gray = np.mean(img, axis=2).astype(np.float32) / 255.0
            else:
                gray = img.astype(np.float32) / 255.0
            gray_resized = resize(gray, (H, W), mode='reflect', anti_aliasing=True)
            tensor_imgs.append(torch.from_numpy(gray_resized))

        tensor_imgs = torch.stack(tensor_imgs).unsqueeze(1).to(self.device)  # [B,1,H,W]
        with torch.no_grad():
            feat_vector = self.model(tensor_imgs)
        return feat_vector.cpu().numpy()

# ----------------- 使用示例 -----------------
def deep_relief_seq_features(images, output_dim=8, batch_size=4, hidden_dim=64):
    extractor = DeepReliefSeqExtractor(output_dim=output_dim, batch_size=batch_size, hidden_dim=hidden_dim)
    return extractor.extract_batch(images)