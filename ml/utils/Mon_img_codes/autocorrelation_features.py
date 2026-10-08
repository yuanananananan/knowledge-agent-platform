import numpy as np
from scipy.signal import correlate2d
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize


# 1. 二维自相关
def autocorrelation_2d(img, lag_h=1, lag_w=1):
    """
    计算二维自相关在指定滞后 (lag_h, lag_w) 下的值
    img: (B,H,W) or (B,C,H,W)
    return: (B,C)
    """
    if img.ndim == 3:
        img = img[:, None, :, :]
    B, C, H, W = img.shape
    res = []
    for b in range(B):
        ch_vals = []
        for c in range(C):
            x = img[b,c] - np.mean(img[b,c])
            lh = min(lag_h, H-1)
            lw = min(lag_w, W-1)
            ac = np.sum(x[:H-lh, :W-lw]*x[lh:, lw:]) / (np.sum(x**2)+1e-12)
            ch_vals.append(ac)
        res.append(ch_vals)
    return np.array(res)  # (B,C)


# 2. 聚合自相关（将二维自相关按半径聚合成一维向量）
def aggregated_autocorrelation_2d(img, lag_list=[0,1,2,3]):
    """
    对二维自相关按指定 lag_h, lag_w 取值聚合
    return: (B, C, len(lag_list))
    """
    B, C, H, W = img.shape if img.ndim==4 else (*img.shape, 1)
    results = []
    for lag in lag_list:
        ac = autocorrelation_2d(img, lag, lag)  # 或者单独指定 lh, lw
        results.append(ac)
    return np.stack(results, axis=-1)  # (B,C,L)


# 0902 二维自协方差
def autocovariance_2d(img, lag_h=1, lag_w=1):
    """
    二维自协方差指定滞后
    return: (B,C)
    """
    if img.ndim == 3:
        img = img[:, None, :, :]
    B, C, H, W = img.shape
    res = []
    for b in range(B):
        ch_vals = []
        for c in range(C):
            x = img[b,c] - np.mean(img[b,c])
            lh = min(lag_h, H-1)
            lw = min(lag_w, W-1)
            acov = np.sum(x[:H-lh,:W-lw]*x[lh:,lw:])
            ch_vals.append(acov)
        res.append(ch_vals)
    return np.array(res)


# 4. 二维偏自相关（简化版：直接用自相关消去部分中间滞后）
def partial_autocorrelation_2d(img, lag_h=1, lag_w=1):
    """
    计算指定滞后 (lag_h, lag_w) 的二维偏自相关
    简化：PAC = AC(lag_h, lag_w) - AC(lag_h, 0)*AC(0, lag_w) / AC(0,0)
    """
    if img.ndim == 3:  # (B, H, W)
        img = img[:, None, :, :]
    B, C, H, W = img.shape
    pac_values = []
    for b in range(B):
        ch_vals = []
        for c in range(C):
            x = img[b, c] - np.mean(img[b, c])
            def ac(lh, lw):
                return np.sum(
                    x[:H-lh, :W-lw] * x[lh:, lw:]
                ) / np.sum(x**2)
            pac = ac(lag_h, lag_w) - ac(lag_h, 0)*ac(0, lag_w)/ac(0,0)
            ch_vals.append(pac)
        pac_values.append(ch_vals)
    return np.array(pac_values)  # (B, C)


class PatchEmbed(nn.Module):
    """
    将图像分块为 patch 并线性映射到 embedding 空间
    """
    def __init__(self, in_channels=1, embed_dim=64, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, embed_dim, H/patch, W/patch]
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim)
        )

    def forward(self, x):
        # 自注意力
        x_res = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = x + x_res

        # MLP
        x_res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + x_res
        return x


class TransformerAutocorrExtractor(nn.Module):
    """
    使用 Transformer 提取自相关特征（随机初始化，无训练）
    """
    def __init__(self, in_channels=1, embed_dim=64, num_heads=4, depth=2, patch_size=16):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)  # 对 token 维度做平均池化

    def forward(self, x):
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        for blk in self.blocks:
            x = blk(x)
        # 全局平均池化 -> [B, embed_dim]
        x = x.transpose(1, 2)  # [B, embed_dim, num_patches]
        x = self.pool(x).squeeze(-1)
        return x


class DeepAutocorrTransformer:
    def __init__(self, device=None, out_channels=64, batch_size=4):
        self.device = "cuda:0" if (device is None and torch.cuda.is_available()) else "cpu"
        self.batch_size = batch_size
        self.model = TransformerAutocorrExtractor(
            in_channels=1, embed_dim=out_channels, num_heads=4, depth=2, patch_size=16
        ).to(self.device)
        self.model.eval()

    def extract_batch(self, images):
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
            if img.ndim == 3:
                gray = np.mean(img, axis=2).astype(np.float32) / 255.0
            else:
                gray = img.astype(np.float32) / 255.0
            gray_resized = resize(gray, (H, W), mode='reflect', anti_aliasing=True)
            tensor_imgs.append(torch.from_numpy(gray_resized))
        tensor_imgs = torch.stack(tensor_imgs).unsqueeze(1).to(self.device)
        with torch.no_grad():
            feat_vector = self.model(tensor_imgs)
        return feat_vector.cpu().numpy()


def deep_autocorrelation_features(images, out_channels=64, batch_size=4):
    extractor = DeepAutocorrTransformer(out_channels=out_channels, batch_size=batch_size)
    return extractor.extract_batch(images)