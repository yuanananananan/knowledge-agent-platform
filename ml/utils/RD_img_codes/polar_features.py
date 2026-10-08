import numpy as np
from joblib import Parallel, delayed
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize
from sklearn.decomposition import PCA
import scipy.stats as stats

def fS2toT3(S2):
    """
    Transform to form T3 matrix and apply flat-earth removal.

    Parameters:
        S2: ndarray
            Scattering matrix in H-V polarization basis.
            Dimensions can be:
                - (2, 2): single matrix
                - (N, 2, 2): multiple matrices
                - (N, M, 2, 2): grid of matrices

    Returns:
        T3: ndarray
            Covariance matrix T3.
            Dimensions will correspond to the input:
                - (3, 3) for (2, 2) input
                - (N, 3, 3) for (N, 2, 2) input
                - (N, M, 3, 3) for (N, M, 2, 2) input
    """
    if S2.ndim == 2:  # Single 2x2 matrix
        T3 = np.zeros((3, 3), dtype=np.complex128)

        # Pauli scattering vector
        s1 = (1 / np.sqrt(2)) * (S2[0, 0] + S2[1, 1])
        s2 = (1 / np.sqrt(2)) * (S2[0, 0] - S2[1, 1])
        s3 = (1 / np.sqrt(2)) * 2 * S2[0, 1]

        # Form T3 matrix
        T3[0, 0] = s1 * np.conj(s1)
        T3[0, 1] = s1 * np.conj(s2)
        T3[0, 2] = s1 * np.conj(s3)
        T3[1, 0] = s2 * np.conj(s1)
        T3[1, 1] = s2 * np.conj(s2)
        T3[1, 2] = s2 * np.conj(s3)
        T3[2, 0] = s3 * np.conj(s1)
        T3[2, 1] = s3 * np.conj(s2)
        T3[2, 2] = s3 * np.conj(s3)

    elif S2.ndim == 3:  # Multiple 2x2 matrices
        n = S2.shape[0]
        T3 = np.zeros((n, 3, 3), dtype=np.complex128)

        # Pauli scattering vector
        s1 = (1 / np.sqrt(2)) * (S2[:, 0, 0] + S2[:, 1, 1])
        s2 = (1 / np.sqrt(2)) * (S2[:, 0, 0] - S2[:, 1, 1])
        s3 = (1 / np.sqrt(2)) * 2 * S2[:, 0, 1]

        # Form T3 matrix
        T3[:, 0, 0] = s1 * np.conj(s1)
        T3[:, 0, 1] = s1 * np.conj(s2)
        T3[:, 0, 2] = s1 * np.conj(s3)
        T3[:, 1, 0] = s2 * np.conj(s1)
        T3[:, 1, 1] = s2 * np.conj(s2)
        T3[:, 1, 2] = s2 * np.conj(s3)
        T3[:, 2, 0] = s3 * np.conj(s1)
        T3[:, 2, 1] = s3 * np.conj(s2)
        T3[:, 2, 2] = s3 * np.conj(s3)

    elif S2.ndim == 4:  # Grid of 2x2 matrices
        n, m = S2.shape[:2]
        T3 = np.zeros((n, m, 3, 3), dtype=np.complex128)

        # Pauli scattering vector
        s1 = (1 / np.sqrt(2)) * (S2[:, :, 0, 0] + S2[:, :, 1, 1])
        s2 = (1 / np.sqrt(2)) * (S2[:, :, 0, 0] - S2[:, :, 1, 1])
        s3 = (1 / np.sqrt(2)) * 2 * S2[:, :, 0, 1]

        # Form T3 matrix
        T3[:, :, 0, 0] = s1 * np.conj(s1)
        T3[:, :, 0, 1] = s1 * np.conj(s2)
        T3[:, :, 0, 2] = s1 * np.conj(s3)
        T3[:, :, 1, 0] = s2 * np.conj(s1)
        T3[:, :, 1, 1] = s2 * np.conj(s2)
        T3[:, :, 1, 2] = s2 * np.conj(s3)
        T3[:, :, 2, 0] = s3 * np.conj(s1)
        T3[:, :, 2, 1] = s3 * np.conj(s2)
        T3[:, :, 2, 2] = s3 * np.conj(s3)

    else:
        raise ValueError("Input dimensions not supported!")
    return T3

def fHAlphaADecomp_Modified(T):
    """
    H/alpha/A decomposition based on Cloude and Pottier (1997).

    Input:
        T: Coherency matrix (2D, 3D, or 4D numpy array).

    Output:
        H: Entropy
        alpha: Mean alpha angle
        Ani: Anisotropy
        P: Polarization degree
    """

    dim = T.ndim
    if dim == 2:
        nr, na = T.shape
        if nr != 3 or na != 3:
            raise ValueError("Coherency matrix should be 3x3!")
        nfftr = 1
    elif dim == 3:
        nr, na = T[0, :, :].shape
        if nr != 3 or na != 3:
            raise ValueError("Coherency matrix should be n x 3 x 3!")
        nfftr = T.shape[0]
        H = np.zeros(nfftr)
        alpha = np.zeros(nfftr)
        Ani = np.zeros(nfftr)
        P = np.zeros(nfftr)
    elif dim == 4:
        nr, na = T[0, 0, :, :].shape
        if nr != 3 or na != 3:
            raise ValueError("Coherency matrix should be n x m x 3 x 3!")
        nfftr, nffta = T.shape[:2]
        H = np.zeros((nfftr, nffta))
        alpha = np.zeros((nfftr, nffta))
        Ani = np.zeros((nfftr, nffta))
        P = np.zeros((nfftr, nffta))

    # Decomposition
    if dim == 2:
        T0 = T
        D, V = np.linalg.eig(T0)  # V: eigenvectors, D: eigenvalues
        d = np.real(D)  # Ensure all eigenvalues are real

        p = d / np.sum(d)  # Scattering mechanism probabilities
        p = np.clip(p, 0, 1)  # Ensure probabilities are in range [0, 1]
        H = -np.sum(p * np.log(p + np.finfo(float).eps) / np.log(3))

        # print(V.shape)
        alpha0 = np.arccos(np.abs(V[0, :]))  # Abs of the first row
        alpha = np.sum(p * alpha0)

        p_sorted = np.sort(p)[::-1]
        if p_sorted[1] == 0 and p_sorted[2] == 0:
            Ani = 0
        else:
            Ani = (p_sorted[1] - p_sorted[2]) / (p_sorted[1] + p_sorted[2])

        if p_sorted[0] == 0 and p_sorted[1] == 0:
            P = 0
        else:
            P = (p_sorted[0] - p_sorted[1]) / (p_sorted[0] + p_sorted[1])

    elif dim == 3:
        for n in range(nfftr):
            T0 = T[n, :, :]
            D, V = np.linalg.eig(T0)
            d = np.real(D)

            p = d / np.sum(d)
            p = np.clip(p, 0, 1)
            H[n] = -np.sum(p * np.log(p + np.finfo(float).eps) / np.log(3))

            alpha0 = np.arccos(np.abs(V[0, :]))
            alpha[n] = np.sum(p * alpha0)

            p_sorted = np.sort(p)[::-1]
            if p_sorted[1] == 0 and p_sorted[2] == 0:
                Ani[n] = 0
            else:
                Ani[n] = (p_sorted[1] - p_sorted[2]) / (p_sorted[1] + p_sorted[2])

            if p_sorted[0] == 0 and p_sorted[1] == 0:
                P[n] = 0
            else:
                P[n] = (p_sorted[0] - p_sorted[1]) / (p_sorted[0] + p_sorted[1])

    elif dim == 4:
        for n in range(nfftr):
            for m in range(nffta):
                T0 = T[n, m, :, :]
                D, V = np.linalg.eig(T0)
                d = np.real(D)

                p = d / np.sum(d)
                p = np.clip(p, 0, 1)
                H[n, m] = -np.sum(p * np.log(p + np.finfo(float).eps) / np.log(3))

                alpha0 = np.arccos(np.abs(V[0, :]))
                alpha[n, m] = np.sum(p * alpha0)

                p_sorted = np.sort(p)[::-1]
                if p_sorted[1] == 0 and p_sorted[2] == 0:
                    Ani[n, m] = 0
                else:
                    Ani[n, m] = (p_sorted[1] - p_sorted[2]) / (p_sorted[1] + p_sorted[2])

                if p_sorted[0] == 0 and p_sorted[1] == 0:
                    P[n, m] = 0
                else:
                    P[n, m] = (p_sorted[0] - p_sorted[1]) / (p_sorted[0] + p_sorted[1])

    alpha = alpha * 180 / np.pi  # Convert alpha to degrees
    return H, alpha, Ani, P

def fCalcuPolSimilarity(S1, S2):
    """
    计算批量极化散射矩阵 S1 与一个参考矩阵 S2 的相似性指标。

    Parameters:
        S1: ndarray of shape (N, L, 2, 2)
            批量极化散射矩阵。
        S2: ndarray of shape (2, 2)
            单个参考极化散射矩阵。

    Returns:
        r: ndarray of shape (N, L)
            每个 S1[i,j,:,:] 与 S2 的相似性指标。
    """
    sqrt2_inv = 1 / np.sqrt(2)

    # 计算 S2 的 Pauli 矢量（长度为3）
    k2 = sqrt2_inv * np.array([
        S2[0, 0] + S2[1, 1],
        S2[0, 0] - S2[1, 1],
        S2[0, 1] + S2[1, 0]
    ], dtype=np.complex128)

    # 提取 S1 的各分量
    S_HH = S1[:, :, 0, 0]
    S_HV = S1[:, :, 0, 1]
    S_VH = S1[:, :, 1, 0]
    S_VV = S1[:, :, 1, 1]

    # 构造 Pauli 矢量 k1: shape (N, L, 3)
    k1 = sqrt2_inv * np.stack([
        S_HH + S_VV,
        S_HH - S_VV,
        S_HV + S_VH
    ], axis=-1)  # shape (N, L, 3)

    # 点积并取模平方：r = |k1·k2*|² / (||k1||² * ||k2||²)
    dot_product = np.sum(k1 * k2.conj(), axis=-1)  # shape (N, L)
    numerator = np.abs(dot_product) ** 2

    norm_k1_sq = np.sum(np.abs(k1) ** 2, axis=-1)  # shape (N, L)
    norm_k2_sq = np.sum(np.abs(k2) ** 2)  # scalar

    r = numerator / (norm_k1_sq * norm_k2_sq + 1e-12)  # 防止除0
    return r



# -------------------- H/Alpha/Ani/P 特征 --------------------
def _compute_H(img):
    H, W = img.shape[1], img.shape[2]
    S2 = np.zeros((H, W, 2, 2), dtype=np.complex128)
    S2[:, :, 0, 0] = img[0]; S2[:, :, 0, 1] = img[1]
    S2[:, :, 1, 0] = img[2]; S2[:, :, 1, 1] = img[3]
    T3 = fS2toT3(S2)
    T = np.mean(T3, axis=(0, 1))
    H_val, alpha, Ani, P = fHAlphaADecomp_Modified(T)

    out = [H_val, alpha, Ani, P]
    # return H_val
    return out

def polar_H(data, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(delayed(_compute_H)(data[i]) for i in range(data.shape[0])))


def _compute_alpha(img):
    H, W = img.shape[1], img.shape[2]
    S2 = np.zeros((H, W, 2, 2), dtype=np.complex128)
    S2[:, :, 0, 0] = img[0]; S2[:, :, 0, 1] = img[1]
    S2[:, :, 1, 0] = img[2]; S2[:, :, 1, 1] = img[3]
    T3 = fS2toT3(S2)
    T = np.mean(T3, axis=(0, 1))
    _, alpha, _, _ = fHAlphaADecomp_Modified(T)
    return alpha

def polar_alpha(data, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(delayed(_compute_alpha)(data[i]) for i in range(data.shape[0])))


def _compute_Ani(img):
    H, W = img.shape[1], img.shape[2]
    S2 = np.zeros((H, W, 2, 2), dtype=np.complex128)
    S2[:, :, 0, 0] = img[0]; S2[:, :, 0, 1] = img[1]
    S2[:, :, 1, 0] = img[2]; S2[:, :, 1, 1] = img[3]
    T3 = fS2toT3(S2)
    T = np.mean(T3, axis=(0, 1))
    _, _, Ani, _ = fHAlphaADecomp_Modified(T)
    return Ani

def polar_Ani(data, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(delayed(_compute_Ani)(data[i]) for i in range(data.shape[0])))


def _compute_P(img):
    H, W = img.shape[1], img.shape[2]
    S2 = np.zeros((H, W, 2, 2), dtype=np.complex128)
    S2[:, :, 0, 0] = img[0]; S2[:, :, 0, 1] = img[1]
    S2[:, :, 1, 0] = img[2]; S2[:, :, 1, 1] = img[3]
    T3 = fS2toT3(S2)
    T = np.mean(T3, axis=(0, 1))
    _, _, _, P = fHAlphaADecomp_Modified(T)
    return P

def polar_P(data, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(delayed(_compute_P)(data[i]) for i in range(data.shape[0])))


def extract_statistical_features(similarity_map):
    features = []
    
    # 全局统计特征
    features.extend([np.mean(similarity_map), np.std(similarity_map), 
                    np.median(similarity_map), np.max(similarity_map), 
                    np.min(similarity_map)])
    
    # 分块统计特征
    h, w = similarity_map.shape
    block_h, block_w = h//4, w//4 # 分为4x4=16个块
    
    for i in range(4):
        for j in range(4):
            block = similarity_map[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            block = block.flatten()
            features.extend([np.mean(block), 
                             np.std(block), 
                             np.median(block), 
                             np.max(block), 
                             np.min(block),
                             stats.skew(block),
                             stats.kurtosis(block),    
                             stats.variation(block),
                             ])
            # features.extend([np.mean(block), np.std(block)])
    
    return np.array(features)

def stats_features(x):
    return [np.mean(x), np.std(x), np.min(x), np.max(x), np.median(x), np.var(x)]

# -------------------- 散射矩阵相似性 r1~r7 --------------------
def _compute_r(img, S2_ref):
    H, W = img.shape[1], img.shape[2]
    S2 = np.zeros((H, W, 2, 2), dtype=np.complex128)
    S2[:, :, 0, 0] = img[0]; S2[:, :, 0, 1] = img[1]
    S2[:, :, 1, 0] = img[2]; S2[:, :, 1, 1] = img[3]
    similarity_map = fCalcuPolSimilarity(S2, S2_ref)
    similarity_map = extract_statistical_features(similarity_map)
    # return np.mean(fCalcuPolSimilarity(S2, S2_ref))
    return similarity_map

def polar_r(data, S2_ref, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(delayed(_compute_r)(data[i], S2_ref) for i in range(data.shape[0])))

def polar_r1(data, n_jobs=-1):
    S2_ref = np.array([[1,0],[0,1]], dtype=np.complex128)
    return polar_r(data, S2_ref, n_jobs)

def polar_r2(data, n_jobs=-1):
    S2_ref = np.array([[1,0],[0,-1]], dtype=np.complex128)
    return polar_r(data, S2_ref, n_jobs)

def polar_r3(data, n_jobs=-1):
    S2_ref = np.array([[1,0],[0,0]], dtype=np.complex128)
    return polar_r(data, S2_ref, n_jobs)

def polar_r4(data, n_jobs=-1):
    S2_ref = np.array([[2,0],[0,1]], dtype=np.complex128)
    return polar_r(data, S2_ref, n_jobs)

def polar_r5(data, n_jobs=-1):
    S2_ref = np.array([[1,1j],[1j,-1]], dtype=np.complex128)
    return polar_r(data, S2_ref, n_jobs)

def polar_r6(data, n_jobs=-1):
    S2_ref = np.array([[1,-1j],[-1j,-1]], dtype=np.complex128)
    return polar_r(data, S2_ref, n_jobs)

def polar_r7(data, n_jobs=-1):
    S2_ref = np.array([[0,1],[1,0]], dtype=np.complex128)
    return polar_r(data, S2_ref, n_jobs)


# -------------------- 散射能量、行列式、去极化 --------------------
def _compute_power(img):
    H, W = img.shape[1], img.shape[2]
    hrrp = img.reshape(4, H*W)
    power_trace = np.sum(np.abs(hrrp)**2, axis=0)
    # return np.mean(power_trace)
    return stats_features(power_trace)

def polar_power(data, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(delayed(_compute_power)(data[i]) for i in range(data.shape[0])))


def _compute_det(img):
    H, W = img.shape[1], img.shape[2]
    hrrp = img.reshape(4, H*W)
    det = np.abs(hrrp[0]*hrrp[3] - hrrp[1]*hrrp[2])
    # return np.mean(det)
    return stats_features(det)

def polar_det(data, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(delayed(_compute_det)(data[i]) for i in range(data.shape[0])))


def _compute_depol(img):
    H, W = img.shape[1], img.shape[2]
    hrrp = img.reshape(4, H*W)
    SHH, SHV, SVH, SVV = hrrp
    hv_equal = np.allclose(SHV, SVH, atol=1e-8)
    if hv_equal:
        numerator = 0.5 * (np.abs(SHH - SVV)**2 + 2*np.abs(SHV)**2)
        denominator = np.abs(SHH)**2 + np.abs(SVV)**2 + 2*np.abs(SHV)**2
    else:
        numerator = np.abs(SHH - SVV)**2 + 2*(np.abs(SHV)**2 + np.abs(SVH)**2)
        denominator = 2*(np.abs(SHH)**2 + np.abs(SHV)**2 + np.abs(SVH)**2 + np.abs(SVV)**2)
    P3 = numerator / (denominator + 1e-12)
    # return np.mean(P3)
    return stats_features(P3)

def polar_depol(data, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(delayed(_compute_depol)(data[i]) for i in range(data.shape[0])))


# -------------------- 本征极化椭圆率、方向角 --------------------
def _compute_ellipticity(img, theta):
    H, W = img.shape[1], img.shape[2]
    hrrp = img.reshape(4, H*W)
    SHH, SHV, SVH, SVV = hrrp
    c, s = np.cos(theta), np.sin(theta)
    HH_rot = c**2*SHH + s**2*SVV + 2*c*s*SHV
    VV_rot = s**2*SHH + c**2*SVV - 2*c*s*SHV
    HV_rot = -c*s*SHH + c*s*SVV + (c**2 - s**2)*SHV
    numerator = 1j * 2 * HV_rot
    denominator = HH_rot + VV_rot
    P4 = 0.5 * np.arctan2(np.imag(numerator/denominator),
                           np.real(numerator/denominator))
    # return np.mean(P4)
    return stats_features(P4)


def polar_ellipticity(data, theta=np.pi / 2, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(delayed(_compute_ellipticity)(data[i], theta) for i in range(data.shape[0])))


def _compute_orientation(img, theta):
    H, W = img.shape[1], img.shape[2]
    hrrp = img.reshape(4, H*W)
    SHH, SHV, SVH, SVV = hrrp
    c, s = np.cos(theta), np.sin(theta)
    HH_rot = c**2*SHH + s**2*SVV + 2*c*s*SHV
    VV_rot = s**2*SHH + c**2*SVV - 2*c*s*SHV
    HV_rot = -c*s*SHH + c*s*SVV + (c**2 - s**2)*SHV
    numerator = 2 * np.real(np.conj(HH_rot + VV_rot) * HV_rot)
    denominator = np.real(np.conj(HH_rot + VV_rot) * (HH_rot - VV_rot))
    P5 = 0.5 * np.arctan2(numerator, denominator)
    # return np.mean(P5)
    return stats_features(P5)

def polar_orientation(data, theta=0.0, n_jobs=-1):
    return np.array(Parallel(n_jobs=n_jobs)(delayed(_compute_orientation)(data[i], theta) for i in range(data.shape[0])))



# -------------------- 极化注意力 --------------------
class PolarizationAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1)

    def forward(self, x):
        # x: [B, C, H, W]
        pol_feat = torch.mean(x, dim=2)  # 沿 H 池化 -> [B, C, W]
        attn = F.relu(self.fc1(pol_feat))
        attn = torch.sigmoid(self.fc2(attn))
        attn = attn.unsqueeze(2)  # [B, C, 1, W]
        return x * attn

# -------------------- 极化特征网络 --------------------
class PolarFeature2D(nn.Module):
    def __init__(self, in_channels=4, out_channels=64):
        super().__init__()
        c1 = out_channels // 3
        c2 = out_channels // 3
        c3 = out_channels - c1 - c2

        self.conv3 = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, c2, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, c3, kernel_size=7, padding=3)

        self.pol_attn = PolarizationAttention(out_channels)

        self.pool_energy = nn.AdaptiveAvgPool2d(1)
        self.pool_peak = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        f3 = F.relu(self.conv3(x))
        f5 = F.relu(self.conv5(x))
        f7 = F.relu(self.conv7(x))
        feat = torch.cat([f3, f5, f7], dim=1)

        feat = self.pol_attn(feat)

        energy_feat = self.pool_energy(feat).squeeze(-1).squeeze(-1)
        peak_feat = self.pool_peak(feat).squeeze(-1).squeeze(-1)

        return torch.cat([energy_feat, peak_feat], dim=1)

# -------------------- 极化深度特征提取器 --------------------
class DeepPolarExtractor:
    def __init__(self, device=None, out_channels=64, batch_size=4):
        self.device = "cuda:0" if (device is None and torch.cuda.is_available()) else "cpu"
        self.batch_size = batch_size
        self.model = PolarFeature2D(in_channels=4, out_channels=out_channels).to(self.device)
        self.model.eval()

    def extract_batch(self, pol_images):
        N = len(pol_images)
        feat_list = []

        for i in range(0, N, self.batch_size):
            mini_batch = pol_images[i:i+self.batch_size]
            feat_mini = self._extract_mini_batch(mini_batch)
            feat_list.append(feat_mini)
            torch.cuda.empty_cache()

        return np.vstack(feat_list)

    def _extract_mini_batch(self, pol_images):
        H, W = 128, 128
        tensor_imgs = []

        for img in pol_images:
            img_norm = img.astype(np.float32)
            img_norm /= (np.max(np.abs(img_norm)) + 1e-6)
            # 保持通道数不变，resize H, W
            img_resized = np.stack([resize(img_norm[c], (H, W), mode='reflect', anti_aliasing=True)
                                    for c in range(img_norm.shape[0])], axis=0)
            tensor_imgs.append(torch.from_numpy(img_resized))

        tensor_imgs = torch.stack(tensor_imgs).to(self.device)  # [B, 9, H, W]
        with torch.no_grad():
            feat_vector = self.model(tensor_imgs)
        return feat_vector.cpu().numpy()

# -------------------- 调用接口 --------------------
def deep_polar_features(pol_images, out_channels=16, batch_size=4):
    extractor = DeepPolarExtractor(out_channels=out_channels, batch_size=batch_size)
    return extractor.extract_batch(pol_images)