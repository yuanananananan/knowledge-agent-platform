import numpy as np

# 径向长度
def radial_length(x, threshold_ratio=0.5):
    """
    批量计算 HRRP 四极化通道的径向长度
    Args:
        x: ndarray, shape [B, 4, D]
        threshold_ratio: 阈值比例 (默认 0.5)
    Returns:
        radial_lengths: ndarray, shape [B, 4]  每个样本每个通道的径向长度
    """
    B, C, D = x.shape
    radial_lengths = np.zeros((B, C), dtype=int)

    for i in range(B):
        for j in range(C):
            seq = x[i, j]
            threshold = threshold_ratio * np.max(seq)
            indices = np.where(seq >= threshold)[0]
            if len(indices) > 0:
                radial_lengths[i, j] = indices.max() - indices.min()
            else:
                radial_lengths[i, j] = 0

    return radial_lengths


# 散射质心
def scattering_centroid(x, n1=None, n2=None):
    """
    批量计算 HRRP 四极化通道的散射质心归一化位置
    Args:
        x: ndarray, shape [B, 4, D]
        n1, n2: 计算区间 [n1, n2]
    Returns:
        M: ndarray, shape [B, 4] 每个样本每个通道的归一化质心位置
    """
    B, C, D = x.shape

    if n1 is None:
        n1 = 0
    if n2 is None:
        n2 = D - 1

    positions = np.arange(n1, n2 + 1)  # [L]
    M = np.zeros((B, C), dtype=float)

    for i in range(B):
        for j in range(C):
            segment = x[i, j, n1:n2 + 1]
            seg_sum = np.sum(segment)
            if seg_sum != 0:
                weighted_mean_pos = np.sum(positions * segment) / seg_sum
                M[i, j] = (weighted_mean_pos - n1) / (n2 - n1)
            else:
                M[i, j] = 0.0

    return M

# 强散射中心数
def strong_scattering_centers(x, n1=None, n2=None):
    """
    批量计算 HRRP 四极化通道的强散射中心数量（完全向量化）
    
    Args:
        x: ndarray, shape [B, 4, D]
        n1: 区间起始索引，默认从 0 开始
        n2: 区间结束索引，默认到 D-1
    Returns:
        counts: ndarray, shape [B, 4] 每个样本每个通道的强散射中心数量
    """
    B, C, D = x.shape
    
    if n1 is None:
        n1 = 0
    if n2 is None:
        n2 = D - 1
    
    # 截取区间
    segment = x[:, :, n1:n2 + 1]  # [B, 4, L]
    
    # 左右移一位比较，判断峰值
    left = segment[:, :, :-2]     # [B, 4, L-2]
    center = segment[:, :, 1:-1]  # [B, 4, L-2]
    right = segment[:, :, 2:]      # [B, 4, L-2]
    
    peaks = (center > left) & (center > right)  # [B, 4, L-2]
    
    # 统计每个样本每个通道的峰值数量
    counts = np.sum(peaks, axis=-1)  # [B, 4]
    
    return counts

# 两最强散射中心的距离
def two_strongest_scattering_distance(x, n1=None, n2=None, delta_d=1.0):
    """
    批量计算 HRRP 四极化通道最强两个散射中心的距离
    Args:
        x: ndarray, shape [B, 4, D]
        n1: 区间起始索引，默认从 0 开始
        n2: 区间结束索引，默认到 D-1
        delta_d: 距离分辨率
    Returns:
        DPK: ndarray, shape [B, 4] 每个样本每个通道最强两个散射中心的距离
    """
    B, C, D = x.shape
    if n1 is None:
        n1 = 0
    if n2 is None:
        n2 = D - 1

    # 截取区间
    segment = x[:, :, n1:n2 + 1]  # [B, 4, L]
    L = n2 - n1 + 1

    # 获取每个样本每个通道的 top2 索引
    top2_indices = np.argpartition(segment, -2, axis=-1)[..., -2:]  # [B, 4, 2]

    # 将索引映射到原始 positions
    positions = np.arange(n1, n2 + 1)  # [L]
    # 取对应的原始位置
    m_vals = positions[top2_indices]   # [B, 4, 2]

    # 计算两个最大值的绝对距离
    DPK = delta_d * np.abs(m_vals[..., 0] - m_vals[..., 1])  # [B, 4]

    return DPK

# 最强散射中心与前端距离
def strongest_scattering_front_distance(x, n1=0, delta_d=1.0):
    """
    批量计算 HRRP 四极化通道最强散射中心到前端的距离
    Args:
        x: ndarray, shape [B, 4, D]
        n1: 区间起始索引，用于计算相对距离
        delta_d: 距离分辨率
    Returns:
        DEP: ndarray, shape [B, 4] 每个样本每个通道最强散射中心到前端的距离
    """
    # 截取从 n1 到序列末尾的区间
    segment = x[:, :, n1:]  # [B, 4, L]

    # 找到每个样本每个通道最大值的索引
    max_indices = np.argmax(segment, axis=-1)  # [B, 4]

    # 计算距离
    DEP = delta_d * max_indices

    return DEP

# 强散射中心的幅值分布熵
def strong_scattering_entropy(x, n1=0, n2=None):
    """
    批量计算 HRRP 四极化通道强散射中心的幅值分布熵 EA
    Args:
        x: ndarray, shape [B, 4, D]
        n1: 区间起始索引
        n2: 区间结束索引，如果为 None 则到序列末尾
    Returns:
        ea: ndarray, shape [B, 4] 每个样本每个通道的强散射幅值熵
    """
    B, C, D = x.shape
    if n2 is None:
        n2 = D - 1
    
    segment = x[:, :, n1:n2 + 1]  # [B, 4, L]
    
    # 找峰值（比左右相邻点大）
    left = segment[:, :, :-2]      # [B, 4, L-2]
    center = segment[:, :, 1:-1]   # [B, 4, L-2]
    right = segment[:, :, 2:]      # [B, 4, L-2]
    
    peaks_mask = (center > left) & (center > right)  # [B, 4, L-2]
    
    # 获取峰值幅度
    center_vals = center * peaks_mask  # 非峰值位置置 0
    
    # 为每个样本每个通道计算熵
    # 先计算归一化概率 p'_mi
    sum_vals = np.sum(center_vals, axis=-1, keepdims=True)  # [B, 4, 1]
    p_prime = np.where(sum_vals != 0, center_vals / sum_vals, 0.0)  # [B, 4, L-2]
    
    # 计算熵
    ea = -np.sum(p_prime * np.log2(p_prime + 1e-10), axis=-1)  # [B, 4]
    
    return ea

# 强散射中心的位置分布熵
def strong_scattering_position_entropy(x, n1=0, n2=None):
    """
    批量计算 HRRP 四极化通道强散射中心的位置分布熵 EP
    Args:
        x: ndarray, shape [B, 4, D]
        n1: 区间起始索引
        n2: 区间结束索引，如果为 None 则到序列末尾
    Returns:
        ep: ndarray, shape [B, 4] 每个样本每个通道的强散射位置熵
    """
    B, C, D = x.shape
    if n2 is None:
        n2 = D - 1
    
    segment = x[:, :, n1:n2 + 1]  # [B, 4, L]
    
    # 找峰值（比左右相邻点大）
    left = segment[:, :, :-2]      # [B, 4, L-2]
    center = segment[:, :, 1:-1]   # [B, 4, L-2]
    right = segment[:, :, 2:]      # [B, 4, L-2]
    
    peaks_mask = (center > left) & (center > right)  # [B, 4, L-2]
    
    # 获取峰值的归一化位置
    positions = np.arange(1, segment.shape[2] - 1) + n1  # 相对于整个序列
    m_prime = np.where(peaks_mask, (positions[np.newaxis, np.newaxis, :] - n1) / (n2 - n1), 0.0)
    
    # 计算熵
    ep = -np.sum(m_prime * np.log2(m_prime + 1e-10), axis=-1)  # [B, 4]
    
    return ep

def strong_scattering_position_entropy(x, n1=0, n2=None):
    B, C, D = x.shape
    if n2 is None:
        n2 = D - 1
    segment = x[:, :, n1:n2 + 1]
    left = segment[:, :, :-2]    
    center = segment[:, :, 1:-1] 
    right = segment[:, :, 2:]  
    peaks_mask = (center > left) & (center > right)
    positions = np.arange(1, segment.shape[2] - 1) + n1
    m_prime = np.where(peaks_mask, (positions[np.newaxis, np.newaxis, :] - n1) / (n2 - n1), 0.0)
    ep = -np.sum(m_prime * np.log2(m_prime + 1e-10), axis=-1) 

    # # 最大熵
    # max_entropy = np.log2(segment.shape[2] - 2 + 1e-10)

    # # 归一化到 [0,1]
    # return ep / max_entropy

    return ep
















