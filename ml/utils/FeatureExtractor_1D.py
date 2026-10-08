import pywt
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
from scipy.stats import skew, kurtosis

feature_name = [
    # 统计域
    'mean',
    'std',
    'var',
    'max',
    'min',
    'median',
    'skew',
    'kurtosis',
    'mean_absolute_error_from_median',
    'mean_absolute_error_from_mean',
    'median_absolute_deviation',

    # 谱域
    'fft_mean',
    'spectral_fundamental',
    'spectral_max',
    'spectral_median',
    'spectral_max_peak',
    'spectral_distance',
    'wavelet_abs_mean',
    'wavelet_std',
    'wavelet_var',

    # 时域
    'mean_diff',
    'mean_abs_diff',
    'median_diff',
    'median_abs_diff',
    'sum_abs_diff',
    'autocorr_lag1',
    'centroid',
    'entropy',
    'peak2peak',
    'area',
    'zcr',
    'n_peaks',
    'n_valleys'
]


class FeatureExtractor:

    def __init__(self):
        """
        初始化特征提取器
        :param feature_names: List[str]，例如 ['mean', 'std', 'skew']
        """
        self.feature_fuctions = {
            # 统计域
            'mean': lambda x: np.mean(x, axis=-1),
            'std': lambda x: np.std(x, axis=-1),
            'var': lambda x: np.var(x, axis=-1),
            'max': lambda x: np.max(x, axis=-1),
            'min': lambda x: np.min(x, axis=-1),
            'median': lambda x: np.median(x, axis=-1),
            'skew': lambda x: skew(x, axis=-1),
            'kurtosis': lambda x: kurtosis(x, axis=-1),
            'mean_absolute_error_from_median': self.mean_absolute_error_from_median,
            'mean_absolute_error_from_mean': self.mean_absolute_error_from_mean,
            'median_absolute_deviation': self.median_absolute_deviation,

            # 谱域
            'fft_mean': self.fft_mean_coefficient,
            'spectral_fundamental': self.spectral_fundamental_frequency,
            'spectral_max': self.spectral_max_frequency,
            'spectral_median': self.spectral_median_frequency,
            'spectral_max_peak': self.spectral_max_peak,
            'spectral_distance': self.spectral_distance,
            'wavelet_abs_mean': self.wavelet_absolute_mean,
            'wavelet_std': self.wavelet_std,
            'wavelet_var': self.wavelet_var,

            # 时域
            'mean_diff': self.mean_differences,
            'mean_abs_diff': self.mean_absolute_differences,
            'median_diff': self.median_differences,
            'median_abs_diff': self.median_absolute_differences,
            'sum_abs_diff': self.sum_absolute_differences,
            'autocorr_lag1': self.autocorrelation_lag1,
            'centroid': self.signal_centroid,
            'entropy': self.signal_entropy,
            'peak2peak': self.peak_to_peak_distance,
            'area': self.area_under_curve,
            'zcr': self.zero_crossing_rate,
            'n_peaks': self.count_max_peaks,
            'n_valleys': self.count_min_peaks,
        }

    # 基于均值的平均绝对误差
    @staticmethod
    def mean_absolute_error_from_mean(data):
        mean = np.mean(data, axis=-1, keepdims=True)
        return np.mean(np.abs(data - mean), axis=-1)

    # 基于中位数的平均绝对误差
    @staticmethod
    def mean_absolute_error_from_median(data):
        median = np.median(data, axis=-1, keepdims=True)
        return np.mean(np.abs(data - median), axis=-1)

    # 中位绝对偏差（MAD）
    @staticmethod
    def median_absolute_deviation(data):
        median = np.median(data, axis=-1, keepdims=True)
        return np.median(np.abs(data - median), axis=-1)

    # FFT 幅度均值
    # 含义：频域中所有频率分量的平均强度（幅值）。
    # 物理意义：衡量信号整体在频域上的“能量分布水平”，越大表示高频和低频均有显著能量。
    @staticmethod
    def fft_mean_coefficient(data):
        fft_vals = np.abs(fft(data, axis=-1))
        half_len = data.shape[-1] // 2
        return np.mean(fft_vals[..., :half_len], axis=-1)

    # 小波绝对均值
    # 含义：小波分解系数绝对值的均值。
    # 物理意义：反映信号在多尺度上的整体能量大小，对突变（边缘/异常）也敏感。
    @staticmethod
    def wavelet_absolute_mean(data, wavelet='db4'):
        B, R, _ = data.shape
        return np.array([[np.mean(np.abs(np.concatenate(pywt.wavedec(data[b, r], wavelet)))) for r in range(R)]
                         for b in range(B)])

    # 小波标准差
    # 含义：小波分解系数的标准差。
    # 物理意义：反映不同尺度上信号能量的离散程度，高值说明信号变化剧烈或多频率混合。
    @staticmethod
    def wavelet_std(data, wavelet='db4'):

        B, R, _ = data.shape
        return np.array([[np.std(np.concatenate(pywt.wavedec(data[b, r], wavelet))) for r in range(R)]
                         for b in range(B)])

    # 小波方差
    # 含义：小波分解系数的方差。
    # 物理意义：与标准差类似，代表多尺度下信号强度变化的离散程度。
    @staticmethod
    def wavelet_var(data, wavelet='db4'):

        B, R, _ = data.shape
        return np.array([[np.var(np.concatenate(pywt.wavedec(data[b, r], wavelet))) for r in range(R)]
                         for b in range(B)])

    # 频谱基频
    # 含义：频谱中第一个主要峰值对应的频率。
    # 物理意义：最主要的振动频率，通常表示周期性现象的基本频率。
    @staticmethod
    def spectral_fundamental_frequency(data, fs=1.0):

        B, R, D = data.shape
        freqs = fftfreq(D, d=1 / fs)[:D // 2]
        result = np.zeros((B, R))
        for b in range(B):
            for r in range(R):
                mag = np.abs(fft(data[b, r]))[:D // 2]
                peaks, _ = find_peaks(mag)
                result[b, r] = freqs[peaks[0]] if len(peaks) > 0 else 0
        return result

    # 频谱最大频率
    # 含义：频谱中幅值最大的点对应的频率。
    # 物理意义：信号最强烈成分出现在哪个频率位置，表征主导振动。
    @staticmethod
    def spectral_max_frequency(data, fs=1.0):
        D = data.shape[-1]
        freqs = fftfreq(D, d=1 / fs)[:D // 2]
        mag = np.abs(fft(data, axis=-1))[..., :D // 2]
        return freqs[np.argmax(mag, axis=-1)]

    # 频谱中频
    # 含义：将频谱能量一分为二的频率点。
    # 物理意义：反映频域上的“能量中心”，与质心类似。
    @staticmethod
    def spectral_median_frequency(data, fs=1.0):
        B, R, D = data.shape
        freqs = fftfreq(D, d=1 / fs)[:D // 2]
        results = np.zeros((B, R))
        for b in range(B):
            for r in range(R):
                mag = np.abs(fft(data[b, r]))[:D // 2]
                cum = np.cumsum(mag)
                idx = np.searchsorted(cum, cum[-1] / 2)
                results[b, r] = freqs[idx] if idx < len(freqs) else 0
        return results

    # 谱最大峰值
    # 含义：频域中最大幅度。
    # 物理意义：衡量信号中最强的频率分量强度。
    @staticmethod
    def spectral_max_peak(data):
        return np.max(np.abs(fft(data, axis=-1))[..., :data.shape[-1] // 2], axis=-1)

    # 谱距离（相对于全1参考谱）
    # 含义：频谱与理想平坦谱之间的距离。
    # 物理意义：越大表示频谱集中、变化剧烈；越小表示能量分布均匀。
    @staticmethod
    def spectral_distance(data, reference=None):
        spec = np.abs(fft(data, axis=-1))[..., :data.shape[-1] // 2]
        if reference is None:
            reference = np.ones(spec.shape[-1])
        return np.linalg.norm(spec - reference, axis=-1)

    # 自相关（lag=1）
    # 含义：当前值与前一时刻值的相关程度。
    # 物理意义：衡量时间序列的平稳性/周期性，越接近 1 表示趋势平稳。
    @staticmethod
    def autocorrelation_lag1(data):
        x1 = data[..., :-1]
        x2 = data[..., 1:]
        mean1 = np.mean(x1, axis=-1, keepdims=True)
        mean2 = np.mean(x2, axis=-1, keepdims=True)
        num = np.sum((x1 - mean1) * (x2 - mean2), axis=-1)
        den = np.sqrt(np.sum((x1 - mean1)**2, axis=-1) * np.sum((x2 - mean2)**2, axis=-1))
        return np.where(den == 0, 0, num / (den + 1e-8))

    # 质心（信号重心）
    # 含义：加权平均位置（按振幅绝对值加权）。
    # 物理意义：信号“能量”在时间轴上的集中位置，是时间的“重心”。
    @staticmethod
    def signal_centroid(data):
        D = data.shape[-1]
        index = np.arange(D)
        weighted = np.sum(np.abs(data) * index, axis=-1)
        total = np.sum(np.abs(data), axis=-1)
        return np.where(total == 0, 0, weighted / total)

    # 差分均值
    # 含义：相邻值的平均变化。
    # 物理意义：表示趋势的平均增长或下降速度（斜率趋势）。
    @staticmethod
    def mean_differences(data):
        return np.mean(np.diff(data, axis=-1), axis=-1)

    # 差分绝对值均值
    # 含义：相邻值变化的平均绝对值。
    # 物理意义：信号“抖动”或波动幅度的度量，稳定性指标。
    @staticmethod
    def mean_absolute_differences(data):
        return np.mean(np.abs(np.diff(data, axis=-1)), axis=-1)

    # 差分中位数
    # 含义：一阶差分的中位数。
    # 物理意义：比均值更稳健地衡量局部趋势变化。
    @staticmethod
    def median_differences(data):
        return np.median(np.diff(data, axis=-1), axis=-1)

    # 差分绝对值中位数
    # 含义：一阶差分绝对值的中位数。
    # 物理意义：稳健度量信号局部波动大小，受异常值影响小。
    @staticmethod
    def median_absolute_differences(data):
        return np.median(np.abs(np.diff(data, axis=-1)), axis=-1)

    # 差分绝对值之和
    # 含义：所有相邻差分绝对值的总和。
    # 物理意义：总变化量（全程变化幅度），反映整体活跃程度。
    @staticmethod
    def sum_absolute_differences(data):
        return np.sum(np.abs(np.diff(data, axis=-1)), axis=-1)

    # 熵（直方图）
    # 含义：信号概率分布的熵值。
    # 物理意义：衡量信号的复杂性或不确定性。高熵 = 混乱；低熵 = 规则性强。
    @staticmethod
    def signal_entropy(data, bins=50):
        B, R, _ = data.shape
        results = np.zeros((B, R))
        for b in range(B):
            for r in range(R):
                hist, _ = np.histogram(data[b, r], bins=bins, density=True)
                hist = hist[hist > 0]
                results[b, r] = -np.sum(hist * np.log(hist))
        return results

    # 波峰与波谷距离
    # 含义：最大值和最小值之间的差。
    # 物理意义：信号的“动态范围”或最大摆幅。
    @staticmethod
    def peak_to_peak_distance(data):
        return np.max(data, axis=-1) - np.min(data, axis=-1)

    # 曲线覆盖面积
    # 含义：信号曲线的绝对面积（数值积分）。
    # 物理意义：整体信号能量（积分意义下的能量或强度）。
    @staticmethod
    def area_under_curve(data):
        return trapezoid(np.abs(data), axis=-1)

    # 跨零率
    # 含义：信号穿过零点的次数。
    # 物理意义：衡量高频成分或振荡活跃度（如音频高频特征指标）。
    @staticmethod
    def zero_crossing_rate(data):
        signs = np.signbit(data)
        return np.mean(np.diff(signs, axis=-1) != 0, axis=-1)

    # 最大峰个数
    # 含义：局部最大值的个数。
    # 物理意义：反映振荡次数或局部突变次数。
    @staticmethod
    def count_max_peaks(data):
        B, R, _ = data.shape
        return np.array([[len(find_peaks(data[b, r])[0]) for r in range(R)] for b in range(B)])

    # 最小峰个数
    # 含义：局部最小值的个数。
    # 物理意义：同上，反映谷底/极小点的活跃程度，和峰值配合分析信号形状。
    @staticmethod
    def count_min_peaks(data):
        B, R, _ = data.shape
        return np.array([[len(find_peaks(-data[b, r])[0]) for r in range(R)] for b in range(B)])


if __name__ == '__main__':
    B, R, D = 1, 100, 100  # 2个样本，每个样本3行，每行一个长度为100的时间序列
    data = np.random.randn(B, R, D)

    # extractor = TimeSeriesFeatureExtractor(feature)
    # output = extractor.transform(data)

    # print("输出特征 shape:", output.shape)  # 应为 (2, 3, 3)
    # print(output)

    feature_extractor = FeatureExtractor()
    for i in feature_name:
        function = feature_extractor.feature_fuctions[i]
        features = function(data)
        print(i, features.shape)
