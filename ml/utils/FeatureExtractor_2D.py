import cv2
import pywt
import numpy as np
import skimage.feature as feature
from scipy.fftpack import dct

feature_name = [
    'color_histogram', 'color_moments', 'color_coherence_vector', 'glcm_features', 'lbp_features', 'gabor_features',
    'area_features', 'aspectratio_features', 'rectangularity_features', 'hu_moments_features', 'sphericity_features',
    'hog_features', 'canny_features', 'laplacian_features', 'log_features', 'prewitt_features', 'sobel_features',
    'sift_features', 'orb_features', 'akaze_features', 'brisk_features', 'harris_features', 'fft_features',
    'wavelet_features', 'dct_features'
]


class FeatureExtractor:

    def __init__(self):
        self.feature_fuctions = {
            # 图像颜色特征
            'color_histogram': self.extract_color_histogram,
            'color_moments': self.extract_color_moments,
            'color_coherence_vector': self.extract_color_coherence_vector,

            # 纹理特征
            'glcm_features': self.extract_glcm_features,
            'lbp_features': self.extract_lbp_features,
            'gabor_features': self.extract_gabor_features,

            # 形状特征
            'area_features': self.extract_area_features,
            'aspectratio_features': self.extract_aspectratio_features,
            'hog_features': self.extract_hog_features,
            'rectangularity_features': self.extract_rectangularity_features,
            'hu_moments_features': self.extract_hu_moments_features,
            'sphericity_features': self.extract_sphericity_features,

            # 边缘特征
            'canny_features': self.extract_canny_features,
            'laplacian_features': self.extract_laplacian_features,
            'log_features': self.extract_log_features,
            'prewitt_features': self.extract_prewitt_features,
            'sobel_features': self.extract_sobel_features,

            # 关键点特征
            'sift_features': self.extract_sift_features,
            'orb_features': self.extract_orb_features,
            'akaze_features': self.extract_akaze_features,
            'brisk_features': self.extract_brisk_features,
            'harris_features': self.extract_harris_features,

            # 其他特征（如频域特征）
            'fft_features': self.extract_fft_features,
            'wavelet_features': self.extract_wavelet_features,
            'dct_features': self.extract_dct_features
        }

    ## 图像颜色特征
    # 颜色直方图
    @staticmethod
    def extract_color_histogram(image, bins=(16, 16, 16)):
        features = []
        for img in image:
            # 计算颜色直方图
            hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
            # 归一化直方图
            hist = cv2.normalize(hist, hist).flatten()
            features.append(np.array(hist))
        return np.array(features)

    # 颜色矩
    @staticmethod
    def extract_color_moments(image):
        features = []
        for img in image:
            r, g, b = cv2.split(img)
            r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
            r_std, g_std, b_std = np.std(r), np.std(g), np.std(b)
            r_skew = np.mean((r - r_mean)**3) / (r_std**3)
            g_skew = np.mean((g - g_mean)**3) / (g_std**3)
            b_skew = np.mean((b - b_mean)**3) / (b_std**3)
            moments = [r_mean, r_std, r_skew, g_mean, g_std, g_skew, b_mean, b_std, b_skew]
            features.append(np.array(moments))
        return np.array(features)

    # 颜色聚合向量
    @staticmethod
    def extract_color_coherence_vector(image, color_bins=64, tau=10):
        features = []
        for img in image:
            quantized = (img[..., 0] // (256 / color_bins)).astype(np.uint8)
            h, w = quantized.shape
            # 统计连通区域
            ccv = np.zeros((color_bins, 2))  # 每颜色：[聚合像素数, 散乱像素数]
            visited = np.zeros((h, w), dtype=bool)
            for i in range(h):
                for j in range(w):
                    if not visited[i, j]:
                        color = quantized[i, j]
                        stack = [(i, j)]
                        count = 0
                        while stack:
                            x, y = stack.pop()
                            if visited[x, y]:
                                continue
                            visited[x, y] = True
                            count += 1
                            # 检查四个方向
                            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < h and 0 <= ny < w and not visited[nx, ny] and quantized[nx, ny] == color:
                                    stack.append((nx, ny))
                        ccv[color][0] += count
                        if count < tau:
                            ccv[color][1] += count
            # 归一化
            ccv = ccv / np.sum(ccv, axis=0, keepdims=True)
            features.append(np.array(ccv).flatten())
        return np.array(features)

    ## 纹理特征
    # 灰度共生矩阵
    @staticmethod
    def extract_glcm_features(image, distances=[1], angles=[0], levels=256):
        features = []
        for img in image:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            glcm = feature.graycomatrix(gray,
                                        distances=distances,
                                        angles=angles,
                                        levels=levels,
                                        symmetric=True,
                                        normed=True)
            contrast = feature.graycoprops(glcm, 'contrast').flatten()
            dissimilarity = feature.graycoprops(glcm, 'dissimilarity').flatten()
            homogeneity = feature.graycoprops(glcm, 'homogeneity').flatten()
            energy = feature.graycoprops(glcm, 'energy').flatten()
            correlation = feature.graycoprops(glcm, 'correlation').flatten()
            gclm_features = np.hstack([contrast, dissimilarity, homogeneity, energy, correlation])
            features.append(np.array(gclm_features))
        return np.array(features)

    # 局部二值模式
    @staticmethod
    def extract_lbp_features(image, radius=1, n_points=8, method='uniform'):
        features = []
        for img in image:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lbp = feature.local_binary_pattern(gray, n_points, radius, method)
            hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
            features.append(np.array(hist))
        return np.array(features)

    # gabor纹理特征
    @staticmethod
    def extract_gabor_features(image, sigma=1.0, theta=0, lambd=1.0, gamma=0.5, psi=0, ktype=cv2.CV_32F):
        features = []
        for img in image:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gabor_kernel = cv2.getGaborKernel((16, 16), sigma, theta, lambd, gamma, psi, ktype)
            gabor_feature = cv2.filter2D(gray, cv2.CV_32F, gabor_kernel).flatten()
            features.append(np.array(gabor_feature))
        return np.array(features)

    ## 形状特征
    # 面积
    @staticmethod
    def extract_area_features(image):
        features = []
        for img in image:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                area = cv2.contourArea(contours[0])
                features.append(np.array([area]))
            else:
                features.append(np.array([0]))
        return np.array(features)

    # 长宽比
    @staticmethod
    def extract_aspectratio_features(image):
        features = []
        for img in image:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                _, _, w, h = cv2.boundingRect(contours[0])
                aspect_ratio = float(w) / h if h != 0 else 0
                features.append(np.array([aspect_ratio]))
            else:
                features.append(np.array([0]))
        return np.array(features)

    # 矩形度
    @staticmethod
    def extract_rectangularity_features(image):
        features = []
        for img in image:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                rect = cv2.minAreaRect(contours[0])
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                area = cv2.contourArea(box)
                perimeter = cv2.arcLength(box, True)
                rectangularity = area / (perimeter**2) if perimeter != 0 else 0
                features.append(np.array([rectangularity]))
            else:
                features.append(np.array([0]))
        return np.array(features)

    # Hu矩
    @staticmethod
    def extract_hu_moments_features(image):
        features = []
        for img in image:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                moments = cv2.moments(contours[0])
                hu_moments = cv2.HuMoments(moments).flatten()
                features.append(np.array(hu_moments))
            else:
                features.append(np.zeros(7))  # Hu矩有7个特征
        return np.array(features)

    # 球状性
    @staticmethod
    def extract_sphericity_features(image):
        features = []
        for img in image:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                area = cv2.contourArea(contours[0])
                perimeter = cv2.arcLength(contours[0], True)
                sphericity = (4 * np.pi * area) / (perimeter**2) if perimeter != 0 else 0
                features.append(np.array([sphericity]))
            else:
                features.append(np.array([0]))
        return np.array(features)

    ## 边缘特征
    # HOG形状特征
    @staticmethod
    def extract_hog_features(image, win_size=(64, 128), cell_size=(8, 8), block_size=(16, 16), nbins=9):
        features = []
        for img in image:
            # 转换为灰度图像
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 调整图像大小以匹配 HOG 窗口大小
            gray_image = cv2.resize(gray_image, win_size)
            # 初始化 HOG 描述符
            hog = cv2.HOGDescriptor(_winSize=win_size,
                                    _blockSize=block_size,
                                    _blockStride=(cell_size[0], cell_size[1]),
                                    _cellSize=cell_size,
                                    _nbins=nbins)
            # 计算 HOG 特征
            hog_features = hog.compute(gray_image)
            features.append(np.array(hog_features).flatten())
        return np.array(features)

    # Canny边缘特征
    @staticmethod
    def extract_canny_features(image):
        features = []
        for img in image:
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(img, 100, 200).flatten()
            features.append(np.array(edges))
        return np.array(features)

    # laplacian边缘特征
    @staticmethod
    def extract_laplacian_features(image):
        features = []
        for img in image:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F).flatten()
            features.append(np.array(laplacian))
        return np.array(features)

    # LoG边缘特征
    @staticmethod
    def extract_log_features(image, sigma=1.0):
        features = []
        for img in image:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            log_kernel = cv2.getGaussianKernel(ksize=5, sigma=sigma)
            log_kernel = -log_kernel @ log_kernel.T + (1 / (np.pi * sigma**4))
            log_feature = cv2.filter2D(gray, cv2.CV_64F, log_kernel).flatten()
            features.append(np.array(log_feature))
        return np.array(features)

    # prewitt边缘特征
    @staticmethod
    def extract_prewitt_features(image):
        features = []
        prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

        for img in image:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            prewitt_x_feature = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, prewitt_x).flatten()
            prewitt_y_feature = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, prewitt_y).flatten()
            prewitt_feature = np.hstack([prewitt_x_feature, prewitt_y_feature])
            features.append(np.array(prewitt_feature))
        return np.array(features)

    # sobel边缘特征
    @staticmethod
    def extract_sobel_features(image):
        features = []
        for img in image:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3).flatten()
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3).flatten()
            sobel_feature = np.hstack([sobel_x, sobel_y])
            features.append(np.array(sobel_feature))
        return np.array(features)

    ## 点特征提取
    # SIFT关键点特征
    @staticmethod
    def extract_sift_features(image):
        features = []
        sift = cv2.SIFT_create()
        descriptor_dim = 128

        for img in image:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = sift.detectAndCompute(gray, None)

            if descriptors is None or len(descriptors) == 0:
                descriptors_mean = np.zeros(descriptor_dim)
            else:
                descriptors_mean = np.mean(descriptors, axis=0)
                if descriptors_mean.size < descriptor_dim:
                    descriptors_mean = np.pad(descriptors_mean, (0, descriptor_dim - descriptors_mean.size), 'constant')
                elif descriptors_mean.size > descriptor_dim:
                    descriptors_mean = descriptors_mean[:descriptor_dim]
            features.append(np.array(descriptors_mean))
        return np.array(features)

    # ORB关键点特征
    @staticmethod
    def extract_orb_features(image):
        features = []
        orb = cv2.ORB_create()
        descriptor_dim = 32

        for img in image:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = orb.detectAndCompute(gray, None)

            if descriptors is None or len(descriptors) == 0:
                descriptors_mean = np.zeros(descriptor_dim)
            else:
                descriptors_mean = np.mean(descriptors, axis=0)
                if descriptors_mean.size < descriptor_dim:
                    descriptors_mean = np.pad(descriptors_mean, (0, descriptor_dim - descriptors_mean.size), 'constant')
                elif descriptors_mean.size > descriptor_dim:
                    descriptors_mean = descriptors_mean[:descriptor_dim]
            features.append(np.array(descriptors_mean))
        return np.array(features)

    # AKAZE关键点特征
    @staticmethod
    def extract_akaze_features(image):
        features = []
        akaze = cv2.AKAZE_create()
        descriptor_dim = 64

        for img in image:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = akaze.detectAndCompute(gray, None)

            if descriptors is None or len(descriptors) == 0:
                descriptors_mean = np.zeros(descriptor_dim)
            else:
                descriptors_mean = np.mean(descriptors, axis=0)
                if descriptors_mean.size < descriptor_dim:
                    descriptors_mean = np.pad(descriptors_mean, (0, descriptor_dim - descriptors_mean.size), 'constant')
                elif descriptors_mean.size > descriptor_dim:
                    descriptors_mean = descriptors_mean[:descriptor_dim]
            features.append(np.array(descriptors_mean))
        return np.array(features)

    # BRISK关键点特征
    @staticmethod
    def extract_brisk_features(image):
        features = []
        brisk = cv2.BRISK_create()
        descriptor_dim = 64

        for img in image:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = brisk.detectAndCompute(gray, None)

            if descriptors is None or len(descriptors) == 0:
                descriptors_mean = np.zeros(descriptor_dim)
            else:
                descriptors_mean = np.mean(descriptors, axis=0)
                if descriptors_mean.size < descriptor_dim:
                    descriptors_mean = np.pad(descriptors_mean, (0, descriptor_dim - descriptors_mean.size), 'constant')
                elif descriptors_mean.size > descriptor_dim:
                    descriptors_mean = descriptors_mean[:descriptor_dim]
            features.append(np.array(descriptors_mean))
        return np.array(features)

    # Harris角点特征
    @staticmethod
    def extract_harris_features(image, block_size=2, ksize=3, k=0.04):
        features = []
        descriptor_dim = 30
        for img in image:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray)
            harris_corners = cv2.cornerHarris(gray, block_size, ksize, k)
            harris_corners = cv2.dilate(harris_corners, None)
            corners = np.argwhere(harris_corners > 0.01 * harris_corners.max())
            if corners.size == 0:
                feature_vec = np.zeros(descriptor_dim)  # 假设我们需要30个特征
            else:
                feature_vec = corners.flatten()
                if feature_vec.size < descriptor_dim:
                    feature_vec = np.pad(feature_vec, (0, descriptor_dim - feature_vec.size), 'constant')
                elif feature_vec.size > descriptor_dim:
                    feature_vec = feature_vec[:descriptor_dim]
            features.append(np.array(feature_vec))
        return np.array(features)

    ## 频率特征
    # 快速傅里叶变换
    @staticmethod
    def extract_fft_features(image):
        features = []
        for img in image:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)  # 避免 log(0)
            hist, _ = np.histogram(magnitude_spectrum.ravel(), bins=64, density=True)
            features.append(np.array(hist))
        return np.array(features)

    # 小波变换特征
    @staticmethod
    def extract_wavelet_features(image, wavelet='haar', level=2):
        features = []
        for img in image:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            coeffs = pywt.wavedec2(gray, wavelet=wavelet, level=level)
            # 只使用细节系数：LH、HL、HH
            coeff_features = []
            for i in range(1, len(coeffs)):
                for subband in coeffs[i]:  # subband = LH, HL, HH
                    coeff_features.extend([np.mean(subband), np.std(subband), np.max(subband), np.min(subband)])
            features.append(np.array(coeff_features))
        return np.array(features)

    # 离散余弦变换
    @staticmethod
    def extract_dct_features(image, top_k=20):
        features = []
        for img in image:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = np.float32(gray) / 255.0
            dct_coeffs = dct(dct(gray.T, norm='ortho').T, norm='ortho')  # 2D DCT
            flat = dct_coeffs.flatten()
            top_features = np.sort(np.abs(flat))[-top_k:]  # 取前K个能量最大的系数
            features.append(np.array(top_features))
        return np.array(features)


if __name__ == "__main__":
    # image = [cv2.imread('image/jf/Unite_1/monopulse_pic_Unite_1_PA0_AZ0_V3.png')]  # Replace with your image path
    image = [np.zeros((100, 100, 3), dtype=np.uint8)]

    feature_extractor = FeatureExtractor()
    for i in feature_name:
        function = feature_extractor.feature_fuctions[i]
        features = function(image)
        print(i,features.shape)
