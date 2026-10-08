import numpy as np
from utils.SAR_img_codes.wrapper import *
from utils.SAR_img_codes.amplitude_features import *
from utils.SAR_img_codes.autocorrelation_features import *
from utils.SAR_img_codes.complexity_features import *
from utils.SAR_img_codes.fluctuation_features import *
from utils.SAR_img_codes.fourier_features import *
from utils.SAR_img_codes.geometric_features import *
from utils.SAR_img_codes.intensity_features import *
from utils.SAR_img_codes.keypoint_features import *
from utils.SAR_img_codes.scatteringCenter_features import *
from utils.SAR_img_codes.texture_features import *
from utils.SAR_img_codes.wavelet_features import *
from PIL import Image
import os


def multiFeaturesExtractor(feature_functions, images):
    #循环调用特征函数
    feature_all = []
    
    for feature_name in feature_functions:
        # features = feature_functions[feature_name](images)
        features = singleFeatureExtractor(feature_functions, feature_name, images)
        features = features.reshape(features.shape[0], -1)
        if features is not None and features.size > 0:
            feature_all.append(features)
    #进行特征拼接
    feature_all= np.hstack(feature_all)

    return align(feature_all)   

def excator1(images):
    feature_functions = {
        "autocovariance_2d": autocovariance_2d,
        "extract_sobel_features": extract_sobel_features,
        "fft_entropy": fft_entropy,
        "num_centers": num_centers
    }
    
    return multiFeaturesExtractor(feature_functions, images)

def excator2(images):
    feature_functions = {
        "autocovariance_2d": autocovariance_2d,
        "extract_sample_entropy": extract_sample_entropy,
        "extract_sobel_features": extract_sobel_features,
        "fft_entropy": fft_entropy,
        "num_centers": num_centers
    }
    return multiFeaturesExtractor(feature_functions, images)
    
    
    
def excator3(images):
    feature_functions = {
        "autocovariance_2d": autocovariance_2d,
        "extract_sobel_features": extract_sobel_features,
        "get_variance": get_variance,
        "glcm_correlation": glcm_correlation,
        "num_centers": num_centers
    }
    return multiFeaturesExtractor(feature_functions, images)
    
def excator4(images):
    feature_functions = {
        "autocovariance_2d": autocovariance_2d,
        "extract_sobel_features": extract_sobel_features,
        "fft_entropy": fft_entropy,
        "glcm_correlation": glcm_correlation,
        "num_centers": num_centers
    }
    return multiFeaturesExtractor(feature_functions, images)
    
def excator5(images):
    feature_functions = {
        "extract_sample_entropy": extract_sample_entropy,
        "get_variance": get_variance,
        "glcm_correlation": glcm_correlation,
        "wavelet_subband_kurt": wavelet_subband_kurt
    }
    return multiFeaturesExtractor(feature_functions, images)


if __name__ == "__main__":
    # ******************************************************************************
    # *建议一次性将特征提取完保存到 pkl形式的序列化文件再进行读取，不要一边训练一边提取特征*
    # ******************************************************************************
    
    # 数据地址前缀
    pathPrefix = 'data/SAR'
    
    # 图片具体名称
    img_path = 'Ship_C01S02N0001.tiff'
    
    image_size = (224, 224)
    # 建议将图片一次性读取到内存内计算，形成[Batchsize, H, W, channel]的形式，
    # 算法对图片按Batch计算进行了加速，单张图片的计算再拼接速度比较慢。
    # 用于存储所有的，其维度应该为[Batchsize, H, W, channel]
    images = []
    # 存储单张数据！维度应该为[ H, W, channel]
    img_channels = []

    # --- 图像路径 ---
    img_completed_path = os.path.join(pathPrefix, img_path)

    if not os.path.exists(img_completed_path):
        raise FileNotFoundError(f"图像未找到: {img_completed_path}")

    img = Image.open(img_completed_path).convert('RGB')
    img = img.resize(image_size)
    image = np.array(img)
    images.append(image)
    images = np.array(images)
    # print(images.shape)
    # 这里提取出来的一张图片的维度为[1, H, W, 3] 对应 [Batchsize, H, W, channel]
    
    # 特征提取
    features = excator2(images)