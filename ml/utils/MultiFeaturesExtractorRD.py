import numpy as np
from utils.RD_img_codes.wrapper import *
from utils.RD_img_codes.doppler_features import *
from utils.RD_img_codes.keypoint_features import *
from utils.RD_img_codes.fourier_features import *
from utils.RD_img_codes.intensity_features import *
from utils.RD_img_codes.geometric_features import *
from utils.RD_img_codes.scatteringCenter_features import *
from utils.RD_img_codes.wavelet_features import *
from utils.RD_img_codes.fluctuation_features import *
from utils.RD_img_codes.polar_features import *
from utils.RD_img_codes.autocorrelation_features import *
from utils.RD_img_codes.complexity_features import *
from utils.RD_img_codes.texture_features import *
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
        "doppler_spectrum_entropy": doppler_spectrum_entropy,
        "extract_akaze_features": extract_akaze_features,
        "fft_directional_energy_ratio": fft_directional_energy_ratio,
        "gray_autocorrelogram": gray_autocorrelogram,
        "extract_hu_moments": extract_hu_moments,
        "scattering_centroid": scattering_centroid
    }
    
    return multiFeaturesExtractor(feature_functions, images)

def excator2(images):
    feature_functions = {
        "extract_akaze_features": extract_akaze_features,
        "get_std_dev": get_std_dev,
        "gray_autocorrelogram": gray_autocorrelogram,
        "wavelet_subband_mean": wavelet_subband_mean 
    }
    return multiFeaturesExtractor(feature_functions, images)
    
    
    
def excator3(images):
    feature_functions = {
        "doppler_spectrum_entropy": doppler_spectrum_entropy,
        "extract_akaze_features": extract_akaze_features,
        "polar_r1": polar_r1,
        "wavelet_subband_mean": wavelet_subband_mean
    }
    return multiFeaturesExtractor(feature_functions, images)
    
def excator4(images):
    feature_functions = {
        "aggregated_autocorrelation_2d": aggregated_autocorrelation_2d,
        "doppler_spectrum_entropy": doppler_spectrum_entropy,
        "extract_akaze_features": extract_akaze_features,
        "fft_directional_energy_ratio": fft_directional_energy_ratio,
        "lbp_histogram": lbp_histogram,
        "wavelet_subband_mean": wavelet_subband_mean 
    }
    return multiFeaturesExtractor(feature_functions, images)
    
def excator5(images):
    feature_functions = {
        "extract_akaze_features": extract_akaze_features,
        "extract_multiscale_entropy": extract_multiscale_entropy,
        "fft_directional_energy_ratio": fft_directional_energy_ratio,
        "lbp_histogram": lbp_histogram,
        "polar_r1": polar_r1,
        "wavelet_subband_mean": wavelet_subband_mean 
    }
    return multiFeaturesExtractor(feature_functions, images)
    
    
    
if __name__ == "__main__":
    # ******************************************************************************
    # *建议一次性将特征提取完保存到 pkl形式的序列化文件再进行读取，不要一边训练一边提取特征*
    # ******************************************************************************
    
    # 数据地址前缀
    pathPrefix = 'data/Mon'
    
    # 极化通道名称 也是对应的文件名
    polars = ['Shh', 'Shv', 'Svh', 'Svv']
    
    # 图片具体名称
    img_path = 'jfa5_RD_ship1_jfa5_jfb6_PA30_AZ0_p010_V300.png'
    
    image_size = (224, 224)
    # 建议将图片一次性读取到内存内计算，形成[Batchsize, polar, H, W, channel]的形式，
    # 算法对图片按Batch计算进行了加速，单张图片的计算再拼接速度比较慢。
    # 用于存储所有的，其维度应该为[Batchsize, polar, H, W, channel]
    images = []
    # 存储单张数据！维度应该为[polar, H, W, channel]
    img_channels = []
    for polar in polars:
    # --- 图像路径 ---
        img_completed_path = os.path.join(pathPrefix, polar, img_path)

        if not os.path.exists(img_completed_path):
            raise FileNotFoundError(f"图像未找到: {img_completed_path}")

        img = Image.open(img_completed_path).convert('RGB')
        img = img.resize(image_size)
        img_np = np.array(img)
        img_channels.append(img_np)
        
    images.append(img_channels)
    images = np.array(images)
    # 这里提取出来的一张图片的维度为[1, 4, H, W, 3] 对应 [Batchsize, polar, H, W, channel]
    
    # 特征提取
    features = excator1(images)