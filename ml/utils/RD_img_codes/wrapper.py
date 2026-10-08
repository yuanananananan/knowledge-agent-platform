import numpy as np
from sklearn.decomposition import NMF


def align(X, n_components=30):
    X_nmf_input = np.nan_to_num(X, nan=0.0)
    X_nmf_input = np.clip(X_nmf_input, a_min=0, a_max=None) 
    nmf = NMF(
        n_components= min(n_components, X.shape[1] - 1), 
        init='random',             
        solver='cd',             
        max_iter=200,             
        random_state=42           
            )
    X_nmf = nmf.fit_transform(X_nmf_input)
    
    return X_nmf


# 根据不同特征，事先进行数据预处理
def singleFeatureExtractor(feature_functions, feature_name, images):
    features_polar = {
        'polar_H',  
        'polar_alpha',  
        'polar_Ani',  
        'polar_P',  
        'polar_r1',  
        'polar_r2',  
        'polar_r3', 
        'polar_r4',
        'polar_r5', 
        'polar_r6', 
        'polar_r7', 
        'polar_power', 
        'polar_det', 
        'polar_ellipticity', 
        'polar_depol', 
        'polar_orientation', 
    }
    
    if feature_name in features_polar:
        images = images.mean(axis=-1)
    else:
        images = images[:,0,:,:,:].mean(axis=-1)
        
    feature = feature_functions[feature_name](images)
    return feature        