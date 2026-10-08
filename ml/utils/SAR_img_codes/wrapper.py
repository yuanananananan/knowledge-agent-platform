import numpy as np
from sklearn.decomposition import NMF


def align(X, n_components=60):
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
    # print(X_nmf)
    return X_nmf


# 根据不同特征，事先进行数据预处理
def singleFeatureExtractor(feature_functions, feature_name, images):
    images = images.mean(axis=-1)
    feature = feature_functions[feature_name](images)
    return feature        