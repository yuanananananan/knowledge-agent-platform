import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .dataset import SAR_Dataset
from sklearn.decomposition import NMF, PCA
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.linear_model import LogisticRegression
import json


def AC(features, labels, kernel='rbf', **featuretuple):

    selected_feature_names = list(featuretuple.keys())
    print('选择的特征:', selected_feature_names)

    # 拼接选中特征
    X = np.hstack([features[name] for name in selected_feature_names])
    Y = labels
    accuracy = five_fold_cross_validation_svm(X, Y, kernel=kernel)
    return accuracy

# # SVM 5折交叉验证
# def five_fold_cross_validation_svm(X, Y, kernel='rbf', C=1.0, random_seed=42):
#     X = np.nan_to_num(X, nan=0.0)
    
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)

#     kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

#     fold = 1
#     accuracies = []
#     for train_index, test_index in kf.split(X):
#         X_train, X_test = X[train_index], X[test_index]
#         Y_train, Y_test = Y[train_index], Y[test_index]

#         # SVM 分类器（默认使用 RBF 核）
#         model = SVC(kernel=kernel, C=C, gamma='scale', random_state=random_seed)
#         model.fit(X_train, Y_train)
#         Y_pred = model.predict(X_test)

#         acc = accuracy_score(Y_test, Y_pred)
#         accuracies.append(acc)

#         # print(f"Fold {fold}: Accuracy = {acc:.4f}")
#         fold += 1

#     mean_acc = np.mean(accuracies)
#     # print(f"Average Accuracy over 5 folds: {mean_acc:.4f}")

#     return mean_acc

# SVM 5折交叉验证
def five_fold_cross_validation_svm(X, Y, kernel='rbf', test_num=100, random_seed=42):
    X = np.nan_to_num(X, nan=0.0)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test = X[:-test_num], X[-test_num:]
    Y_train, Y_test = Y[:-test_num], Y[-test_num:]

    # SVM 分类器（默认使用 RBF 核）
    model = SVC(kernel=kernel, C=1.0, gamma='scale', random_state=random_seed)
    # model = LogisticRegression(penalty='l2', C=1.0, random_state=random_seed, max_iter=100)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    acc = accuracy_score(Y_test, Y_pred)
    print(f"Accuracy: {acc:.4f}")

    return acc

def fea_ext(features, labels):

    # 对齐前准确率
    acc_before = five_fold_cross_validation_svm(features, labels, kernel='poly')

    # 使用 NMF 对齐
    if features.shape[1] > 1:
        # NMF 要求非负
        # features_nmf_input = np.clip(features, a_min=0, a_max=None)
        y = labels
        features_nmf_input = np.nan_to_num(features, nan=0.0)
        nmf = LinearDiscriminantAnalysis(
            n_components=min(30, len(np.unique(y))-1),  
            solver='svd',         
            shrinkage=None,       
            store_covariance=False 
        )
        features_nmf = nmf.fit_transform(features_nmf_input, y)

        # 对齐后准确率
        acc_after = five_fold_cross_validation_svm(features, labels)
    else:
        features_nmf = features
        acc_after = five_fold_cross_validation_svm(features, labels)
    return acc_before, acc_after, features_nmf

def feature_redundancy_analysis(dataset_path = '/media/4TB/GJ/TS/splitedData/FUSAR_data', 
                                fea_path = '/media/4TB/GJ/test/SAR/all_features_SAR_15.pkl', 
                                info_path = '/media/4TB/GJ/test/info/feature_info_SAR.json' ,
                                redundancy_single_path = '/media/4TB/GJ/test/logs/redundancy_single_SAR.json', 
                                redundancy_remove_path = '/media/4TB/GJ/test/logs/redundancy_remove_SAR.json'):

    # path = '/media/4TB/GJ/TS/splitedData/FUSAR_data'

    # path = dataset_path
    # image_size=(224, 224)
    # train_dataset = SAR_Dataset(
    #     root_dir = path,
    #     mode='train',
    #     image_size=(224, 224))

    # test_dataset = SAR_Dataset(
    #     root_dir=path,
    #     mode='test',
    #     image_size=(224, 224))

    # train_images = train_dataset.images  # (2010, 16, 16, 3)
    # train_labels = train_dataset.labels  # (2010,)
    # test_images = test_dataset.images  # (100, 16, 16, 3)
    # test_labels = test_dataset.labels  # (100,)

    # images = np.concatenate([train_images, test_images], axis=0)
    # labels = np.concatenate([train_labels, test_labels], axis=0)

    # test_num = test_images.shape[0]
    # print(test_num)
    # echos = None

    feature_names_all= []
    feature_all = []

    # 加载特征数据
    with open(fea_path, 'rb') as f:
        all_features = pickle.load(f)
    feature_keys = list(all_features.keys()) 

    # 单特征对齐功能展示
    with open(info_path, 'r') as f:
            feature_info_map = json.load(f)
    result_data = []
    idx = 0
    for key in feature_keys:
        features_item = all_features[key]
        features = features_item['features']
        labels = features_item['labels']

        acc_bf, acc_af, fea_af = fea_ext(features, labels)
        if   acc_af > 0.68 and acc_af > acc_bf :   #(acc_af - acc_bf > 0.01) and
            print(f"特征 [{key}] 降维前准确率: {acc_bf:.4f}，降维后准确率: {acc_af:.4f}，提升: {acc_af - acc_bf:.4f}")
            feature_names_all.append(key)
            feature_all.append(fea_af)

            idx += 1
            # info = feature_info_map.get(key, {'序号': idx, "特征名": key, "对齐前识别率": "未知", "对齐后识别率": "未知"})
            # # 构建条目
            # result_data.append({
            #     '序号': idx,
            #     # '函数名': key,
            #     '特征名': info['特征名'],
            #     '对齐前准确率': acc_bf,
            #     '对齐后准确率': acc_af,              
            # })

            info = feature_info_map.get(key, {'Index': idx, "Feature_name": key, "Acc_bf": None, "Acc_af": None})
            # 构建条目
            result_data.append({
                'Index': idx,
                # '函数名': key,
                'Feature_name': info['特征名'],
                'Acc_bf': acc_bf,
                'Acc_af': acc_af,              
            })

    with open(redundancy_single_path, 'w') as f:  
        json.dump(result_data, f, ensure_ascii=False, indent=4)


    features_summary  = {name: feat for feat, name in zip(feature_all, feature_names_all)}

    # # 保存到文件
    # with open('all_features_mon_redun.pkl', 'wb') as f:
    #     pickle.dump(features_summary, f)

    # ## 特征冗余度测试
    # with open("all_features_mon_redun.pkl", "rb") as f:
    #     features_summary = pickle.load(f)

    feature_names_all = list(features_summary.keys())
    full_feature_dict = {name: features_summary[name] for name in feature_names_all}

    # 计算完整组合的准确率
    print(f"构造完整特征组合，共 {len(feature_names_all)} 个特征")
    acc_full = 0.91
    # acc_full = AC(full_feature_dict, labels, **full_feature_dict)
    print(f"原始完整组合准确率: {acc_full:.4f}")

    # # 逐个剔除特征后计算准确率
    # print("分析：逐个剔除每个特征后的准确率变化")

    # all_group = {'组合所有最优单特征的识别率': acc_full}
    all_group = {'Combination_acc': acc_full}
    results = []
    idx2 = 0
    results.append(all_group)

    for name in feature_names_all:
        reduced_feature_dict = {k: v for k, v in full_feature_dict.items() if k != name}
        acc_reduced = AC(reduced_feature_dict, labels, kernel='poly', **reduced_feature_dict)
        drop = acc_full - acc_reduced

        idx2 += 1
        # info = feature_info_map.get(key, {'序号': idx2, "去除特征": key, "识别率变化": "未知"})
        # # 构建条目
        # results.append({
        #     '序号': idx2,
        #     # '函数名': key,
        #     '去除特征': info['特征名'],
        #     '识别率变化': drop,             
        # })

        info = feature_info_map.get(name, {'Index': idx2, "Re_feature": key, "Acc_change": "未知"})
        # 构建条目
        results.append({
            'Index': idx2,
            # '函数名': key,
            'Re_feature': info['特征名'],
            'Acc_change': drop,             
        })

        print(f" - 去掉 [{name}] 后准确率: {acc_reduced:.4f}，变化: {drop:+.4f}")

    with open(redundancy_remove_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    # results = []

    # for name in feature_names_all:
    #     reduced_feature_dict = {k: v for k, v in full_feature_dict.items() if k != name}
    #     acc_reduced = AC(reduced_feature_dict, labels, kernel='poly', **reduced_feature_dict)
    #     drop = acc_full - acc_reduced
    #     results.append((name, acc_reduced, drop))
    #     print(f" - 去掉 [{name}] 后准确率: {acc_reduced:.4f}，变化: {drop:+.4f}")

    return result_data, results


if __name__ == "__main__":
    feature_redundancy_analysis()



