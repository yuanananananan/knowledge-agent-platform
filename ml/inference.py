import os
import scipy.io
import torch
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from dataset.sar import get_sar_dataset_full
from model import resnet18, resnet50, KED
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import load_config, get_dataloader, get_model, save_model, load_model
from utils.get_feature import get_outra_feature


def get_data(img, cfg):
    '''
    输入 img:128*128
    输出: img 3,128,128
            feature :list[ 6*[1*80]]
    '''
    img = np.array(img)
    img = cv2.resize(img, (128, 128))
    data = img.astype(np.float32) / 255.
    feature = get_outra_feature(data, cfg)

    img = torch.from_numpy(img).unsqueeze(0)  # 1,128,128
    feature = torch.from_numpy(feature).unsqueeze(0).float()  # 1,n
    # normalize img to [0,1] and repeat to 3 channels
    img = img.repeat(3, 1, 1).unsqueeze(0).float() / 255.  # 1,3,128,128
    return img, feature


def infer(cfg, model, device, img_path):
    img = Image.open(img_path).convert('L')
    img, feature = get_data(img, cfg)
    is_ked = cfg['model']['GNN']['is_GNN']
    with torch.no_grad():
        model.eval()
        im = img.to(device)
        feature = feature.to(device)
        if is_ked:
            result, feats = model(im, feature)
        else:
            result, feats = model(im)

        result = result.max(-1)[1].float()
        return result.item()


if __name__ == '__main__':

    cfg = load_config('configs/sar_config.yaml')
    model_path = 'model_ckp\\test_ckp\\model_best.pth'

    train_loader, test_loader = get_dataloader(cfg)
    model = get_model(cfg)
    model = load_model(model, path=model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # val(cfg, model, test_loader, model_path)

    img_path = 'data\\2026split\\SAR\\test\\likeship\\A5_4864.png'
    # img_path = 'data\\2026split\\SAR\\train\\Cargo\\Ship_C01S02N0013.tiff'
    pred_cls = infer(cfg, model, device, img_path=img_path)
    print(f"Predicted class for {img_path}: {pred_cls}")
