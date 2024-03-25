import os
import sys
import gc
import ast
import cv2
import time
import timm
import pickle
import random
import pydicom
import argparse
import warnings
import numpy as np
import pandas as pd
import glob
import nibabel as nib
from PIL import Image
from tqdm import tqdm
import albumentations
from pylab import rcParams
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold, StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn.functional as F
from monai.inferers import sliding_window_inference

from monai.transforms import Resize
import monai.transforms as transforms
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.data import decollate_batch, DataLoader, Dataset
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandSpatialCropd,
    SpatialPadd,
    ScaleIntensityRanged,
    Spacingd,
)
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

os.chdir("../") #Set dir to "Transfer_Unets"
from conv3d_same import *
from model_convert import TimmSegModel, convert_3d

from utils import train_transforms, val_transforms,test_transforms, encode_rle_multiclass

os.chdir('../')

class CFG:
    fold = 0,
    lr = 1e-4,
    best_model_path = "", # ex. "Dyn_best_metric_model.pth"
    backbone = "seresnext50_32x4d",
    model_path = "Transfer_Unetpp_seresnext50_32x4d.pth",
    sub = "Transfer_Unetpp_seresnext50_32x4d.csv"

rcParams['figure.figsize'] = 20, 8
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

data_dir = 'ai_contest2024'
train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii")))
test_images = sorted(glob.glob(os.path.join(data_dir, "imagesTs", "*.nii")))

data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

sub_df = pd.read_csv('ai_contest2024/sample_submission.csv')
print('num of train images: ',len(train_images),'\nnum of train labels: ',len(train_labels),'\nnum of test images: ',len(test_images))
print('data path\n',train_images[0],'\n',train_labels[0])
sub_df.head()

os.chdir('Transfer_Unets/UnetPP')
    
model = TimmSegModel(CFG.backbone, segtype='link',pretrained=True)
model = convert_3d(model)

if (CFG.best_model_path == ""):
    model.load_state_dict(torch.load(CFG.best_model_path))

model.to(device)

import gc
gc.collect()

test_images = sorted(glob.glob(os.path.join(data_dir, "imagesTs", "*.nii")))
test_ids = [path.split('_')[-2] for path in test_images]##　idを取得する。
test_data = [{"image": image, "id": ID} for image,ID in zip(test_images,test_ids)]
post_pred = Compose([AsDiscrete(argmax=True,keepdim=False)])

submission_df = pd.DataFrame()

test_ds = Dataset(data=test_data, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=1)

model.eval()

with torch.no_grad():
    for test_data in tqdm(test_loader):
        test_inputs = test_data["image"].to(device)
        file_id = str(test_data["id"][0])
        roi_size = (128, 128, 128)
        sw_batch_size = 4
        with torch.cuda.amp.autocast():
            test_outputs = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
        test_outputs = [post_pred(i) for i in decollate_batch(test_outputs)]
        for j, output in enumerate(test_outputs):
            # transform output to numpy array
            output_np = output.cpu().numpy()
            # Encode RLE and write to submission dataframe
            rle_encoded_data = encode_rle_multiclass(output_np)
            for cls_name, rle in rle_encoded_data:
                submission_df = pd.concat([submission_df,pd.DataFrame([f'{file_id}_{cls_name}', rle]).T], ignore_index=True)

# Save
submission_df.columns=['id', 'prediction']
submission_df.to_csv(CFG.sub, index=False)