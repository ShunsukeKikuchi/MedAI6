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
    model_path = "TransferUnetAttPP_seresnext50_32x4d_best_metric_model.pth",
    sub = "TransferUnetAttPP_seresnext50_32x4d_submission.csv"

rcParams['figure.figsize'] = 20, 8
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

data_dir = 'ai_contest2024'
train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii")))
test_images = sorted(glob.glob(os.path.join(data_dir, "imagesTs", "*.nii")))

data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

sub_df = pd.read_csv('../ai_contest2024/sample_submission.csv')
print('num of train images: ',len(train_images),'\nnum of train labels: ',len(train_labels),'\nnum of test images: ',len(test_images))
print('data path\n',train_images[0],'\n',train_labels[0])
sub_df.head()

os.chdir('Transfer_Unets/Transfer_UnetAttPP')

# Train Setting
kf = KFold(n_splits=5)
count = 0
for train_index, val_index in kf.split(data_dicts):
    train_files, val_files = np.array(data_dicts)[train_index], np.array(data_dicts)[val_index]
    if count == CFG.fold:
        break
    count += 1

train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2,pin_memory=True, drop_last=True)
val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=1,pin_memory=True)

model = TimmSegModel(CFG.backbone, segtype='unetpp', attention_type="scse", pretrained=True)
model = convert_3d(model)
if (CFG.best_model_path == ""):
    model.load_state_dict(torch.load(CFG.best_model_path))

loss_function = DiceCELoss(to_onehot_y=True, softmax=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), CFG.lr)
scaler = torch.cuda.amp.GradScaler()
dice_metric = DiceMetric(include_background=False, reduction="mean")
scheduler = optim.lr_scheduler.StepLR(optimizer, 150, gamma=0.5)

max_epochs = 300
val_interval = 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=12)])
post_label = Compose([AsDiscrete(to_onehot=12)])

model.to(device)
            
for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        del outputs, labels
        scaler.scale(loss).backward()
        epoch_loss += loss.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()        
        print(f"{step}/{len(train_ds) // train_loader.batch_size}", f"train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = (128, 128, 128)
                sw_batch_size = 4
                with torch.cuda.amp.autocast():
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                del val_inputs
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)
                del val_outputs, val_labels
            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()
            print(metric)
            metric_values.append(metric)
            if (metric > best_metric):
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), CFG.model_path)
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )
            gc.collect()
            torch.cuda.empty_cache()
    if epoch%30==0:
        model.load_state_dict(torch.load(CFG.model_path))
    scheduler.step()


import gc
del train_ds, val_ds
del train_loader, val_loader

# ガーベージコレクションの実行
gc.collect()

test_images = sorted(glob.glob(os.path.join(data_dir, "imagesTs", "*.nii")))
test_ids = [path.split('_')[-2] for path in test_images]##　idを取得する。
test_data = [{"image": image, "id": ID} for image,ID in zip(test_images,test_ids)]
post_pred = Compose([AsDiscrete(argmax=True,keepdim=False)])

submission_df = pd.DataFrame()

test_ds = Dataset(data=test_data, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=1)

model.load_state_dict(torch.load(CFG.model_path))
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