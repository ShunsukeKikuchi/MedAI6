# %%
import os
import torch
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import nibabel as nib
from sklearn.model_selection import KFold

from monai.utils import first
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
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

from monai.config import print_config
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import DynUNet
from monai.networks.layers import Norm

from monai.data import (
    DataLoader,
    Dataset,
    decollate_batch,
)

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import test_transforms, encode_rle_multiclass

os.chdir('../')

class CFG:
    fold = 0,
    lr = 1e-4,
    best_model_path = "", # ex. "Dyn_best_metric_model.pth"
    model_path = "Dyn_best_metric_model0.pth",
    sub = "Dyn1_submission.csv"

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
print_config()
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

img = nib.load(train_images[0])
img = img.get_fdata()
label = nib.load(train_labels[0])
label = label.get_fdata()
print('image: ',img.shape, img.max(),img.min())
print('ground-truth: ',label.shape, label.max(),label.min())

os.chdir('DynUnet')

model = DynUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=12,
    strides=(1, 2, 2,2),
    kernel_size=(3,3,3,3),
    upsample_kernel_size=(2,2,2),
).to(device)

import gc
gc.collect()

test_images = sorted(glob.glob(os.path.join(data_dir, "imagesTs", "*.nii")))
test_ids = [path.split('_')[-2] for path in test_images]##　idを取得する。
test_data = [{"image": image, "id": ID} for image,ID in zip(test_images,test_ids)]
post_pred = Compose([AsDiscrete(argmax=True,keepdim=False)])

submission_df = pd.DataFrame()

test_ds = Dataset(data=test_data, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=1)

model.load_state_dict(torch.load(CFG.model_path))##上のセルで学習したモデルを読み込むときは、パスを"best_metric_model.pth"に変更してください。
model.eval()

with torch.no_grad():
    for test_data in tqdm(test_loader):
        test_inputs = test_data["image"].to(device)
        file_id = str(test_data["id"][0])
        del test_data
        torch.cuda.empty_cache()
        roi_size = (128, 128, 128)
        sw_batch_size = 16
        with torch.cuda.amp.autocast():
            test_outputs = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
        del test_inputs
        torch.cuda.empty_cache()
        test_outputs = [post_pred(i) for i in decollate_batch(test_outputs)]

        for j, output in enumerate(test_outputs):
            # 予測をNumpy配列に変換
            output_np = output.cpu().numpy()
            del output
            torch.cuda.empty_cache()
            #print(output_np.shape)
            # RLEエンコーディングとデータフレームへの追加
            rle_encoded_data = encode_rle_multiclass(output_np)
            for cls_name, rle in rle_encoded_data:
                # ここでの 'file_id' は適切なファイル識別子に置き換える
                submission_df = pd.concat([submission_df,pd.DataFrame([f'{file_id}_{cls_name}', rle]).T], ignore_index=True)

submission_df.columns=['id', 'prediction']
# CSVファイルに保存
submission_df.to_csv(CFG.sub, index=False)

