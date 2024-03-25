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
from utils import train_transforms, val_transforms,test_transforms, encode_rle_multiclass

os.chdir('../') # Set current dir to "MedAI6"

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

# Import Data
data_dir = 'ai_contest2024'
train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii")))
test_images = sorted(glob.glob(os.path.join(data_dir, "imagesTs", "*.nii")))

data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]

sub_df = pd.read_csv('ai_contest2024/sample_submission.csv')
print('num of train images: ',len(train_images),'\nnum of train labels: ',len(train_labels),'\nnum of test images: ',len(test_images))
print('data path\n',train_images[0],'\n',train_labels[0])
sub_df.head()

# Display image and label information
img = nib.load(train_images[0])
img = img.get_fdata()
label = nib.load(train_labels[0])
label = label.get_fdata()
print('image: ',img.shape, img.max(),img.min())
print('ground-truth: ',label.shape, label.max(),label.min())

# Change Directory to right place
os.chdir('DynUnet')

kf = KFold(n_splits=5)
count = 0
for train_index, val_index in kf.split(data_dicts):
    train_files, val_files = np.array(data_dicts)[train_index], np.array(data_dicts)[val_index]
    if count == CFG.fold:
        break
    count += 1

train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2,pin_memory=True)
val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=1,pin_memory=True)

model = DynUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=12,
    strides=(1, 2, 2,2),
    kernel_size=(3,3,3,3),
    upsample_kernel_size=(2,2,2), 
).to(device)

if (CFG.best_model_path == ""):
    model.load_state_dict(torch.load(CFG.best_model_path))

loss_function = DiceCELoss(to_onehot_y=True, softmax=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), CFG.lr)
scaler = torch.cuda.amp.GradScaler()
dice_metric = DiceMetric(include_background=False, reduction="mean")
            
max_epochs = 50
val_interval = 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=12)])
post_label = Compose([AsDiscrete(to_onehot=12)])

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
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        del inputs, labels
        torch.cuda.empty_cache()
        scaler.scale(loss).backward()
        epoch_loss += loss.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
        print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
    del loss, outputs, batch_data
    torch.cuda.empty_cache()
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
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)
                del val_inputs, val_labels, val_outputs
                torch.cuda.empty_cache()
            # aggregate the final mean dice result
            del val_data
            torch.cuda.empty_cache()
            metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()
            print(metric)
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), CFG.model_path)
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )

import gc
del train_ds, val_ds
del train_loader, val_loader
gc.collect()
torch.cuda.empty_cache()

test_images = sorted(glob.glob(os.path.join(data_dir, "imagesTs", "*.nii")))
test_ids = [path.split('_')[-2] for path in test_images]
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


