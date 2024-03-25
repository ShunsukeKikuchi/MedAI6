# Medical AI Contest 2024
This repository contains my 4th-Place Solution for Medical AI Contest 2024. (https://www.kaggle.com/competitions/medical-ai-contest2024/submissions)

## Overview
初学者目線で具体的なコードを見れるのはかなりありがたいと感じるため、こちらに今回のコンペで使用していたコードを公開します。コードの可読性を多少修正しましたが、諸々の事情(サーバーのメンテ)につきコードの再現性は確認が取れていないのと、環境設定が取得できていません。3月中にはに更新できるかと思います。<br>

I thought it was very helpful to be able to see concrete code, so I will published the code I used for this competition here. I had slightly modified the readability of the code, but due to various circumstances (server maintenance), I have not been able to confirm the reproducibility of the code, and I have not been able to obtain the environment settings.

### Submissions/Scores
Final Submission：(UNet + VNet + SwinRUNet + DynUNet) + <br>
　　　　(Pretrained / Attention-UNet) + (Pretrained / Attention-UNet++) <br>
　　　　LB 0.897, Private: 0.891

最初の4モデルはmonaiベース、残り2つは[RSNA-2022 1st-solution](https://www.kaggle.com/code/haqishen/rsna-2022-1st-place-solution-train-stage1)に修正を加え、segmentation model pytorchのモデルを2D→3Dにしたものを用いました。<br>

The first 4 models are based on monai, and the remaining 2 are modified from the RSNA-2022 1st-solution, using the modification code to let the model deal with 3D and modified some models on segmentation model pytorch.

#### MONAI-based
公開ベースラインを参考にmonaiのモデルを利用。

Referred to the public baseline and used the monai model.
- Unet: cv 0.85, lb 0.83 (epoch=380)
- Vnet: cv 0.85 lb 0.82 (epoch=250)
- SwinRUnet: cv 0.863, lb 0.832 (epoch=300)
- DynUNet: cv 0.86 lb 0.84 (epoch=350) <br>

→ Ensemble: LB 0.881, Private: 0.876
<br>
#### Pretrained + Segmentation models pytorch
RSNA2022 1st solを参考に、segmentation models pytorchのモデルを3D入力に対応できるように修正。Encoderは事前学習済みモデルを使用。

Referred to the RSNA2022 1st solution and modified the segmentation models pytorch model to deal with 3D input. The encoder is a pre-trained model.

- Attention Unet <br>
  Encoder: resnext50d_32x4d (pre-trained) <br>
  cv 0.913, LB 0.885
- Attention Unet++ <br>
  Encoder: 	seresnext50_32x4d (pre-trained)<br>
  cv 0.904, LB 0.879



## Downloading the repository
```
git clone https://github.com/ShunsukeKikuchi/MedAI6.git
cd MedAI6
```

## Setting up the environment:
```
# I'm gonna upload a configuration file soon, sorry for the inconvenience
```
