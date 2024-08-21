# UNetMamba
## Introduction

**UNetMamba** is the official code of 


## Folder Structure

Prepare the following folders to organize this repo:
```none
├── UNetMamba
|   ├──config 
|   ├──tools
|   ├──unetmamba_model
|   ├──train.py
|   ├──loveda_test.py
|   ├──vaihingen_test.py
├── pretrain_weights (pretrained weights of backbones)
├── model_weights (model weights trained on ISPRS vaihingen, LoveDA, etc)
├── fig_results (the segmentation results)
├── data
│   ├── LoveDA
│   │   ├── Train
│   │   │   ├── Urban
│   │   │   │   ├── images_png (original)
│   │   │   │   ├── masks_png (original)
│   │   │   │   ├── masks_png_convert (converted masks used for training)
│   │   │   │   ├── masks_png_convert_rgb (original rgb format masks)
│   │   │   ├── Rural
│   │   │   │   ├── images_png 
│   │   │   │   ├── masks_png 
│   │   │   │   ├── masks_png_convert
│   │   │   │   ├── masks_png_convert_rgb
│   │   ├── Val (the same with Train)
│   │   ├── Test
│   │   ├── train_val (merge Train and Val)
│   ├── vaihingen
│   │   ├── train_images (original)
│   │   ├── train_masks (original)
│   │   ├── test_images (original_ID:: 1, 5, 11, 15, 21, 26, 30, 33, 38)
│   │   ├── test_masks (original_ID:: 1, 5, 11, 15, 21, 26, 30, 33, 38)
│   │   ├── test_masks_eroded (original_ID:: 1, 5, 11, 15, 21, 26, 30, 33, 38)
│   │   ├── train_1024 (train set at 1024*1024)
│   │   ├── test_1024 (test set at 1024*1024)
│   │   ├── ...
```

## Install


## Pretrained Weights of Backbones

[pretrain_weights](https://pan.baidu.com/s/19TRZVfz6M9v0VYxiHB6mSA?pwd=82cj) 

## Pretrained Weights of UNetMamba

[model_weights](https://pan.baidu.com/s/1wVVI1MPY_fnVSYg_5bLIlQ?pwd=mdwe) 

## Data Preprocessing

Download the datasets from the official website and split them as follows.

**LoveDA** ([LoveDA official](https://github.com/Junjue-Wang/LoveDA))
```
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Rural/masks_png --output-mask-dir data/LoveDA/Train/Rural/masks_png_convert
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Urban/masks_png --output-mask-dir data/LoveDA/Train/Urban/masks_png_convert

python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Rural/masks_png --output-mask-dir data/LoveDA/Val/Rural/masks_png_convert
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Urban/masks_png --output-mask-dir data/LoveDA/Val/Urban/masks_png_convert

python tools/loveda_mask_convert.py --mask-dir data/LoveDA/train_val/Rural/masks_png --output-mask-dir data/LoveDA/train_val/Rural/masks_png_convert
python tools/loveda_mask_convert.py --mask-dir data/LoveDA/train_val/Urban/masks_png --output-mask-dir data/LoveDA/train_val/Urban/masks_png_convert
```

**Vaihingen** ([Vaihingen official](https://www.isprs.org/education/benchmarks/UrbanSemLab/Default.aspx))

Generate the train set.
```
python tools/vaihingen_patch_split.py 
--img-dir "data/vaihingen/train_images" --mask-dir "data/vaihingen/train_masks" 
--output-img-dir "data/vaihingen/train_1024/images" --output-mask-dir "data/vaihingen/train_1024/masks" 
--mode "train" --split-size 1024 --stride 512
```
Generate the test set. (Tip: the eroded one.)
```
python tools/vaihingen_patch_split.py 
--img-dir "data/vaihingen/test_images" --mask-dir "data/vaihingen/test_masks_eroded" 
--output-img-dir "data/vaihingen/test_1024/images" --output-mask-dir "data/vaihingen/test_1024/masks"
--mode "val" --split-size 1024 --stride 1024 --eroded
```
Generate the masks_1024_rgb (RGB format ground truth labels) for visualization.
```
python tools/vaihingen_patch_split.py 
--img-dir "data/vaihingen/test_images" --mask-dir "data/vaihingen/test_masks" 
--output-img-dir "data/vaihingen/test_1024/images" --output-mask-dir "data/vaihingen/test_1024/masks_rgb" 
--mode "val" --split-size 1024 --stride 1024 --gt
```
As for the validation set, you can select some images from the training set to build it.

## Training

"-c" means the path of the config, use different **config** to train different models in different datasets.

```
python train.py -c config/loveda/unetmamba.py
python train.py -c config/vaihingen/unetmamba.py
```

## Testing

"-c" denotes the path of the config, Use different **config** to test different models in different datasets. 

"-o" denotes the output path 

"-t" denotes the test time augmentation (TTA), can be [None, 'lr', 'd4'], default is None, 'lr' is flip TTA, 'd4' is multiscale TTA

"--rgb" denotes whether to output masks in RGB format


**LoveDA** ([Online Testing](https://codalab.lisn.upsaclay.fr/competitions/421))
```
python loveda_test.py -c config/loveda/unetmamba.py -o fig_results/loveda/unetmamba_test
python loveda_test.py -c config/loveda/unetmamba.py -o fig_results/loveda/unetmamba_test -t 'd4'
python loveda_test.py -c config/loveda/unetmamba.py -o fig_results/loveda/unetmamba_rgb -t 'd4' --rgb --val
```

**Vaihingen**
```
python vaihingen_test.py -c config/vaihingen/unetmamba.py -o fig_results/vaihingen/unetmamba_test
python vaihingen_test.py -c config/vaihingen/unetmamba.py -o fig_results/vaihingen/unetmamba_test -t 'lr'
python vaihingen_test.py -c config/vaihingen/unetmamba.py -o fig_results/vaihingen/unetmamba_rgb --rgb
```

## Citation

If you find this project useful in your research, please consider citing：


## Acknowledgement

- [GeoSeg](https://github.com/WangLibo1995/GeoSeg)
- [SSRS](https://github.com/sstary/SSRS)
- [VMamba](https://github.com/MzeroMiko/VMamba)
- [Swin-UMamba](https://github.com/JiarunLiu/Swin-UMamba)
- [LoveDA](https://github.com/Junjue-Wang/LoveDA)
