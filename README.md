# UNetMamba
## Introduction

**UNetMamba** is the official code of 



## Folder Structure

Prepare the following folders to organize this repo:
```none
├── UNetMamba (code)
|   ├──config 
|   ├──tools
|   ├──unetmamba_model
|   ├──train.py
|   ├──loveda_test.py
|   ├──vaihingen_test.py
├── pretrain_weights (pretrained weights of backbones)
├── model_weights (model weights trained on ISPRS vaihingen, LoveDA, etc)
├── fig_results (the segmentation results predicted by models)
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
│   │   ├── test_images (original)
│   │   ├── test_masks (original)
│   │   ├── test_masks_eroded (original)
│   │   ├── train_1024 (train set split into 1024*1024)
│   │   ├── test_1024 (test set split into 1024*1024)
│   │   ├── ...
```

## Install


## Pretrained Weights of Backbones


## Data Preprocessing

Download the datasets from the official website and split them.

**Vaihingen**

Generate the training set.
```
python GeoSeg-main/tools/vaihingen_patch_split.py 
--img-dir "autodl-tmp/data/vaihingen/train_images" --mask-dir "autodl-tmp/data/vaihingen/train_masks" 
--output-img-dir "autodl-tmp/data/vaihingen/train_1024/images" --output-mask-dir "autodl-tmp/data/vaihingen/train_1024/masks" 
--mode "train" --split-size 1024 --stride 512
```
Generate the testing set.
```
python GeoSeg-main/tools/vaihingen_patch_split.py 
--img-dir "autodl-tmp/data/vaihingen/test_images" --mask-dir "autodl-tmp/data/vaihingen/test_masks_eroded" 
--output-img-dir "autodl-tmp/data/vaihingen/test_1024/images" --output-mask-dir "autodl-tmp/data/vaihingen/test_1024/masks"
--mode "val" --split-size 1024 --stride 1024 --eroded
```
Generate the masks_1024_rgb (RGB format ground truth labels) for visualization.
```
python GeoSeg-main/tools/vaihingen_patch_split.py 
--img-dir "autodl-tmp/data/vaihingen/test_images" --mask-dir "autodl-tmp/data/vaihingen/test_masks" 
--output-img-dir "autodl-tmp/data/vaihingen/test_1024/images" --output-mask-dir "autodl-tmp/data/vaihingen/test_1024/masks_rgb" 
--mode "val" --split-size 1024 --stride 1024 --gt
```
As for the validation set, you can select some images from the training set to build it.


**LoveDA**
```
python GeoSeg-main/tools/loveda_mask_convert.py --mask-dir autodl-tmp/data/LoveDA/Train/Rural/masks_png --output-mask-dir autodl-tmp/data/LoveDA/Train/Rural/masks_png_convert
python GeoSeg-main/tools/loveda_mask_convert.py --mask-dir autodl-tmp/data/LoveDA/Train/Urban/masks_png --output-mask-dir autodl-tmp/data/LoveDA/Train/Urban/masks_png_convert
python GeoSeg-main/tools/loveda_mask_convert.py --mask-dir autodl-tmp/data/LoveDA/Val/Rural/masks_png --output-mask-dir autodl-tmp/data/LoveDA/Val/Rural/masks_png_convert
python GeoSeg-main/tools/loveda_mask_convert.py --mask-dir autodl-tmp/data/LoveDA/Val/Urban/masks_png --output-mask-dir autodl-tmp/data/LoveDA/Val/Urban/masks_png_convert
python GeoSeg-main/tools/loveda_mask_convert.py --mask-dir autodl-tmp/data/LoveDA/train_val/Rural/masks_png --output-mask-dir autodl-tmp/data/LoveDA/train_val/Rural/masks_png_convert
python GeoSeg-main/tools/loveda_mask_convert.py --mask-dir autodl-tmp/data/LoveDA/train_val/Urban/masks_png --output-mask-dir autodl-tmp/data/LoveDA/train_val/Urban/masks_png_convert
```

## Training

"-c" means the path of the config, use different **config** to train different models.

```
python GeoSeg-main/train.py -c GeoSeg-main/config/loveda/unetmamba.py
python GeoSeg-main/train.py -c GeoSeg-main/config/vaihingen/unetmamba.py
```

## Testing

"-c" denotes the path of the config, Use different **config** to test different models. 

"-o" denotes the output path 

"-t" denotes the test time augmentation (TTA), can be [None, 'lr', 'd4'], default is None, 'lr' is flip TTA, 'd4' is multiscale TTA

"--rgb" denotes whether to output masks in RGB format


**LoveDA** ([Online Testing](https://codalab.lisn.upsaclay.fr/competitions/421))
```
python GeoSeg-main/loveda_test.py -c GeoSeg-main/config/loveda/unetmamba.py -o autodl-tmp/fig_results/loveda/unetmamba_test
python GeoSeg-main/loveda_test.py -c GeoSeg-main/config/loveda/unetmamba.py -o autodl-tmp/fig_results/loveda/unetmamba_test -t 'd4'
python GeoSeg-main/loveda_test.py -c GeoSeg-main/config/loveda/unetmamba.py -o autodl-tmp/fig_results/loveda/unetmamba_test -t 'd4' --rgb --val
```

**Vaihingen**
```
python GeoSeg-main/vaihingen_test.py -c GeoSeg-main/config/vaihingen/unetmamba.py -o autodl-tmp/fig_results/vaihingen/unetmamba_1024_test --rgb
```

## Citation

If you find this project useful in your research, please consider citing：

- [UNetFormer: A UNet-like transformer for efficient semantic segmentation of remote sensing urban scene imagery](https://authors.elsevier.com/a/1fIji3I9x1j9Fs)

## Acknowledgement

We wish **GeoSeg** could serve the growing research of remote sensing by providing a unified benchmark 
and inspiring researchers to develop their own segmentation networks. Many thanks the following projects's contributions to **GeoSeg**.
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
