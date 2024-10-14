# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .mamba_sys import VSSM
# from .mamba_sys_swin import VSSM_SWIN
# from .mamba_sys_vmamba import VSSMMamba
# from .mamba_sys_res import VSSMRes

logger = logging.getLogger(__name__)

class MambaUnet(nn.Module):
    def __init__(self, PATCH_SIZE,
                    EMBED_DIM, DEPTHS, MLP_RATIO,
                    DROP_RATE, DROP_PATH_RATE, PATCH_NORM,
                    USE_CHECKPOINT, PRETRAIN_CKPT=None,
                    num_classes=6, zero_head=False, vis=False):
        super(MambaUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head

        self.mamba_unet = VSSM(patch_size=PATCH_SIZE,
                                in_chans=3,
                                num_classes=self.num_classes,
                                embed_dim=EMBED_DIM,
                                depths=DEPTHS,
                                mlp_ratio=MLP_RATIO,
                                drop_rate=DROP_RATE,
                                drop_path_rate=DROP_PATH_RATE,
                                patch_norm=PATCH_NORM,
                                use_checkpoint=USE_CHECKPOINT)

        # weight_path = "/root/folder/stseg_base.pth"
        # old_dict = torch.load(weight_path)['state_dict']
        # model_dict = self.mamba_unet.state_dict()
        # old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        # model_dict.update(old_dict)
        # msg = self.mamba_unet.load_state_dict(model_dict)
        # print(msg)


    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.mamba_unet(x)
        # print()
        return logits

    def load_from(self, PRETRAIN_CKPT=None):
        pretrained_path = PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.mamba_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.mamba_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.mamba_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")


# class MambaSwin(nn.Module):
#     def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
#         super(MambaSwin, self).__init__()
#         self.num_classes = num_classes
#         self.zero_head = zero_head
#         self.config = config
#
#         self.mamba_unet =  VSSM_SWIN(
#                                 patch_size=img_size,
#                                 in_chans=3,
#                                 num_classes=self.num_classes,
#                                 embed_dim=config.MODEL.VSSM.EMBED_DIM,
#                                 depths=config.MODEL.VSSM.DEPTHS,
#                                 mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
#                                 drop_rate=config.MODEL.DROP_RATE,
#                                 drop_path_rate=config.MODEL.DROP_PATH_RATE,
#                                 patch_norm=config.MODEL.SWIN.PATCH_NORM,
#                                 use_checkpoint=config.TRAIN.USE_CHECKPOINT)
#
#         weight_path = "/root/folder/stseg_base.pth"
#         old_dict = torch.load(weight_path)['state_dict']
#         model_dict = self.mamba_unet.state_dict()
#         old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
#         model_dict.update(old_dict)
#         msg = self.mamba_unet.load_state_dict(model_dict)
#         print(msg)
#         import time
#         time.sleep(20)
#
#     def forward(self, x):
#         if x.size()[1] == 1:
#             x = x.repeat(1,3,1,1)
#         logits = self.mamba_unet(x)
#         return logits
#
#     def load_from(self, config):
#         pretrained_path = config.MODEL.PRETRAIN_CKPT
#         if pretrained_path is not None:
#             print("pretrained_path:{}".format(pretrained_path))
#             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             pretrained_dict = torch.load(pretrained_path, map_location=device)
#             if "model"  not in pretrained_dict:
#                 print("---start load pretrained modle by splitting---")
#                 pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
#                 for k in list(pretrained_dict.keys()):
#                     if "output" in k:
#                         print("delete key:{}".format(k))
#                         del pretrained_dict[k]
#                 msg = self.mamba_unet.load_state_dict(pretrained_dict,strict=False)
#                 # print(msg)
#                 return
#             pretrained_dict = pretrained_dict['model']
#             print("---start load pretrained modle of swin encoder---")
#
#             model_dict = self.mamba_unet.state_dict()
#             full_dict = copy.deepcopy(pretrained_dict)
#             for k, v in pretrained_dict.items():
#                 if "layers." in k:
#                     current_layer_num = 3-int(k[7:8])
#                     current_k = "layers_up." + str(current_layer_num) + k[8:]
#                     full_dict.update({current_k:v})
#             for k in list(full_dict.keys()):
#                 if k in model_dict:
#                     if full_dict[k].shape != model_dict[k].shape:
#                         print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
#                         del full_dict[k]
#
#             msg = self.mamba_unet.load_state_dict(full_dict, strict=False)
#             # print(msg)
#         else:
#             print("none pretrain")
#
#
# class Mambamamba(nn.Module):
#     def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
#         super(Mambamamba, self).__init__()
#         self.num_classes = num_classes
#         self.zero_head = zero_head
#         self.config = config
#
#         self.mamba_unet =  VSSMMamba(
#                                 patch_size=img_size,
#                                 in_chans=3,
#                                 num_classes=self.num_classes,
#                                 embed_dim=config.MODEL.VSSM.EMBED_DIM,
#                                 depths=config.MODEL.VSSM.DEPTHS,
#                                 mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
#                                 drop_rate=config.MODEL.DROP_RATE,
#                                 drop_path_rate=config.MODEL.DROP_PATH_RATE,
#                                 patch_norm=config.MODEL.SWIN.PATCH_NORM,
#                                 use_checkpoint=config.TRAIN.USE_CHECKPOINT)
#
#         # weight_path = "/root/folder/stseg_base.pth"
#         # old_dict = torch.load(weight_path)['state_dict']
#         # model_dict = self.mamba_unet.state_dict()
#         # old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
#         # model_dict.update(old_dict)
#         # msg = self.mamba_unet.load_state_dict(model_dict)
#         # print(msg)
#         # import time
#         # time.sleep(20)
#
#     def forward(self, x):
#         if x.size()[1] == 1:
#             x = x.repeat(1,3,1,1)
#         logits = self.mamba_unet(x)
#         return logits
#
#     def load_from(self, config):
#         pretrained_path = config.MODEL.PRETRAIN_CKPT
#         if pretrained_path is not None:
#             print("pretrained_path:{}".format(pretrained_path))
#             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             pretrained_dict = torch.load(pretrained_path, map_location=device)
#             if "model"  not in pretrained_dict:
#                 print("---start load pretrained modle by splitting---")
#                 pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
#                 for k in list(pretrained_dict.keys()):
#                     if "output" in k:
#                         print("delete key:{}".format(k))
#                         del pretrained_dict[k]
#                 msg = self.mamba_unet.load_state_dict(pretrained_dict,strict=False)
#                 print(msg)
#                 import time
#                 time.sleep(20)
#                 return
#             pretrained_dict = pretrained_dict['model']
#             print("---start load pretrained modle of swin encoder---")
#
#             model_dict = self.mamba_unet.state_dict()
#             full_dict = copy.deepcopy(pretrained_dict)
#             for k, v in pretrained_dict.items():
#                 if "layers." in k:
#                     current_layer_num = 3-int(k[7:8])
#                     current_k = "layers_up." + str(current_layer_num) + k[8:]
#                     full_dict.update({current_k:v})
#             for k in list(full_dict.keys()):
#                 if k in model_dict:
#                     if full_dict[k].shape != model_dict[k].shape:
#                         print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
#                         del full_dict[k]
#
#             msg = self.mamba_unet.load_state_dict(full_dict, strict=False)
#             print(msg)
#         else:
#             print("none pretrain")
#
# class MambaRes(nn.Module):
#     def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
#         super(MambaRes, self).__init__()
#         self.num_classes = num_classes
#         self.zero_head = zero_head
#         self.config = config
#
#         self.mamba_unet =  VSSMRes(
#                                 patch_size=img_size,
#                                 in_chans=3,
#                                 num_classes=self.num_classes,
#                                 embed_dim=config.MODEL.VSSM.EMBED_DIM,
#                                 depths=config.MODEL.VSSM.DEPTHS,
#                                 mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
#                                 drop_rate=config.MODEL.DROP_RATE,
#                                 drop_path_rate=config.MODEL.DROP_PATH_RATE,
#                                 patch_norm=config.MODEL.SWIN.PATCH_NORM,
#                                 use_checkpoint=config.TRAIN.USE_CHECKPOINT)
#
#         # weight_path = "/root/folder/stseg_base.pth"
#         # old_dict = torch.load(weight_path)['state_dict']
#         # model_dict = self.mamba_unet.state_dict()
#         # old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
#         # model_dict.update(old_dict)
#         # msg = self.mamba_unet.load_state_dict(model_dict)
#         # print(msg)
#
#
#     def forward(self, x):
#         if x.size()[1] == 1:
#             x = x.repeat(1,3,1,1)
#         logits = self.mamba_unet(x)
#         return logits
#
#     def load_from(self, config):
#         pretrained_path = config.MODEL.PRETRAIN_CKPT
#         if pretrained_path is not None:
#             print("pretrained_path:{}".format(pretrained_path))
#             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             pretrained_dict = torch.load(pretrained_path, map_location=device)
#             if "model"  not in pretrained_dict:
#                 print("---start load pretrained modle by splitting---")
#                 pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
#                 for k in list(pretrained_dict.keys()):
#                     if "output" in k:
#                         print("delete key:{}".format(k))
#                         del pretrained_dict[k]
#                 msg = self.mamba_unet.load_state_dict(pretrained_dict,strict=False)
#                 # print(msg)
#                 return
#             pretrained_dict = pretrained_dict['model']
#             print("---start load pretrained modle of swin encoder---")
#
#             model_dict = self.mamba_unet.state_dict()
#             full_dict = copy.deepcopy(pretrained_dict)
#             for k, v in pretrained_dict.items():
#                 if "layers." in k:
#                     current_layer_num = 3-int(k[7:8])
#                     current_k = "layers_up." + str(current_layer_num) + k[8:]
#                     full_dict.update({current_k:v})
#             for k in list(full_dict.keys()):
#                 if k in model_dict:
#                     if full_dict[k].shape != model_dict[k].shape:
#                         print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
#                         del full_dict[k]
#
#             msg = self.mamba_unet.load_state_dict(full_dict, strict=False)
#             # print(msg)
#         else:
#             print("none pretrain")