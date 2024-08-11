from unetmamba_model.classification.models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Optional, Union, Type, List, Tuple, Callable, Dict
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from unetmamba_model.models.ResT import ResT


def rest_lite(pretrained=True, weight_path='pretrain_weights/rest_lite.pth',  **kwargs):
    model = ResT(embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], apply_transform=True, **kwargs)
    if pretrained and weight_path is not None:
        old_dict = torch.load(weight_path)
        model_dict = model.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        model.load_state_dict(model_dict)
    return model


class PatchExpand(nn.Module):
    """
    Reference: https://arxiv.org/pdf/2105.05537.pdf
    """
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # B, C, H, W ==> B, H, W, C
        x = self.expand(x)
        B, H, W, C = x.shape

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x = self.norm(x)
        x = x.reshape(B, H*2, W*2, C//4)

        return x


class FinalPatchExpand_X4(nn.Module):
    """
    Reference:
        - GitHub: https://github.com/HuCaoFighting/Swin-Unet/blob/main/networks/swin_transformer_unet_skip_expand_decoder_sys.py
        - Paper: https://arxiv.org/pdf/2105.05537.pdf
    """
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # H, W = self.input_resolution
        x = x.permute(0, 2, 3, 1)  # B, C, H, W ==> B, H, W, C
        x = self.expand(x)
        B, H, W, C = x.shape
        # B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        # x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x = self.norm(x)
        x = x.reshape(B, H*self.dim_scale, W*self.dim_scale, self.output_dim)

        return x#.permute(0, 3, 1, 2)


class VSSLayer(nn.Module):
    """
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class LocalSupervision(nn.Module):
    def __init__(self, in_channels=128, num_classes=6):
        super().__init__()
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, dilation=1, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.ReLU6())
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, dilation=1, stride=1, padding=0, bias=False),
                                   nn.BatchNorm2d(in_channels),
                                   nn.ReLU6())
        self.drop = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(in_channels, num_classes, kernel_size=1, dilation=1, stride=1, padding=0, bias=False)

    def forward(self, x, h, w):
        local1 = self.conv3(x)
        local2 = self.conv1(x)
        x = self.drop(local1 + local2)
        x = self.conv_out(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x


class MambaSegDecoder(nn.Module):
    def __init__(
            self,
            num_classes: int,
            encoder_channels: Union[Tuple[int, ...], List[int]] = None,
            decode_channels: int = 64,
            drop_path_rate: float = 0.2,
            d_state: int = 16,
    ):
        super().__init__()

        encoder_output_channels = encoder_channels
        self.num_classes = num_classes
        n_stages_encoder = len(encoder_output_channels)

        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, (n_stages_encoder - 1) * 2)]
        depths = [2, 2, 2, 2]

        stages = []
        expand_layers = []
        lsm_layers = []
        concat_back_dim = []

        for s in range(1, n_stages_encoder):
            input_features_below = encoder_output_channels[-s]
            input_features_skip = encoder_output_channels[-(s + 1)]
            expand_layers.append(PatchExpand(
                input_resolution=None,
                dim=input_features_below,
                dim_scale=2,
                norm_layer=nn.LayerNorm,
            ))
            stages.append(VSSLayer(
                dim=input_features_skip,
                depth=2,
                attn_drop=0.,
                drop_path=dpr[sum(depths[:s - 1]):sum(depths[:s])],
                d_state=math.ceil(2 * input_features_skip / 6) if d_state is None else d_state,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=False,
            ))
            concat_back_dim.append(nn.Linear(2 * input_features_skip, input_features_skip))
            lsm_layers.append(LocalSupervision(encoder_channels[-(s + 1)], num_classes))

        expand_layers.append(FinalPatchExpand_X4(
            input_resolution=None,
            dim=encoder_output_channels[0],
            dim_scale=4,
            norm_layer=nn.LayerNorm,
        ))
        stages.append(nn.Identity())

        self.stages = nn.ModuleList(stages)
        self.expand_layers = nn.ModuleList(expand_layers)
        self.concat_back_dim = nn.ModuleList(concat_back_dim)
        if self.training:
            self.lsm = nn.ModuleList(lsm_layers)
        self.seg = nn.Conv2d(encoder_channels[-4], num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.init_weight()

    def forward(self, skips, h, w):
        lres_input = skips[-1]
        if self.training:
            ls = []
            for s in range(len(self.stages)):
                x = self.expand_layers[s](lres_input)

                if s < (len(self.stages) - 1):
                    x = torch.cat((x, skips[-(s + 2)].permute(0, 2, 3, 1)), -1)
                    x = self.concat_back_dim[s](x)
                x = self.stages[s](x).permute(0, 3, 1, 2)

                if s == (len(self.stages) - 1):
                    seg_out = self.seg(x)
                else:
                    ls.append(self.lsm[s](x, h, w))
                lres_input = x

            return seg_out, sum(ls)

        else:
            for s in range(len(self.stages)):
                x = self.expand_layers[s](lres_input)

                if s < (len(self.stages) - 1):
                    x = torch.cat((x, skips[-(s + 2)].permute(0, 2, 3, 1)), -1)
                    x = self.concat_back_dim[s](x)
                x = self.stages[s](x).permute(0, 3, 1, 2)

                if s == (len(self.stages) - 1):
                    seg_out = self.seg(x)
                lres_input = x

            return seg_out

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class UNetMamba(nn.Module):
    def __init__(self,
                 pretrained,
                 decode_channels=64,
                 # backbone_name="resnet18",
                 backbone_path='pretrain_weights/rest_lite.pth',
                 embed_dim=64,
                 num_classes=6,
                 **kwargs
                 ):
        super().__init__()

        # self.encoder = timm.create_model(backbone_name, features_only=True, output_stride=32,
        #                                   out_indices=(1, 2, 3, 4), pretrained=pretrained)
        # encoder_channels = self.encoder.feature_info.channels()
        self.encoder = rest_lite(weight_path=backbone_path)
        encoder_channels = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
        self.decoder = MambaSegDecoder(num_classes=num_classes, encoder_channels=encoder_channels, decode_channels=decode_channels)

    def forward(self, x):
        h, w = x.size()[-2:]
        outputs = self.encoder(x)

        if self.training:
            x, lsm = self.decoder(outputs, h, w)
            return x, lsm

        else:
            x = self.decoder(outputs, h, w)
            return x
