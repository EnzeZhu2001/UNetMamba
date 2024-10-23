from torch.utils.data import DataLoader
from unetmamba_model.losses import *
from unetmamba_model.datasets.vaihingen_dataset import *
from unetmamba_model.models.UNetMamba import UNetMamba
from catalyst.contrib.nn import Lookahead
from catalyst import utils
from fvcore.nn import flop_count, parameter_count
import copy


# training hparam
max_epoch = 100
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 1
lr = 6e-4
weight_decay = 2.5e-4
backbone_lr = 6e-5
backbone_weight_decay = 2.5e-4
num_classes = len(CLASSES)
classes = CLASSES
image_size = 1024
crop_size = int(512*float(image_size/1024))

weights_name = "unetmamba-"+str(image_size)+"-e"+str(max_epoch)
weights_path = "model_weights/vaihingen/{}".format(weights_name)
test_weights_name = "unetmamba-"+str(image_size)+"-e"+str(max_epoch)
log_name = 'vaihingen/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = False
check_val_every_n_epoch = 1
pretrained_ckpt_path = None# the path for the pretrained unetmamba weight
gpus = 'auto'  # default or gpu ids:[0] or gpu nums: 2, more setting can refer to pytorch_lightning
resume_ckpt_path = None  # whether continue training with the checkpoint, default None

# VSSM parameters
PATCH_SIZE = 4
IN_CHANS = 3
DEPTHS = [2, 2, 9, 2]
EMBED_DIM = 96
SSM_D_STATE = 16
SSM_RATIO = 2.0
SSM_RANK_RATIO = 2.0
SSM_DT_RANK = "auto"
SSM_ACT_LAYER = "silu"
SSM_CONV = 3
SSM_CONV_BIAS = True
SSM_DROP_RATE = 0.0
SSM_INIT = "v0"
SSM_FORWARDTYPE = "v4"
MLP_RATIO = 4.0
MLP_ACT_LAYER = "gelu"
MLP_DROP_RATE = 0.0
DROP_PATH_RATE = 0.1
PATCH_NORM = True
NORM_LAYER = "ln"
DOWNSAMPLE = "v2"
PATCHEMBED = "v2"
GMLP = False
USE_CHECKPOINT = False

#  define the network
net = UNetMamba(pretrained=pretrained_ckpt_path,
                num_classes=num_classes,
                patch_size=PATCH_SIZE,
                in_chans=IN_CHANS,
                depths=DEPTHS,
                dims=EMBED_DIM,
                ssm_d_state=SSM_D_STATE,
                ssm_ratio=SSM_RATIO,
                ssm_rank_ratio=SSM_RANK_RATIO,
                ssm_dt_rank=("auto" if SSM_DT_RANK == "auto" else int(SSM_DT_RANK)),
                ssm_act_layer=SSM_ACT_LAYER,
                ssm_conv=SSM_CONV,
                ssm_conv_bias=SSM_CONV_BIAS,
                ssm_drop_rate=SSM_DROP_RATE,
                ssm_init=SSM_INIT,
                forward_type=SSM_FORWARDTYPE,
                mlp_ratio=MLP_RATIO,
                mlp_act_layer=MLP_ACT_LAYER,
                mlp_drop_rate=MLP_DROP_RATE,
                drop_path_rate=DROP_PATH_RATE,
                patch_norm=PATCH_NORM,
                norm_layer=NORM_LAYER,
                downsample_version=DOWNSAMPLE,
                patchembed_version=PATCHEMBED,
                gmlp=GMLP,
                use_checkpoint=USE_CHECKPOINT)
# define the loss
loss = UnetMambaLoss(ignore_index=ignore_index)
use_aux_loss = True

# define the dataloader
def get_training_transform():
    train_transform = [
        albu.RandomRotate90(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[0.5, 0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=crop_size, max_ratio=0.75, ignore_index=len(CLASSES), nopad=False)])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask

train_dataset = VaihingenDataset(data_root='data/vaihingen/train_'+str(image_size), mode='train',
                                 mosaic_ratio=0.25, transform=train_aug)

val_dataset = VaihingenDataset(data_root='data/vaihingen/val_'+str(image_size),
                               transform=val_aug)

test_dataset = VaihingenDataset(data_root='data/vaihingen/test_'+str(image_size),
                                transform=val_aug)

pin_memory = True
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=pin_memory,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=pin_memory,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

model = copy.deepcopy(net)
model.cuda().eval()
input = torch.randn((1, 3, image_size, image_size), device=next(model.parameters()).device)
params = parameter_count(model)[""]
Gflops, unsupported = flop_count(model=model, inputs=(input,))
print('GFLOPs: ', sum(Gflops.values()), 'G')
print('Params: ', params/1e6,'M')
