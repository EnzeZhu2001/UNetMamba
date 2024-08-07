_base_ = ['twins-pcpvt-base_8xb128_in1k.py']

# unetmamba settings
model = dict(backbone=dict(arch='large'), head=dict(in_channels=512))

# dataset settings
train_dataloader = dict(batch_size=64)
