_base_ = ['twins-pcpvt-base_8xb128_in1k.py']

# unetmamba settings
model = dict(backbone=dict(arch='small'), head=dict(in_channels=512))
