_base_ = ['./regnetx-400mf_8xb128_in1k.py']

# unetmamba settings
model = dict(
    backbone=dict(type='RegNet', arch='regnetx_800mf'),
    head=dict(in_channels=672, ))
