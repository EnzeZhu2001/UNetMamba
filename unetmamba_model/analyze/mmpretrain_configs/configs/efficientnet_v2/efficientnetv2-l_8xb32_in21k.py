_base_ = ['./efficientnetv2-s_8xb32_in21k.py']

# unetmamba setting
model = dict(backbone=dict(arch='l'), )
