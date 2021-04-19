#encoding=utf-8
import os
import cv2
import sys
import time
import collections
import torch
import argparse
import numpy as np
import models

path_model = "./checkpoint.pth.tar"
path_save_wts = "./psenet0419.wts"

args_scale = 1.0
model = models.resnet50(pretrained=True, num_classes=7, scale=args_scale)
for param in model.parameters():
        param.requires_grad = False

checkpoint = torch.load(path_model)

d = collections.OrderedDict()
for key, value in checkpoint['state_dict'].items():
    tmp = key[7:]
    d[tmp] = value
model.load_state_dict(d)

model.eval()

import struct
f = open(path_save_wts, 'w')
f.write('{}\n'.format(len(model.state_dict().keys())))
for k, v in model.state_dict().items():
    vr = v.reshape(-1).cpu().numpy()
    f.write('{} {} '.format(k, len(vr)))
    for vv in vr:
        f.write(' ')
        f.write(struct.pack('>f',float(vv)).hex())
    f.write('\n')

print("success generate wts!")


