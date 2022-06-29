# -*- coding: utf-8 -*-
"""
    AUTHOR: lujinhong
CREATED ON: 2022年06月24日 10:39
   PROJECT: machine-learning-project
DESCRIPTION: 提取模型的中间输出，作为图像的特征向量；以及冻结某些层然后继续训练模型的方法。详见jupyter笔记。
"""
import os.path

import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from utils.vision_utils import FeatureExtractor

import torchvision.models as models
import torchvision.transforms as transforms

from ..utils import constants

img_path = os.path.join(constants.dataset_root, 'flower_data/flower_photos/daisy/450128527_fd35742d44.jpg')


extract_list = ["conv1", "maxpool", "layer1", "avgpool", "fc"]
resnet = models.resnet50(pretrained=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet.to(device)
# 打印网络结构，可以得到所有层的名字。
print(resnet)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()]
)
img = Image.open(img_path)
img = transform(img)

x = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False).to(device)
extract_result = FeatureExtractor(resnet, extract_list)
# 其中avgpool后的结果就是我们所需要的图像向量。
print(extract_result(x)[3].shape)  # [0]:conv1  [1]:maxpool  [2]:layer1  [3]:avgpool  [4]:fc
#
# # 继续训练模型：冻结某些层
# for layer in resnet.children():
#     # 有些层没有可以训练的参数，所以没有weight属性，如relu。
#     if hasattr(layer, 'weight'):
#         layer.weight.requires_grad = False
#
# # 设置optimizer。
# my_model = torch.optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), lr=0.005, betas=(0.5, 0.999))
