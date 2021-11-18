# -*- coding: utf-8 -*- 

"""
    AUTHOR: lujinhong
CREATED ON: 2021年11月18日 16:47
   PROJECT: machine-learning-project
   DESCRIPTION: 常用的数据预处理功能。
"""

import torch
from torchvision import transforms, datasets


# 加载特定目录格式的图像生产dataset
def image_folder_dataset(image_dir):
    data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    flower_dataset = datasets.ImageFolder(root=image_dir,
                                          transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(flower_dataset,
                                                 batch_size=4, shuffle=True,
                                                 num_workers=4)
    return dataset_loader


def test_image_folder_dataset():
    train_dataloader  = image_folder_dataset('~/datasets/flower_data/train')
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")


if __name__ == '__main__':
    test_image_folder_dataset()
