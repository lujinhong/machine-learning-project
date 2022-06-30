# -*- coding: utf-8 -*- 

"""
    AUTHOR: lujinhong
CREATED ON: 2021年11月18日 14:39
   PROJECT: machine-learning-project
   DESCRIPTION: 训练resnet模型
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from resnet import resnet34
import sys
sys.path.append('../../..')
from utils.constants import model_root, dataset_root

epochs = 10
save_path = os.path.join(model_root, 'resNet34.pth')
best_acc = 0.0


def get_dataloader():
    batch_size = 16
    image_path = os.path.join(dataset_root, "flower_data")  # flower data set path
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process to load image in {}'.format(num_workers, image_path))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=num_workers)
    train_num, val_num = len(train_dataset), len(validate_dataset)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # print(len(train_dataset), len(train_loader), len(validate_dataset), len(validate_loader), batch_size, num_workers)
    # 将类别与ID的映射关系写入json文件
    save_flower_dict(train_dataset)
    return train_loader, val_num, validate_loader


def save_flower_dict(train_dataset):
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)


def train_one_epoch(device, epoch, loss_function, net, optimizer, train_loader,
                    validate_loader):
    # train
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader)
    train_steps = len(train_loader) # len(dataloader) = len(dataset)/batch_size
    for step, data in enumerate(train_bar):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)
    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    val_num = len(validate_loader.dataset)
    # 在python，如果全局变量在函数中没有提前用global申明，就修改其值，结果是这个全局变量不会被修改，
    # 会在这个函数中另外产生一个局部变量（名字相同）。
    global best_acc
    with torch.no_grad():
        val_bar = tqdm(validate_loader)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            # loss = loss_function(outputs, test_labels)
            # 使用准确率代替损失函数
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
            val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)
    val_accurate = acc / val_num
    print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
          (epoch + 1, running_loss / train_steps, val_accurate))
    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(net.state_dict(), save_path)


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # 1、数据集
    train_loader, val_num, validate_loader = get_dataloader()

    # 2、创建模型，及加载预训练的模型
    net = resnet34()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    # model_weight_path = os.path.join(model_root, "resnet34-pre.pth")
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # for param in net.parameters():
    #     param.requires_grad = False

    # 3、改变全连接层的输出数量，对应待分类的类别数量
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 5)
    net.to(device)

    # 4、损失函数及优化器
    loss_function = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    # 5、训练
    for epoch in range(epochs):
        train_one_epoch(device, epoch, loss_function, net, optimizer, train_loader,
                        validate_loader)
    print('Finished Training')


if __name__ == '__main__':
    main()
