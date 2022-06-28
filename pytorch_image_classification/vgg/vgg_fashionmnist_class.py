# -*- coding: utf-8 -*-
"""
    AUTHOR: lujinhong
CREATED ON: 2022年06月22日 17:16
   PROJECT: machine-learning-project
DESCRIPTION: 使用类的方式定义VGG。详细注解。
"""

import torch
from matplotlib import pyplot as plt
from torch import nn
# jupyter中使用
import sys
sys.path.append('../..')
from utils import my_utils, constants


class VGG(nn.Module):
    def __init__(self, vgg_type, num_classes):
        """
        1、模型所有带参数的层都要放到__init__()中，否则调用model.to(device)时，这些层的参数不会被放到GPU，
        从而出现报错。
        :param vgg_type: VGG的常见类型。
        :param num_classes: 待分类图片的类别数量。
        """
        super(VGG, self).__init__()
        # （每个block中有多少个卷积层，每个卷积层的通道数）
        vgg_structs = {
            'vgg11': ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)),
            'vgg13': ((2, 64), (2, 128), (2, 256), (2, 512), (2, 512)),
            'vgg16': ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512)),
            'vgg19': ((2, 64), (2, 128), (4, 256), (4, 512), (4, 512))
        }
        # 卷积层部分
        self.conv_arch = vgg_structs[vgg_type]
        self.conv_blocks = []
        in_channels = 1
        for(num_convs, out_channels) in self.conv_arch:
            self.conv_blocks.append(self.vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels
        # 列表前面加星号作用是将列表内的元素展开成独立的参数，传入函数。
        self.cnn = nn.Sequential(*self.conv_blocks)
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def vgg_block(self, num_convs, in_channels, out_channels):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())  # 不用参数
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


def train(net, train_dataloader, test_dataloader, num_epochs, lr, device):
    # 使用指定的方式进行参数初始化
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    # nn.Module.apply()函数将参数指定的函数应用在model自身及其每一个子层
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    # jupyter画图使用
    # animator = my_utils.Animator(xlabel='epoch', xlim=[1, num_epochs],
    #                         legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = my_utils.Timer(), len(train_dataloader)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = my_utils.Accumulator(3)
        # 告诉模型现在是训练阶段，有些层（如dropout, batchnorm等）在训练和测试有不同的行为。
        # 对应的，可以使用net.eval()或者net.train(mode=False)告诉模型现在是测试阶段。
        net.train()
        for i, (X, y) in enumerate(train_dataloader):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], my_utils.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            # if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            #     animator.add(epoch + (i + 1) / num_batches,
            #                  (train_l, train_acc, None))
        test_acc = my_utils.evaluate_accuracy(net, test_dataloader, device=device)
        # animator.add(epoch + 1, (None, None, test_acc))
        print(f'epoch:{epoch}, train_loss:{train_l:.3f}, train_accuracy:{train_acc:.3f}, test_accuracy:{test_acc:.3f}')
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


if __name__ == '__main__':
    model = VGG(vgg_type='vgg11', num_classes=10)
    device = my_utils.try_gpu(i=1)
    lr, num_epochs, batch_size = 0.05, 10, 32
    train_dataloader, test_dataloader = my_utils.load_data_fashion_mnist_pytorch(batch_size, resize=224, num_workers=1)
    train(model, train_dataloader, test_dataloader, num_epochs, lr, device)
