# -*- coding: utf-8 -*-
"""
    AUTHOR: lujinhong
CREATED ON: 2022年06月22日 15:28
   PROJECT: machine-learning-project
DESCRIPTION: vgg的简单实现方式，使用了函数来构建模型。参考d2l。另有其它代码使用类来构建模型。
"""

import torch
from matplotlib import pyplot as plt

dir='../..'
import sys
sys.path.append(dir)
from utils import my_utils, constants
from torch import nn


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())  # 不用参数
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(conv_arch):
    conv_blocks = []
    in_channels = 1

    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blocks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blocks,
        nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))


def train(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = my_utils.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = my_utils.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        print(f'epoch {epoch}')
        metric = my_utils.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
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
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = my_utils.evaluate_accuracy(net, test_iter, device=device)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


if __name__ == '__main__':
    # （每个block中有多少个卷积层，每个卷积层的通道数）
    vgg11 = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    vgg13 = ((2, 64), (2, 128), (2, 256), (2, 512), (2, 512))
    vgg16 = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))
    vgg19 = ((2, 64), (2, 128), (4, 256), (4, 512), (4, 512))
    net = vgg(vgg11)

    device = my_utils.try_gpu(i=1)
    lr, num_epochs, batch_size = 0.05, 10, 32
    train_iter, test_iter = my_utils.load_data_fashion_mnist_pytorch(batch_size, resize=224, num_workers=1)
    train(net, train_iter, test_iter, num_epochs, lr, device)
