# -*- coding: utf-8 -*- 

"""
    AUTHOR: lujinhong
CREATED ON: 2022年05月13日 14:41
   PROJECT: machine-learning-project
   DESCRIPTION: 使用全连接网络模型对fashionmnist进行分类。
"""
import torch
from utils import my_utils

from torch import nn
from torchvision import transforms


def train_epoch(net, train_iter, loss, updater):
    """训练模型一个迭代周期（定义见第3章）
    Defined in :numref:`sec_softmax_scratch`"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = my_utils.Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), my_utils.accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


def train(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型（定义见第3章）
    Defined in :numref:`sec_softmax_scratch`"""
    animator = my_utils.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = my_utils.evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    # train_loss, train_acc = train_metrics
    # assert train_loss < 0.5, print(train_loss)
    # assert train_acc <= 1 and train_acc > 0.7, print(train_acc)
    # assert test_acc <= 1 and test_acc > 0.7, print(test_acc)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


batch_size = 256
train_dataloader, test_dataloader = my_utils.load_data_fashion_mnist_pytorch(batch_size)

# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
net.apply(init_weights);
loss = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
num_epochs = 10
train(net, train_dataloader, test_dataloader, loss, num_epochs, optimizer)
