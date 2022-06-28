# -*- coding: utf-8 -*- 

"""
    AUTHOR: lujinhong
CREATED ON: 2022年05月13日 14:45
   PROJECT: machine-learning-project
   DESCRIPTION: TODO
"""
import numpy as np
import platform
import pandas as pd
from utils import constants
import torch
from matplotlib import pyplot as plt
import time
import torchvision
from IPython import display
from matplotlib_inline import backend_inline
from collections.abc import Iterable


from torchvision import transforms

def try_gpu(use_gpu=True, i=0):
    """如果存在且使用gpu，则返回gpu(i)，否则返回cpu
    """
    if use_gpu is True and platform.system().lower() == 'darwin':
        return torch.device('mps')
    elif use_gpu is True and platform.system().lower() == 'linux':
        if torch.cuda.device_count() >= i + 1:
            return torch.device(f'cuda:{i}')
    return torch.device('cpu')

    # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Timer:
    """记录程序多次运行时间
    使用示例见utils.ipynb。
    """
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


def synthetic_linear_data_numpy(w, b, num_examples):
    """使用numpy人造线性数据集，并加上一些随机噪声。
    根据num_example的数值及w的长度，生成shape = [num_expample, len(w)]的数据x，另y = w * x + b + 噪声
    使用示例见utils.ipynb。

    使用numpy和pytorch两种方式生成线性数据集。
    一般情况下，使用numpy的方式即可。但如果使用pytorch计算模型，则建议使用pytorch方式，否则经常会有数据格式错误的问题。
    """
    X = np.random.normal(0.0, 1.0, (num_examples, len(w)))
    y = np.matmul(X, w) + b
    y += np.random.normal(0.0, 0.01, y.shape)
    return X, y.reshape(-1,1)


def synthetic_linear_data_pytorch(w, b, num_examples):
    """使用pytorch人造线性数据集，并加上一些随机噪声。
    根据num_example的数值及w的长度，生成shape = [num_expample, len(w)]的数据x，另y = w * x + b + 噪声
    使用示例见utils.ipynb。
    """
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1,1)


def load_data_fashion_mnist_pytorch(batch_size, resize=None, num_workers=4):  #@save
    """下载pytorch Fashion-MNIST数据集，然后将其加载到内存中
    返回dataloader。
    """
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root=constants.dataset_root, train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root=constants.dataset_root, train=False, transform=trans, download=True)
    return (torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                        num_workers = num_workers),
            torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                        num_workers = num_workers))


def accuracy(y_hat, y):
    """计算分类问题预测正确的数量。
    y_hat为各个类别的概率，y为正确的类别。
    y_hat及y均为torch.tensor类型。
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float((cmp.type(y.dtype).sum()))


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴
    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def evaluate_accuracy(net, data_iter, device=None):
    """计算在指定数据集上模型的精度
    Defined in :numref:`sec_softmax_scratch`"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            # 将X和y也放入GPU，否则会报错。Input type (torch.FloatTensor) and weight type
            # (torch.cuda.FloatTensor) should be the same or input should be a MKLDNN
            # tensor and weight is a dense tensor
            if device is not None:
                X = X.to(device)
                y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())

    return metric[0] / metric[1]

#
# def evaluate_accuracy_gpu(net, data_iter, device=None):
#     """使用GPU计算模型在数据集上的精度
#
#     Defined in :numref:`sec_lenet`"""
#     if isinstance(net, torch.nn.Module):
#         net.eval()  # 设置为评估模式
#         if not device:
#             device = next(iter(net.parameters())).device
#     # 正确预测的数量，总预测的数量
#     metric = Accumulator(2)
#     with torch.no_grad():
#         for X, y in data_iter:
#             if isinstance(X, list):
#                 # BERT微调所需的（之后将介绍）
#                 X = [x.to(device) for x in X]
#             else:
#                 X = X.to(device)
#             y = y.to(device)
#             metric.add(accuracy(net(X), y), y.numel())
#     return metric[0] / metric[1]


class Accumulator:
    """在n个变量上累加，常用于计算分类正确的数量等。
    """
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    _, _, patches = plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    plt.legend(legend)

class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """Defined in :numref:`sec_softmax_scratch`"""
        # 增量地绘制多条线
        if legend is None:
            legend = []
        # 使用svg格式在Jupyter中显示绘图
        backend_inline.set_matplotlib_formats('svg')
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def sgd(params, lr, batch_size):
    """小批量随机梯度下降

    Defined in :numref:`sec_linear_scratch`"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器，读入array，返回DataLoader。
    """
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)


# 冻结模型中的某些层
def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze


def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)


def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)


def set_freeze_by_idxs(model, idxs, freeze=True):
    if not isinstance(idxs, Iterable):
        idxs = [idxs]
    num_child = len(list(model.children()))
    idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
    for idx, child in enumerate(model.children()):
        if idx not in idxs:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze


def freeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, True)


def unfreeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, False)