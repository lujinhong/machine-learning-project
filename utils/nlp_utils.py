# -*- coding: utf-8 -*- 

"""
    AUTHOR: lujinhong
CREATED ON: 2022年05月30日 10:51
   PROJECT: machine-learning-project
   DESCRIPTION: 文本处理中的常用功能。
"""
import collections
import re,os
import requests
import random
import torch
from torch import nn
from matplotlib import pyplot as plt


from utils.constants import DATASET_URL,dataset_root
from utils import my_utils


def download_dataset(url, download_dir):
    """从网络中把文件下载到本地，返回本地文件的路径。
    若数据集已经存在则不再重复下载。
    """
    os.makedirs(download_dir, exist_ok=True)
    file_name = os.path.join(download_dir + url.split('/')[-1])
    if os.path.exists(file_name):
        print(f'file {file_name} already exist, skip download.')
        return file_name
    print(f'Downloading {file_name} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(file_name, 'wb') as f:
        f.write(r.content)
    return file_name




def load_dataset(file):
    """
    读取文件中的内容，做预处理后返回行组成的列表。
    :param file: 带读取的文件名称。
    :return: line组成的列表
    """
    with open(file, 'r') as f:
        lines = f.readlines()
    # 只保留英文字母，并全部转为小写
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def tokenize(lines, token='word'):
    """
    将lines的每一行拆分为单词，返回列表
    :param lines:
    :param token:
    :return:
    """
    if token == 'word':
        line_tokens = [line.split() for line in lines]
    elif token == 'char':
        line_tokens = [list(line) for line in lines]
    else:
        print(f'error token {token}')
        return -1
    return [token for line in line_tokens for token in line]


def count_corpus(tokens):
    """
    统计每个token出现的频次
    :param tokens:
    :return:
    """
    print(type(tokens))
    return collections.Counter(tokens)


class Vocab:
    """
    记录所有token，并按频次排序
    """
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """通过文本查找对应的索引。可以直接调用vocal['word']"""
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """通过索引查找对饮的文本"""
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def load_corpus_time_machine(max_tokens=-1):  #@save
    """整合上述功能，返回时光机器数据集的词元索引列表和词表"""
    file_name = download_dataset(DATASET_URL['time_machine'], os.path.join(dataset_root,'time_machine/'))
    line_list = load_dataset(file_name)
    tokens = tokenize(line_list, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


# corpus, vocab = load_corpus_time_machine()
# print(len(corpus), len(vocab))

# # deomo for load_dataset()
# file_name = download_dataset(DATASET_URL['time_machine'], os.path.join(dataset_root,'time_machine/'))
# line_list = load_dataset(file_name)
# print(len(line_list))
# for i in range(10):
#     print(line_list[i])
# tokens = tokenize(line_list)
# print(tokens[:10])
# print(count_corpus(tokens))
# vocab = Vocab(tokens)
# print(list(vocab.token_to_idx.items())[:10])
# for i in range(10):
#     print('文本:', tokens[i])
#     print('索引:', vocab[tokens[i]])


def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y


class SeqDataLoader:
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def preprocess_nmt(text):
    """预处理机器翻译数据集"""
    # 只保留数据集中的第一第二列
    sub_text = ''
    for line in text:
        sub_text += line.split('\t')[0] + '\t' + line.split('\t')[1] + '\n'
    # 使用一个空格代替连续的多个空格
    # 字母改为小写
    sub_text = sub_text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and (char in set(',.!?') and sub_text[i-1] != ' ') else char
           for i,char in enumerate(sub_text)]
    return ''.join(out)


def tokenize_nmt(text):
    """词元化：将句子拆分为词元"""
    source, target = [], []
    for line in text.split('\n'):
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def truncate_pad(line, num_step, padding_token):
    """将文本截断或者填充至固定长度"""
    if len(line) > num_step:
        return line[:num_step]
    return line + [padding_token] * (num_step-len(line))


def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


# 下载英法翻译数据集
machine_translation_dataset_url = DATASET_URL['machine_translation_fra-eng']
machine_translation_dataset_dir = os.path.join(dataset_root, 'machine_translation_fra-eng/')
def load_data_nmt(batch_size, num_steps):
    # 下载文本数据集
    download_dataset(machine_translation_dataset_url, machine_translation_dataset_dir)
    with open(os.path.join(machine_translation_dataset_dir, 'fra.txt')) as f:
        origin_text = f.readlines()
    text = preprocess_nmt(origin_text)
    # 词元化
    source, target = tokenize_nmt(text)
    # source,target是嵌套list，将其展开，用于计算词元
    source_expand = [i for k in source for i in k]
    source_vocab = Vocab(source_expand, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    target_expand = [i for k in target for i in k]
    target_vocab = Vocab(target_expand, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # 转成小批量数据集
    source_array, source_valid_len = build_array_nmt(source, source_vocab, num_steps)
    target_array, target_valid_len = build_array_nmt(target, target_vocab, num_steps)
    data_arrays = (source_array, source_valid_len, target_array, target_valid_len)
    # 生成dataloader，用于后续训练
    dataloader = my_utils.load_array(data_arrays, batch_size)
    return dataloader, source_vocab, target_vocab


def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """显示矩阵热图

    Defined in :numref:`sec_attention-cues`"""
    my_utils.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);