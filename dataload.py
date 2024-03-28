import torch
import torchvision
from torch.utils import data

import pickle
import numpy as np
from scipy.io import loadmat


def load_minst(batch_size, flatten=False, rnn=False):
    datas = loadmat('data/MNISTData.mat')
    device = torch.device("cuda:0")

    X_train = torch.tensor(datas['X_Train'], dtype=torch.float32, device=device)
    X_test = torch.tensor(datas['X_Test'], dtype=torch.float32, device=device)
    y_train = torch.tensor(datas['D_Train'], dtype=torch.float32, device=device)
    y_test = torch.tensor(datas['D_Test'], dtype=torch.float32, device=device)

    X_train = X_train.permute(2, 0, 1)  # (60000, 28, 28)
    X_test = X_test.permute(2, 0, 1)
    y_train = y_train.T.argmax(axis=1)  # (60000,)
    y_test = y_test.T.argmax(axis=1)

    if not rnn:
        if flatten:
            X_train = X_train.reshape(X_train.shape[0], -1)  # (60000, 784)
            X_test = X_test.reshape(X_test.shape[0], -1)
        else:
            X_train = torch.unsqueeze(X_train, dim=1)  # (60000, 1, 28, 28)
            X_test = torch.unsqueeze(X_test, dim=1)

    train_set = data.TensorDataset(X_train, y_train)
    train_db, val_db = data.random_split(train_set, [54000, 6000])

    train_iter = data.DataLoader(train_db, batch_size, shuffle=True)
    val_iter = data.DataLoader(val_db, batch_size)
    test_iter = data.DataLoader(data.TensorDataset(X_test, y_test), batch_size)
    return train_iter, val_iter, test_iter


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar_10(batch_size):
    # 依次加载batch_data_i,并合并到x,y
    x, y = [], []
    for i in range(1, 6):
        batch_path = f'data/cifar-10-batches-py/data_batch_{i}'
        batch_dict = unpickle(batch_path)
        train_batch = batch_dict[b'data']
        train_label = np.array(batch_dict[b'labels'])
        x.append(train_batch)
        y.append(train_label)
    # 将5个训练样本batch合并为50000x3x32x32，标签合并为50000x1
    train_data = np.concatenate(x).reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    train_labels = torch.tensor(np.concatenate(y))

    # 分割训练集和验证集
    val_num = int(0.1*train_labels.shape[0])
    s = np.arange(train_labels.shape[0])
    np.random.shuffle(s)
    val_data = train_data[s[:val_num]]
    val_labels = train_labels[s[:val_num]]
    train_data = train_data[s[val_num:]]
    train_labels = train_labels[s[val_num:]]

    # 创建测试样本
    test_dict = unpickle('data/cifar-10-batches-py/test_batch')
    test_data = test_dict[b'data'].reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = torch.tensor(np.array(test_dict[b'labels']))

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        # 在高度和宽度上将图像放大到40像素的正方形
        torchvision.transforms.Resize(40),
        # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
        # 生成一个面积为原始图像面积0.64～1倍的小正方形，
        # 然后将其缩放为高度和宽度均为32像素的正方形
        torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                 ratio=(1.0, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        # 标准化图像的每个通道
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                         [0.2023, 0.1994, 0.2010])])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                         [0.2023, 0.1994, 0.2010])])

    return data.DataLoader(CifarData(train_data, train_labels, transform_train), batch_size, shuffle=True), \
           data.DataLoader(CifarData(val_data, val_labels, transform_test), batch_size), \
           data.DataLoader(CifarData(test_data, test_labels, transform_test), batch_size)


class CifarData(data.Dataset):
    def __init__(self, dataset, labels, transform):
        self.device = torch.device("cuda:0")
        self.dataset = dataset
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx]).to(self.device), self.labels[idx].to(self.device)

    def __len__(self):
        return len(self.labels)


def load_minst_random(batch_size):
    datas = loadmat('data/MNISTData.mat')
    device = torch.device("cuda:0")

    X_train = torch.tensor(datas['X_Train'], dtype=torch.float32, device=device)
    y_train = datas['D_Train'].T.argmax(axis=1)
    y_true = torch.tensor(y_train, dtype=torch.int64, device=device)
    np.random.shuffle(y_train)
    y_random = torch.tensor(y_train, dtype=torch.int64, device=device)

    X_train = X_train.permute(2, 0, 1)  # (60000, 28, 28)
    X_train = torch.unsqueeze(X_train, dim=1)  # (60000, 1, 28, 28)

    random_iter = data.DataLoader(data.TensorDataset(X_train, y_random), batch_size, shuffle=True)
    true_iter = data.DataLoader(data.TensorDataset(X_train, y_true), batch_size, shuffle=True)
    return true_iter, random_iter
