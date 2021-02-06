import torch
import numpy as np
from torch import nn
from torch.nn import init

from dl import load_data_fashion_mnist, sgd, softmax, load_data_fashion_mnist_small
from dl.model import FlattenLayer
from dl.train import train_fashion_mnist
from collections import OrderedDict

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
# train_iter, test_iter = load_data_fashion_mnist_small(batch_size)
num_inputs = 784
num_outputs = 10
num_epochs, lr = 5, 100.0

__all__ = ("softmax_zero", "softmax_simple")


def softmax_zero():
    def net(X):
        return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

    W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)),
                     dtype=torch.float, requires_grad=True)
    b = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)
    params = [W, b]
    for param in params:
        param.requires_grad_(requires_grad=True)
    optimizer = sgd
    loss = torch.nn.CrossEntropyLoss()
    train_fashion_mnist(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr, optimizer)


def softmax_simple():
    net = nn.Sequential(
        OrderedDict([
            ('flatten', FlattenLayer()),
            ('linear', nn.Linear(num_inputs, num_outputs))
        ])
    )
    init.normal_(net.linear.weight, mean=0, std=0.01)
    init.constant_(net.linear.bias, val=0)
    loss = nn.CrossEntropyLoss()
    # PyTorch提供了一个包括softmax运算和交叉熵损失计算的函数。它的数值稳定性更好
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    train_fashion_mnist(net, train_iter, test_iter, loss, num_epochs, batch_size, optimizer=optimizer)


if __name__ == '__main__':
    # softmax_zero()
    softmax_simple()
    pass
