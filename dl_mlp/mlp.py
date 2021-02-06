import torch
import numpy as np
from torch import nn
from torch.nn import init

from dl import load_data_fashion_mnist, sgd
from dl.model import FlattenLayer
from dl.train import train_fashion_mnist

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
num_inputs, num_outputs, num_hiddens = 784, 10, 256
num_epochs, lr = 5, 100

__all__ = ("mlp_zero", "mlp_simple")


def mlp_zero():
    def net(X):
        X = X.view((-1, num_inputs))
        from dl.model import relu
        H = relu(torch.matmul(X, W1) + b1)
        return torch.matmul(H, W2) + b2

    W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
    b1 = torch.zeros(num_hiddens, dtype=torch.float)
    W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
    b2 = torch.zeros(num_outputs, dtype=torch.float)
    params = [W1, b1, W2, b2]
    for param in params:
        param.requires_grad_(requires_grad=True)
    optimizer = sgd
    loss = torch.nn.CrossEntropyLoss()
    train_fashion_mnist(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr, optimizer)


def mlp_simple():
    net = nn.Sequential(
        FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs),
    )

    for params in net.parameters():
        init.normal_(params, mean=0, std=0.01)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    train_fashion_mnist(net, train_iter, test_iter, loss, num_epochs, batch_size, optimizer=optimizer)


# 本函数已保存在d2lzh包中方便以后使用


if __name__ == '__main__':
    # mlp_zero()
    mlp_simple()
