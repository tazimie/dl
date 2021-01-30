"""The model module contains neural network building blocks"""
from abc import abstractmethod

import torch
import torch.nn as nn

__all__ = ['linreg', 'LinearNet', 'softmax']


def linreg(X, w, b):
    """Linear regression."""
    return torch.mm(X, w) + b


def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制


def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))


class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

    # forward 定义前向传播
    def forward(self, x):
        y = self.linear(x)
        return y


# 本函数已保存在d2lzh_pytorch包中方便以后使用
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)
