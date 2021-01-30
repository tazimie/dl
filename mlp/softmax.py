import torch
import numpy as np
from torch import nn
from torch.nn import init

from d2lzh_pytorch import load_data_fashion_mnist, cross_entropy, sgd, softmax
from d2lzh_pytorch.model import FlattenLayer
from d2lzh_pytorch.optimizer import evaluate_accuracy

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float, requires_grad=True)
b = torch.zeros(num_outputs, dtype=torch.float, requires_grad=True)
num_epochs, lr = 5, 0.1





def softmax_zero():
    def net(X):
        return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            print(y_hat)
            l = cross_entropy(y_hat, y).sum()
            # 梯度清零
            W.grad.data.zero_()
            b.grad.data.zero_()
            l.backward()
            sgd([W, b], lr, batch_size)
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


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
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    num_epochs, lr = 5, 0.1

    # 本函数已保存在d2lzh包中方便以后使用
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for x, y in train_iter:
            y_hat = net(x)
            l = loss(y_hat, y).sum()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()  # “softmax回归的简洁实现”一节将用到
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

    pass


from collections import OrderedDict

if __name__ == '__main__':
    softmax_zero()
    # softmax_simple()
    pass
