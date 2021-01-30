from torch import nn

from d2lzh_pytorch import data_iter, linreg, squared_loss, sgd

import torch
import numpy as np
__all__=['linear_regression_zero','linear_regression_simple']

true_w = [2, -3.4]
true_b = 4.2
# data
num_inputs = 2
num_examples = 1000

w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32, requires_grad=True)
b = torch.zeros(1, dtype=torch.float32, requires_grad=True)

features = torch.randn(num_examples, num_inputs,
                       dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),
                       dtype=torch.float32)
lr = 0.03
num_epochs = 3
batch_size = 10


def linear_regression_zero():
    net = linreg
    loss = squared_loss

    for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
        # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
        # 和y分别是小批量样本的特征和标签
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
            l.backward()  # 小批量的损失对模型参数求梯度
            sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数
            # 不要忘了梯度清零
            w.grad.data.zero_()
            b.grad.data.zero_()
        train_l = loss(net(features, w, b), labels)
        print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))


def linear_regression_simple():
    import torch.utils.data as Data
    dataset = Data.TensorDataset(features,labels)
    data_iter = Data.DataLoader(dataset,batch_size,shuffle=True)

    net = nn.Sequential(nn.Linear(num_inputs, 1))

    from torch.nn import init
    init.normal_(net[0].weight, mean=0, std=0.01)
    init.constant_(net[0].bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)

    loss = nn.MSELoss()
    import torch.optim as optim
    optimizer = optim.SGD(net.parameters(), lr=0.03)
    for epoch in range(num_epochs):
        for X, y in data_iter:
            output = net(X)
            l = loss(output, y.view(-1, 1))
            optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
            l.backward()
            optimizer.step()
        print('epoch %d, loss: %f' % (epoch, l.item()))





if __name__ == '__main__':
    linear_regression_simple()









