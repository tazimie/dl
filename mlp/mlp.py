import torch
import numpy as np

from d2lzh_pytorch import load_data_fashion_mnist, sgd
from d2lzh_pytorch.optimizer import evaluate_accuracy

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)
num_epochs, lr = 5, 100.0

params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)


def net(X):
    X = X.view((-1, num_inputs))
    from d2lzh_pytorch.model import relu
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2


loss = torch.nn.CrossEntropyLoss()


def mlp_zero():
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            # 梯度清零
            params = [W1, b1, W2, b2]
            for param in params:
                param.grad.data.zero_()
            l.backward()
            sgd(params, lr, batch_size)
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


if __name__ == '__main__':
    mlp_zero()
