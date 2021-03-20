import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torchvision
import os


class Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, nonlinearity='relu')

    def forward(self, x):
        return self.rnn(x)


path = os.path.join('~', 'Datasets', 'MNIST')
path = os.path.expanduser(path)
transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.1307,), (0.3081,))
    ]
)
train = torchvision.datasets.MNIST(path, train=True, transform=transforms, download=True)
batch_size = 32
input_size = 28 * 28
hidden_size = 10

data_loader = DataLoader(train, shuffle=False, batch_size=batch_size)

net = Model(input_size, hidden_size)

loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())
if __name__ == '__main__':
    for x, y in data_loader:
        optimizer.zero_grad()
        y_hat, _ = net(x.view(batch_size, -1, input_size))
        y_hat = y_hat.view(batch_size, -1)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
        with torch.no_grad():
            print((y_hat.max(dim=1).indices == y).sum()/32)
        print(l.item())
