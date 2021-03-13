import math
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import dl as d2l


def to_one_hot(x, num_classes):
    x = torch.LongTensor(x).to(torch.int64)
    return F.one_hot(x, num_classes=num_classes)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
(corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()

epochs, seq_len, batch_size, lr, clipping_theta, hidden_size = 25, 35, 10, 1e2, 1e-2, 256
input_size = 1027

model = nn.RNN(input_size=vocab_size, hidden_size=hidden_size)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

if __name__ == '__main__':
    for epoch in range(epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = d2l.data_iter_consecutive(corpus_indices, batch_size, num_steps)
        for X, Y in data_iter:
            x = to_one_hot(X, vocab_size)
