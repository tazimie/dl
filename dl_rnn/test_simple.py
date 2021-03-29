import os
import time

import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import dl as d2l


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size, hidden_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state):  # inputs: (batch, seq_len)
        X = F.one_hot(inputs.to(torch.int64), num_classes=self.vocab_size).float()
        Y, self.state = self.rnn(X, state)
        output = self.dense(Y.view(-1, self.hidden_size))
        return output, self.state


def predict_rnn_pytorch(prefix, num_chars, model, idx_to_char, char_to_idx):
    output = [char_to_idx[prefix[0]]]  # output会记录prefix加上输出
    state = None
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]]).view(1, 1)
        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


if __name__ == '__main__':
    (corpus_indices, char_to_idx, idx_to_char, vocab_size) = d2l.load_data_jay_lyrics()
    num_hiddens = 256
    num_steps = 35
    rnn = nn.RNN(input_size=vocab_size, hidden_size=256)
    num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e-3, 1e-2  # 注意这里的学习率设置
    pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
    model = RNNModel(rnn, vocab_size, num_hiddens)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    state = None
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = d2l.data_iter_consecutive(corpus_indices, batch_size, num_steps, 'cpu')  # 相邻采样
        for X, Y in data_iter:
            if state is not None:
                state = state.detach()
            (output, state) = model(X, state)  # output: 形状为(num_steps * batch_size, vocab_size)
            y = torch.transpose(Y, 0, 1).contiguous().view(-1)
            l = loss(output, y.long())
            optimizer.zero_grad()
            l.backward()
            d2l.grad_clipping(model.parameters(), clipping_theta, 'cpu')
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            perplexity = float('inf')
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, perplexity, time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_pytorch(
                    prefix, pred_len, model, idx_to_char,
                    char_to_idx))