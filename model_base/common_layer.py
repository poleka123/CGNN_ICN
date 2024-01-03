import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SpatialTransformerModel(torch.nn.Module):
    def __init__(self, K, d):
        super(SpatialTransformerModel, self).__init__()
        D = K*d
        self.FC_q = torch.nn.Linear(2*D, D)
        self.FC_k = torch.nn.Linear(2*D, D)
        self.FC_v = torch.nn.Linear(2*D, D)
        self.FC_o1 = torch.nn.Linear(D, D)
        self.FC_o2 = torch.nn.Linear(D, D)
        self.K = K
        self.d = d
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, Q, K, V, STE, mask):
        Q = torch.cat((Q, STE), dim=-1)
        K = torch.cat((K, STE), dim=-1)
        V = torch.cat((V, STE), dim=-1)
        query = F.relu(self.FC_q(Q))
        key = F.relu(self.FC_k(K))
        value = F.relu(self.FC_v(V))
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        attention = torch.matmul(query, torch.transpose(key, 2, 3))
        attention /= (self.d ** 0.5)
        attention = self.softmax(attention)
        X = torch.matmul(attention, value)
        X = torch.cat(torch.split(X, X.shape[0]//self.K, dim=0), dim=-1)
        X = self.FC_o2(F.relu(self.FC_o1(X)))
        return X, attention

class TemporalTransformerModel(torch.nn.Module):
    def __init__(self, K, d):
        super(TemporalTransformerModel, self).__init__()
        D = K*d
        self.FC_q = torch.nn.Linear(2*D, D)
        self.FC_k = torch.nn.Linear(2*D, D)
        self.FC_v = torch.nn.Linear(2*D, D)
        self.FC_o1 = torch.nn.Linear(D, D)
        self.FC_o2 = torch.nn.Linear(D, D)
        self.K = K
        self.d = d
        self.softmax = torch.nn.Softmax(dim=-1)


    def forward(self, Q, K, V, STE, mask):
        Q = torch.cat((Q, STE), dim=-1)
        K = torch.cat((K, STE), dim=-1)
        V = torch.cat((V, STE), dim=-1)
        query = F.relu(self.FC_q(Q))
        key = F.relu(self.FC_k(K))
        value = F.relu(self.FC_v(V))
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        query = torch.transpose(query, 2, 1)
        key = torch.transpose(torch.transpose(key, 1, 2), 2, 3)
        value = torch.transpose(value, 2, 1)
        attention = torch.matmul(query, key)
        attention /= (self.d ** 0.5)
        if mask:
            batch_size = Q.shape[0]
            num_step = Q.shape[1]
            num_vertex = Q.shape[2]
            mask = torch.ones(num_step, num_step).to('cuda')
            mask = torch.tril(mask)
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0)
            mask = mask.repeat(self.K * batch_size, num_vertex, 1, 1)
            mask = mask.to(torch.bool)
            attention = torch.where(mask, attention, -2 ** 15 + torch.tensor([1],dtype=torch.float32).to('cuda'))
        attention = self.softmax(attention)
        X = torch.matmul(attention, value)
        X = torch.transpose(X, 2, 1)
        X = torch.cat(torch.split(X, X.shape[0]//self.K, dim=0), dim=-1)
        X = self.FC_o2(F.relu(self.FC_o1(X)))
        return X, attention


class Conv(nn.Module):
    """
    Convenience class that does padding and convolution for inputs in the format

    """

    def __init__(self, input_size, output_size, kernel_size, pad_type):
        super(Conv, self).__init__()
        padding = (kernel_size - 1, 0) if pad_type == 'left' else (kernel_size // 2, (kernel_size - 1) // 2)
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, padding=0)

    def forward(self, inputs):
        inputs = self.pad(inputs.permute(0, 2, 3, 1))
        outputs = self.conv(inputs).permute(0, 2, 3, 1)

        return outputs


class PositionwiseFeedForward(nn.Module):
    """
    Does a Linear + RELU + Linear on each of the timesteps
    """

    def __init__(self, input_depth, filter_size, output_depth, layer_config='cc', padding='left', dropout=0.0):

        super(PositionwiseFeedForward, self).__init__()

        layers = []
        sizes = ([(input_depth, filter_size)] +
                 [(filter_size, filter_size)] * (len(layer_config) - 2) +
                 [(filter_size, output_depth)])

        for lc, s in zip(list(layer_config), sizes):
            if lc == 'l':
                layers.append(nn.Linear(*s))
            elif lc == 'c':
                layers.append(Conv(*s, kernel_size=3, pad_type=padding))
            else:
                raise ValueError("Unknown layer type {}".format(lc))

        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers):
                x = self.relu(x)
                x = self.dropout(x)

        return x


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def _gen_embedding(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(np.arange(num_timescales).astype(np.float64) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]],
                    'constant', constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)
