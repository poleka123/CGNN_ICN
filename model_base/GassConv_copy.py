import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F


class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class GaussBlack(nn.Module):
    def __init__(self, n_input, n_output, g_kernel_data, kernel_size, stride, dilation, padding, dropout=0.2):
        super(GaussBlack, self).__init__()
        self.g_data = g_kernel_data
        self.g_data = nn.Parameter(data=torch.FloatTensor(self.g_data).expand(n_input, n_input, self.g_data.shape[0], self.g_data.shape[1]),
            requires_grad=False
        )
        # 更新权重，进行学习
        self.conv1 = weight_norm(nn.Conv2d(
            n_input, n_output, (1, kernel_size), stride=stride, padding=padding, dilation=dilation))
        # f.conv2d() weight卷积核(Cout, Cin, H, W)
        # (1, x)对x进行操作
        self.chomp1 = Chomp2d(padding[1])
        self.relu1 = nn.Tanh()
        self.dropout1 = nn.Dropout(dropout)

        # 更新权重，进行学习
        self.conv2 = weight_norm(nn.Conv2d(
            n_output, n_output, (1, kernel_size), stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp2d(padding[1])
        self.relu2 = nn.Tanh()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                               self.conv2, self.chomp2, self.relu2, self.dropout2)
        # self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)

        # gass
        # self.downsample = nn.Conv2d(n_input, n_output, (1, kernel_size))
        self.downsample = nn.Conv2d(n_input, n_output, (1, 1))
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # g_data (out_channel, in_channel, H, W)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x shape (batch, channel, H, W), H:nodes, W:input_timesteps
        # Gauss Conv
        # out = nn.functional.conv2d(x, self.g_data, stride=1)
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)

class GaussConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, g_kernel, kernel_size=2, dropout=0.2):
        super(GaussConvNet, self).__init__()
        layer = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            # i = 0 第一层等于，输入；后面等于隐层的数量
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layer += [GaussBlack(in_channels, out_channels, g_kernel, kernel_size, stride=1, dilation=dilation_size,
                                 padding=(0, (kernel_size-1)*dilation_size), dropout=dropout)]
        self.network = nn.Sequential(*layer)

    def forward(self, x):
        return self.network(x)

class GTCN(nn.Module):
    def __init__(self, input_size, input_timesteps, output_size, num_channels, g_kernel, kernel_size, dropout):
        super(GTCN, self).__init__()
        self.GTCN = GaussConvNet(input_size, num_channels, g_kernel=g_kernel, kernel_size=kernel_size, dropout=dropout)
        # gass
        # self.linear = nn.Linear(num_channels[-1]*(input_timesteps-len(num_channels)*kernel_size+len(num_channels)), output_size)

        self.linear = nn.Linear(num_channels[-1] * input_timesteps, output_size)

    def forward(self, inputs):
        inputs = inputs.reshape(inputs.shape[0], inputs.shape[3], inputs.shape[1], inputs.shape[2])
        y1 = self.GTCN(inputs)
        y1 = y1.reshape(y1.shape[0], y1.shape[2], y1.shape[1]*y1.shape[3])
        out = self.linear(y1)
        return out



