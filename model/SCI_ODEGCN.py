import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from model.odegcn import ODEG

# TCN
class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()

class TemporalConvNet(nn.Module):
    """
    time dilation convolution
    """

    def __init__(self, graph, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        Args:
            num_inputs : channel's number of input data's feature
            num_channels : numbers of data feature tranform channels, the last is the output channel
            kernel_size : using 1d convolution, so the real kernel is (1, kernel_size)
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=(1, dilation_size),
                                  padding=(0, padding))
            self.conv.weight.data.normal_(0, 0.01)
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]

        self.network = nn.Sequential(*layers)
        self.downsample = nn.Conv2d(num_inputs, num_channels[-1], (1, 1)) if num_inputs != num_channels[-1] else None
        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        like ResNet
        Args:
            X : input data of shape (B, N, T, F)
        """
        # permute shape to (B, F, N, T)
        y = x.permute(0, 3, 1, 2)
        y = F.relu(self.network(y) + self.downsample(y) if self.downsample else y)
        y = y.permute(0, 2, 3, 1)
        return y


class GCN(nn.Module):
    def __init__(self, A_hat, in_channels, out_channels, ):
        super(GCN, self).__init__()
        self.A_hat = A_hat
        self.theta = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.reset()

    def reset(self):
        stdv = 1. / math.sqrt(self.theta.shape[1])
        self.theta.data.uniform_(-stdv, stdv)

    def forward(self, X):
        y = torch.einsum('ij, kjlm-> kilm', self.A_hat, X)
        return F.relu(torch.einsum('kjlm, mn->kjln', y, self.theta))

# ICN
class Splitting(nn.Module):
    def __init__(self):
        super(Splitting, self).__init__()

    def even(self, x):
        return x[:, :, :, ::2]

    def odd(self, x):
        return x[:, :, :, 1::2]

    def forward(self, x):
        '''Returns the odd and even part'''
        return (self.even(x), self.odd(x))


class Interactor(nn.Module):
    def __init__(self,
                 num_input, num_channel, splitting=True, kernel=5, dropout=0.5,  dilation=1):
        super(Interactor, self).__init__()
        self.kernel_size = kernel
        self.dilation = 1
        self.dropout = dropout
        self.hidden_size = num_channel

        # 设置步长为1
        pad_l = self.dilation * (self.kernel_size-1)
        pad_r = self.dilation * (self.kernel_size-1)

        self.splitting = splitting
        self.split = Splitting()

        modules_P = []
        modules_U = []
        modules_psi = []
        modules_phi = []
        prev_size = 1

        modules_P += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(num_channel, num_channel,
                      kernel_size=(1, self.kernel_size), dilation=self.dilation, stride=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),

            nn.Dropout(self.dropout),
            nn.Conv2d(num_channel, num_channel,
                      kernel_size=(1, self.kernel_size), dilation=self.dilation, stride=1),
            nn.Tanh()
        ]
        modules_U += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(num_channel, num_channel,
                      kernel_size=(1, self.kernel_size), dilation=self.dilation, stride=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(num_channel, num_channel,
                      kernel_size=(1, self.kernel_size), dilation=self.dilation, stride=1),
            nn.Tanh()
        ]

        modules_phi += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(num_input, num_channel,
                      kernel_size=(1, self.kernel_size), dilation=self.dilation, stride=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(num_channel, num_channel,
                      kernel_size=(1, self.kernel_size), dilation=self.dilation, stride=1),
            nn.Tanh()
        ]
        modules_psi += [
            nn.ReplicationPad2d((pad_l, pad_r, 0, 0)),
            nn.Conv2d(num_input, num_channel,
                      kernel_size=(1, self.kernel_size), dilation=self.dilation, stride=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(self.dropout),
            nn.Conv2d(num_channel, num_channel,
                      kernel_size=(1, self.kernel_size), dilation=self.dilation, stride=1),
            nn.Tanh()
        ]
        self.phi = nn.Sequential(*modules_phi)
        self.psi = nn.Sequential(*modules_psi)
        self.P = nn.Sequential(*modules_P)
        self.U = nn.Sequential(*modules_U)

    def forward(self, x):
        if self.splitting:
            (x_even, x_odd) = self.split(x)
        else:
            (x_even, x_odd) = x

        d = x_odd.mul(torch.exp(self.phi(x_even)))
        c = x_even.mul(torch.exp(self.psi(x_odd)))

        x_even_update = c + self.U(d)
        x_odd_update = d - self.P(c)

        return (x_even_update, x_odd_update)

#用一层然后拼接
class InteractorandConcat(nn.Module):
    def __init__(self, num_input, num_channel, kernel, dropout, dilation):
        super(InteractorandConcat, self).__init__()
        self.level = Interactor(num_input=num_input, num_channel=num_channel, splitting=True, kernel=kernel, dropout=dropout, dilation=dilation)
        self.relu = nn.ReLU()
        # 下采样
        self.downsample = nn.Conv2d(num_input, num_channel, (1, 1))

    def zip_up_the_pants(self, even, odd):
        # even odd shape is (B, C, N, T)
        even_len = even.shape[3]
        odd_len = odd.shape[3]
        mlen = min((odd_len, even_len))
        _ = []
        for i in range(mlen):
            _.append(even[:, :, :, i].unsqueeze(3))
            _.append(odd[:, :, :, i].unsqueeze(3))
        if odd_len < even_len:
            _.append(even[-1].unsqueeze(3))
        return torch.cat(_, 3)  # B, L, D

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        res = x if self.downsample is None else self.downsample(x)
        (x_even_update, x_odd_update) = self.level(x)
        x_concat = self.zip_up_the_pants(x_even_update, x_odd_update)
        return self.relu((x_concat + res).permute(0, 2, 3, 1))





class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, num_nodes, A_hat, type):
        """
        Args:
            in_channels: Number of input features at each node in each time step.
            out_channels: a list of feature channels in timeblock, the last is output feature channel
            num_nodes: Number of nodes in the graph
            A_hat: the normalized adjacency matrix
        """
        super(STGCNBlock, self).__init__()
        self.A_hat = A_hat

        # self.temporal1 = TemporalConvNet(A_hat, num_inputs=in_channels,
        #                                  num_channels=out_channels)
        #
        # self.temporal2 = TemporalConvNet(A_hat, num_inputs=out_channels[-1],
        #                                  num_channels=out_channels)
        self.temporal1 = InteractorandConcat(num_input=in_channels, num_channel=out_channels,
                                             kernel=kernel, dropout=0.05, dilation=1)

        self.temporal2 = InteractorandConcat(num_input=out_channels, num_channel=out_channels,
                                             kernel=kernel, dropout=0.05, dilation=1)

        self.odeg1 = ODEG(num_nodes, type, out_channels, 12, A_hat, time=6)
        self.batch_norm = nn.BatchNorm2d(num_nodes)


    def forward(self, X):
        """
        Args:
            X: Input data of shape (batch_size, num_nodes, num_timesteps, num_features)
        Return:
            Output data of shape(batch_size, num_nodes, num_timesteps, out_channels[-1])
        """

        b, n, t, c = X.shape
        X = self.temporal1(X)
        X = self.odeg1(X)
        X = self.temporal2(X)


        return self.batch_norm(X)




class ODEGCN(nn.Module):
    """ the overall network framework """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, A_sp_hat, A_se_hat):
        """
        Args:
            num_nodes : number of nodes in the graph
            num_features : number of features at each node in each time step
            num_timesteps_input : number of past time steps fed into the network
            num_timesteps_output : desired number of future time steps output by the network
            A_sp_hat : nomarlized adjacency spatial matrix
            A_se_hat : nomarlized adjacency semantic matrix
        """
        # self.graph=
        super(ODEGCN, self).__init__()
        # adjacency graph branch
        self.sp_blocks = nn.ModuleList(
            [nn.Sequential(
                STGCNBlock(in_channels=num_features, out_channels=32, kernel=3,
                           num_nodes=num_nodes, A_hat=A_sp_hat, type='sp'),
                STGCNBlock(in_channels=32, out_channels=32, kernel=3,
                           num_nodes=num_nodes, A_hat=A_sp_hat, type='sp')) for _ in range(1)
            ])
        # dtw graph branch
        self.se_blocks = nn.ModuleList([nn.Sequential(
            STGCNBlock(in_channels=num_features, out_channels=32, kernel=3,
                       num_nodes=num_nodes, A_hat=A_se_hat, type='se'),
            STGCNBlock(in_channels=32, out_channels=32, kernel=3,
                       num_nodes=num_nodes, A_hat=A_se_hat, type='se')) for _ in range(1)
        ])
        #Attention Module 6 = 3*2
        self.pred = MultiHeadSelfAttention(12 * 32 * 2, 6 * 32, 3)

    def forward(self, x):
        """
        Args:
            x : input data of shape (batch_size, num_nodes, num_timesteps, num_features) == (B, N, T, F)
        Returns:
            prediction for future of shape (batch_size, num_nodes, num_timesteps_output)
        """
        outs = []
        # spatial graph
        for blk in self.sp_blocks:
            outs.append(blk(x))
        # semantic graph
        for blk in self.se_blocks:
            outs.append(blk(x))

        outs = torch.stack(outs, dim=-1)
        b, n, t, c, s = outs.shape
        x = outs.reshape(b, n, t, c * s)
        x = x.reshape((x.shape[0], x.shape[1], -1))


        return self.pred(x)


class MultiHeadSelfAttention(nn.Module):
    # dim_in: int  # input dimension
    # dim_k: int   # key and query dimension
    # dim_v: int   # value dimension
    # num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim_in, dim_k, dim_v, num_heads=3):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=True)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=True)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=True)
        self._norm_fact = 1 / math.sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n
        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        return att
