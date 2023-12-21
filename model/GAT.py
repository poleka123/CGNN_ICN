import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        """
        :param in_features: 输入特征向量通道数
        :param out_features: 输出通道数
        :param dropout: dropout参数
        :param alpha: LeakRelu参数
        :param concat: 如果为True，则再进行elu激活
        """
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        #定义可训练参数， 论文中的w和a
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414) #xavier初始化
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        #定义leakyReLU激活函数
        self.leakrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        """
        :param x: 输入特征
        :param adj: 邻接矩阵
        :return:
        """
        Wh = torch.matmul(x, self.W)
        e = self._prepare_attentional_mechanism_input(Wh) #对应LeakyReLU(eij)计算公式
        zero_vec = -9e15*torch.ones_like(e) #将没有链接的边设置为负无穷
        attention = torch.where(adj > 0, e, zero_vec)  # [N,N]
        # 表示如果邻接矩阵元素大于0时，则两个节点有连接，该位置的注意力系数保留
        # 否则需要mask设置为非常小的值，因为softmax的时候这个最小值会不考虑
        attention = F.softmax(attention, dim=1) #softmax形状保持不变[N,N]，得到归一化的注意力权重
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh) #[N, N] * [N, out_features]=> [N, out_features]

        # 得到由周围节点通过注意力权重进行更新后的表示
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


    def _prepare_attentional_mechanism_input(self, Wh):
        """

        :param Wh:
        :return:
        """
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])

        #broadcast add
        # e = Wh1 + Wh2.T
        e = Wh1 + Wh2.permute(0, 2, 1)

        return self.leakrelu(e)

