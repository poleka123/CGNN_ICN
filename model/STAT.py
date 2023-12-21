import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from model.common_layer import *

class AttentionLayer(nn.Module):
    """
    encoderlayer
    Represents one layer of the Transformer

    """

    def __init__(self, flag, hidden_size, num_heads, layer_dropout=0.0, relu_dropout=0.0):
        super(AttentionLayer, self).__init__()

        if flag == 'T':
            self.multi_head_attention = TemporalTransformerModel(num_heads, hidden_size // num_heads)
        if flag == 'S':
            self.multi_head_attention = SpatialTransformerModel(num_heads, hidden_size // num_heads)

        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, hidden_size,
                                                                 layer_config='ll', padding='both',
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs, STE, mask):
        x = inputs
        # Layer Normalization
        x_norm = self.layer_norm_mha(x)
        # Multi-head attention
        y, enc_self_attns = self.multi_head_attention(x_norm, x_norm, x_norm, STE, mask)
        # Dropout and residual
        x = self.dropout(x + y)
        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)
        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)
        # Dropout and residual
        y = self.dropout(x + y)
        return y, enc_self_attns

class DecoderLayer(nn.Module):
    def __init__(self, flag, hidden_size, num_heads, layer_dropout=0.0, relu_dropout=0.0):
        super(DecoderLayer, self).__init__()
        if flag == 'T':
            self.multi_head_attention1 = TemporalTransformerModel(num_heads, hidden_size // num_heads)
        if flag == 'S':
            self.multi_head_attention1 = SpatialTransformerModel(num_heads, hidden_size // num_heads)

        if flag == 'T':
            self.multi_head_attention2 = TemporalTransformerModel(num_heads, hidden_size // num_heads)
        if flag == 'S':
            self.multi_head_attention2 = SpatialTransformerModel(num_heads, hidden_size // num_heads)


        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, hidden_size,
                                                                 layer_config='ll', padding='both',
                                                                 dropout=relu_dropout)

        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs, enc_inputs, STE, mask):
        x = inputs
        # Layer Normalization
        x_norm = self.layer_norm_mha(x)
        # Multi-head attention1
        y1, dec_self_attns = self.multi_head_attention1(x_norm, x_norm, x_norm, STE, mask)
        # Dropout and residual
        x = self.dropout(x + y1)
        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)
        # Multi-head attention2
        y2, dec_enc_attns = self.multi_head_attention2(x_norm, enc_inputs, enc_inputs, STE, mask)
        x = self.dropout(x + y2)
        x_norm = self.layer_norm_ffn(x)
        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)
        # Dropout and residual
        y = self.dropout(x + y)
        return y, dec_self_attns, dec_enc_attns


# 时间特征与空间特征嵌入
class STEmbModel(nn.Module):
    def __init__(self, SEDims, TEDims, OutDims, device):
        super(STEmbModel, self).__init__()
        self.TEDims = TEDims
        self.FC_se1 = torch.nn.Linear(SEDims, OutDims)
        self.FC_se2 = torch.nn.Linear(OutDims, OutDims)
        self.FC_te1 = torch.nn.Linear(TEDims, OutDims)
        self.FC_te2 = torch.nn.Linear(OutDims, OutDims)
        self.device = device


    def forward(self, SE, TE):
        SE = SE.unsqueeze(0).unsqueeze(0)
        SE = self.FC_se2(F.relu(self.FC_se1(SE)))
        dayofweek = F.one_hot(TE[..., 0], num_classes = 7)
        timeofday = F.one_hot(TE[..., 1], num_classes = self.TEDims-7)
        TE = torch.cat((dayofweek, timeofday), dim=-1)
        TE = TE.unsqueeze(2).type(torch.FloatTensor).to(self.device)
        TE = self.FC_te2(F.relu(self.FC_te1(TE)))
        sum_tensor = torch.add(SE, TE)
        return sum_tensor


class EntangleModel(torch.nn.Module):
    def __init__(self, K, d):
        super(EntangleModel, self).__init__()
        D = K * d
        self.FC_xs = torch.nn.Linear(D, D)
        self.FC_xt = torch.nn.Linear(D, D)
        self.FC_h1 = torch.nn.Linear(D, D)
        self.FC_h2 = torch.nn.Linear(D, D)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, HS, HT):
        XS = self.FC_xs(HS)
        XT = self.FC_xt(HT)
        z = self.sigmoid(torch.add(XS, XT))
        H = torch.add((z * HS), ((1 - z) * HT))
        H = self.FC_h2(F.relu(self.FC_h1(H)))
        return H

class STATModel(nn.Module):
    """
    one layer encoder
    """
    def __init__(self, K, d, hidden_size, device):
        super(STATModel, self).__init__()
        self.spatialTransformer = AttentionLayer('S', hidden_size, K, layer_dropout=0.0, relu_dropout=0.0)

        self.temporalTransformer = AttentionLayer('T', hidden_size, K, layer_dropout=0.0, relu_dropout=0.0)

        self.entangel = EntangleModel(K, d)

    def forward(self, X, STE, mask):
        HS, SAtt = self.spatialTransformer(X, STE, mask)
        HT, TAtt = self.temporalTransformer(X, STE, mask)
        H = self.entangel(HS, HT)
        return torch.add(X, H), SAtt, TAtt

class STATDecModel(nn.Module):
    def __init__(self, K, d, hidden_size, device):
        super(STATDecModel, self).__init__()
        self.device = device
        # self.spatialTransformer = DecoderLayer('S', hidden_size, K, layer_dropout=0.0, relu_dropout=0.0)

        self.temporalTransformer = DecoderLayer('T', hidden_size, K, layer_dropout=0.0, relu_dropout=0.0)

        self.entangel = EntangleModel(K, d)

    def forward(self, X, enc_outputs, STE, mask):
        # HS, SAtt1, SAtt2 = self.spatialTransformer(X, enc_outputs, STE, mask)
        HT, TAtt1, TAtt2 = self.temporalTransformer(X, enc_outputs, STE, mask)
        H = self.entangel(HT, HT)
        return torch.add(X, H)



class BISTAT(torch.nn.Module):
    def __init__(self, features, K, d, SEDims, TEDims, P, F, H, L, hidden_size, device):
        super(BISTAT, self).__init__()
        D = K * d
        self.P = P
        self.F = F
        self.H = H
        self.L = L
        self.FC_1 = torch.nn.Linear(features, D)
        self.FC_2 = torch.nn.Linear(D, D)
        self.STEmb = STEmbModel(SEDims, TEDims, K * d, device)
        self.STATBlockEnc = torch.nn.ModuleList(
            [STATModel(K, d, hidden_size, device) for _ in range(self.L)])
        self.STATBlockDec1 = torch.nn.ModuleList(
            [STATDecModel(K, d, hidden_size, device) for _ in range(self.L)])

        self.FC_dec1_1 = torch.nn.Linear(D, D)
        self.FC_dec1_2 = torch.nn.Linear(D, 1)

    def forward(self, enc_input, dec_input, SE, TE):
        # input
        enc_input = enc_input.unsqueeze(3)
        enc_input = self.FC_2(F.relu(self.FC_1(enc_input))) 

        dec_input = dec_input.unsqueeze(3)
        dec_input = self.FC_2(F.relu(self.FC_1(dec_input)))

        # STE for the Historical, Present and Future condition
        STE = self.STEmb(SE, TE)
        STE_H = STE[:, : self.H]
        STE_P = STE[:, self.H: self.H + self.P]
        STE_F = STE[:, self.H + self.P: self.H + self.P + self.F]

        # output from the last layers in the encoder, which is used for the future-present cross-attention
        for l in range(0, len(self.STATBlockEnc)):
            enc_input, _, _ = self.STATBlockEnc[l](enc_input, STE_P, mask=False)
        X_enc_out = enc_input

        for net in self.STATBlockDec1:
            X_dec1_out = net(dec_input, X_enc_out, STE_F, mask=True)

        X_dec1_out = self.FC_dec1_2(F.relu(self.FC_dec1_1(X_dec1_out))).squeeze()

        return X_dec1_out