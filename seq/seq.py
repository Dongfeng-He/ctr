import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import math


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, kernel_size), stride=stride, padding=0, dilation=dilation))
        self.pad = torch.nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, (1, kernel_size), stride=stride, padding=0, dilation=dilation))
        self.net = nn.Sequential(self.pad, self.conv1, self.relu, self.dropout, self.pad, self.conv2, self.relu, self.dropout)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x.unsqueeze(2)).squeeze(2)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    def __init__(self, input_size, num_inputs=64, num_channels=(32, 32, 32, 32), kernel_size=2, dropout=0.2, pool="max"):
        # num_inputs: 输入特征维度, num_channels: 每层输出特征维度
        super(TCNModel, self).__init__()
        self.embedding = nn.Embedding(input_size, num_inputs)
        self.pool = pool
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=torch.long)
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.network(x)
        if self.pool == "max":
            x = torch.max(x, 2)[0]
        elif self.pool == "avg":
            x = torch.mean(x, 2)
        elif self.pool == "both":
            x = torch.cat([torch.max(x, 2)[0],  torch.mean(x, 2)], 1)
        else:
            x = x[:, :, -1]
        return x


class LSTMModel(nn.Module):
    def __init__(self, input_size, num_inputs=64, hidden_size=32, pool="max"):
        # num_inputs: 输入特征维度, num_channels: 每层输出特征维度
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_size, num_inputs)
        self.pool = pool
        self.lstm = nn.LSTM(input_size=num_inputs,
                            hidden_size=hidden_size,
                            num_layers=1,
                            bias=True,
                            batch_first=True,
                            dropout=0,
                            bidirectional=True)

    def forward(self, x):
        # x = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=torch.long)
        x = self.embedding(x)
        x = self.lstm(x)[0]
        if self.pool == "max":
            x = torch.max(x, 1)[0]
        elif self.pool == "avg":
            x = torch.mean(x, 1)
        elif self.pool == "both":
            x = torch.cat([torch.max(x, 1)[0],  torch.mean(x, 1)], 1)
        else:
            x = x[:, -1, :]
        return x


class AVGModel(nn.Module):
    def __init__(self, input_size, num_inputs=64):
        super(AVGModel, self).__init__()
        self.embedding = nn.Embedding(input_size, num_inputs)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        return x


class AttentionModel(nn.Module):
    def __init__(self, vip_size, input_size, num_inputs=64, hidden_units=(64, 32), dropout=0.2, activation="tanh", weight_norm=True):
        super(AttentionModel, self).__init__()
        self.embedding = nn.Embedding(input_size, num_inputs)
        self.weight_norm = weight_norm
        if activation == "tanh":
            activation_layer = nn.Tanh()
        elif activation == "sigmoid":
            activation_layer = nn.Sigmoid()
        else:
            activation_layer = nn.ReLU()
        layers = []
        for i in range(len(hidden_units) + 1):
            if i == 0:
                layers += [nn.Linear(num_inputs + vip_size, hidden_units[0])]
                layers += [activation_layer]
                layers += [nn.Dropout(dropout)]
            elif i == len(hidden_units):
                layers += [nn.Linear(hidden_units[-1], 1)]
            else:
                layers += [nn.Linear(hidden_units[i-1], hidden_units[i])]
                layers += [activation_layer]
                layers += [nn.Dropout(dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, vip_emb, seq_input):
        # vip_emb: torch.tensor([[0.2, 0.2, 0.3, 0.4], [0.2, 0.2, 0.3, 0.4]], dtype=torch.float32)
        # seq_input: torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=torch.long)
        batch_size, seq_len = seq_input.shape
        vip_emb = vip_emb.unsqueeze(1).repeat(1, seq_len, 1)
        seq_emb = self.embedding(seq_input)
        cat_emb = torch.cat([vip_emb, seq_emb], 2)
        att_score = self.network(cat_emb)
        att_score = att_score.permute(0, 2, 1)
        if self.weight_norm:
            att_score = nn.Softmax(dim=2)(att_score)
        output = torch.matmul(att_score, seq_emb).squeeze()
        return output

