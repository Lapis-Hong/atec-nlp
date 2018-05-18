#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/5/15
from numpy.linalg import norm

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingCNN(nn.Module):
    def __init__(self, args):
        super(EmbeddingCNN, self).__init__()
        self.args = args

        self.embed = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, args.num_kernels, (ks, args.embedding_dim)) for ks in args.kernel_sizes])

    def forward(self, x):
        x = self.embed(x)  # (batch_size, sequence_length, embedding_dim)
        if self.args.word_embedding_type == 'static':
            x = torch.tensor(x)
        x = x.unsqueeze(1)  # (batch_size, 1, sequence_length, embedding_dim)
        # #  input size (N,Cin,H,W)  output size (N,Cout,Hout,1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        output = torch.cat(x, 1)  # (batch_size, len(kernel_sizes)*kernel_num)
        return output


class EmbeddingRNN(nn.Module):

    def __init__(self, args):
        super(EmbeddingRNN, self).__init__()
        self.hidden_units = args.hidden_units
        self.batch_size = args.batch_size

        self.embeds = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.lstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_units, num_layers=args.num_layers,
                            batch_first=True, bidirectional=True)
    #     self.hidden = self.init_hidden()
    #
    # def init_hidden(self):
    #     h0 = Variable(torch.zeros(self.batch_size, num_layers, self.hidden_units))
    #     c0 = Variable(torch.zeros(self.batch_size, num_layers, self.hidden_units))
    #     return h0, c0

    def forward(self, sentence):
        embeds = self.embeds(sentence)
        # print(embeds)  # [torch.FloatTensor of size batch_zise*seq_len*embedding_dim]
        # x = embeds.view(len(sentence), self.batch_size, -1)
        # Inputs: input, (h_0, c_0) Outputs: output, (h_n, c_n) (batch, seq_len, hidden_size * num_directions)
        # If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        lstm_out, _ = self.lstm(embeds)
        # print(lstm_out)
        output = lstm_out[:, -1, :]
        return output


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        out1 = self.embedding_net(x1)
        out2 = self.embedding_net(x2)

        # out1_norm = torch.sqrt(torch.sum(torch.pow(out1, 2), dim=1))
        # out2_norm = torch.sqrt(torch.sum(torch.pow(out2, 2), dim=1))
        # cosine = (out1*out2).sum(1) / (out1_norm*out2_norm)
        sim = F.cosine_similarity(out1, out2, dim=1)
        # pdist = F.pairwise_distance(out1, out2, p=2, eps=1e-06, keepdim=False)

        return out1, out2, sim


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=0.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, y, y_):  # y, y_ must be same type float (*)
        loss = y * torch.pow(1-y_, 2) + (1 - y) * torch.pow(y_-self.margin, 2)
        loss = torch.sum(loss) / 2.0 / len(y)   #y.size()[0]
        return loss


