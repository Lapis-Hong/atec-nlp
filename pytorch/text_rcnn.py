#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/5/15
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.args = args

        self.embed = nn.Embedding(args.sequence_length, args.embed_dim)
        self.convs1 = nn.ModuleList([nn.Conv2d(1, args.kernel_num, (ks, args.embed_dim)) for ks in args.kernel_sizes])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(args.kernel_sizes) * args.kernel_num, args.class_num)

    def forward(self, x):
        x = self.embed(x)  # (batch_size, sequence_length, embedding_dim)
        if self.args.static:
            x = torch.tensor(x)
        x = x.unsqueeze(1)  # (batch_size, 1, sequence_length, embedding_dim)
        # #  input size (N,Cin,H,W)  output size (N,Cout,Hout,1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)  # (batch_size, len(kernel_sizes)*kernel_num)
        x = self.dropout(x)
        logit = self.fc1(x)
        return logit


class TextRNN(nn.Module):

    def __init__(self, args):
        super(TextRNN, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size

        self.embeds = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.lstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_dim, num_layers=args.num_layers,
                            batch_first=True, bidirectional=True)
        self.hidden2label = nn.Linear(args.hidden_dim, args.num_classes)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim))
        return h0, c0

    def forward(self, sentence):
        embeds = self.embeds(sentence)
        # x = embeds.view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        y = self.hidden2label(lstm_out[-1])
        return y


class TextRCNN(nn.Module):

    def __init__(self, args):
        super(TextRCNN, self).__init__()
        self.interaction = args.interaction
        self.model_type = args.model_type
        self.cnn = TextCNN(args)
        self.rnn = TextRNN(args)

    def forward(self, x):
        if self.model_type == 'cnn':
            out = self.cnn.forward(x)
        elif self.model_type == 'rnn':
            out = self.rnn.forward(x)
        elif self.model_type == 'rcnn':
            out = self.cnn.forward(x) + self.rnn.forward(x)

        if self.interaction == 'multiply':
            pass

        return out

