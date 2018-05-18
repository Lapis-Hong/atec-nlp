#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/5/15
import os
import re

import jieba
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class Dictionary(object):
    def __init__(self, infile, char_vocab_file=None, word_vocab_file=None, char_level=True):
        self.infile = infile
        self.char_level = char_level
        self.word2idx = {}
        self.idx2word = []
        vocab_file = char_vocab_file if char_level else word_vocab_file

        if not vocab_file or os.path.exists(vocab_file):
            print('Vocabluary file not found. Building vocabulary...')
            self.build_vocab()
        else:
            self.idx2word = open(vocab_file).read().decode('utf-8').strip().split('\n')
            self.word2idx = dict(zip(self.idx2word, range(len(self.idx2word))))

    @staticmethod
    def _clean_text(text):
        """Text filter for Chinese corpus, only keep CN character."""
        re_non_ch = re.compile(ur'[^\u4e00-\u9fa5]+')
        text = text.decode('utf-8').strip(' ')
        text = re_non_ch.sub('', text)
        return text

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def build_vocab(self):
        self.add_word('<PAD>')  # pad index: 0
        for line in open(self.infile, 'r'):
            _, s1, s2, label = line.strip().split('\t')
            s1, s2 = map(self._clean_text, [s1, s2])
            if not self.char_level:
                s1 = list(jieba.cut(s1))
                s2 = list(jieba.cut(s2))
            for token in s1+s2:
                # build vocabulary
                self.add_word(token)
        self.add_word('UNK')  # unk index: len(word2idx)-1

    def __len__(self):
        return len(self.idx2word)


class MyDataset(Dataset):

    def __init__(self, data_file, sequence_length, word2idx, char_level=True):
        self.word2idx = word2idx
        self.seq_len = sequence_length

        x1, x2, y = [], [], []
        for line in open(data_file, 'r'):
            _, s1, s2, label = line.strip().split('\t')
            s1, s2 = map(self._clean_text, [s1, s2])
            if not char_level:
                s1 = list(jieba.cut(s1))
                s2 = list(jieba.cut(s2))
            x1.append(s1)
            x2.append(s2)
            y.append(1) if label == '1' else y.append(0)
        self.x1 = x1
        self.x2 = x2
        self.y = y

    @staticmethod
    def _clean_text(text):
        """Text filter for Chinese corpus, only keep CN character."""
        re_non_ch = re.compile(ur'[^\u4e00-\u9fa5]+')
        text = text.decode('utf-8').strip(' ')
        text = re_non_ch.sub('', text)
        return text

    def __getitem__(self, index):
        s1, s2 = self.x1[index], self.x2[index]
        s1_id = torch.LongTensor(np.zeros(self.seq_len, dtype=np.int64))
        s2_id = torch.LongTensor(np.zeros(self.seq_len, dtype=np.int64))
        label = torch.LongTensor([self.y[index]])
        for idx, (w1, w2) in enumerate(zip(s1, s2)):
            if idx > self.seq_len - 1:
                break
            s1_id[idx] = self.word2idx.get(w1, self.word2idx["UNK"])
            s2_id[idx] = self.word2idx.get(w2, self.word2idx["UNK"])
        return s1_id, s2_id, label

    def __len__(self):
        return len(self.y)


if __name__ == '__main__':
    dic = Dictionary('../data/atec_nlp_sim_train.csv', '../data/cha.vocab', '../data/word.vocab')
    dataset = MyDataset('../data/atec_nlp_sim_train.csv', 15, dic.word2idx)
    x1, x2, y = dataset[3]
    print(x1)
    print(y)