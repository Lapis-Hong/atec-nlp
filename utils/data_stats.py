#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/5/6
"""This scripts calculate some statistical information about the original text data file.

Main results:
    * positive sample percentage: train1->21.73%    train2->16.06%  total->18.23%
    * length from 5 to 97, 
    * frequency distribution: (5, 0.0071), (6, 0.0292), (7, 0.0392), (8, 0.0752), (9, 0.0877), (10, 0.1072), (11, 0.1069), 
                              (12, 0.0994), (13, 0.0833), (14, 0.0682), (15, 0.0554), (16, 0.043), (17, 0.0338), (18, 0.0285), 
                              (19, 0.0229), (20, 0.0181), (21, 0.0153), (22, 0.0119), (23, 0.01), (24, 0.0088) ...
    * there is no significant frequency difference between positive samples and negative samples.
    * pair length diff: (0, 0.0902), (1, 0.1782), (2, 0.1557), (3, 0.1285), (4, 0.1004), (5, 0.0818), (6, 0.0587) ...
    * the positive pairs length diff at {0, 1, 2} is slightly higher than negative pairs. 
"""
from collections import Counter


def positive_sample_percentage(filename):
    tot, pos = 0, 0
    for line in open(filename):
        line = line.strip().split('\t')
        if line[-1] == '1':
            pos += 1
        tot += 1
    print pos / float(tot)


def sentence_length_distribution(filename):
    tot, pos = 0, 0
    pos_seq_len = []
    neg_seq_len = []
    tot_seq_len = []
    for line in open(filename):
        line = line.strip().split('\t')
        s1 = line[1].decode('utf-8')
        s2 = line[2].decode('utf-8')
        tot_seq_len.extend([len(s1), len(s2)])
        tot += 2
        if line[-1] == '1':
            pos_seq_len.extend([len(s1), len(s2)])
            pos += 2
        else:
            neg_seq_len.extend([len(s1), len(s2)])
    tot_counter = Counter(tot_seq_len)
    pos_counter = Counter(pos_seq_len)
    neg_counter = Counter(neg_seq_len)
    tot_freq = sorted(map(lambda x: (x[0], round(x[1]/float(tot), 4)), tot_counter.items()))
    pos_freq = sorted(map(lambda x: (x[0], round(x[1]/float(pos), 4)), pos_counter.items()))
    neg_freq = sorted(map(lambda x: (x[0], round(x[1]/float(tot-pos), 4)), neg_counter.items()))
    print('Total sample length distribution: {}'.format(tot_freq))
    print('Positive sample length distribution: {}'.format(pos_freq))
    print('Negetive sample length distribution: {}'.format(neg_freq))


def pair_length_diff_distribution(filename):
    tot, pos = 0, 0
    tot_diff = []
    pos_diff = []
    neg_diff = []
    for line in open(filename):
        line = line.strip().split('\t')
        s1 = line[1].decode('utf-8')
        s2 = line[2].decode('utf-8')
        len_diff = abs(len(s1) - len(s2))
        tot_diff.append(len_diff)
        tot += 1
        if line[-1] == '1':
            pos_diff.append(len_diff)
            pos += 1
        else:
            neg_diff.append(len_diff)
    tot_counter = Counter(tot_diff)
    pos_counter = Counter(pos_diff)
    neg_counter = Counter(neg_diff)
    tot_freq = sorted(map(lambda x: (x[0], round(x[1] / float(tot), 4)), tot_counter.items()))
    pos_freq = sorted(map(lambda x: (x[0], round(x[1] / float(pos), 4)), pos_counter.items()))
    neg_freq = sorted(map(lambda x: (x[0], round(x[1] / float(tot - pos), 4)), neg_counter.items()))
    print('Total pair length diff distribution: {}'.format(tot_freq))
    print('Positive pair length diff distribution: {}'.format(pos_freq))
    print('Negetive pair length diff distribution: {}'.format(neg_freq))

if __name__ == '__main__':
    filename = '../data/atec_nlp_sim_train.csv'
    positive_sample_percentage(filename)
    # sentence_length_distribution(filename)
    # pair_length_diff_distribution(filename)

