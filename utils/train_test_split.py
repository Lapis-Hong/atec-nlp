#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/5/16
import random


def train_test_split(infile, test_rate=0.2):
    with open('data/train.csv', 'w') as f_train, \
            open('data/test.csv', 'w') as f_test:
        for line in open(infile):
            if random.random() > test_rate:
                f_train.write(line)
            else:
                f_test.write(line)


if __name__ == '__main__':
    train_test_split('data/atec_nlp_sim_train.csv')

