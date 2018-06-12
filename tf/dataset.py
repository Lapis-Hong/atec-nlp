#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/5/6
"""This module provide an elegant data process class."""
from __future__ import unicode_literals

import logging
import multiprocessing
import os
import re
import sys
import time
from collections import Counter

import jieba
import jieba.analyse
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec

sys.path.insert(0, '../')
from utils.langconv import Converter
reload(sys)
sys.setdefaultencoding('utf-8')

# jieba.enable_parallel(4)  # This is a bug, make add_word no use
jieba.load_userdict('../data/UserDict.txt')
stopwords = ['的', '了']


class Dataset(object):
    """Custom dataset class to deal with input text data."""
    def __init__(self,
                 data_file='../data/atec_nlp_sim_train.csv',
                 npy_char_data_file='../data/train_char.npy',
                 npy_word_data_file='../data/train_word.npy',
                 char_vocab_file='../data/vocab.char',
                 word_vocab_file='../data/vocab.word',
                 char2vec_file='../data/char_vec',
                 word2vec_file='../data/word_vec',
                 char_level=True,
                 embedding_dim=128,
                 is_training=True,
                 ):
        self.data_file = data_file
        self.npy_char_data_file = npy_char_data_file
        self.npy_word_data_file = npy_word_data_file
        self.char_vocab_file = char_vocab_file
        self.word_vocab_file = word_vocab_file
        self.word2vec_file = word2vec_file
        self.char2vec_file = char2vec_file
        self.char_level = char_level
        self.embedding_dim = embedding_dim
        self.is_training = is_training
        if self.char_level:
            print('Using character level model.')
        else:
            print('Using word level model.')
        self.w2v_file = self.char2vec_file if self.char_level else self.word2vec_file
        self.vocab_file = self.char_vocab_file if self.char_level else self.word_vocab_file
        self.npy_file = self.npy_char_data_file if self.char_level else self.npy_word_data_file

    @staticmethod
    def _clean_text(text):
        """Text filter for Chinese corpus, keep CN character and remove stopwords."""
        re_non_ch = re.compile(ur'[^\u4e00-\u9fa5]+')
        text = text.strip(' ')
        text = re_non_ch.sub('', text)
        for w in stopwords:
            text = re.sub(w, '', text)
        return text

    @staticmethod
    def _tradition2simple(text):
        """Tradition Chinese corpus to simplify Chinese."""
        text = Converter('zh-hans').convert(text)
        return text

    def _load_data(self, data_file):
        """Load origin train data and do text pre-processing (converting and cleaning)
        Returns:
            A generator
            if self.is_training:
                train sentence pairs and labels (s1, s2, y).
            else:
                train sentence pairs and None (s1, s2, None).
        """
        for line in open(data_file):
            line = line.strip().decode('utf-8').split('\t')
            s1, s2 = map(self._clean_text, map(self._tradition2simple, line[1:3]))
            if not self.char_level:
                s1 = list(jieba.cut(s1))
                s2 = list(jieba.cut(s2))
            if self.is_training:
                y = int(line[-1])  # 1 or [1]
                yield s1, s2, y
            else:
                yield s1, s2, None  # for consistent

    def _save_token_data(self):
        data_iter = self._load_data(self.data_file)
        with open('../data/atec_token.csv', 'w') as f:
            for s1, s2, _ in data_iter:
                f.write(' '.join(s1) + '|' + ' '.join(s2) + '\n')

    def _build_vocab(self, max_vocab_size=100000, min_count=2):
        """Build vocabulary list."""
        data_iter = self._load_data(self.data_file)
        token = []
        for s1, s2, _ in data_iter:
            if self.char_level:
                for words in s1+s2:
                    for char in words:
                        token.append(char)
            else:
                token.extend(s1+s2)
        print("Number of tokens: {}".format(len(token)))
        counter = Counter(token)
        word_count = counter.most_common(max_vocab_size - 1)  # sort by word freq.
        vocab = ['UNK']  # for oov words
        vocab += [w[0] for w in word_count if w[1] >= min_count]
        vocab.append('<PAD>')  # add word '<PAD>' for padding
        print("Vocabulary size: {}".format(len(vocab)))
        with open(self.vocab_file, 'w') as fo:
            fo.write('\n'.join(vocab))

    def read_vocab(self):
        """Read vocabulary list
        Returns:
             tuple (id2word, word2id).
        """
        if not os.path.exists(self.vocab_file):
            print('Vocabulary file not found. Building vocabulary...')
            self._build_vocab()
        else:
            print("Reading vocabulary file from {}".format(self.vocab_file))
        id2word = open(self.vocab_file).read().split('\n')  # list
        word2id = dict(zip(id2word, range(len(id2word))))  # dict
        return id2word, word2id

    def _word2vec(self, window=5, min_count=2):
        """Train and save word vectors"""
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        s_time = time.time()
        s1, s2, _ = zip(*list(self._load_data(self.data_file)))
        sentences = s1 + s2
        size = self.embedding_dim
        # trim unneeded model memory = use(much) less RAM
        # model.init_sims(replace=True)
        model = Word2Vec(sentences, sg=1, size=size, window=window, min_count=min_count,
                         negative=3, sample=0.001, hs=1, workers=multiprocessing.cpu_count(), iter=20)
        # model.save(output_model_file)
        model.wv.save_word2vec_format(self.w2v_file, binary=False)
        print("Word2vec training time: %d s" % (time.time() - s_time))

    def load_word2vec(self):
        """mapping the words to word vectors.
        Returns:
            tuple (words, vectors)
        """
        if not os.path.exists(self.w2v_file):
            print('Word vectors file not found. Training word vectors...')
            self._word2vec()
        words, vecs = [], []
        fr = open(self.w2v_file)
        word_dim = int(fr.readline().strip().split(' ')[1])  # first line
        print("Pre-trained word vectors dim: {}".format(word_dim))
        if word_dim != self.embedding_dim:
            print("Inconsistent word embedding dim, retrain word vectors...")
            self._word2vec()
            return self.load_word2vec()
        else:
            words.append("UNK")
            vecs.append([0] * word_dim)
            words.append("<PAD>")
            vecs.append([0] * word_dim)
            for line in fr:
                line = line.decode('utf-8').strip().split(' ')
                words.append(line[0])
                vecs.append(line[1:])
            print("Loaded pre-trained word vectors.")
        return words, vecs

    def process_data(self, data_file, sequence_length=20):
        """Process text data file to word-id matrix representation.
        Args:
            data_file: process data file.
            sequence_length: int, max sequence length. (default 20)
        Returns:
            2-D List.
            if self.is_training:
                each element of list is [s1_pad, s2_pad, y]
            else:
                each element of list is [s1_pad, s2_pad]
        """
        if data_file == self.data_file and os.path.exists(self.npy_file):  # only for all train data
            dataset = np.load(self.npy_file)
            # check sequence length same or not
            if len(dataset[0][0]) == sequence_length:
                print("Loaded saved npy word-id matrix train file.")
                return dataset
            else:
                print("Found inconsistent sequence length with npy file.")

        _, word2id = self.read_vocab()
        data_iter = self._load_data(data_file)
        dataset = []
        print('Converting word-index matrix...')
        for s1, s2, y in data_iter:
            # oov words id is 0, token is either a single char or word.
            s1_id = [word2id.get(token, 0) for token in s1]
            s2_id = [word2id.get(token, 0) for token in s2]
            # "pre" or "post" important, "pre" much better, why ?
            s1_pad = tf.keras.preprocessing.sequence.pad_sequences(
                [s1_id], maxlen=sequence_length, padding='post', truncating='post', value=len(word2id)-1)
            s2_pad = tf.keras.preprocessing.sequence.pad_sequences(
                [s2_id], maxlen=sequence_length, padding='post', truncating='post', value=len(word2id)-1)
            # y = tf.keras.utils.to_categorical(y)  # turn label into onehot
            if self.is_training:
                dataset.append([s1_pad[0], s2_pad[0], y])
            else:
                dataset.append([s1_pad[0], s2_pad[0]])
        print("Saving npy...")
        dataset = np.asarray(dataset)
        np.save(self.npy_file, dataset)
        # np.savez(save_file, x1=x1, x2=x2, y=y)  # save multiple arrays as zip file.
        # np.savetxt(save_file, np.concatenate([x1, x2, y], axis=1), fmt="%d")  # or use np.hstack()
        return dataset

    @staticmethod
    def train_test_split(dataset, test_size=0.2, random_seed=123):
        """Split train data into train and test sets.
        Args:
            dataset: 2-D list, each element is a sample list [x1, x2, y, len(s1), len(s2)]
            test_size: float, int. (default 0.2)
                If float, should be between 0.0 and 1.0 and represent the proportion of test set. 
                If int, represents the absolute number of test samples. 
            random_seed: int or None. (default 123)
                If None, do not use random seed.
        Returns
            A tuple (trainset, testset)
        """
        dataset = np.asarray(dataset)
        num_samples = len(dataset)
        test_size = int(num_samples * test_size) if isinstance(test_size, float) else test_size
        print('Total number of samples: {}'.format(num_samples))
        print('Test data size: {}'.format(test_size))
        if random_seed:
            np.random.seed(random_seed)
        shuffle_indices = np.random.permutation(np.arange(num_samples))
        dataset_shuffled = dataset[shuffle_indices]
        trainset = dataset_shuffled[test_size:]
        testset = dataset_shuffled[:test_size]
        print('Train eval data split done.')
        return trainset, testset

    @staticmethod
    def batch_iter(dataset, batch_size, num_epochs, shuffle=True):
        """Generates a batch iterator for a dataset.
        Args:
            dataset: 2-D list, each element is a sample list [x1, x2, y]
        Returns:
            list of batch samples [x1, x2, y].
            use zip(*return) to generate x1_batch, x2_batch, y_batch
        """
        dataset = np.asarray(dataset)
        data_size = len(dataset)
        num_batches_per_epoch = int((len(dataset)-1)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = dataset[shuffle_indices]
            else:
                shuffled_data = dataset
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]


if __name__ == '__main__':
    d_char = Dataset(char_level=True)
    d_word = Dataset(char_level=False, embedding_dim=128)
    # s1, s2, y = d_word._load_data('../data/train.csv').next()

    # d_word._build_vocab()
    d_word._save_token_data()
    # id2w, w2id = d_word.read_vocab()
    # dataset = Dataset().process_data('../data/atec_nlp_sim_train.csv')
    # data = Dataset().batch_iter(dataset, 5, 1, shuffle=False).next()
    # print(data)
    # d_word.load_word2vec()





