#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/5/6
import os
import re
import sys
from collections import Counter

import numpy as np
import tensorflow as tf
import jieba
import jieba.analyse

reload(sys)
sys.setdefaultencoding('utf-8')

jieba.enable_parallel(20)
jieba.add_word(u'花呗')
jieba.add_word(u'借呗')
jieba.add_word(u'余额宝')


class Dataset():
    """Custom dataset class to deal with input text data."""
    def __init__(self, data_file='data/atec_nlp_sim_train.csv',
                 npz_char_data_file='data/train_char.npz', npz_word_data_file='data/train_word.npz',
                 char_vocab_file='data/char.vocab', word_vocab_file='data/word.vocab',
                 word2vec_file='data/w2v.txt',
                 char_level=True,
                 ):
        self.data_file = data_file
        self.npz_char_data_file = npz_char_data_file
        self.npz_word_data_file = npz_word_data_file
        self.char_vocab_file = char_vocab_file
        self.word_vocab_file = word_vocab_file
        self.word2vec_file = word2vec_file
        self.char_level = char_level

    @staticmethod
    def _clean_text(text):
        """Text filter for Chinese corpus, only keep CN character."""
        re_non_ch = re.compile(ur'[^\u4e00-\u9fa5]+')
        text = text.decode('utf-8').strip(' ')
        text = re_non_ch.sub('', text)
        return text

    def _load_data(self):
        """Load origin train data and do text cleaning, token (optional).
        Args: 
            token: Bool, set True to tokenize, default False, which means using Char RNN.
        Returns train pairs and labels.
        """
        x1, x2, y = [], [], []
        for line in open(self.data_file):
            line = line.strip().split('\t')
            if len(line) != 4:
                continue
            text1, text2 = map(self._clean_text, line[1:3])

            if not self.char_level:
                text1 = list(jieba.cut(text1))
                text2 = list(jieba.cut(text2))
            x1.append(text1)
            x2.append(text2)
            y.append(1) if line[-1] == '1' else y.append(0)  # 1 or [1]
        return x1, x2, y

    def _build_vocab(self, max_vocab_size=100000, min_count=1):
        """Build vacabulary list."""
        x1, x2, _ = self._load_data()
        vocab = []
        if self.char_level:
            vocab_file = self.char_vocab_file
            for words in x1+x2:
                for char in words.decode('utf-8'):
                    vocab.append(char)
        else:
            vocab_file = self.word_vocab_file
            for words in x1+x2:
                vocab.extend(words)
        counter = Counter(vocab)
        words_count = counter.most_common(max_vocab_size - 1)  # sort by word freq.
        words = ['UNK']  # for oov words
        words += [w[0] for w in words_count if w[1] >= min_count]
        # add word '<PAD>' for padding
        words.append('<PAD>')
        with open(vocab_file, 'w') as fo:
            fo.write('\n'.join(words))

    def read_vocab(self):
        """Read vocabulary list, return tuple(words, word2id dict)."""
        if self.char_level:
            print('Using character level model.')
            vocab_file = self.char_vocab_file
        else:
            print('Using word level model.')
            vocab_file = self.word_vocab_file
        if not os.path.exists(vocab_file):
            print('Vocabluary file not found. Building vocabulary...')
            self._build_vocab()
        words = open(vocab_file).read().decode('utf-8').strip().split('\n')
        word2id = dict(zip(words, range(len(words))))
        return words, word2id

    def process_data(self, sequence_length=20, is_training=True, save=True):
        """Process train text data file to word id matrix representation.
        Args:
            sequence_length: int, max_sequence_length. (default 20)
            
            is_training: bool, set False to deal with pred data. (default True)
        Returns:
            if is_training=True:
                (x1, x2, y)
            if is_training=False:
                (x1, x2)
        """
        if is_training:
            if self.char_level and os.path.exists(self.npz_char_data_file):
                data = np.load(self.npz_char_data_file)
                # check sequence length same or not
                if len(data["x1"][0]) == sequence_length:
                    print("Loaded saved npy file.")
                    return data["x1"], data["x2"], data["y"]
            elif not self.char_level and os.path.exists(self.npz_word_data_file):
                data = np.load(self.npz_word_data_file)
                if len(data["x1"][0]) == sequence_length:
                    print("Loaded saved npy file.")
                    return data["x1"], data["x2"], data["y"]

        print('Loading origin data...')
        x1, x2, y = self._load_data()
        print('Total number of samples: {}'.format(len(x1)))
        print('Reading vocab...')
        _, word2id = self.read_vocab()
        print('Converting word to index...')
        # oov words id is 0, token either a single char or word.
        x1_id = [[word2id.get(token, 0) for token in words] for words in x1]
        x2_id = [[word2id.get(token, 0) for token in words] for words in x2]
        print('Padding...')
        x1_pad = tf.keras.preprocessing.sequence.pad_sequences(
                x1_id, maxlen=sequence_length, padding='post', truncating='post', value=len(word2id)-1)
        x2_pad = tf.keras.preprocessing.sequence.pad_sequences(x2_id, sequence_length, value=len(word2id)-1)
        # y = tf.keras.utils.to_categorical(y)  # turn label into onehot
        if save:
            print("Saving npz...")
            x1, x2, y = map(np.asarray, [x1_pad, x2_pad, y])
            save_file = self.npz_char_data_file if self.char_level else self.npz_word_data_file
            np.savez(save_file, x1=x1, x2=x2, y=y)  # save multiple arrays as zip file.
            # np.savetxt(save_file, np.concatenate([x1, x2, y], axis=1), fmt="%d")  # or use np.hstack()
        print("Loaded words id matrix.")
        if is_training:
            return x1_pad, x2_pad, y
        else:
            return x1_pad, x2_pad

    @staticmethod
    def train_test_split(data, test_size=0.2, random_seed=123):
        """Split train data into train and test sets.
        Args:
            data: 3 elemetns tuple (x1, x2, y), each ele is 2-D array.
            test_size: float, int. (default 0.2)
                If float, should be between 0.0 and 1.0 and represent the proportion of test set. 
                If int, represents the absolute number of test samples. 
            random_seed: int or None. (default 123)
                If None, do not use random seed.
        Returns
            2 list of tuples [(x1_i, x2_i, y_i)...], each tuple ele is 1-D array.
        """
        x1, x2, y = map(np.asarray, data)
        num_samples = len(y)
        test_size = int(num_samples * test_size) if isinstance(test_size, float) else test_size
        print('Total number of samples: {}'.format(num_samples))
        print('Test data size: {}'.format(test_size))
        if random_seed:
            np.random.seed(random_seed)
        shuffle_indices = np.random.permutation(np.arange(num_samples))
        x1_shuffled, x2_shuffled, y_shuffled = x1[shuffle_indices], x2[shuffle_indices], y[shuffle_indices]
        train_data = x1_shuffled[test_size:], x2_shuffled[test_size:], y_shuffled[test_size:]
        test_data = x1_shuffled[:test_size], x2_shuffled[:test_size], y_shuffled[:test_size]
        print('Train eval data split done.')
        return train_data, test_data

    @staticmethod
    def batch_iter(data, batch_size, num_epochs, shuffle=True, data_augmentation=False):
        """Generates a batch iterator for a dataset.
        Args:
            data: list of tuples [(x1_i, x2_i, y_i)...], each tuple ele is 1-D array.
        Returns:
            list of tuples [(x1_i, x2_i, y_i)...], each tuple ele is 1-D array.
            use zip(*return) to generate x1_batch, x2_batch, y_batch
        """
        # num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
        for epoch in range(num_epochs):
            if data_augmentation:
                new_data = []
                for s1, s2, label in data:
                    new_data.append([s1, s2, label])
                    id1 = np.random.randint(0, len(s1))
                    id2 = np.random.randint(0, len(s2))
                    s1, s2 = map(list, [s1, s2])
                    s1.pop(id1)
                    s1.insert(0, 0)
                    s2.pop(id2)
                    s2.insert(0, 0)
                    new_data.append([s1, s2, label])
                dataset = new_data
            else:
                dataset = data
            # Shuffle the data at each epoch
            dataset = np.asarray(dataset)
            data_size = len(dataset)
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = dataset[shuffle_indices]
            else:
                shuffled_data = dataset

            num_batches_per_epoch = int((len(dataset) - 1) / batch_size) + 1
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    def load_word2vec(self):
        """mapping the words to word vectors, return tuple(words, vectors)"""
        words = []
        vecs = []
        fr = open(self.word2vec_file, 'r')
        word_dim = int(fr.readline().strip().split(' ')[1])  # first line
        words.append("UNK")
        vecs.append([0] * word_dim)
        words.append("<PAD>")
        vecs.append([0] * word_dim)

        for line in fr:
            line = line.decode('utf-8').strip().split(' ')
            words.append(line[0])
            vecs.append(line[1:])
        print "Loaded word2vec."
        return words, vecs

if __name__ == '__main__':
    d = Dataset(char_level=True)
    # d = Dataset(char_level=False)
    # x1, x2, y = Dataset().process_data()
