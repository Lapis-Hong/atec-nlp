#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/5/5

# globals()['__doc__']
"""Usage:
        python word2vec.py input_file output_model_file output_vector_file
"""
import os
import sys
import logging
import multiprocessing
import time

import jieba
from gensim.models import Word2Vec
# from gensim.models.word2vec import LineSentence

from dataset import Dataset


def sentences_iterator(input_file, char_level=True):
    for line in open(input_file):
        line = line.strip().decode('utf-8').split('\t')
        s1, s2 = map(Dataset._clean_text, line[1:3])
        if char_level:
            yield [c for c in s1]
            yield [c for c in s2]
        else:
            yield jieba.cut(s1)
            yield jieba.cut(s2)


def word2vec(sentences, output_model_file, output_vector_file, size=128, window=5, min_count=1):
    s_time = time.time()
    # trim unneeded model memory = use(much) less RAM
    # model.init_sims(replace=True)
    model = Word2Vec(sentences, sg=1, size=size, window=window, min_count=min_count,
                     negative=3, sample=0.001, hs=1, workers=multiprocessing.cpu_count())
    model.save(output_model_file)
    model.wv.save_word2vec_format(output_vector_file, binary=False)
    print("Word2vec training time: %d s" % (time.time() - s_time))


def main():
    program = os.path.basename(sys.argv[0])  # 'word2vec.py'
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info("Running %s" % ' '.join(sys.argv))
    # check and process input arguments
    if len(sys.argv) < 4:
        print('Missing arguments.')
        sys.exit(globals()['__doc__'])
    input_file, output_model_file, output_vector_file = sys.argv[1:4]
    sentences = sentences_iterator(input_file)
    word2vec(list(sentences), output_model_file, output_vector_file)


if __name__ == '__main__':
    # s = sentences_iterator('data/atec_nlp_sim_train.csv')
    main()

