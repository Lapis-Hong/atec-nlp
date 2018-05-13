#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/5/10
# !/usr/bin/env python
import os

import tensorflow as tf

from dataset import Dataset
from train import FLAGS


def bad_cases():
    print("\nPredicting...\n")
    graph = tf.Graph()
    with graph.as_default():  # with tf.Graph().as_default() as g:
        sess = tf.Session()
        with sess.as_default():
            # Load the saved meta graph and restore variables
            # saver = tf.train.Saver(tf.global_variables())
            meta_file = os.path.abspath(os.path.join(FLAGS.model_dir, 'checkpoints/model-1000.meta'))
            new_saver = tf.train.import_meta_graph(meta_file)
            new_saver.restore(sess, tf.train.latest_checkpoint(os.path.join(FLAGS.model_dir, 'checkpoints')))
            # graph = tf.get_default_graph()

            # Get the placeholders from the graph by name
            # input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
            input_x1 = graph.get_tensor_by_name("input_x1:0")  # Tensor("input_x1:0", shape=(?, 15), dtype=int32)
            input_x2 = graph.get_tensor_by_name("input_x2:0")
            dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")
            # Tensors we want to evaluate
            sim = graph.get_tensor_by_name("metrics/sim:0")
            y_pred = graph.get_tensor_by_name("metrics/y_pred:0")

            dev_sample = {}
            for line in open(FLAGS.data_file):
                line = line.strip().split('\t')
                dev_sample[line[0]] = line[1]

            # Generate batches for one epoch
            dataset = Dataset(data_file="data/pred.csv")
            x1, x2, y = dataset.process_data(sequence_length=FLAGS.max_document_length, is_training=False)
            with open("result/fp_file", 'w') as f_fp, open("result/fn_file", 'w') as f_fn:
                for lineno, x1_online, x2_online, y_online in enumerate(zip(x1, x2, y)):
                    sim, y_pred_ = sess.run(
                        [sim, y_pred], {input_x1: x1_online, input_x2: x2_online, dropout_keep_prob: 1.0})
                    if y_pred == 1 and y_online == 0:  # low precision
                        f_fp.write(dev_sample[lineno+1] + str(sim) + '\n')
                    elif y_pred == 0 and y_online == 1:  # low recall
                        f_fn.write(dev_sample[lineno + 1] + str(sim) + '\n')

if __name__ == '__main__':
    # Set to INFO for tracking training, default is WARN. ERROR for least messages
    tf.logging.set_verbosity(tf.logging.WARN)
    bad_cases()
