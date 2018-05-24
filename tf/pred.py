# !/usr/bin/env python
import sys
import os

import tensorflow as tf

from dataset import Dataset
from train import FLAGS

FLAGS.model_dir = '/home/hongquan/atec_nlp/model/rnn/'
FLAGS.max_document_length = 20


def main(input_file, output_file):
    print("\nPredicting...\n")
    graph = tf.Graph()
    with graph.as_default():  # with tf.Graph().as_default() as g:
        sess = tf.Session()
        with sess.as_default():
            # Load the saved meta graph and restore variables
            # saver = tf.train.Saver(tf.global_variables())
            meta_file = os.path.abspath(os.path.join(FLAGS.model_dir, 'checkpoints/model-3400.meta'))
            new_saver = tf.train.import_meta_graph(meta_file)
            new_saver.restore(sess, tf.train.latest_checkpoint(os.path.join(FLAGS.model_dir, 'checkpoints')))
            # graph = tf.get_default_graph()

            # Get the placeholders from the graph by name
            # input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
            input_x1 = graph.get_tensor_by_name("input_x1:0")  # Tensor("input_x1:0", shape=(?, 15), dtype=int32)
            input_x2 = graph.get_tensor_by_name("input_x2:0")
            dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")
            # Tensors we want to evaluate
            y_pred = graph.get_tensor_by_name("metrics/y_pred:0")
            # vars = tf.get_collection('vars')
            # for var in vars:
            #     print(var)

            e = graph.get_tensor_by_name("cosine:0")

            # Generate batches for one epoch
            dataset = Dataset(data_file=input_file, is_training=False)
            data = dataset.process_data(data_file=input_file, sequence_length=FLAGS.max_document_length)
            batches = dataset.batch_iter(data, FLAGS.batch_size, 1, shuffle=False)
            with open(output_file, 'w') as fo:
                lineno = 1
                for batch in batches:
                    x1_batch, x2_batch, _, _ = zip(*batch)
                    y_pred_ = sess.run([y_pred], {input_x1: x1_batch, input_x2: x2_batch, dropout_keep_prob: 1.0})
                    for pred in y_pred_[0]:
                        fo.write('{}\t{}\n'.format(lineno, pred))
                        lineno += 1

if __name__ == '__main__':
    # Set to INFO for tracking training, default is WARN. ERROR for least messages
    tf.logging.set_verbosity(tf.logging.WARN)
    main(sys.argv[1], sys.argv[2])
