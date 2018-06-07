# !/usr/bin/env python
# coding: utf-8
from __future__ import unicode_literals
import os
import datetime

import numpy as np
import tensorflow as tf

from dataset import Dataset
from siamese_network import SiameseNets
from text_rcnn import TextRCNN

# Data loading params
tf.flags.DEFINE_string("data_file", "../data/atec_nlp_sim_train.csv", "Training data file path.")
tf.flags.DEFINE_float("val_percentage", .2, "Percentage of the training data to use for validation. (default: 0.2)")
tf.flags.DEFINE_integer("random_seed", 123, "Random seed to split train and test. (default: None)")
tf.flags.DEFINE_integer("max_document_length", 50, "Max document length of each train pair. (default: 15)")

# Model Hyperparameters
tf.flags.DEFINE_string("model_class", "siamese", "Model class, one of {`siamese`, `textrcnn`}")
tf.flags.DEFINE_string("model_type", "cnn", "Model type, one of {`cnn`, `rnn`, `rcnn`} (default: rnn)")
tf.flags.DEFINE_boolean("char_model", True, "Character based syntactic model. if false, word based semantic model. (default: True)")
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character/word embedding (default: 300)")
tf.flags.DEFINE_string("word_embedding_type", "static", "One of `rand`, `static`, `non-static`, random init(rand) vs pretrained word2vec(static) vs pretrained word2vec + training(non-static)")
# If include CNN
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default: 128)")
# If include RNN
tf.flags.DEFINE_string("rnn_cell", "lstm", "Rnn cell type, lstm or gru (default: lstm)")
tf.flags.DEFINE_integer("hidden_units", 100, "Number of hidden units (default: 50)")
tf.flags.DEFINE_integer("num_layers", 2, "Number of rnn layers (default: 3)")
tf.flags.DEFINE_float("clip_norm", None, "Gradient clipping norm value set None to not use (default: 5)")
tf.flags.DEFINE_boolean("use_attention", False, "Whether use self attention or not (default: False)")
# Common
tf.flags.DEFINE_boolean("weight_sharing", True, "Sharing CNN or LSTM weights. (default: True")
tf.flags.DEFINE_string("energy_function", "cosine", "Similarity energy function, one of {`euclidean`, `cosine`, `exp_manhattan`, `combine`} (default: euclidean)")
tf.flags.DEFINE_string("loss_function", "contrasive", "Loss function one of `cross_entrophy`, `contrasive`, (default: contrasive loss)")
# only for contrasive loss
tf.flags.DEFINE_float("scale_pos_weight", 2, "Scale loss function for imbalance data, set it around neg_samples / pos_samples ")
tf.flags.DEFINE_float("margin", 0.0, "Margin for contrasive loss (default: 0.0)")
tf.flags.DEFINE_float("pred_threshold", 0.50, "Threshold for classify.(default: 0.5)")
tf.flags.DEFINE_boolean("dense_layer", False, "Whether to add a fully connected layer before calculate energy function. (default: False)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 1.0)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_string("model_dir", "../model/cnn2", "Model directory (default: ./model)")
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_float("lr", 1e-3, "Initial learning rate (default: 1e-3)")
tf.flags.DEFINE_float("weight_decay_rate", 0.8, "Exponential weight decay rate (default: 0.9) ")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("log_every_steps", 100, "Print log info after this many steps (default: 100)")
tf.flags.DEFINE_integer("evaluate_every_steps", 100, "Evaluate model on dev set after this many steps (default: 100)")
# tf.flags.DEFINE_integer("checkpoint_every_steps", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

# supress tensorflow logging other than errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("\nParameters:")
if tf.__version__ < 1.5:
    FLAGS._parse_flags()  # tf version <= 1.4
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
else:
    for attr in FLAGS:
        value = FLAGS[attr].value
        print("{}={}".format(attr.upper(), value))
print("")

if not FLAGS.data_file:
    exit("Train data file is empty. Set --data_file argument.")

dataset = Dataset(data_file=FLAGS.data_file, char_level=FLAGS.char_model, embedding_dim=FLAGS.embedding_dim)
vocab, word2id = dataset.read_vocab()
print("Vocabulary Size: {:d}".format(len(vocab)))


def train():
    with tf.Graph().as_default():
        tf.set_random_seed(FLAGS.random_seed)
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            if FLAGS.model_class == 'siamese':
                model = SiameseNets(
                    model_type=FLAGS.model_type,
                    sequence_length=FLAGS.max_document_length,
                    vocab_size=len(vocab),
                    embedding_size=FLAGS.embedding_dim,
                    word_embedding_type=FLAGS.word_embedding_type,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    rnn_cell=FLAGS.rnn_cell,
                    hidden_units=FLAGS.hidden_units,
                    num_layers=FLAGS.num_layers,
                    use_attention=FLAGS.use_attention,
                    dense_layer=FLAGS.dense_layer,
                    pred_threshold=FLAGS.pred_threshold,
                    l2_reg_lambda=FLAGS.l2_reg_lambda,
                    energy_func=FLAGS.energy_function,
                    loss_func=FLAGS.loss_function,
                    margin=FLAGS.margin,
                    contrasive_loss_pos_weight=FLAGS.scale_pos_weight,
                    weight_sharing=FLAGS.weight_sharing
                )
                print("Initialized SiameseNets model.")
            else:
                model = TextRCNN(
                    model_type=FLAGS.model_type,
                    sequence_length=FLAGS.max_document_length,
                    embedding_size=FLAGS.embedding_dim,
                    vocab_size=len(vocab),
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    rnn_cell=FLAGS.rnn_cell,
                    hidden_units=FLAGS.hidden_units,
                    num_layers=FLAGS.num_layers,
                    pos_weight=FLAGS.scale_pos_weight,
                    l2_reg_lambda=FLAGS.l2_reg_lambda,
                    weight_sharing=FLAGS.weight_sharing,
                    interaction="multiply",
                    word_embedding_type=FLAGS.word_embedding_type
                )
                print("Initialized TextRCNN model.")

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            learning_rate = tf.train.exponential_decay(FLAGS.lr, global_step, decay_steps=int(40000/FLAGS.batch_size),
                                                       decay_rate=FLAGS.weight_decay_rate, staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            # optimizer = tf.train.RMSPropOptimizer(learning_rate)
            # optimizer = tf.train.AdadeltaOptimizer(learning_rate, epsilon=1e-6)

        # for i, (g, v) in enumerate(grads_and_vars):
        #     if g is not None:
        #         grads_and_vars[i] = (tf.clip_by_global_norm(g, 5), v)  # clip gradients
        # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        if FLAGS.clip_norm:  # improve loss, but small weight cause small score, need to turn threshold for better f1.
            variables = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(model.loss, variables), FLAGS.clip_norm)
            train_op = optimizer.apply_gradients(zip(grads, variables), global_step=global_step)
            grads_and_vars = zip(grads, variables)
        else:
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
        print("Defined gradient summaries.")

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", model.loss)
        f1_summary = tf.summary.scalar("F1-score", model.f1)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, f1_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(FLAGS.model_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, f1_summary])
        dev_summary_dir = os.path.join(FLAGS.model_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(FLAGS.model_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        graph_def = tf.get_default_graph().as_graph_def()
        with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
            f.write(str(graph_def))
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if FLAGS.word_embedding_type != 'rand':
            # initial matrix with random uniform
            # embedding_init = np.random.uniform(-0.25, 0.25, (len(vocab), FLAGS.embedding_dim))
            embedding_init = np.zeros(shape=(len(vocab), FLAGS.embedding_dim))
            # load vectors from the word2vec
            print("Initializing word embedding with pre-trained word2vec.")
            words, vectors = dataset.load_word2vec()
            for idx, w in enumerate(vocab):
                vec = vectors[words.index(w)]
                embedding_init[idx] = np.asarray(vec).astype(np.float32)
            print("Initialized word embedding")
            sess.run(model.W.assign(embedding_init))

        # Generate batches
        data = dataset.process_data(data_file=FLAGS.data_file, sequence_length=FLAGS.max_document_length)  # (x1, x2, y)
        train_data, eval_data = dataset.train_test_split(data, test_size=FLAGS.val_percentage, random_seed=FLAGS.random_seed)
        train_batches = dataset.batch_iter(train_data, FLAGS.batch_size, FLAGS.num_epochs, shuffle=True)

        print("Starting training...")
        F1_best = 0.
        last_improved_step = 0
        for batch in train_batches:
            x1_batch, x2_batch, y_batch = zip(*batch)
            # print(x1_batch[:3])
            # print(y_batch[:3])
            # if random.random() > 0.5:
            #     x1_batch, x2_batch = x2_batch, x1_batch
            feed_dict = {
                model.input_x1: x1_batch,
                model.input_x2: x2_batch,
                model.input_y: y_batch,
                model.dropout_keep_prob: FLAGS.dropout_keep_prob,
            }
            _, step, loss, acc, precision, recall, F1, summaries = sess.run(
                [train_op, global_step, model.loss, model.acc, model.precision, model.recall, model.f1, train_summary_op],  feed_dict)
            time_str = datetime.datetime.now().isoformat()
            if step % FLAGS.log_every_steps == 0:
                train_summary_writer.add_summary(summaries, step)
                print("{} step {} TRAIN loss={:g} acc={:.3f} P={:.3f} R={:.3f} F1={:.6f}".format(
                    time_str, step, loss, acc, precision, recall, F1))
            if step % FLAGS.evaluate_every_steps == 0:
                # eval
                x1_batch, x2_batch, y_batch = zip(*eval_data)

                feed_dict = {
                    model.input_x1: x1_batch,
                    model.input_x2: x2_batch,
                    model.input_y: y_batch,
                    model.dropout_keep_prob: 1.0,
                }
                # x1, out1, out2, sim_euc, sim_cos, sim_ma, e = sess.run(
                #    [model.embedded_1, model.out1, model.out2, model.sim_euc, model.sim_cos, model.sim_ma, model.e], feed_dict)
                # # print(x1)
                # sim_euc = [round(s, 2) for s in sim_euc[:30]]
                # sim_cos = [round(s, 2) for s in sim_cos[:30]]
                # sim_ma = [round(s, 2) for s in sim_ma[:30]]
                # e = [round(s, 2) for s in e[:30]]
                # # print(out1)
                # out1 = [round(s, 3) for s in out1[0]]
                # out2 = [round(s, 3) for s in out2[0]]
                # print(zip(out1, out2))
                # for w in zip(y_batch[:30], e, sim_euc, sim_cos, sim_ma):
                #     print(w)

                loss, acc, cm, precision, recall, F1, summaries = sess.run(
                    [model.loss, model.acc, model.cm, model.precision, model.recall, model.f1, dev_summary_op], feed_dict)
                dev_summary_writer.add_summary(summaries, step)
                if F1 > F1_best:
                    F1_best = F1
                    last_improved_step = step
                    if F1_best > 0.5:
                        path = saver.save(sess, checkpoint_prefix, global_step=step)
                        print("Saved model with F1={} checkpoint to {}\n".format(F1_best, path))
                    improved_token = '*'
                else:
                    improved_token = ''
                print("{} step {} DEV loss={:g} acc={:.3f} cm{} P={:.3f} R={:.3f} F1={:.6f} {}".format(
                    time_str, step, loss, acc, cm, precision, recall, F1, improved_token))
                # if step % FLAGS.checkpoint_every_steps == 0:
                #     if F1 >= F1_best:
                #         F1_best = F1
                #         path = saver.save(sess, checkpoint_prefix, global_step=step)
                #         print("Saved model with F1={} checkpoint to {}\n".format(F1_best, path))
            if step - last_improved_step > 4000:  # 2000 steps
                print("No improvement for a long time, early-stopping at best F1={}".format(F1_best))
                break

if __name__ == '__main__':
    train()