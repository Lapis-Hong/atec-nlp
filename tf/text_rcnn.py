#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/5/12
import tensorflow as tf


class TextRCNN(object):
    """
    A CNN, RNN based deep network for text similarity.
    Uses an character/word level embedding layer, followed by a {`BiLSTM`, `CNN`, `combine`} network.
    """
    def __init__(self, model_type, sequence_length, embedding_size, vocab_size,
                 filter_sizes, num_filters,
                 rnn_cell, hidden_units, num_layers,
                 pos_weight, l2_reg_lambda, weight_sharing=True, interaction="multiply", word_embedding_type="rand"):
        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
        self.input_y = tf.placeholder(tf.int64, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.variable_scope("embedding"):
            if word_embedding_type == "rand":
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    trainable=True, name="W")
            else:
                if word_embedding_type == "static":
                    trainable = False
                else:
                    trainable = True
                self.W = tf.Variable(
                    tf.constant(0.0, shape=[vocab_size, embedding_size]),
                    trainable=trainable, name="W")

            self.embedded_chars1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            self.embedded_chars2 = tf.nn.embedding_lookup(self.W, self.input_x2)
            print('input_x1: ', self.embedded_chars1.get_shape())
            # Input dropout. TODO
            self.embedded_chars1 = tf.nn.dropout(self.embedded_chars1, self.dropout_keep_prob)
            self.embedded_chars2 = tf.nn.dropout(self.embedded_chars2, self.dropout_keep_prob)  # shape(batch_size, seq_len, dim)
            self.embedded_chars1_expanded = tf.expand_dims(self.embedded_chars1, -1)  # shape(batch_size, seq_len, dim, 1)
            self.embedded_chars2_expanded = tf.expand_dims(self.embedded_chars2, -1)

        with tf.variable_scope("CNN", reuse=tf.AUTO_REUSE):  # share cnn weights.
            # shape(batch_size, num_filters*len(filters_sizes))
            self.cnn_out1 = self.cnn(self.embedded_chars1_expanded,
                                 sequence_length, embedding_size, filter_sizes, num_filters)
            self.cnn_out2 = self.cnn(self.embedded_chars2_expanded,
                                 sequence_length, embedding_size, filter_sizes, num_filters)

        with tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):  # share rnn weights.
            # shape(batch_size, 2*hidden_units)
            self.rnn_out1 = self.bi_rnn(self.embedded_chars1, rnn_cell, hidden_units, num_layers,
                                        self.dropout_keep_prob)
            self.rnn_out2 = self.bi_rnn(self.embedded_chars2, rnn_cell, hidden_units, num_layers,
                                        self.dropout_keep_prob)

        with tf.name_scope("output"):
            if model_type.lower() == 'cnn':
                self.out1 = self.cnn_out1
                self.out2 = self.cnn_out2
            elif model_type.lower() == 'rnn':
                self.out1 = self.rnn_out1
                self.out2 = self.rnn_out2
            elif model_type.lower() == 'rcnn':
                self.out1 = tf.concat([self.cnn_out1, self.rnn_out1], axis=1)
                self.out2 = tf.concat([self.cnn_out2, self.rnn_out2], axis=1)

            if interaction == 'concat':
                self.out = tf.concat([self.out1, self.out2], axis=1, name="out")
            elif interaction == 'multiply':
                self.out = tf.multiply(self.out1, self.out2, name="out")
            self.fc = tf.layers.dense(self.out, 128, name='fc1', activation=tf.nn.relu)
            # self.scores = tf.layers.dense(self.fc, 1, activation=tf.nn.sigmoid)
            self.logits = tf.layers.dense(self.fc, 2, name='fc2')
            # self.y_pred = tf.round(tf.nn.sigmoid(self.logits), name="predictions")  # pred class
            self.y_pred = tf.argmax(tf.nn.sigmoid(self.logits), 1, name="predictions")

        with tf.name_scope("loss"):
            # [batch_size, num_classes]
            y = tf.one_hot(self.input_y, 2)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=y)
            self.loss = tf.reduce_mean(cross_entropy)
            # self.loss = tf.losses.sigmoid_cross_entropy(logits=self.scores, multi_class_labels=self.input_y)
            # y = self.input_y
            # y_ = self.scores
            # self.loss = -tf.reduce_mean(pos_weight * y * tf.log(tf.clip_by_value(y_, 1e-10, 1.0))
            #                             + (1-y) * tf.log(tf.clip_by_value(1-y_, 1e-10, 1.0)))

            # add l2 reg except bias anb BN variables.
            self.l2 = l2_reg_lambda * tf.reduce_sum(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if not ("noreg" in v.name or "bias" in v.name)])
            self.loss += self.l2

        # Accuracy computation is outside of this class.
        with tf.name_scope("metrics"):
            TP = tf.count_nonzero(self.input_y * self.y_pred, dtype=tf.float32)
            TN = tf.count_nonzero((self.input_y - 1) * (self.y_pred - 1), dtype=tf.float32)
            FP = tf.count_nonzero(self.y_pred * (self.input_y - 1), dtype=tf.float32)
            FN = tf.count_nonzero((self.y_pred - 1) * self.input_y, dtype=tf.float32)
            # tf.div like python2 division, tf.divide like python3
            self.cm = tf.confusion_matrix(self.input_y, self.y_pred, name="confusion_matrix")
            self.acc = tf.divide(TP + TN, TP + TN + FP + FN, name="accuracy")
            self.precision = tf.divide(TP, TP + FP, name="precision")
            self.recall = tf.divide(TP, TP + FN, name="recall")
            self.f1 = tf.divide(2 * self.precision * self.recall, self.precision + self.recall, name="F1_score")

    @staticmethod
    def bi_rnn(x, rnn_cell, hidden_units, num_layers, dropout, dynamic=True):
        # Prepare data shape to match `static_rnn` function requirements
        # current_batch_of_words does not correspond to a "sentence" of words, but [t_steps, batch_size, num_features]
        # Every word in a batch should correspond to a time t.
        # Unpacks the given dimension of a rank-`R` tensor into rank-`(R-1)` tensors.
        # sequence_length tensors of shape (batch_size, embedding_dim)
        if not dynamic:
            x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
        if rnn_cell.lower() == 'lstm':
            rnn_cell = tf.nn.rnn_cell.LSTMCell
        elif rnn_cell.lower() == 'gru':
            rnn_cell = tf.nn.rnn_cell.GRUCell
        # Define lstm cells with tensorflow
        with tf.variable_scope("fw"):
            # state(c, h), tf.nn.rnn_cell.BasicLSTMCell does not support gradient clipping, use tf.nn.rnn_cell.LSTMCell.
            fw_cells = [rnn_cell(hidden_units) for _ in range(num_layers)]
            fw_cells = tf.nn.rnn_cell.MultiRNNCell(cells=fw_cells, state_is_tuple=True)
            # how to do dropout efficiently ?
            fw_cells = tf.nn.rnn_cell.DropoutWrapper(
                fw_cells, input_keep_prob=1.0, output_keep_prob=dropout, state_keep_prob=1.0, variational_recurrent=False)

        with tf.variable_scope("bw"):
            bw_cells = [rnn_cell(hidden_units) for _ in range(num_layers)]
            bw_cells = tf.nn.rnn_cell.MultiRNNCell(cells=bw_cells, state_is_tuple=True)
            bw_cells = tf.nn.rnn_cell.DropoutWrapper(
                bw_cells, input_keep_prob=1.0, output_keep_prob=dropout, variational_recurrent=False)
        # A tuple (outputs, output_state_fw, output_state_bw)
        # outputs is a list of outputs (one for each input), depth-concatenated forward and backward outputs.
        # outputs, _, _ = tf.nn.static_bidirectional_rnn(fw_cells, bw_cells, x, dtype=tf.float32)

        # A tuple (outputs, output_states)
        # where: outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output Tensor
        # output_fw will be a Tensor shaped: [batch_size, max_time, cell_fw.output_size]
        # tf.concat(outputs, 2)
        # output_states: A tuple (output_state_fw, output_state_bw) containing the forward and the backward final states
        if dynamic:
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(fw_cells, bw_cells, x, dtype=tf.float32)
            outputs = tf.concat(outputs, 2)
            outputs = outputs[:, -1, :]
        else:
            outputs, _, _ = tf.nn.static_bidirectional_rnn(fw_cells, bw_cells, x, dtype=tf.float32)
            outputs = outputs[-1]  # take last hidden states
        return outputs

    @staticmethod
    def cnn(x, sequence_length, embedding_size, filter_sizes, num_filters):
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size, reuse=None):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.get_variable("W", filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
                b = tf.get_variable("bias", [num_filters], initializer=tf.constant_initializer(0.1))
                conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        return h_pool_flat


if __name__ == '__main__':
    TextRCNN(
        model_type="rcnn",
        sequence_length=15,
        embedding_size=64,
        vocab_size=1000,
        filter_sizes=[3,4,5],
        num_filters=32,
        rnn_cell="lstm",
        hidden_units=128,
        num_layers=2,
        pos_weight=1.0,
        weight_sharing=True,
        l2_reg_lambda=0.0,
        interaction="concat",
        word_embedding_type="rand")


