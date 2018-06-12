#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/6/8
"""This module contains two kinds of encoders: CNNEncoder and RNNEncoder."""
import tensorflow as tf


class CNNEncoder(object):

    def __init__(self, sequence_length, embedding_dim, filter_sizes, num_filters):
        self._sequence_length = sequence_length
        self._embedding_dim = embedding_dim
        self._filter_sizes = filter_sizes
        self._num_filters = num_filters

    def forward(self, x, scope="CNN"):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Create a convolution + maxpool layer for each filter size
            x = tf.expand_dims(x, -1)   # shape(batch_size, seq_len, dim, 1)
            pooled_outputs = []
            for i, filter_size in enumerate(self._filter_sizes):
                with tf.variable_scope("conv-maxpool-%s" % filter_size, reuse=None):
                    # Convolution Layer
                    filter_shape = [filter_size, self._embedding_dim, 1, self._num_filters]
                    W = tf.get_variable("W", filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
                    b = tf.get_variable("bias", [self._num_filters], initializer=tf.constant_initializer(0.1))
                    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(h, ksize=[1, self._sequence_length - filter_size + 1, 1, 1],
                                            strides=[1, 1, 1, 1], padding='VALID', name="pool")
                    pooled_outputs.append(pooled)
            # Combine all the pooled features
            num_filters_total = self._num_filters * len(self._filter_sizes)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])  # very sparse !

            # very important, very sensitive to dropout rate 0.7 good!
            with tf.name_scope("dropout"):
                h_drop = tf.nn.dropout(h_pool_flat, 0.7)

            # very important, necessary
            with tf.name_scope("output"):
                W = tf.get_variable("W", shape=[num_filters_total, 128],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[128]), name="b")
                outputs = tf.nn.xw_plus_b(h_drop, W, b, name="outputs")
        return outputs


class RNNEncoder(object):

    def __init__(self, rnn_cell, hidden_units, num_layers, dropout_keep_prob, use_dynamic, use_attention):
        self._rnn_cell = rnn_cell
        self._hidden_units = hidden_units
        self._num_layers = num_layers
        self._dropout_keep_prob = dropout_keep_prob
        self._use_dynamic = use_dynamic
        self._use_attention = use_attention

    def forward(self, x, sequence_length=None, scope="RNN"):
        rnn = tf.nn.rnn_cell
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):  # initializer=tf.orthogonal_initializer(),
            # scope.reuse_variables()  # or tf.get_variable_scope().reuse_variables()
            # current_batch_of_words does not correspond to a "sentence" of words
            # but [t_steps, batch_size, num_features]
            # Unpacks the given dimension of a rank-`R` tensor into rank-`(R-1)` tensors.
            # sequence_length list tensors of shape (batch_size, embedding_dim)
            if not self._use_dynamic:
                x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))  # `static_rnn` input
            if self._rnn_cell.lower() == 'lstm':
                rnn_cell = rnn.LSTMCell
            elif self._rnn_cell.lower() == 'gru':
                rnn_cell = rnn.GRUCell
            elif self._rnn_cell.lower() == 'rnn':
                rnn_cell = rnn.RNNCell
            else:
                raise ValueError("Invalid rnn_cell type.")

            with tf.variable_scope("fw"):
                # state(c, h), tf.nn.rnn_cell.BasicLSTMCell does not support gradient clipping, use tf.nn.rnn_cell.LSTMCell.
                # fw_cells = [rnn_cell(hidden_units) for _ in range(num_layers)]
                fw_cells = []
                for _ in range(self._num_layers):
                    fw_cell = rnn_cell(self._hidden_units)
                    fw_cell = rnn.DropoutWrapper(fw_cell, output_keep_prob=self._dropout_keep_prob,
                                                 variational_recurrent=False, dtype=tf.float32)
                    fw_cells.append(fw_cell)
                fw_cells = rnn.MultiRNNCell(cells=fw_cells, state_is_tuple=True)
            with tf.variable_scope("bw"):
                bw_cells = []
                for _ in range(self._num_layers):
                    bw_cell = rnn_cell(self._hidden_units)
                    bw_cell = rnn.DropoutWrapper(bw_cell, output_keep_prob=self._dropout_keep_prob,
                                                 variational_recurrent=False, dtype=tf.float32)
                    bw_cells.append(bw_cell)
                bw_cells = rnn.MultiRNNCell(cells=bw_cells, state_is_tuple=True)

            if self._use_dynamic:
                # [batch_size, max_time, cell_fw.output_size]
                outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                    fw_cells, bw_cells, x, sequence_length=sequence_length, dtype=tf.float32)
                outputs = tf.concat(outputs, 2)
                # outputs = outputs[:, -1, :]  # take last hidden states  (batch_size, 2*hidden_units)
                # outputs = tf.concat([output_states[-1][0].h, output_states[-1][1].h], 1)
                outputs = self._last_relevant(outputs, sequence_length)
            else:
                # `static_rnn` Returns: A tuple (outputs, output_state_fw, output_state_bw)
                # outputs is a list of timestep outputs, depth-concatenated forward and backward outputs.
                outputs, state_fw, state_bw = tf.nn.static_bidirectional_rnn(
                    fw_cells, bw_cells, x, dtype=tf.float32, sequence_length=sequence_length)
                if self._use_attention:
                    d_a = 300
                    r = 2
                    self.H = tf.transpose(tf.stack(outputs), perm=[1, 0, 2])  # (bs, seq_len, 2*hidden_units)
                    batch_size = tf.shape(x)[0]
                    initializer = tf.contrib.layers.xavier_initializer()
                    with tf.variable_scope("attention"):
                        # shape(W_s1) = d_a * 2u
                        self.W_s1 = tf.get_variable('W_s1', shape=[d_a, 2 * self._hidden_units], initializer=initializer)
                        # shape(W_s2) = r * d_a
                        self.W_s2 = tf.get_variable('W_s2', shape=[r, d_a], initializer=initializer)
                        # shape (d_a, 2u) --> shape(batch_size, d_a, 2u)
                        self.W_s1 = tf.tile(tf.expand_dims(self.W_s1, 0), [batch_size, 1, 1])
                        self.W_s2 = tf.tile(tf.expand_dims(self.W_s2, 0), [batch_size, 1, 1])
                        # attention matrix A = softmax(W_s2*tanh(W_s1*H^T)  shape(A) = batch_siz * r * n
                        self.H_T = tf.transpose(self.H, perm=[0, 2, 1], name="H_T")
                        self.A = tf.nn.softmax(
                            tf.matmul(self.W_s2, tf.tanh(tf.matmul(self.W_s1, self.H_T)), name="A"))
                        # sentences embedding matrix M = AH  shape(M) = (batch_size, r, 2u)
                        self.M = tf.matmul(self.A, self.H, name="M")
                        outputs = tf.reshape(self.M, [batch_size, -1])

                    with tf.variable_scope("penalization"):
                        # penalization term: Frobenius norm square of matrix AA^T-I, ie. P = |AA^T-I|_F^2
                        A_T = tf.transpose(self.A, perm=[0, 2, 1], name="A_T")
                        I = tf.eye(r, r, batch_shape=[batch_size], name="I")
                        self.P = tf.square(tf.norm(tf.matmul(self.A, A_T) - I, axis=[-2, -1], ord='fro'), name="P")
                else:
                    outputs = tf.concat([state_fw[-1].h, state_bw[-1].h], 1)  # good
                    # outputs = tf.reduce_mean(outputs, 0)  # average [batch_size, hidden_units] (mean pooling)
                    # outputs = tf.reduce_max(outputs, axis=0)  # max pooling, bad result.
                    # outputs = outputs[-1]  # take last hidden state [batch_size, hidden_units]
                    # outputs = tf.transpose(tf.stack(outputs), [1, 0, 2])  # shape(batch_size, seq_len, hidden_units)
                    # outputs = self._last_relevant(outputs, sequence_length)
        return outputs

    @staticmethod
    def _last_relevant(outputs, sequence_length):
        """Deprecated"""
        batch_size = tf.shape(outputs)[0]
        max_length = outputs.get_shape()[1]
        output_size = outputs.get_shape()[2]
        index = tf.range(0, batch_size) * max_length + (sequence_length - 1)
        flat = tf.reshape(outputs, [-1, output_size])
        last_timesteps = tf.gather(flat, index)  # very slow
        # mask = tf.sign(index)
        # last_timesteps = tf.boolean_mask(flat, mask)
        # # Creating a vector of 0s and 1s that will specify what timesteps to choose.
        # partitions = tf.reduce_sum(tf.one_hot(index, tf.shape(flat)[0], dtype='int32'), 0)
        # # Selecting the elements we want to choose.
        # _, last_timesteps = tf.dynamic_partition(flat, partitions, 2)  # (batch_size, n_dim)
        # https://stackoverflow.com/questions/35892412/tensorflow-dense-gradient-explanation
        return last_timesteps

if __name__ == '__main__':
    x1 = tf.placeholder(tf.int32, [None, 20], name="input_x1")
    x2 = tf.placeholder(tf.int32, [None, 20], name="input_x2")
    cnn_encoder = CNNEncoder(
        sequence_length=20,
        embedding_dim=128,
        filter_sizes=[3,4,5],
        num_filters=100,
        )
    rnn_encoder = RNNEncoder(
        rnn_cell='lstm',
        hidden_units=100,
        num_layers=2,
        dropout_keep_prob=0.7,
        use_dynamic=False,
        use_attention=False,
        )



