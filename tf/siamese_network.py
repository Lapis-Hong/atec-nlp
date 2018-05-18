#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf


class SiameseNets(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an character/word level embedding layer, followed by a {`BiLSTM`, `CNN`, `combine`} network
    and Energy Loss layer.
    """
    def __init__(self, model_type, sequence_length, embedding_size, vocab_size,
                 filter_sizes, num_filters,
                 rnn_cell, hidden_units, num_layers,
                 l2_reg_lambda, energy_func, pred_threshold=0.5,
                 loss_func='contrasive', contrasive_loss_pos_weight=1.0,
                 dense_layer=False, word_embedding_type="rand", weight_sharing=True):
        self.seqlen1 = tf.placeholder(tf.int32, [None])
        self.seqlen2 = tf.placeholder(tf.int32, [None])
        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.variable_scope("embedding"):
            if word_embedding_type == "rand":
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    trainable=True, name="W")  # # tf.truncated_normal()
                # self.W = tf.get_variable(name='W', shape=[vocab_size, embedding_size],
                # initializer=tf.random_uniform_initializer(-1, 1))
            elif word_embedding_type == "static":
                self.W = tf.Variable(
                    tf.constant(0.0, shape=[vocab_size, embedding_size]),
                    trainable=False, name="W")
                # embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
                # self.embedding_init = self.W.assign(embedding_placeholder)
            elif word_embedding_type == "non-static":
                self.W = tf.Variable(
                    tf.constant(0.0, shape=[vocab_size, embedding_size]),
                    trainable=True, name="W")

            self.embedded_chars1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            self.embedded_chars2 = tf.nn.embedding_lookup(self.W, self.input_x2)
            # Input dropout. data augmentation, invariance to small input change
            self.embedded_chars1 = tf.nn.dropout(self.embedded_chars1, 0.9)
            self.embedded_chars2 = tf.nn.dropout(self.embedded_chars2, 0.9)  # shape(batch_size, seq_len, dim)
            self.embedded_chars1_expanded = tf.expand_dims(self.embedded_chars1, -1)  # shape(batch_size, seq_len, dim, 1)
            self.embedded_chars2_expanded = tf.expand_dims(self.embedded_chars2, -1)

        if weight_sharing:
            with tf.variable_scope("CNN", reuse=tf.AUTO_REUSE):  # share cnn weights.
                # shape(batch_size, num_filters*len(filters_sizes))  # very sparse !
                self.cnn_out1 = self.cnn(self.embedded_chars1_expanded,
                                     sequence_length, embedding_size, filter_sizes, num_filters, self.dropout_keep_prob)
                # scope.reuse_variables()  # or tf.get_variable_scope().reuse_variables()
                self.cnn_out2 = self.cnn(self.embedded_chars2_expanded,
                                     sequence_length, embedding_size, filter_sizes, num_filters, self.dropout_keep_prob)

            with tf.variable_scope("RNN", reuse=tf.AUTO_REUSE):  # share rnn weights.
                # shape(batch_size, 2*hidden_units)
                self.rnn_out1 = self.bi_rnn(self.embedded_chars1, rnn_cell, hidden_units, num_layers,
                                            self.dropout_keep_prob, self.seqlen1)
                self.rnn_out2 = self.bi_rnn(self.embedded_chars2, rnn_cell, hidden_units, num_layers,
                                            self.dropout_keep_prob, self.seqlen2)
        else:
            with tf.variable_scope("CNN_1"):  # not share cnn weights.
                # shape(batch_size, num_filters*len(filters_sizes))
                self.cnn_out1 = self.cnn(self.embedded_chars1_expanded, sequence_length, embedding_size,
                                         filter_sizes, num_filters, self.dropout_keep_prob)
            with tf.variable_scope("CNN_2"):
                self.cnn_out2 = self.cnn(self.embedded_chars2_expanded, sequence_length, embedding_size,
                                         filter_sizes, num_filters, self.dropout_keep_prob)

            with tf.variable_scope("RNN_1"):  # share rnn weights.
                # shape(batch_size, 2*hidden_units)
                self.rnn_out1 = self.bi_rnn(self.embedded_chars1, rnn_cell, hidden_units, num_layers,
                                            self.dropout_keep_prob, self.seqlen1)
            with tf.variable_scope("RNN_2"):
                self.rnn_out2 = self.bi_rnn(self.embedded_chars2, rnn_cell, hidden_units, num_layers,
                                            self.dropout_keep_prob, self.seqlen2)

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

            if dense_layer:
                with tf.variable_scope("fc"):
                    W1 = tf.get_variable("W1", shape=[2*hidden_units, 128], initializer=tf.contrib.layers.xavier_initializer())
                    b1 = tf.Variable(tf.constant(0.1, shape=[128]), name="b1")
                    W2 = tf.get_variable("W2", shape=[2*hidden_units, 128], initializer=tf.contrib.layers.xavier_initializer())
                    b2 = tf.Variable(tf.constant(0.1, shape=[128]), name="b2")
                    self.out1 = tf.nn.xw_plus_b(self.out1, W1, b1, name="out1")
                    self.out2 = tf.nn.xw_plus_b(self.out2, W2, b2, name="out2")

                # sim = 1 - norm(x1-x2, 2) / (norm(x1, 2) + norm(x2, 2))
            # shape(batch_size,), if keep_dims=True shape(batch_size, 1)
            out1_norm = tf.sqrt(tf.reduce_sum(tf.square(self.out1), 1))
            out2_norm = tf.sqrt(tf.reduce_sum(tf.square(self.out2), 1))
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(self.out1-self.out2), 1, keep_dims=False))
            # normalize euclidean distance, think as triangle, so dis range [0,1]
            self.distance = tf.div(self.distance, tf.add(out1_norm, out2_norm))  # Incompatible shapes: [7869] vs. [7869,128]
            self.sim_euc = 1 - self.distance
            self.e = self.sim_euc

            # self.sim = tf.reduce_sum(tf.multiply(self.out1, self.out2), 1) / tf.multiply(out1_norm, out2_norm)
            out1_norm = tf.nn.l2_normalize(self.out1, 1)  # output = x / sqrt(max(sum(x**2), epsilon))
            out2_norm = tf.nn.l2_normalize(self.out2, 1)
            self.sim_cos = tf.reduce_sum(tf.multiply(out1_norm, out2_norm), axis=1, name="cosine")
            self.e = self.sim_cos
            # sim = exp(-||x1-x2||) range (0, 1]
            self.sim_ma = tf.exp(-tf.reduce_sum(tf.abs(self.out1-self.out2), 1))
            self.e = self.sim_ma

            if energy_func == 'euclidean':
                self.e = self.sim_euc
            elif energy_func == 'cosine':
                self.e = self.sim_cos
            elif energy_func == 'exp_manhattan':
                self.e = self.sim_ma

        with tf.name_scope("loss"):
            self.loss = self.contrastive_loss(self.input_y, self.e, pos_weight=contrasive_loss_pos_weight)
            # add l2 reg except bias anb BN variables.
            self.l2 = l2_reg_lambda * tf.reduce_sum(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if not ("noreg" in v.name or "bias" in v.name)])
            self.loss += self.l2

        # Accuracy computation is outside of this class.
        with tf.name_scope("metrics"):
            # for v in tf.trainable_variables():
            #     print(v)
            # for v in tf.global_variables():
            #     print(v)
            # tf.rint: Returns element-wise integer closest to x. auto threshold 0.5
            self.y_pred = tf.cast(tf.greater(self.e, pred_threshold), dtype=tf.float32, name="y_pred")
            # self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_pred, self.input_y), tf.float32), name="accuracy")
            TP = tf.count_nonzero(self.input_y * self.y_pred, dtype=tf.float32)
            TN = tf.count_nonzero((self.input_y - 1) * (self.y_pred - 1), dtype=tf.float32)
            FP = tf.count_nonzero(self.y_pred * (self.input_y - 1), dtype=tf.float32)
            FN = tf.count_nonzero((self.y_pred - 1) * self.input_y, dtype=tf.float32)
            # tf.div like python2 division, tf.divide like python3
            # self.cm = tf.confusion_matrix(self.input_y, self.y_pred, name="confusion_matrix")
            self.acc = tf.divide(TP + TN, TP + TN + FP + FN, name="accuracy")
            self.precision = tf.divide(TP, TP + FP, name="precision")
            self.recall = tf.divide(TP, TP + FN, name="recall")
            self.f1 = tf.divide(2 * self.precision * self.recall, self.precision + self.recall, name="F1_score")

    def bi_rnn(self, x, rnn_cell, hidden_units, num_layers, dropout, sequence_length, dynamic=False):
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

        if dynamic:
            # [batch_size, max_time, cell_fw.output_size]
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(fw_cells, bw_cells, x, sequence_length=sequence_length, dtype=tf.float32)
            # outputs = tf.concat(outputs, 2)
            # outputs = outputs[:, -1, :]  # take last hidden states  (batch_size, 2*hidden_units)
            output_fw = output_states[0][0].h
            output_bw = output_states[0][1].h
            outputs = tf.concat([output_fw, output_bw], 1)

        else:
            # Returns: A tuple (outputs, output_state_fw, output_state_bw)
            # outputs is a list of outputs (one for each input), depth-concatenated forward and backward outputs.
            outputs, _, _ = tf.nn.static_bidirectional_rnn(fw_cells, bw_cells, x, dtype=tf.float32)
            # outputs = tf.reduce_mean(outputs, 0)  # average [batch_size, hidden_units] (mean pooling)
            outputs = outputs[-1]  # take last hidden state [batch_size, hidden_units]
            # outputs = tf.reduce_max(outputs, axis=0)  # max pooling, bad result.
        return outputs

    @staticmethod
    def cnn(x, sequence_length, embedding_size, filter_sizes, num_filters, dropout):
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
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])  # very sparse !

        # Add dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, dropout)

        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, 128],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[128]), name="b")
            outputs = tf.nn.xw_plus_b(h_drop, W, b, name="outputs")
        # outputs = h_pool_flat
        return outputs

    @staticmethod
    def contrastive_loss(y, e, margin=0.0, pos_weight=1.0):
        # margin and pos_weight can directly influence P and R metrics.
        l_1 = pos_weight * tf.square(1-e)
        l_0 = tf.square(tf.maximum(e-margin, 0))
        loss = tf.reduce_mean(y * l_1 + (1 - y) * l_0)
        return loss


if __name__ == '__main__':
    siamese = SiameseNets(
        model_type='rcnn',
        sequence_length=15,
        embedding_size=128,
        vocab_size=1000,
        filter_sizes=[3,4,5],
        num_filters=100,
        rnn_cell='lstm',
        hidden_units=64,
        num_layers=3,
        dense_layer=False,
        l2_reg_lambda=1,
        pred_threshold=0.5,
        energy_func='euclidean',
        loss_func='contrasive',
        contrasive_loss_pos_weight=1.0,
        word_embedding_type='rand',
        weight_sharing=False)
    # get all ops
    # print([node.name for node in tf.get_default_graph().as_graph_def().node])