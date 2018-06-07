#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf


class SiameseNets(object):
    """
    A Siamese CNN/RNN based network for text similarity.
    Uses a character/word level embedding layer, followed by a {`BiLSTM`, `CNN`, `combine`} network
    and Energy Loss layer.
    """
    def __init__(self,
                 model_type,
                 sequence_length,
                 embedding_size,
                 word_embedding_type,
                 vocab_size,
                 filter_sizes,
                 num_filters,
                 rnn_cell,
                 hidden_units,
                 num_layers,
                 l2_reg_lambda,
                 pred_threshold,
                 energy_func,
                 loss_func='contrasive',
                 margin=0.0,
                 contrasive_loss_pos_weight=1.0,
                 dense_layer=False,
                 use_attention=False,
                 weight_sharing=True):
        self.input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x1")
        self.input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name="input_x2")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.seqlen1 = tf.cast(tf.reduce_sum(tf.sign(self.input_x1), 1), tf.int32)
        self.seqlen2 = tf.cast(tf.reduce_sum(tf.sign(self.input_x2), 1), tf.int32)
        # input word level dropout, data augmentation, invariance to small input change
        # self.shape = tf.shape(self.input_x1)
        # self.mask1 = tf.cast(tf.random_uniform(self.shape) > 0.1, tf.int32)
        # self.mask2 = tf.cast(tf.random_uniform(self.shape) > 0.1, tf.int32)
        # self.input_x1 = self.input_x1 * self.mask1
        # self.input_x2 = self.input_x2 * self.mask2
        with tf.variable_scope("embedding"):
            if word_embedding_type == "rand":
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    trainable=True, name="W")  # tf.truncated_normal()
                # self.W = tf.get_variable(name='W', shape=[vocab_size, embedding_size],
                # initializer=tf.random_uniform_initializer(-1, 1))
            else:
                trainable = False if word_embedding_type == "static" else True
                self.W = tf.Variable(
                    tf.constant(0.0, shape=[vocab_size, embedding_size]),
                    trainable=trainable, name="W")
                # embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
                # self.embedding_init = self.W.assign(embedding_placeholder)
            self.embedded_1 = tf.nn.embedding_lookup(self.W, self.input_x1)
            self.embedded_2 = tf.nn.embedding_lookup(self.W, self.input_x2)
            # Input embedding dropout. very sensitive to the dropout rate !
            self.embedded_1 = tf.nn.dropout(self.embedded_1, 0.7)
            self.embedded_2 = tf.nn.dropout(self.embedded_2, 0.7)
            self.embedded_1_expanded = tf.expand_dims(self.embedded_1, -1)  # shape(batch_size, seq_len, dim, 1)
            self.embedded_2_expanded = tf.expand_dims(self.embedded_2, -1)

        if weight_sharing:
            cnn_scope1, cnn_scope2, rnn_scope1, rnn_scope2 = "CNN", "CNN", "RNN", "RNN"
        else:
            cnn_scope1, cnn_scope2, rnn_scope1, rnn_scope2 = "CNN1", "CNN2", "RNN1", "RNN2"
        # shape(batch_size, num_filters*len(filters_sizes))  # very sparse !
        self.cnn_out1 = self.cnn(self.embedded_1_expanded,
            sequence_length, embedding_size, filter_sizes, num_filters, scope=cnn_scope1)
        self.cnn_out2 = self.cnn(self.embedded_2_expanded,
            sequence_length, embedding_size, filter_sizes, num_filters, scope=cnn_scope2)

        # shape(batch_size, 2*hidden_units)
        self.rnn_out1 = self.bi_rnn(self.embedded_1, rnn_cell, hidden_units, num_layers,
                                    self.seqlen1, False, use_attention, rnn_scope1)
        self.rnn_out2 = self.bi_rnn(self.embedded_2, rnn_cell, hidden_units, num_layers,
                                    self.seqlen2, False, use_attention, rnn_scope2)

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
                out_dim = self.out1.get_shape().as_list()[1]
                W1 = tf.get_variable("W1", shape=[out_dim, 128], initializer=tf.contrib.layers.xavier_initializer())
                b1 = tf.Variable(tf.constant(0.1, shape=[128]), name="b1")
                W2 = tf.get_variable("W2", shape=[out_dim, 128], initializer=tf.contrib.layers.xavier_initializer())
                b2 = tf.Variable(tf.constant(0.1, shape=[128]), name="b2")
                self.out1 = tf.nn.xw_plus_b(self.out1, W1, b1, name="out1")
                self.out2 = tf.nn.xw_plus_b(self.out2, W2, b2, name="out2")

        out1_norm = tf.sqrt(tf.reduce_sum(tf.square(self.out1), 1))
        # tf.norm(tensor, ord='euclidean', axis=None, keep_dims=False, name=None)
        out2_norm = tf.sqrt(tf.reduce_sum(tf.square(self.out2), 1))
        self.distance = tf.sqrt(tf.reduce_sum(tf.square(self.out1-self.out2), 1, keep_dims=False))
        # normalize euclidean distance, think as triangle, so dis range [0,1]
        self.distance = tf.div(self.distance, tf.add(out1_norm, out2_norm))
        self.sim_euc = tf.subtract(1.0, self.distance, name="euc")

        # self.sim = tf.reduce_sum(tf.multiply(self.out1, self.out2), 1) / tf.multiply(out1_norm, out2_norm)
        # # shape(batch_size,), if keep_dims=True shape(batch_size, 1)
        out1_norm = tf.nn.l2_normalize(self.out1, 1)  # output = x / sqrt(max(sum(x**2), epsilon))
        out2_norm = tf.nn.l2_normalize(self.out2, 1)
        self.sim_cos = tf.reduce_sum(tf.multiply(out1_norm, out2_norm), axis=1, name="cosine")
        # sim = exp(-||x1-x2||) range (0, 1]
        self.sim_ma = tf.exp(-tf.reduce_sum(tf.abs(self.out1-self.out2), 1), name="manhattan")

        if energy_func == 'euclidean':
            self.e = self.sim_euc
        elif energy_func == 'cosine':
            self.e = self.sim_cos
        elif energy_func == 'exp_manhattan':
            self.e = self.sim_ma
        elif energy_func == 'combine':
            w = tf.Variable(1, dtype=tf.float32)
            self.e = w*self.sim_euc + (1-w)*self.sim_cos
            # self.fc1 = tf.layers.dense(tf.concat(
            #     [tf.expand_dims(self.sim_euc, 1), tf.expand_dims(self.sim_cos, 1)], 1), 128, activation=tf.nn.relu, name='fc1')
            # self.e = tf.layers.dense(self.fc1, 1, activation=tf.nn.relu, name='score')

        with tf.name_scope("loss"):
            if loss_func == 'contrasive':
                self.loss = self.contrastive_loss(self.input_y, self.e, margin, pos_weight=contrasive_loss_pos_weight)
            elif loss_func == 'cross_entrophy':
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.e))
            # add l2 reg except bias anb BN variables.
            self.l2 = l2_reg_lambda * tf.reduce_sum(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if not ("noreg" in v.name or "bias" in v.name)])
            self.loss += self.l2
            if use_attention:
                self.loss += tf.reduce_mean(self.P)

        # Accuracy computation is outside of this class.
        with tf.name_scope("metrics"):
            self.y_pred = tf.cast(tf.greater(self.e, pred_threshold), dtype=tf.float32, name="y_pred")
            # self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_pred, self.input_y), tf.float32), name="accuracy")
            # TP = tf.count_nonzero(self.input_y * self.y_pred, dtype=tf.float32)
            # TN = tf.count_nonzero((self.input_y - 1) * (self.y_pred - 1), dtype=tf.float32)
            # FP = tf.count_nonzero(self.y_pred * (self.input_y - 1), dtype=tf.float32)
            # FN = tf.count_nonzero((self.y_pred - 1) * self.input_y, dtype=tf.float32)
            # # tf.div like python2 division, tf.divide like python3
            # self.acc = tf.divide(TP + TN, TP + TN + FP + FN, name="accuracy")
            # self.precision = tf.divide(TP, TP + FP, name="precision")
            # self.recall = tf.divide(TP, TP + FN, name="recall")
            # https://github.com/tensorflow/tensorflow/issues/15115
            self.cm = tf.confusion_matrix(self.input_y, self.y_pred, name="confusion_matrix")  # [[5036 1109] [842 882]]
            _, self.acc = tf.metrics.accuracy(self.input_y, self.y_pred)
            _, self.precision = tf.metrics.precision(self.input_y, self.y_pred, name='precision')
            _, self.recall = tf.metrics.recall(self.input_y, self.y_pred, name='recall')
            # tf.assert_equal(self.acc, self.acc_)
            self.f1 = tf.divide(2 * self.precision * self.recall, self.precision + self.recall, name="F1_score")

    def bi_rnn(self, x, rnn_cell, hidden_units, num_layers, sequence_length, dynamic, use_attention, scope):
        rnn = tf.nn.rnn_cell
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # scope.reuse_variables()  # or tf.get_variable_scope().reuse_variables()
            # current_batch_of_words does not correspond to a "sentence" of words
            # but [t_steps, batch_size, num_features]
            # Unpacks the given dimension of a rank-`R` tensor into rank-`(R-1)` tensors.
            # sequence_length list tensors of shape (batch_size, embedding_dim)
            if not dynamic:
                x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))  # `static_rnn` input
            if rnn_cell.lower() == 'lstm':
                rnn_cell = rnn.LSTMCell
            elif rnn_cell.lower() == 'gru':
                rnn_cell = rnn.GRUCell
            with tf.variable_scope("fw"):
                # state(c, h), tf.nn.rnn_cell.BasicLSTMCell does not support gradient clipping, use tf.nn.rnn_cell.LSTMCell.
                # fw_cells = [rnn_cell(hidden_units) for _ in range(num_layers)]
                fw_cells = []
                for _ in range(num_layers):
                    fw_cell = rnn_cell(hidden_units)
                    fw_cell = rnn.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob,
                                                            variational_recurrent=False, dtype=tf.float32)
                    fw_cells.append(fw_cell)
                fw_cells = rnn.MultiRNNCell(cells=fw_cells, state_is_tuple=True)
            with tf.variable_scope("bw"):
                bw_cells = []
                for _ in range(num_layers):
                    bw_cell = rnn_cell(hidden_units)
                    bw_cell = rnn.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob,
                                                 variational_recurrent=False, dtype=tf.float32)
                    bw_cells.append(bw_cell)
                bw_cells = rnn.MultiRNNCell(cells=bw_cells, state_is_tuple=True)

            if dynamic:
                # [batch_size, max_time, cell_fw.output_size]
                outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                    fw_cells, bw_cells, x, sequence_length=sequence_length, dtype=tf.float32)
                # outputs = tf.concat(outputs, 2)
                # outputs = outputs[:, -1, :]  # take last hidden states  (batch_size, 2*hidden_units)
                output_fw = output_states[0][0].h
                output_bw = output_states[0][1].h
                outputs = tf.concat([output_fw, output_bw], 1)
            else:
                # `static_rnn` Returns: A tuple (outputs, output_state_fw, output_state_bw)
                # outputs is a list of timestep outputs, depth-concatenated forward and backward outputs.
                outputs, _, _ = tf.nn.static_bidirectional_rnn(
                    fw_cells, bw_cells, x, dtype=tf.float32, sequence_length=sequence_length)
                # outputs = tf.reduce_mean(outputs, 0)  # average [batch_size, hidden_units] (mean pooling)
                # outputs = tf.reduce_max(outputs, axis=0)  # max pooling, bad result.
                if use_attention:
                    d_a = 300
                    r = 2
                    self.H = tf.transpose(tf.stack(outputs), perm=[1, 0, 2])  # (bs, seq_len, 2*hidden_units)
                    batch_size = tf.shape(self.input_y)[0]
                    initializer = tf.contrib.layers.xavier_initializer()
                    with tf.variable_scope("attention"):
                        # shape(W_s1) = d_a * 2u
                        self.W_s1 = tf.get_variable('W_s1', shape=[d_a, 2 * hidden_units], initializer=initializer)
                        # shape(W_s2) = r * d_a
                        self.W_s2 = tf.get_variable('W_s2', shape=[r, d_a], initializer=initializer)
                        # shape (d_a, 2u) --> shape(batch_size, d_a, 2u)
                        self.W_s1 = tf.tile(tf.expand_dims(self.W_s1, 0), [batch_size, 1, 1])
                        self.W_s2 = tf.tile(tf.expand_dims(self.W_s2, 0), [batch_size, 1, 1])
                        # attention matrix A = softmax(W_s2*tanh(W_s1*H^T)  shape(A) = batch_siz * r * n
                        self.H_T = tf.transpose(self.H, perm=[0, 2, 1], name="H_T")
                        self.A = tf.nn.softmax(tf.matmul(self.W_s2, tf.tanh(tf.matmul(self.W_s1, self.H_T)), name="A"))
                        # sentences embedding matrix M = AH  shape(M) = (batch_size, r, 2u)
                        self.M = tf.matmul(self.A, self.H, name="M")
                        outputs = tf.reshape(self.M, [batch_size, -1])

                    with tf.variable_scope("penalization"):
                        # penalization term: Frobenius norm square of matrix AA^T-I, ie. P = |AA^T-I|_F^2
                        A_T = tf.transpose(self.A, perm=[0, 2, 1], name="A_T")
                        I = tf.eye(r, r, batch_shape=[batch_size], name="I")
                        self.P = tf.square(tf.norm(tf.matmul(self.A, A_T) - I, axis=[-2, -1], ord='fro'), name="P")
                else:
                    # outputs = outputs[-1]  # take last hidden state [batch_size, hidden_units]
                    outputs = tf.transpose(tf.stack(outputs), [1, 0, 2])  # shape(batch_size, seq_len, hidden_units)
                    outputs = self.last_relevant(outputs, sequence_length)

        return outputs

    def cnn(self, x, sequence_length, embedding_size, filter_sizes, num_filters, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
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
                    pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                            strides=[1, 1, 1, 1], padding='VALID',name="pool")
                    pooled_outputs.append(pooled)
            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])  # very sparse !

            with tf.name_scope("dropout"):
                h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

            with tf.name_scope("output"):
                W = tf.get_variable("W", shape=[num_filters_total, 128],
                                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[128]), name="b")
                outputs = tf.nn.xw_plus_b(h_drop, W, b, name="outputs")
            # outputs = h_pool_flat
        return outputs

    def contrastive_loss(self, y, e, margin, pos_weight):
        # margin and pos_weight can directly influence P and R metrics.
        l_1 = pos_weight * tf.pow(1-e, 2)
        l_0 = tf.square(tf.maximum(e-margin, 0))
        loss = tf.reduce_mean(y * l_1 + (1 - y) * l_0)
        return loss

    @staticmethod
    def last_relevant(outputs, sequence_length):
        batch_size = tf.shape(outputs)[0]
        max_length = outputs.get_shape()[1]
        output_size = outputs.get_shape()[2]
        index = tf.range(0, batch_size) * max_length + (sequence_length - 1)
        flat = tf.reshape(outputs, [-1, output_size])
        return tf.gather(flat, index)

    @property
    def variables(self):
        # for v in tf.trainable_variables():
        #     print(v)
        return tf.global_variables()


if __name__ == '__main__':
    siamese = SiameseNets(
        model_type='rcnn',
        sequence_length=15,
        embedding_size=128,
        word_embedding_type='rand',
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
        margin=0.0,
        weight_sharing=False)
    # get all ops
    # print([node.name for node in tf.get_default_graph().as_graph_def().node])