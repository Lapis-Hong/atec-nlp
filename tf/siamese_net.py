#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/6/10
"""Siamese Similarity Network regard this task as a sentence similarity problem;
   Siamese Classification Network regard this task as a text classification problem.

References:
    Learning Text Similarity with Siamese Recurrent Networks, 2016
    Siamese Recurrent Architectures for Learning Sentence Similarity, 2016
"""
import tensorflow as tf


class SiameseNets(object):
    """Siamese base nets, input embedding and encoder layer output. """
    def __init__(self, input_x1, input_x2, word_embedding_type, vocab_size, embedding_size,
                 encoder_type, cnn_encoder, rnn_encoder, dense_layer, weight_sharing):
        """
        Args:
            cnn_encoder: instance of CNNEncoder
            rnn_encoder: instance of RNNEncoder
        """
        # input word level dropout, data augmentation, invariance to small input change
        # self.shape = tf.shape(self.input_x1)
        # self.mask1 = tf.cast(tf.random_uniform(self.shape) > 0.1, tf.int32)
        # self.mask2 = tf.cast(tf.random_uniform(self.shape) > 0.1, tf.int32)
        # self.input_x1 = self.input_x1 * self.mask1
        # self.input_x2 = self.input_x2 * self.mask2
        self._encoder_type = encoder_type
        self._rnn_encoder = rnn_encoder
        seqlen1 = tf.cast(tf.reduce_sum(tf.sign(input_x1), 1), tf.int32)
        seqlen2 = tf.cast(tf.reduce_sum(tf.sign(input_x2), 1), tf.int32)
        assert word_embedding_type in {'rand', 'static', 'non-static'}, 'Invalid word embedding type'
        with tf.variable_scope("embedding"):
            if word_embedding_type == "rand":
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    trainable=True, name="W")  # tf.truncated_normal()
            else:
                trainable = False if word_embedding_type == "static" else True
                self.W = tf.Variable(
                    tf.constant(0.0, shape=[vocab_size, embedding_size]),
                    trainable=trainable, name="W")
                # embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
                # self.embedding_init = self.W.assign(embedding_placeholder)
            self.embedded_1 = tf.nn.embedding_lookup(self.W, input_x1)
            self.embedded_2 = tf.nn.embedding_lookup(self.W, input_x2)
            # Input embedding dropout. very sensitive to the dropout rate !
            self.embedded_1 = tf.nn.dropout(self.embedded_1, 0.7)
            self.embedded_2 = tf.nn.dropout(self.embedded_2, 0.7)
        if weight_sharing:
            cnn_scope1, cnn_scope2, rnn_scope1, rnn_scope2 = "CNN", "CNN", "RNN", "RNN"
        else:
            cnn_scope1, cnn_scope2, rnn_scope1, rnn_scope2 = "CNN1", "CNN2", "RNN1", "RNN2"
        if encoder_type.lower() == 'cnn':
            self.out1 = cnn_encoder.forward(self.embedded_1, cnn_scope1)
            self.out2 = cnn_encoder.forward(self.embedded_2, cnn_scope2)
        elif encoder_type.lower() == 'rnn':
            self.out1 = rnn_encoder.forward(self.embedded_1, seqlen1, rnn_scope1)
            self.out2 = rnn_encoder.forward(self.embedded_2, seqlen2, rnn_scope2)
        elif encoder_type.lower() == 'rcnn':
            cnn_out1 = cnn_encoder.forward(self.embedded_1, cnn_scope1)
            cnn_out2 = cnn_encoder.forward(self.embedded_2, cnn_scope2)
            rnn_out1 = rnn_encoder.forward(self.embedded_1, rnn_scope1, seqlen1)
            rnn_out2 = rnn_encoder.forward(self.embedded_2, rnn_scope2, seqlen2)
            self.out1 = tf.concat([cnn_out1, rnn_out1], axis=1)
            self.out2 = tf.concat([cnn_out2, rnn_out2], axis=1)
        else:
            raise ValueError("Invalid encoder type.")

        if dense_layer:
            with tf.variable_scope("fc"):
                out_dim = self.out1.get_shape().as_list()[1]
                W1 = tf.get_variable("W1", shape=[out_dim, 128], initializer=tf.contrib.layers.xavier_initializer())
                b1 = tf.Variable(tf.constant(0.1, shape=[128]), name="b1")
                W2 = tf.get_variable("W2", shape=[out_dim, 128], initializer=tf.contrib.layers.xavier_initializer())
                b2 = tf.Variable(tf.constant(0.1, shape=[128]), name="b2")
                self.out1 = tf.nn.xw_plus_b(self.out1, W1, b1, name="out1")
                self.out2 = tf.nn.xw_plus_b(self.out2, W2, b2, name="out2")

    @property
    def variables(self):
        # for v in tf.trainable_variables():
        #     print(v)
        return tf.global_variables()


class SiameseSimilarityNets(SiameseNets):
    """A siamese based deep network for text similarity.
    Use a character/word level embedding layer, followed by a {`BiLSTM`, `CNN`, `combine`} encoder layer,
    then use euclidean distance/cosine/manhattan distance to measure similarity"""
    def __init__(self, input_x1, input_x2, input_y,
                 word_embedding_type, vocab_size, embedding_size,
                 encoder_type, cnn_encoder, rnn_encoder, dense_layer,
                 l2_reg_lambda, pred_threshold, energy_func, loss_func='contrasive',
                 margin=0.0, contrasive_loss_pos_weight=1.0, weight_sharing=True):
        self.input_y = input_y
        self._l2_reg_lambda = l2_reg_lambda
        self._pred_threshold = pred_threshold
        self._energy_func = energy_func
        self._loss_func = loss_func
        self._margin = margin
        self._contrastive_loss_pos_weight = contrasive_loss_pos_weight
        super(SiameseSimilarityNets, self).__init__(
            input_x1, input_x2, word_embedding_type, vocab_size, embedding_size,
            encoder_type, cnn_encoder, rnn_encoder, dense_layer, weight_sharing)

    def forward(self):
        # out1_norm = tf.sqrt(tf.reduce_sum(tf.square(self.out1), 1))
        # out2_norm = tf.sqrt(tf.reduce_sum(tf.square(self.out2), 1))
        # self.distance = tf.sqrt(tf.reduce_sum(tf.square(self.out1 - self.out2), 1, keepdims=False))
        distance = tf.norm(self.out1-self.out2, ord='euclidean', axis=1, keepdims=False, name='euc-distance')
        distance = tf.div(distance, tf.add(tf.norm(self.out1, 2, axis=1), tf.norm(self.out2, 2, axis=1)))
        self.sim_euc = tf.subtract(1.0, distance, name="euc")

        # self.sim = tf.reduce_sum(tf.multiply(self.out1, self.out2), 1) / tf.multiply(out1_norm, out2_norm)
        out1_norm = tf.nn.l2_normalize(self.out1, 1)  # output = x / sqrt(max(sum(x**2), epsilon))
        out2_norm = tf.nn.l2_normalize(self.out2, 1)
        self.sim_cos = tf.reduce_sum(tf.multiply(out1_norm, out2_norm), axis=1, name="cosine")
        # sim = exp(-||x1-x2||) range (0, 1]
        # self.sim_ma = tf.exp(-tf.reduce_sum(tf.abs(self.out1 - self.out2), 1), name="manhattan")
        self.sim_ma = tf.exp(-tf.norm(self.out1-self.out2, 1, 1), name="manhattan")

        if self._energy_func == 'euclidean':
            self.sim = self.sim_euc
        elif self._energy_func == 'cosine':
            self.sim = self.sim_cos
        elif self._energy_func == 'exp_manhattan':
            self.sim = self.sim_ma
        elif self._energy_func == 'combine':
            w = tf.Variable(1, dtype=tf.float32)
            self.sim = w * self.sim_euc + (1 - w) * self.sim_cos
        else:
            raise ValueError("Invalid energy function name.")
        self.y_pred = tf.cast(tf.greater(self.sim, self._pred_threshold), dtype=tf.float32, name="y_pred")

        with tf.name_scope("loss"):
            if self._loss_func == 'contrasive':
                self.loss = self.contrastive_loss(self.input_y, self.sim)
            elif self._loss_func == 'cross_entrophy':
                self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y, logits=self.sim))
            # add l2 reg except bias anb BN variables.
            self.l2 = self._l2_reg_lambda * tf.reduce_sum(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() if not ("noreg" in v.name or "bias" in v.name)])
            self.loss += self.l2
            if self._encoder_type != 'cnn' and self._rnn_encoder._use_attention:
                self.loss += tf.reduce_mean(self._rnn_encoder.encoder.P)

        # Accuracy computation is outside of this class.
        # self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.y_pred, self.input_y), tf.float32), name="accuracy")
        TP = tf.count_nonzero(self.input_y * self.y_pred, dtype=tf.float32)
        TN = tf.count_nonzero((self.input_y - 1) * (self.y_pred - 1), dtype=tf.float32)
        FP = tf.count_nonzero(self.y_pred * (self.input_y - 1), dtype=tf.float32)
        FN = tf.count_nonzero((self.y_pred - 1) * self.input_y, dtype=tf.float32)
        # tf.div like python2 division, tf.divide like python3
        self.acc = tf.divide(TP + TN, TP + TN + FP + FN, name="accuracy")
        self.precision = tf.divide(TP, TP + FP, name="precision")
        self.recall = tf.divide(TP, TP + FN, name="recall")
        self.cm = tf.confusion_matrix(self.input_y, self.y_pred, name="confusion_matrix")
        # tf.assert_equal(self.acc, self.acc_)
        # https://github.com/tensorflow/tensorflow/issues/15115, be careful!
        # _, self.acc = tf.metrics.accuracy(self.input_y, self.y_pred)
        # _, self.precision = tf.metrics.precision(self.input_y, self.y_pred, name='precision')
        # _, self.recall = tf.metrics.recall(self.input_y, self.y_pred, name='recall')
        self.f1 = tf.divide(2 * self.precision * self.recall, self.precision + self.recall, name="F1_score")

    def contrastive_loss(self, y, e):
        # margin and pos_weight can directly influence P and R metrics.
        l_1 = self._contrastive_loss_pos_weight * tf.pow(1-e, 2)
        l_0 = tf.square(tf.maximum(e-self._margin, 0))
        loss = tf.reduce_mean(y * l_1 + (1 - y) * l_0)
        return loss


class SiameseClassificationNets(SiameseNets):
    """A Siamese based deep network for text similarity.
    Uses  character/word level embedding layer, followed by a {`BiLSTM`, `CNN`, `combine`} encoder layer,
    then use multiply/concat interaction to feed for classification layers.
    """
    def __init__(self, input_x1, input_x2, input_y,
                 word_embedding_type, vocab_size, embedding_size,
                 encoder_type, cnn_encoder, rnn_encoder, dense_layer,
                 l2_reg_lambda, interaction="multiply", weight_sharing=True):
        self.input_y = input_y
        self._l2_reg_lambda = l2_reg_lambda
        self._interaction = interaction
        super(SiameseClassificationNets, self).__init__(
            input_x1, input_x2, word_embedding_type, vocab_size, embedding_size,
            encoder_type, cnn_encoder, rnn_encoder, dense_layer, weight_sharing)

    def forward(self):
        if self._interaction == 'concat':
            self.out = tf.concat([self.out1, self.out2], axis=1, name="out")
        elif self._interaction == 'multiply':
            self.out = tf.multiply(self.out1, self.out2, name="out")
        fc = tf.layers.dense(self.out, 128, name='fc1', activation=tf.nn.relu)
        # self.scores = tf.layers.dense(self.fc, 1, activation=tf.nn.sigmoid)
        self.logits = tf.layers.dense(fc, 2, name='fc2')
        # self.y_pred = tf.round(tf.nn.sigmoid(self.logits), name="predictions")  # pred class
        self.y_pred = tf.cast(tf.argmax(tf.nn.sigmoid(self.logits), 1, name="predictions"), tf.float32)

        with tf.name_scope("loss"):
            # [batch_size, num_classes]
            y = tf.one_hot(tf.cast(self.input_y, tf.int32), 2)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=y)
            self.loss = tf.reduce_mean(cross_entropy)
            # self.loss = tf.losses.sigmoid_cross_entropy(logits=self.logits, multi_class_labels=y)

            # y = self.input_y
            # y_ = self.scores
            # self.loss = -tf.reduce_mean(pos_weight * y * tf.log(tf.clip_by_value(y_, 1e-10, 1.0))
            #                             + (1-y) * tf.log(tf.clip_by_value(1-y_, 1e-10, 1.0)))
            # add l2 reg except bias anb BN variables.
            self.l2 = self._l2_reg_lambda * tf.reduce_sum(
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


if __name__ == '__main__':
    from encoder import CNNEncoder, RNNEncoder
    x1 = tf.placeholder(tf.int32, [None, 20], name="input_x1")
    x2 = tf.placeholder(tf.int32, [None, 20], name="input_x2")
    y = tf.placeholder(tf.float32, [None], name="input_y")
    cnn_encoder = CNNEncoder(
        sequence_length=20,
        embedding_dim=128,
        filter_sizes=[3, 4, 5],
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
    model1 = SiameseSimilarityNets(
        input_x1=x1,
        input_x2=x2,
        input_y=y,
        word_embedding_type='rand',
        vocab_size=10000,
        embedding_size=128,
        encoder_type='cnn',
        cnn_encoder=cnn_encoder,
        rnn_encoder=rnn_encoder,
        dense_layer=False,
        l2_reg_lambda=0,
        pred_threshold=0.5,
        energy_func='cosine',
        loss_func='contrasive',
        margin=0.0,
        contrasive_loss_pos_weight=1.0,
        weight_sharing=True)
    model2 = SiameseClassificationNets(
        input_x1=x1,
        input_x2=x2,
        input_y=y,
        word_embedding_type='rand',
        vocab_size=10000,
        embedding_size=128,
        encoder_type='cnn',
        cnn_encoder=cnn_encoder,
        rnn_encoder=rnn_encoder,
        dense_layer=False,
        l2_reg_lambda=0,
        interaction='multiply',
        weight_sharing=True
    )