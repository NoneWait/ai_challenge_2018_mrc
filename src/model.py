import tensorflow as tf
from func import cudnn_gru, native_gru, dot_attention, summ, dropout, cl_ptr

class Model(object):
    def __init__(self, config, batch, word_mat=None, trainable=True, opt=True):
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        self.c, self.q, self.alternatives, self.feat, self.y, self.qa_id = batch.get_next()
        self.is_train = tf.get_variable(
            "is_train", shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
            word_mat, dtype=tf.float32), trainable=True)
        # self.char_mat = tf.get_variable(
        #     "char_mat", initializer=tf.constant(char_mat, dtype=tf.float32))

        self.c_mask = tf.cast(self.c, tf.bool)
        self.q_mask = tf.cast(self.q, tf.bool)
        self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
        self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)

        if opt:
            N, CL = config.batch_size, config.char_limit
            self.c_maxlen = tf.reduce_max(self.c_len)
            self.q_maxlen = tf.reduce_max(self.q_len)
            self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen])
            self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen])
            self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
            self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])
            self.alternatives = tf.slice(self.alternatives, [0, 0], [N, 3])
            self.y = tf.slice(self.y, [0, 0], [N, 3])
            self.feat = tf.slice(self.feat, [0, 0], [N, self.q_maxlen])

        else:
            self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit

        self.ready()

        if trainable:
            self.lr = tf.get_variable(
                "lr", shape=[], dtype=tf.float32, trainable=False)
            self.opt = tf.train.AdadeltaOptimizer(
                learning_rate=self.lr, epsilon=1e-6)
            grads = self.opt.compute_gradients(self.loss)
            gradients, variables = zip(*grads)
            capped_grads, _ = tf.clip_by_global_norm(
                gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)

    def ready(self):
        config = self.config
        # N, PL, QL, d= config.batch_size, self.c_maxlen, self.q_maxlen, config.hidden,
        N, PL, QL, CL, d, dc, dg = config.batch_size, self.c_maxlen, self.q_maxlen, config.char_limit,\
                                   config.hidden, config.char_dim, config.char_hidden
        gru = cudnn_gru if config.use_cudnn else native_gru

        with tf.variable_scope("emb"):
            with tf.name_scope("word"):
                c_emb = tf.nn.embedding_lookup(self.word_mat, self.c)
                q_emb = tf.nn.embedding_lookup(self.word_mat, self.q)
                alter_emb = tf.nn.embedding_lookup(self.word_mat, self.alternatives)
                # [batch, 3, ?, emb_size]
            with tf.name_scope("feat"):
                 q_emb = tf.concat([q_emb, tf.expand_dims(self.feat, axis=2)], axis=2)

        with tf.variable_scope("encoding"):
            rnn = gru(num_layers=3, num_units=d, batch_size=N, input_size=c_emb.get_shape(
                    ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            # [batch, c_size, h]
            c = rnn(c_emb, seq_len=self.c_len)
        with tf.variable_scope("q_encoding"):
            rnn = gru(num_layers=3, num_units=d, batch_size=N, input_size=q_emb.get_shape(
                    ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            q = rnn(q_emb, seq_len=self.q_len)

        with tf.variable_scope("attention"):
            qc_att = dot_attention(c, q, mask=self.q_mask, hidden=d,
                                   keep_prob=config.keep_prob, is_train=self.is_train)
            rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=qc_att.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            att = rnn(qc_att, seq_len=self.c_len)

        with tf.variable_scope("match"):
            self_att = dot_attention(
                att, att, mask=self.c_mask, hidden=d, keep_prob=config.keep_prob, is_train=self.is_train)
            rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=self_att.get_shape(
            ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            match = rnn(self_att, seq_len=self.c_len)

        with tf.variable_scope("pointer"):
            cls = cl_ptr(batch=N, hidden=match.get_shape().as_list()[-1], keep_prob=config.ptr_keep_prob,
                         is_train=self.is_train)
            # [batch,h]
            res = cls(match, self.c_len)
            # [batch, 1, h]
            res = tf.expand_dims(res, 1)
        with tf.variable_scope("predict"):
            # [batch, 3, h]
            dense1 = tf.layers.dense(alter_emb, units=res.get_shape().as_list()[-1], activation=tf.nn.relu)
            logits = tf.matmul(res, dense1, transpose_b=True)
            logits = tf.squeeze(logits, 1)
            # logits = tf.matmul()
            #
            # dense1 = tf.layers.dense(res, units=80, activation=tf.nn.relu)
            # # dense1 = tf.reduce_mean(dense1, axis=1)
            # logits = tf.layers.dense(dense1, units=3, activation=tf.nn.tanh)
            scores = tf.nn.softmax(logits)
            self.yp = tf.argmax(scores, axis=1)

            # losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            #     logits=logits, labels=tf.stop_gradient(self.y)
            # )
            # self.loss = tf.reduce_mean(losses)
            pp = tf.slice(scores, [0, 0], [N, 1])
            self.loss = -tf.reduce_mean(tf.log(pp))
            # print("test ")

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step
