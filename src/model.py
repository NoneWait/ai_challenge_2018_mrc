import tensorflow as tf
from func import cudnn_lstm, native_lstm,cudnn_gru, native_gru, dot_attention, summ, dropout, cl_ptr,\
    fuse, dense

class Model(object):
    """
    实现了
    <<Multi-Granularity Hierarchical Attention Fusion Networks for Reading Comprehension and Question Answering>>
    Tips,些许改变，question中融入了选项信息，去除了question的to a single vector

    """
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
        # 设置梯度的最大范数max_grad_norm, gradient clipping方法，某种程度上起到正则化的效果
            capped_grads, _ = tf.clip_by_global_norm(
                gradients, config.grad_clip)
            self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.global_step)
        # if trainable:
        #     self.lr = tf.get_variable(
        #         "lr", shape=[], dtype=tf.float32, trainable=False)
        #     # 初始化一个优化器
        #     self.opt = tf.contrib.opt.AdaMaxOptimizer()
        #     #     learning_rate=self.lr)
        #     tf.train.AdamOptimizer()
        #     print(self.lr)
        #     # # 对变量计算loss的梯度
        #     # # 返回以(gradient, variable)组成的列表
        #     grads = self.opt.compute_gradients(self.loss)
        #     gradients, variables = zip(*grads)
        #     capped_grads, _ = tf.clip_by_global_norm(
        #         gradients, config.grad_clip)
        #     self.train_op = self.opt.apply_gradients(
        #         zip(capped_grads, variables), global_step=self.global_step)

    def ready(self):
        config = self.config
        # N, PL, QL, d= config.batch_size, self.c_maxlen, self.q_maxlen, config.hidden,
        N, PL, QL, CL, d, dc, dg = config.batch_size, self.c_maxlen, self.q_maxlen, config.char_limit,\
                                   config.hidden, config.char_dim, config.char_hidden
        gru = cudnn_gru if config.use_cudnn else native_gru

        # 词向量层
        with tf.variable_scope("emb"):
            with tf.name_scope("word"):
                c_emb = tf.nn.embedding_lookup(self.word_mat, self.c)
                q_emb = tf.nn.embedding_lookup(self.word_mat, self.q)
                alter_emb = tf.nn.embedding_lookup(self.word_mat, self.alternatives)
                # [batch, 3, ?, emb_size]
            # with tf.name_scope("feat"):
            #      q_emb = tf.concat([q_emb, tf.expand_dims(self.feat, axis=2)], axis=2)

        with tf.variable_scope("encoding"):
            rnn = gru(num_layers=3, num_units=d, batch_size=N, input_size=c_emb.get_shape(
                    ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            # [batch, c_size, h]
            c = rnn(c_emb, seq_len=self.c_len)
        with tf.variable_scope("q_encoding"):
            rnn = gru(num_layers=3, num_units=d, batch_size=N, input_size=q_emb.get_shape(
                    ).as_list()[-1], keep_prob=config.keep_prob, is_train=self.is_train)
            q = rnn(q_emb, seq_len=self.q_len)

        with tf.variable_scope("co-attention"):
            """
            pp => 融合了问题信息的文章语义表示
            qq => 融合了文章信息的问题语义表示
            """
            dim = q.get_shape().as_list()[-1]
            q_d = tf.nn.relu(tf.reshape(dense(q, dim, use_bias=False), [N, -1, dim]))
            c_d = tf.nn.relu(tf.reshape(dense(c, dim, use_bias=False, reuse=True), [N, -1, dim]))
            # [batch, q_len, c_len]
            s = tf.matmul(q_d, c_d, transpose_b=True)/(dim ** 0.5)

            # for Q'
            # 0 axis不变， 1 axis不变， 2 axis 变
            mask = tf.tile(tf.expand_dims(self.q_mask, axis=2), [1, 1, self.c_maxlen])
            s_mask = -1e30 * (1 - tf.cast(mask, tf.float32)) + s
            p2q = tf.nn.softmax(s_mask, axis=1)
            # Q' q prime [batch, c_len, h]
            # [batch, q_len, c_len] [batch, q_len, h]
            qp = tf.matmul(p2q, q_d, transpose_a=True)
            print("qp", qp.get_shape().as_list())
            # for P'
            mask = tf.tile(tf.expand_dims(self.c_mask, axis=2), [1, 1, self.q_maxlen])
            # [batch, c_len, q_len]
            s_mask = -1e30 * (1 - tf.cast(mask, tf.float32)) + tf.transpose(s, [0, 2, 1])
            q2p = tf.nn.softmax(s_mask, axis=1)
            # [batch, q_len] =>Q' q prime [batch, q_len, h]
            # [batch, c_len, q_len] [batch, c_len, h]
            pq = tf.matmul(q2p, c_d, transpose_a=True)
            print("pq", pq.get_shape().as_list())

            # http://wing.comp.nus.edu.sg/~antho/P/P16/P16-2022.pdf
            p_q_ = tf.concat([c, qp, c*qp, c-qp], axis=2)

            q_p_ = tf.concat([q, pq, q*pq, q-pq], axis=2)

            p_q_ = tf.reshape(p_q_, [-1, p_q_.get_shape().as_list()[-1]])
            q_p_ = tf.reshape(q_p_, [-1, q_p_.get_shape().as_list()[-1]])

            p_sigmoid = dense(p_q_, hidden=dim, use_bias=True, scope="f_p_g")
            pp = (tf.nn.sigmoid(p_sigmoid)) * (tf.nn.tanh(dense(p_q_, hidden=dim, use_bias=True, scope="f_p_m"))) - \
                tf.nn.sigmoid(p_sigmoid) * tf.reshape(c_d, [-1, c_d.get_shape().as_list()[-1]]) + \
                tf.reshape(c_d, [-1, c_d.get_shape().as_list()[-1]])
            # [batch, c_len, h]
            pp = tf.reshape(pp, [N, -1, pp.get_shape().as_list()[-1]])

            q_sigmoid = dense(q_p_, hidden=dim, use_bias=True, scope="f_q_g")
            qq = (tf.nn.sigmoid(q_sigmoid)) * (tf.nn.tanh(dense(q_p_, hidden=dim, use_bias=True, scope="f_q_m"))) - \
                tf.nn.sigmoid(q_sigmoid) * tf.reshape(q_d, [-1, q_d.get_shape().as_list()[-1]]) + \
                tf.reshape(q_d, [-1, q_d.get_shape().as_list()[-1]])
            # [batch, q_len, h]
            qq = tf.reshape(qq, [N, -1, qq.get_shape().as_list()[-1]])

        with tf.variable_scope("self-attention-passage"):
            rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=pp.get_shape().as_list()[-1],
                      keep_prob=config.keep_prob, is_train=self.is_train)
            # [batch,c_len,2*d]
            D = rnn(pp, seq_len=self.c_len)
            dim = D.get_shape().as_list()[-1]
            L = tf.nn.softmax(tf.matmul(dense(D, hidden=dim, use_bias=False, scope="L"), D, transpose_b=True), axis=1)
            D_ = tf.matmul(L, D)

            dd = tf.concat([D, D_, D*D_, D-D_], axis=2)
            dd = tf.reshape(dd, [-1, dd.get_shape().as_list()[-1]])
            # d_sigmoid = dense(dd, hidden=D.get_shape().as_list()[-1], use_bias=True, scope="f_d_g")
            d_sigmoid = dense(dd, hidden=dim, use_bias=True, scope="f_d_g")
            # D'= g(D,D_)m(D,D_)+D-g(D,D_)D
            dd = (tf.nn.sigmoid(d_sigmoid)) * (tf.nn.tanh(dense(dd, hidden=dim, use_bias=True, scope="f_d_m"))) - \
                 tf.nn.sigmoid(d_sigmoid) * tf.reshape(D, [-1, dim]) + \
                 tf.reshape(D, [-1, dim])
            dd = tf.reshape(dd, [N, -1, dd.get_shape().as_list()[-1]])
            with tf.variable_scope("bi-gru"):
                rnn_d = gru(num_layers=1, num_units=d, batch_size=N, input_size=dd.get_shape().as_list()[-1],
                        keep_prob=config.keep_prob, is_train=self.is_train)
                ddd = rnn_d(dd, seq_len=self.c_len)
            ddd = tf.reshape(ddd, [N, -1, ddd.get_shape().as_list()[-1]])
            sc = tf.nn.softmax(tf.reshape(dense(ddd, hidden=1, use_bias=False, scope="dv"), [N, -1, 1]), axis=1)
            # [batch, 1, 2*d]
            dv = tf.matmul(sc, ddd, transpose_a=True)

        with tf.variable_scope("self_attention_question"):
            rnn = gru(num_layers=1, num_units=d, batch_size=N, input_size=qq.get_shape().as_list()[-1],
                     keep_prob=config.keep_prob, is_train=self.is_train)
            # [batch, q_len, 2*d]
            qqq = rnn(qq, seq_len=self.q_len)

            # scores = tf.reshape(dense(qqq, hidden=1, use_bias=False), [N, -1, 1])
            # # axis=1 => 面向列
            # s_f = tf.nn.softmax(scores, axis=1)
            # # [batch,q_len,1] [batch, q_len, 2*d] => [batch,1,2*d]
            # qv = tf.matmul(s_f, qqq, transpose_a=True)

        with tf.variable_scope("Q2O_attention"):
            """
            question-to-option：结合问题信息的观点表示
            """
            dim = qqq.get_shape().as_list()[-1]
            # question
            qq_d = tf.nn.relu(tf.reshape(dense(qqq, dim, use_bias=True, scope="qqq_d"), [N, -1, dim]))
            # alter
            alter_d = tf.nn.relu(tf.reshape(dense(alter_emb, dim, use_bias=True, scope="alter_d"), [N, 3, dim]))
            # [batch, q_len, 3]
            s_a = tf.matmul(qq_d, alter_d, transpose_b=True)/(dim ** 0.5)
            # for Q'
            # 0 axis不变， 1 axis不变， 2 axis 变
            mask_a = tf.tile(tf.expand_dims(self.q_mask, axis=2), [1, 1, 3])
            s_mask_a = -1e30 * (1 - tf.cast(mask_a, tf.float32)) + s_a
            # alter to question
            a2q = tf.nn.softmax(s_mask_a, axis=1)
            # Q' q prime [batch, 3, h]
            # [batch, q_len, 3] [batch, q_len, h] => [batch,3,h]
            qa = tf.matmul(a2q, qq_d, transpose_a=True)
            print("qa", qa.get_shape().as_list())

        # with tf.variable_scope("O2P_attention"):
        #     """
        #     利用包含问题信息选项qa,summarize the evidence ddd into a fixed-size vector
        #     """
        #     # ddd [batch,c_len,h] =>passage
        #     # qa [batch, 3, h] =>options
        #     dim = ddd.get_shape().as_list()[-1]
        #     print("ddd", ddd.get_shape())
        #     #
        #     ddd_d = tf.nn.relu(tf.reshape(dense(ddd, dim, use_bias=False, scope="ddd_d"), [N, -1, dim]))
        #     print("ddd_", ddd_d.get_shape())
        #     # [batch, 3 , h]
        #     qa_dd = tf.nn.relu(tf.reshape(dense(qa, dim, use_bias=False, scope="qa_d"), [N, 3, dim]))
        #     # [batch, 3, c_len]
        #     s = tf.matmul(ddd_d, qa_dd, transpose_b=True) / (dim ** 0.5)
        #     # for Q'
        #     # 0 axis不变， 1 axis不变， 2 axis 变
        #     mask = tf.tile(tf.expand_dims(self.c_mask, axis=2), [1, 1, 3])
        #     s_mask = -1e30 * (1 - tf.cast(mask, tf.float32)) + s
        #     # alter to d
        #     # [batch, c_len, 3]
        #     # s_mask = tf.transpose(s_mask, [0, 2, 1])
        #     a2d = tf.nn.softmax(s_mask, axis=1)
        #     # [batch,c_len,3], [batch, c_len, h] => [batch, 3, h]
        #     ad = tf.matmul(a2d, ddd_d)
        #     # [batch,1,h]
        #     ad = tf.expand_dims(tf.reduce_mean(ad, axis=1), axis=1)

        # with tf.variable_scope("OC_attention"):
        #     # [batch,3,h] [batch,3,h]
        #     dim = qa.get_shape().as_list()[-1]
        #     qa_d = tf.nn.relu(tf.reshape(dense(qa, dim, use_bias=True, scope="oc_qa")), [N, 3, dim])
        #     # [3,3]
        #     s = tf.matmul(qa_d, qa_d, transpose_b=True)
        #     mask = tf.constant([], shape=[N, ])

        with tf.variable_scope("predict"):
            # [batch, 3, h]
            # dense1 = tf.layers.dense(alter_emb, units=dv.get_shape().as_list()[-1], activation=tf.nn.relu)
            # rP*W
            dense1 = tf.nn.relu(dense(dv, hidden=qa.get_shape().as_list()[-1], use_bias=False))
            # [batch, 1, h] ,[batch, h, 3] => [batch,1,3]
            logits = tf.matmul(dense1, qa, transpose_b=True)
            # [batch, 3]
            logits = tf.squeeze(logits, 1)
            scores = tf.nn.softmax(logits, axis=1)
            self.yp = tf.argmax(scores, axis=1)
            pp = tf.slice(scores, [0, 0], [N, 1])
            self.loss = -tf.reduce_mean(tf.log(pp))
            # print("test ")

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step