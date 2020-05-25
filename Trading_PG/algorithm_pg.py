import numpy as np
import tensorflow as tf
from models import BaseRLTFModel


class AlgorithmPG(BaseRLTFModel):

    def __init__(self, session, env, a_space, s_space, **options):
        super(AlgorithmPG, self).__init__(session, env, a_space, s_space, **options)

        self.loss = .0

        self.a_buffer = []
        self.r_buffer = []
        self.s_buffer = []

        self._init_input()
        self._init_nn()
        self._init_op()
        self._init_saver()
        self._init_summary_writer()

    def _init_input(self):
        self.a = tf.placeholder(tf.int32, [None, ])
        self.r = tf.placeholder(tf.float32, [None, ])
        self.s = tf.placeholder(tf.float32, [None, self.s_space])
        self.s_next = tf.placeholder(tf.float32, [None, self.s_space])

    def _init_nn(self):
        # Initialize predict actor and critic.
        w_init, b_init = tf.random_normal_initializer(.0, .001), tf.constant_initializer(0.1)

        with tf.variable_scope('nn'):

            first_dense = tf.layers.dense(self.s,
                                          64,
                                          tf.nn.relu,
                                          kernel_initializer=w_init,
                                          bias_initializer=b_init)

            second_dense = tf.layers.dense(first_dense,
                                           64,
                                           tf.nn.relu,
                                           kernel_initializer=w_init,
                                           bias_initializer=b_init)

            # third_dense = tf.layers.dense(second_dense,
            #                               64,
            #                               tf.nn.relu,
            #                               kernel_initializer=w_init,
            #                               bias_initializer=b_init)

            action_prob = tf.layers.dense(second_dense,
                                          self.a_space,
                                          # tf.nn.tanh,
                                          kernel_initializer=w_init,
                                          bias_initializer=b_init)

            self.a_prob = action_prob
            self.a_s_prob = tf.nn.softmax(action_prob)

    def _init_op(self):
        with tf.variable_scope('loss'):
            # a_one_hot = tf.one_hot(self.a, self.a_space)
            # negative_cross_entropy = -tf.reduce_sum(tf.log(self.a_prob) * a_one_hot)
            negative_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.a_prob, labels=self.a)
            self.loss_fn = tf.reduce_mean(negative_cross_entropy * self.r)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learning_rate * 2).minimize(self.loss_fn)
        self.session.run(tf.global_variables_initializer())

    def run(self):
        if self.mode != 'train':
            self.restore()
        else:
            for episode in range(self.episodes):
                self.log_loss(episode)
                s = self.env.reset(self.mode)
                while True:
                    cas, a_index = self.predict(s)
                    # ca, a_index = self.predict(s)
                    # print(ca)
                    for ca in cas:
                        s_next, r, status, _ = self.env.forward(ca[0], ca[1])
                    self.save_transition(s, a_index, r, s_next)
                    s = s_next
                    if status == self.env.Done:
                        self.train()
                        self.env.trader.log_asset(episode)
                        break
                if self.enable_saver and episode % 5 == 0:
                    self.save(episode)
                    self.eval()
                    self.mode = 'train'
                    self.env.reset()

    def train(self):
        # Copy r_buffer
        r_buffer = self.r_buffer
        # Init r_tau
        r_tau = 0
        # Calculate r_tau
        for index in reversed(range(0, len(r_buffer))):
            r_tau = r_tau * self.gamma + r_buffer[index]
            self.r_buffer[index] = r_tau
        _, self.loss = self.session.run([self.train_op, self.loss_fn], {
            self.s: np.array(self.s_buffer),
            self.a: np.array(self.a_buffer),
            self.r: np.array(self.r_buffer)
        })
        self.s_buffer = []
        self.a_buffer = []
        self.r_buffer = []

    def predict(self, s):
        a = self.session.run(self.a_s_prob, {self.s: s})
        # print("******", s)
        # print("######", a)
        return self._get_stock_code_and_action(a, use_greedy=False, use_prob=True)
        # return self._get_stock_code_and_action(a)

    def save_transition(self, s, a, r, s_next):
        self.s_buffer.append(s.reshape((-1, )))
        self.a_buffer.append(a)
        self.r_buffer.append(r)

    def log_loss(self, episode):
        self.logger.warning("Episode: {0} | Actor Loss: {1:.2f}".format(episode, self.loss))
