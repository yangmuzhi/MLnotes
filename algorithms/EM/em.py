"""

EM 框架

X: data
Z: hidden var
theta: params

"""

import tensorflow as tf
import numpy as np


class EM:

    def __init__(self, X):
        """应该定义好 model 里的tensor"""
        self.X = X

    def model(self):
        """this should be override"""
        K = 3
        self.mu = tf.Variable(initial_value=np.random.normal(0,0.1,[K,1]),dtype=tf.float32)
        self.pi = tf.Variable(initial_value=np.random.normal(0,0.1,[K,1]),dtype=tf.float32)
        self.sigma =  tf.Variable(initial_value=np.random.normal(1,0.1,1),dtype=tf.float32)
        self.log_P = tf.divide(- tf.squared_difference(self.X, self.mu) / 2 / self.sigma)
        self.q_i = tf.multiply(tf.exp(tf.divide(- tf.squared_difference(self.X,
                    self.mu) / 2 / self.sigma)), self.pi )
        self.q = tf.divide(self.q_i, tf.reduce_sum(self.q_i))

    def _E_step(self):

        return q

    def _M_step(self, max_step=10):
        # argmax theta
        for i in range(max_step):
            _ = self.sess.run(self.Q, feed_dict={})
        theta = self.sess.run(self.theta, feed_dict={})
        return theta

    def _Q_func(self):
        # compute q_ and define Q: E[p(theta_{t-1}) * log(theta)]
        q = self.sess.run(self.q, feed_dict={})
        self.Q = tf.reduce_sum(q * tf.log(self.log_Ps))

        return Q

    def inference(self, episode=100):
        for i in range(episode):
            self._E_step()
            self._M_step()

    def predict(self, X):
        return
