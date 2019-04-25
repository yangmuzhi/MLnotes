"""
最小二乘法--线性回归
"""
import numpy as np
import tensorflow as tf

class linear_reg_ls:

    def __init__(self, X, y):

        self.data = np.matrix(X)
        self.data = np.column_stack((np.ones_like(X[:,0]), X))
        self.y = y
        self.theta = None

    def matrix_inference(self):
        """
        X * theta = y
        X^T * X * theta = X^T * y
        theta = (X^T * X)^(-1) * X^T * y
        """
        _inv = np.linalg.inv(np.dot(self.data.transpose(), self.data))
        self.theta = np.dot(np.dot(_inv, self.data.transpose()), self.y)
        return self.theta

    def _tensor_init(self, x_shape, y_shape, learning_rate):
        self.t_data = tf.placeholder(shape=[None,x_shape[1]],dtype=tf.float32)
        self.t_y = tf.placeholder(shape=[None,y_shape[1]],dtype=tf.float32)
        self.t_theta = tf.Variable(initial_value=np.random.normal(0,0.1,[x_shape[1],1]),dtype=tf.float32)
        self.loss = tf.reduce_mean(tf.squared_difference(
                                    tf.matmul(self.t_data, self.t_theta), self.t_y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.train_op = self.optimizer.minimize(loss=self.loss)
        self.sess = tf.Session()
        self.sess.run(fetches=tf.global_variables_initializer())

    def inference(self, learning_rate, epochs=10):
        self._tensor_init(self.data.shape, self.y.shape, learning_rate=learning_rate)
        for i in range(epochs):
            self.sess.run(fetches=[self.train_op], feed_dict={
                                    self.t_data:self.data, self.t_y:self.y})
            if i % 10 == 0:
                print("loss:", self.sess.run(fetches=self.loss, feed_dict={
                                        self.t_data:self.data, self.t_y:self.y}))
                print("theta:", self.sess.run(fetches=self.t_theta))
