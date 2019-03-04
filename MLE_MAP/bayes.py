"""
极大似然估计
"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as ds
# 生成数据
N = 1000
mu = 1
sigma = 0.5
print(mu, sigma)

X = np.random.uniform(-5,7,N)

def normal(X, mu=mu, sigma=sigma):
    y = 1/(np.sqrt(2 * np.pi) * sigma) * np.exp(-(X - mu)**2 / (2*sigma**2))
    return y

##
y = normal(X)
np.log(y).sum()
#
t_x = tf.placeholder(tf.float32, shape=None)
#
t_mu = tf.Variable(0.0, dtype=tf.float32)
t_sigma = tf.Variable(0.0, dtype=tf.float32)

def t_normal(X, mu=t_mu, sigma=t_sigma):
    y = tf.multiply(1/tf.multiply(tf.sqrt(2 * np.pi),sigma),
        tf.exp(-tf.square(X - mu) / (2*tf.square(sigma))))
    return y


cost = tf.reduce_mean(tf.log(t_normal(t_x)))

sess.run(cost,  {t_x: X})

# 优化
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_op = optimizer.minimize(-cost)

#
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for _ in range(500):
    sess.run(train_op, {t_x: X})
    print('cost: ',sess.run(cost,  {t_x: X}))
    # print('\n mu: {0}\t   sigma: {1}'.format(
    #         sess.run(t_mu,  {t_x: X}),
    #         sess.run(t_sigma, {t_x: X})))
