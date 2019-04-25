"""
利用极大似然估计对一元高斯分布进行参数估计

An example learned from  https://github.com/kyleclo/tensorflow-mle
"""

import numpy as np
import tensorflow as tf

TRUE_MU = 10.0
TRUE_SIGMA = 5.0
SAMPLE_SIZE = 100

INIT_MU_PARAMS = {'loc': 0.0, 'scale': 0.1}
INIT_PHI_PARAMS = {'loc': 1.0, 'scale': 0.1}
LEARNING_RATE = 0.01
MAX_ITER = 10000
TOL_PARAM, TOL_LOSS, TOL_GRAD = 1e-8, 1e-8, 1e-8
RANDOM_SEED = 0

MAX_CHARS = 15

# generate sample
np.random.seed(0)
x_obs = np.random.normal(loc=TRUE_MU, scale=TRUE_SIGMA, size=SAMPLE_SIZE)

# center and scale the data
# CENTER = x_obs.min()
# SCALE = x_obs.max() - x_obs.min()
# x_obs = (x_obs - CENTER) / SCALE

# tensor for data
x = tf.placeholder(dtype=tf.float32)

# tensors for parameters
np.random.seed(RANDOM_SEED)
mu = tf.Variable(initial_value=np.random.normal(**INIT_MU_PARAMS),
                 dtype=tf.float32)
phi = tf.Variable(initial_value=np.random.normal(**INIT_PHI_PARAMS),
                  dtype=tf.float32)
sigma = tf.square(phi)

# loss function
gaussian_dist = tf.contrib.distributions.Normal(loc=mu, scale=sigma)
log_prob = gaussian_dist.log_prob(value=x)
neg_log_likelihood = -1.0 * tf.reduce_sum(log_prob)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss=neg_log_likelihood)
sess = tf.Session()

sess.run(fetches=tf.global_variables_initializer())
obs_loss = sess.run(fetches=[neg_log_likelihood], feed_dict={x: x_obs})
for i in range(MAX_ITER):
    sess.run(fetches=train_op, feed_dict={x: x_obs})
    new_loss = sess.run(fetches=neg_log_likelihood, feed_dict={x: x_obs})
    if i % 100 == 0:
        new_mu, new_sigma = sess.run([mu,sigma])
        print(' i: {} | mu: {} | sigma{} |  loss:{}'
              .format(i,
                      new_mu,
                      new_sigma,
                      new_loss))
