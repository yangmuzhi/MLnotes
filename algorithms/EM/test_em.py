import tensorflow as tf

y = tf.constant(5)
x = tf.constant(2)


op = tf.divide(y,x)
sess = tf.Session()
sess.run(op)
