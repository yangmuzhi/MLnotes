from algorithms.linear_models.linear_reg_ls import linear_reg_ls
import numpy as np
import tensorflow as tf

#-------------------------data-----------------------------------
N = 1000
num_feature = 5
X = np.random.uniform(0,10,[N,5])

X_add_one = np.column_stack((np.ones_like(X[:,0]), X))
true_theta = np.matrix([0.5,0.6,0.6,0.6,0.6,0.6]).transpose()
y = np.dot(X_add_one, true_theta) + np.random.normal(0,0.1,[N,1])


#-------------------------test model 最小二乘-----------------------------------

# test matrix inference
lm = linear_reg_ls(X,y)
theta = lm.matrix_inference()
print(theta)

# test  mle inference

lm.inference(learning_rate=0.001,epochs=10000)
