from algorithms.decision_tree.cart import CART
import numpy as np

# --------------------data-----------------------

N = 1000
num_feature = 5
X = np.random.uniform(0,10,[N,5])

X_add_one = np.column_stack((np.ones_like(X[:,0]), X))
true_theta = np.matrix([0.5,0.6,0.6,0.6,0.6,0.6]).transpose()
y = np.dot(X_add_one, true_theta) + np.random.normal(0,0.1,[N,1])


# --------------------test reg-----------------------
cart = CART(X,y)

cart.train_reg()
