from algorithms.tree_based_model.cart import CART
from algorithms.tree_based_model.cart import Node, Tree
import numpy as np
import scipy.optimize
# import pysnooper
# --------------------data-----------------------

N = 1000
num_feature = 5
X = np.random.uniform(0,10,[N,5])

X_add_one = np.column_stack(( X, X**2 ))
true_theta = np.matrix(np.random.randint(1,5,X_add_one.shape[1])).transpose()
y = np.dot(X_add_one, true_theta) + np.random.normal(0,0.1,[N,1])

X = X_add_one
# --------------------test Node tree-----------------------
root = Node(True)
# tree = Tree(max_depth=3, debug=True)
tree = Tree(max_depth=3)

tree.split(tree.root,x_data=X_add_one,y=y,depth=0)

x = X[0]
x
tree.reg_lookup(x)
tree.root.left.y_value.mean()



# --------------------test scipy-----------------------


# --------------------test reg-----------------------
import matplotlib.pyplot as plt
cart = CART()
cart.train_reg(X, y, max_depth=6)
y_pred = cart.predict(X)
y_test = y
type(y_pred)
# y_pred - np.array(y).reshape(-1)
i = 3
y_pred[i]
y[i]

cart.reg_loss()

y_pred.shape
loss = y_pred - np.array(y_test).reshape(-1)
plt.plot(np.arange(loss.shape[0]), loss)
plt.show()
dir(cart)
