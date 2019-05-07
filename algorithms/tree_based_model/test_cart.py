from algorithms.tree_based_model.cart import CART
from algorithms.tree_based_model.cart import Node, Tree


import numpy as np
import scipy

# --------------------data-----------------------

N = 1000
num_feature = 5
X = np.random.uniform(0,10,[N,5])

X_add_one = np.column_stack(( X, X**2 ))
true_theta = np.matrix(np.random.randint(1,5,X_add_one.shape[1])).transpose()
y = np.dot(X_add_one, true_theta) + np.random.normal(0,0.1,[N,1])

# --------------------test Node tree-----------------------
root = Node(True)
# tree = Tree(max_depth=3, debug=True)
tree = Tree(max_depth=3)

tree.split(tree.root,x_data=X_add_one,y=y,depth=0)

x = X[0]
tree.lookup(x)


# --------------------test scipy-----------------------


tree.root.left = Node()
tree.root.right = Node()
split_var = 0
left_ind = X[:,split_var] <= 1
right_ind = X[:,split_var] > 1
tree.root.left.x_data = X[left_ind,:]
tree.root.left.y_value = y[left_ind,:]
tree.root.right.x_data = X[right_ind,:]
tree.root.right.y_value = y[right_ind,:]


tree._reg_error_func(1, split_var, X, y)

tree._reg_error_func(5, split_var, X, y)

var_space = X[:,split_var]

scipy.optimize.fminbound(tree._reg_error_func, 
                         min(var_space), max(var_space), 
                         args=(split_var,X,y),full_output = 1)
tree._greedy_split(tree.root.left)
list(tree.root.left.x_data[:,0])

x_data = X
split_value = 1
ind_left = x_data[:,split_var] >= split_value
ind_right = x_data[:,split_var] < split_value
error = np.square(y[ind_left] - np.mean(y[ind_left])).sum() + np.square(y[ind_right] - np.mean(y[ind_right])).sum()

tree.error
minimum = scipy.optimize.fminbound(f, -1, 2, args=([data]))

# --------------------test reg-----------------------
# cart = CART(X,y)

# cart.train_reg()
