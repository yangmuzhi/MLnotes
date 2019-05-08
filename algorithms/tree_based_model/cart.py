"""
Cart for both regression and classifying

min[ min sum (y_i - c1)^2 + min sum (y_j - c2)^2]

二叉树

"""
import numpy as np
import scipy
import time
# import pysnooper

class Node:

    """
    node
    left right Node类
    """

    def __init__(self, is_root=False):
        self.is_root = is_root
        self.x_data = None
        self.y_value = None
        # l,r node类
        self.left = None
        self.right = None
        self.split_var = None # 分割变量 str
        self.split_value = None

    @property
    def isend(self):
        """判断是否是一棵树的end"""
        return self.left is None and self.right is None

class Tree:
    def __init__(self, max_depth=10, debug=False):
        self.root = Node(True)
        self.max_depth = max_depth
        self.debug = debug

    # @pysnooper.snoop()
    def reg_lookup(self, x):
        """
        data 从root进入，然后输出end node的value
        x 是numpy.array
        """
        assert not self.root.isend, "root未分裂"
        next_node = self.root

        while not next_node.isend:
            node = next_node
            if x[node.split_var] <= node.split_value:
                next_node = node.left
            else:
                next_node = node.right
            next_end = node.isend
        return node.y_value.mean()

    def predict(self, data):
        """在lookup基础上"""
        raiseNotImplementedError("tree.predict: not implemented!")

    def _random_split(self, node):
        var = np.random.choice(node.x_data.shape[1])
        print(var)
        return var, np.random.uniform(
                min(node.x_data[:,var]), max(node.x_data[:,var]))

    def _greedy_split(self, node):
        """贪心的方式分裂, 返回split_var 和 split_value"""
        data_num, feature_num = node.x_data.shape
        split_list = []
        error_list = []
        split_var_list = []
        error = None
        #
        min_error = np.inf

        for split_var in range(feature_num):
            # 使用scipy 寻找最优的分节点
            # get optimal value
            var_space = list(node.x_data[:,split_var])
            if (not var_space) or (min(var_space) == max(var_space)):
                continue
            split, error, ierr, numf = scipy.optimize.fminbound(
                    self._reg_error_func, min(var_space), max(var_space),
                    args = (split_var, node.x_data, node.y_value), full_output = 1)
            split_list.append(split)
            error_list.append(error)
            split_var_list.append(split_var)
            if min_error > error:
                min_error = error
                best_split_var = split_var
                best_split_value = split
        if self.debug:
            time.sleep(1)

            print("split_var: ", split_var_list, "\n")
            print("error_list: ", error_list, "\n")
            if error:
                print("best_split_value:", best_split_value, "\n")
                print("best_split_var:", best_split_var, "\n")
            print("-"*10,"\n")
        if error:
            return best_split_var, best_split_value
        else:
            return

    def _reg_error_func(self, split_value, split_var, x_data, y):
        ind_left = x_data[:,split_var] >= split_value
        ind_right = x_data[:,split_var] < split_value
        error = np.square(y[ind_left] - np.mean(y[ind_left])).sum() + np.square(
                    y[ind_right] - np.mean(y[ind_right])).sum()
        return error

    def split(self,node,x_data,y,depth):
        """
        递归的分裂
        """
        # 分裂节点

        assert node.isend, "分裂节点必须是树的终结点"

        if self.debug:
            print("depth", depth)
        if depth >= self.max_depth:
            # print("done")
            return

        # 赋值
        node.x_data = x_data
        node.y_value = y
        split_var, split_value = self._greedy_split(node)
        # split_var, split_value = self._random_split(node)

        node.split_var = split_var
        node.split_value = split_value

        left_ind = x_data[:,split_var] <= split_value
        right_ind = x_data[:,split_var] > split_value
        x_data_left = x_data[left_ind]
        x_data_right = x_data[right_ind]
        y_left = y[left_ind]
        y_right = y[right_ind]

        node.left = Node()
        node.right = Node()
        self.split(node.left,x_data_left,y_left,depth+1)
        self.split(node.right,x_data_right,y_right,depth+1)




#----------------------------------------------------------------------

class CART:
    """
    cart封装
    """
    def __init__(self):
        pass

    def train_reg(self, X, y, max_depth=5, regularization=False, debug=False):
        """
        Args:
            X: train data
            y: label
        """
        self.y = y
        self.X = X
        self.tree = Tree(max_depth=max_depth, debug=debug)
        self.tree.split(self.tree.root, x_data=X, y=y, depth=0)

    def train_clf(self):
        pass

    def predict(self, X):
        """
        Args:
            X: data
        """
        self.y_pred = np.array(list(map(lambda x: self.tree.reg_lookup(x), X)))

        return self.y_pred

    def reg_loss(self, y_pred=None, y=None):
        if not y_pred:
            y_pred = self.y_pred
            y = np.array(self.y).reshape(-1)
            #
        return np.mean(np.abs(y_pred - y))
