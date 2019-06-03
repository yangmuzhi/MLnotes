"""

特征数量：
用作分类时，m默认取 ，最小取1；
用作回归时，m默认取 p / 3，最小取5

样本数量与训练数据相同
"""

from algorithms.tree_based_model.cart import CART
import numpy as np
from tqdm import tqdm

class RandomForest:

    def __init__(self):
        self.forest = []

    def train_reg(self, X, y, max_depth=3,tree_num=100):
        self.y = y

        def gen_one_tree(X,y,max_depth):

            if int(X.shape[1] / 3):
                m = int(X.shape[1] / 3)
            else:
                m = 1
            rand_ind = np.random.choice(X.shape[1],m, replace=False)
            X = X[:, rand_ind]
            cart = CART()
            cart.train_reg(X, y, max_depth=max_depth)
            return cart

        tqdm_range = tqdm(range(tree_num))

        for i in tqdm_range:
            self.forest.append(gen_one_tree(X,y,max_depth=max_depth))

        # self.forest = list(map(gen_one_tree,x_data_list,y_list,max_depth=max_depth_list))
        #

    def predict(self, X, pred_method="mean"):
        """使用均值或者投票的方式来选择
        """
        if pred_method == "mean":
            func = self._mean_predict
        pred = []
        for cart in self.forest:
            pred.append(cart.predict(X))

        self.y_pred = func(pred)

        return self.y_pred

    def _mean_predict(self,y):
        return np.mean(y, axis=0)

    def loss(self):
        y = np.array(self.y).reshape(-1)
        y_pred = np.array(self.y_pred).reshape(-1)
        return np.mean(np.abs(y_pred - y))
