from algorithms.tree_based_model.cart import CART
from numpy as np

class RandomForest:

    def __init__(self):
        self.tree = []

    def train_reg(self, cart = CART, tree_num=100):

        for i in range(tree_num):
            self.tree.append(cart)
            X, y = self.sampling()
            self.tree[-1].train_reg(self, X, y, max_depth=5, regularization=False, debug=False)

    def predict(self, X, pred_method="mean"):
        """使用均值或者投票的方式来选择
        """
        if pred_method == "mean":
            func = self._mean_predict
        pred = []
        for i in range(len(self.tree)):
            pred.append(self.tree[i].predict(X))

        return func(pred)

    def _mean_predict(self,y):
        return np.mean(y, axis=1)


    def sampling(self, X_full, y_full):
        """采样变量和数据量,
        return  X,y,
        """
        return X_full, y_full
