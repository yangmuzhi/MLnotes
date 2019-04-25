"""
Cart for both regression and classifying

min[ min sum (y_i - c1)^2 + min sum (y_j - c2)^2]
每次节点是二叉树

一个难点 如何优雅的表示树？？使用类

"""



class CART:

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def _reg_node_loss(self, j, s, regularization=False):
        """
        每次节点是二叉树, avg(y|x in R(j,s))
        """
        self.X[]

    def train_reg(self, regularization=False):
        pass

    def train_clf(self):
        pass

Node = {"left":None, "right":None, "parent":None, "value":None}
