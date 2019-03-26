# using utf-8
# naive bayes Classifiers
# binary feature

import numpy as np

class Naive_bayes(object):

    def __init__(self, num_feature, num_class):

        self.N_C = np.zeros(num_class)
        self.N_X = np.zeros([num_class, num_feature])
        self.num_class = num_class
        self.num_feature = num_feature

    def inference(self, X, y):
        num_date, num_feature = X.shape

		# 计算参数
        for i in range(self.num_class):
            class_x = X[(y==i).ravel()]
            self.N_C[i] = len(class_x)
            for j in range(num_feature):
                self.N_X[i, j] = len(class_x[class_x[:, j] == 1])
        self.theta_x = self.N_X / self.N_C.reshape(-1, 1)
        self.theta_c = self.N_C / num_date
        return self.theta_c, self.theta_x

    def predict(self, X):
        num_date, num_feature = X.shape
        assert num_feature == self.num_feature

		# 计算后验分布分子的log值
        predict_y = np.log(self.theta_c.reshape(-1, 1)) + \
                    np.dot(np.log(self.theta_x), X.T) + \
                    np.dot(np.log(1 - self.theta_x), 1 - X.T)
        self.y = predict_y.argmax(axis=0).reshape(num_date, )
        return self.y

    def predict_prob(self, X):
        num_date, num_feature = X.shape
        assert num_feature == self.num_feature
		# 计算后验分布分子的log值
        predict_y = np.log(self.theta_c.reshape(-1, 1)) + \
                    np.dot(np.log(self.theta_x), X.T) + \
                    np.dot(np.log(1 - self.theta_x), 1 - X.T)
		# 计算每个样本分子最大值的类别索引
        max_prob = predict_y.max(axis=0).reshape(1, -1)
		# 防止下溢
        denominator = np.exp(predict_y - max_prob).sum(axis=0).reshape(1, -1)
		# 计算每个样本从属类别概率
        self.p = np.exp((predict_y - denominator - max_prob).T)
        self.p = self.p / (self.p.sum(axis=1).reshape(-1, 1))
        return self.p
