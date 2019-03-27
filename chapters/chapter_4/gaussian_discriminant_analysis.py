# using utf-8

import numpy as np

class Gaussian_discriminant_analysis():

    def __init__(self):
        self.C = None  # 类别数
        self.p = None  # 维度
        self.category = None  # 类别名称

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)  # shape=(N, 1)

        assert len(X) == len(y)

        self.category = np.unique(y)
        self.C = len(self.category)
        N, self.p = X.shape  # N表示样本数量，p表示维度

        category_param = np.zeros(self.C).reshape(-1, 1)  # shape=(self.C, 1)
        mean_param = np.zeros(shape=(self.C, self.p))
        covariance_param = np.zeros(shape=(self.C, self.p, self.p))
        for i in range(self.C):
            X_category = X[(y == self.category[i]).ravel()]  # 取出每一类的样本
            category_param[i, 0] = len(X_category) / N  # 类别参数
            mean_param[i, :] = X_category.mean(axis=0)  # 均值参数
            stand_X = X_category - mean_param[i, :]  # 标准化
            covariance_param[i] = np.dot(stand_X.T, stand_X) / (category_param[i, 0] * N) # 计算样本协方差

        self.category_param = category_param
        self.mean_param = mean_param
        self.covariance_param = covariance_param
        return self.category_param, self.mean_param, self.covariance_param

    def predict(self, X):
        X = np.array(X)
        N, p = X.shape
        assert p == self.p

        self.inverse_covariance = np.linalg.pinv(self.covariance_param)  # 存储逆协方差矩阵
        determinant_covariance = np.linalg.det(self.covariance_param)  # 存储协方差矩阵的行列式
        y_prob = np.zeros(shape=(self.C, N))  # 存储每个样本在每个类别下的概率

        for c in range(self.C):
            stand_X = X - self.mean_param[c,:]
            for i in range(N):
                # 计算概率密度
                y_prob[c, i] = determinant_covariance[c] ** (-1/2) * \
                               np.exp((-1/2) * np.dot(np.dot(stand_X[i, :].T, self.inverse_covariance[c]),
                                                    stand_X[i, :]))

        y_predict = self.category[y_prob.argmax(axis=0)].reshape(-1, 1)
        y_prob = y_prob / y_prob.sum(axis=0)
        return y_predict, y_prob