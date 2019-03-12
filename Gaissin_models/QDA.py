"""
QDA
Quadratic discrimnant Analysis
"""
import numpy as np
import math

class QDA:

    def __init__(self, num_feature, num_class):

        # 初始化 mu sigma
        self.num_feature = num_feature
        self.num_class = num_class
        self.mu = []
        self.sigma = []

    def inference(self, X, y):
        for i in range(num_class):
            self.mu.append(X[y==i,:].mean(axis=0))
        for i in range(num_class):
            self.sigma.append(np.corrcoef(np.transpose(X[y==i,:])))

        return self.mu, self.sigma

    def predict(self, X):
        num_date, num_feature = X.shape
        assert self.num_feature == num_feature

        prob = np.zeros([num_date, num_class])
        for i in range(self.num_class):
            prob[:,i] = self._normal(X, mu=self.mu,
                        sigm=self.sigma).reshape(-1,1)
        prob_sum = prob.sum(axis=1)
        for i in range(self.num_class):
            prob[:,i] = prob[:,i] / prob_sum
        return prob

    def _normal(X, mu=mu, sigma=sigma):
        y = 1/(np.sqrt(2 * np.pi) * sigma) * np.exp(-(X - mu)**2 / (2*sigma**2))
        return y
