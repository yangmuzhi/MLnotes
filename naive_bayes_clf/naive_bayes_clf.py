"""
naive bayes Classifiers
binary feature
"""
import numpy as np

class Naive_bayes(object):

    def __init__(self, num_feature, num_class):

        self.N_C = np.zeros(num_class)
        self.N_X = np.zeros([num_class,num_feature])
        self.num_class = num_class
        self.num_feature = num_feature

    def inference(self, X, y):
        num_date, num_feature = X.shape
        for i in range(num_date):
            self.N_C[y[i]] += 1
            for j in range(num_feature):
                if X[i,j] == 1:
                    self.N_X[y[i], j] += 1
        self.theta_c = self.N_C / num_date
        self.theta_x = np.array([self.N_X[i,:] / self.N_C[i] for i in range(self.num_class)])
        return self.theta_c, self.theta_x

    def predict(self, X):
        num_date, num_feature = X.shape
        self.y = np.zeros(num_date)
        self.p = np.zeros([num_date, self.num_class])
        L = np.zeros([num_date, self.num_class])
        assert num_feature == self.num_feature
        for i in range(num_date):
            for c in range(self.num_class):
                l = np.log(self.theta_c[c])
                for j in range(num_feature):
                    if X[i,j] == 1:
                        l += np.log(self.theta_x[c,j])
                    else:
                        l += np.log(1 - self.theta_x[c,j])
                L[i,c] = l
            for c in range(self.num_class):
                self.p[i,c] = np.exp(L[i,c] - self._logsumexp(L[i,:]))
            self.y[i] = np.argmax(self.p[i,:])
        return self.y

    def _logsumexp(self, L):
        """防止溢出
        change np.log(np.sum(np.exp(L[i]))) >>
        max_l + np.log(np.sum(np.exp(L[i]-max_l)))"""
        max_l = L.max()
        return max_l + np.log(np.sum(np.exp(L-max_l)))
