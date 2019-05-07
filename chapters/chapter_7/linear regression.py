# using utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

class linear_regression():

    def __init__(self):
        self.D = None  # 数据维度
        self.N = None  # 样本数量
        self.w = None  # 待估计系数

    def fit(self, X, y, method="MLE", **kwargs):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        self.N, self.D = X.shape
        W = np.zeros((self.D, 1))
        X = X - X.mean(axis=0)
        y = y - y.mean(axis=0)
        if np.all(X[:, 0] == 1):
            X = X[:, 1:]

        if method == "MLE":
            if self.N > self.D * 2:
                Q, R = np.linalg.qr(X)
                self.w = np.linalg.inv(R).dot(Q.T).dot(y)
            else:
                # X.T.dot(X) = V Σ Σ V.T
                # inv(X.T.dot(X)) = V inv(Σ Σ) V.T
                # 对X进行SVD分解，X=U Σ V.T
                # w = inv(X.T X) X.T y = V inv(Σ) U.T y
                U, S, V_h = np.linalg.svd(X, full_matrices=False)
                V = V_h.T
                self.w = V.dot(np.diag(1/S)).dot(U.T).dot(y)
            return self.w
        elif method == "MAP":
            noisy_sigma = kwargs["noisy_sigma"]  # y的noisy
            prior_sigma = kwargs["prior_sigma"]  # 参数w的先验
            penalty = noisy_sigma / prior_sigma  # 惩罚系数
            if self.N > self.D * 2:
                X_stack = np.vstack((X, 1 / noisy_sigma * np.eye(self.D)))
                y_stack = np.vstack((y, np.zeros((self.D, 1))))
                Q, R = np.linalg.qr(X_stack)
                self.w = np.linalg.inv(R).dot(Q.T).dot(y_stack)
            else:
                # X = U S V.T, dim(U) = N × N, dim(S) = N × D, dim(V) = D × D
                U, S_ravel, V_h = np.linalg.svd(X, full_matrices=True)
                V = V_h.T
                S = np.zeros((self.N, self.D))
                for i in range(len(S_ravel)):
                    S[i, i] = S_ravel[i]
                self.w = V.dot(np.linalg.inv(S.T.dot(S) + penalty * np.eye(self.D))).dot(S.T).dot(U.T).dot(y)
            return self.w
        elif method == "EB":
            noisy_sigma = kwargs["noisy_sigma"]  # y的观测误差
            U, S_ravel, V_h = np.linalg.svd(X, full_matrices=False)
            V = V_h

            # y = X * mu_prior, mu_prior = inv(X.T.dot(X)).dot(X.T).dot(y)
            # X.T.dot(X) = V.dot(S).dot(S).dot(V.T); inv(X.T.dot(X)) = V.dot(inv(S)).dot(inv(S)).dot(V.T)
            mu_prior = V.dot(np.diag(1/S_ravel)).dot(U.T).dot(y)  # 参数w均值的先验

            # sigma_prior = (y.var() - noisy_sigma) * inv(X.T.dot(X))
            sigma_prior = (y.var() - noisy_sigma) * V.dot(np.diag(1 / S_ravel**2)).dot(V.T)  # 参数w协方差的先验


            # sigma_posterior = inv(inv(sigma_prior) + X.T.dot(1/noisy_sigma).dot(X))
            # = inv(X.T.dot(X)/(y.var() - noisy_sigma) + X.T.dot(X)/ noisy_sigma) = inv(X.T.dot(X)) * ()
            sigma_posterior = V.dot(np.diag(1/ S_ravel**2)).dot(V.T) * \
                              (noisy_sigma * (y.var() - noisy_sigma) / y.var())  # 参数w的后验协方差
            mu_posterior = sigma_posterior.dot(X.T.dot(X).dot(mu_prior) / (y.var() - noisy_sigma) + 1/noisy_sigma * X.T.dot(y))
            self.w = mu_posterior
            return mu_posterior, sigma_posterior

if __name__ == "__main__":
    np.random.seed(3)

    # 对MLE 中 N <= 2*D 的情况进行测试
    X = 1.2 * np.random.randn(9, 1)
    y_obs = 0.6 * X **2 - 0.8 * X + 4 * np.random.randn(len(X), 1)
    error_var = 4
    linear_estimator_1 = linear_regression()
    result = linear_estimator_1.fit(np.hstack((X, X**2, X**3, X**4, X**5, X**6, X**7)), y_obs, method="MLE")
    print("测试1结果:", result)
    linear_sklearn = LinearRegression()
    sklearn_result = linear_sklearn.fit(np.hstack((X, X**2, X**3, X**4, X**5, X**6, X**7)), y_obs)
    print("sklearn结果1:", sklearn_result.coef_)

    # 对MLE 中 N > 2*D 的情况进行测试
    X = 1.2 * np.random.randn(9, 1)
    y_obs = 0.6 * X ** 2 - 0.8 * X + 4 * np.random.randn(len(X), 1)
    error_var = 4
    linear_estimator_2 = linear_regression()
    result = linear_estimator_2.fit(X, y_obs, method="MLE")
    print("测试2结果:", result)
    linear_sklearn = LinearRegression()
    sklearn_result = linear_sklearn.fit(X, y_obs)
    print("sklearn结果2:", sklearn_result.coef_)
    
    # 对MAP中 N <= D * 2 的情况进行测试
    X = 1.2 * np.random.randn(9, 1)
    y_obs = 0.6 * X **2 - 0.8 * X + 4 * np.random.randn(len(X), 1)
    error_var = 4
    linear_estimator_3 = linear_regression()
    result = linear_estimator_3.fit(np.hstack((X, X**2, X**3, X**4, X**5, X**6, X**7)), y_obs, method="MAP",
                                    noisy_sigma=4, prior_sigma=4)
    print("测试结果3:", result)
    sklearn_ridge = Ridge(alpha=1)
    sklearn_result = sklearn_ridge.fit(np.hstack((X, X**2, X**3, X**4, X**5, X**6, X**7)), y_obs)
    print("sklearn结果3:", sklearn_ridge.coef_)
    
    # 对MAP中 N > D * 2 的情况进行测试
    X = 1.2 * np.random.randn(9, 1)
    y_obs = 0.6 * X ** 2 - 0.8 * X + 4 * np.random.randn(len(X), 1)
    error_var = 4
    linear_estimator_4 = linear_regression()
    result = linear_estimator_4.fit(np.hstack((X, X**2, X**3)), y_obs, method="MAP", noisy_sigma=1, prior_sigma=3)
    print("测试结果4:", result)
    sklearn_ridge = Ridge(alpha=1)
    sklearn_result = sklearn_ridge.fit(np.hstack((X, X**2, X**3)), y_obs)
    print("sklearn结果4:", sklearn_ridge.coef_)

    X = 1.2 * np.random.randn(9, 1)
    y_obs = 0.6 * X ** 2 - 0.8 * X + 4 * np.random.randn(len(X), 1)
    error_var = 4
    linear_estimator_5 = linear_regression()
    result_EB, _ = linear_estimator_5.fit(np.hstack((X, X ** 2, X ** 3)), y_obs, method="EB", noisy_sigma=1)
    linear_estimator_6 = linear_regression()
    result_MAP = linear_estimator_6.fit(np.hstack((X, X ** 2, X ** 3)), y_obs, method="MAP", noisy_sigma=1, prior_sigma=3)
    print("EB结果:", result_EB)
    print("MAP结果:", result_MAP)