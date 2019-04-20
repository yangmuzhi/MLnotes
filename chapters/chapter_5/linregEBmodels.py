# using utf-8

"""
所建立的模型为p(y) = N(x'w, 4),采用EB方法估计w
设置w先验为参数是μ，Σ的高斯分布
"""

import numpy as np
import matplotlib.pyplot as plt

Ns = 5  # 观测的样本数
np.random.seed(2)
x = np.random.randn(Ns, 1) * 5  # 观测点x值
error = np.random.randn(Ns, 1)  # 误差
y_train = (x - 4) ** 2 + 2 * error  # y观测值，观测的方差为4
y_true = (x - 4) ** 2  # y真实值

# 先对两个变量的进行设置

# 减去均值忽略掉截距项
x_expand = np.hstack((x, x**2))
x_expand_center = x_expand - x_expand.mean(axis=0)
y_train_center = y_train - y_train.mean(axis=0)
error_var = 4
"""
估计μ和Σ的取值
p(D|μ,Σ) = ∫ p(D|w) p(w|μ,Σ) dw
p(D|w) = N(y_mean|x_mean w,σ**2/N)
p(D|μ,Σ) = N(y_mean|x_mean u, σ**2/N + x_mean Σ x_mean')
x_mean μ = y_mean； μ = (x_mean' x_mean) ** -1 x_mean' y_mean
s ** 2 =  σ**2/N + x_mean Σ x_mean'; Σ = (s ** 2 - σ ** 2 /N)(x_mean' x_mean) ** -1 
"""
mu_prior = (np.linalg.pinv((x_expand_center.T.dot(x_expand_center))).dot(x_expand_center.T)).dot(y_train_center)
sigma_prior = (y_train_center.var() - (error_var / Ns)) * np.linalg.pinv(x_expand_center.T.dot(x_expand_center))

"""
在了解先验μ和Σ后，对参数w进行估计
再利用对高斯线性模型
p(D|w) = N(y_mean| x_mean w, σ ** 2/N) p(w) = N(w|μ,Σ)
w_sigma = (Σ ** -1 + x_mean' (σ ** 2/N)**-1 x_mean) ** -1
w_mu = w_sigma(Σ **-1 μ + x_mean' (σ ** 2/N) ** -1 y_mean)
p(w|D) = N(w|, (σ**2)**-1+x_mean' Σ ** {-1} x_mean)
"""
w_sigma = np.linalg.pinv(np.linalg.pinv(sigma_prior).dot(mu_prior) +
                   1/(error_var / Ns) * x_expand_center.T.dot(x_expand_center))
w_mu = w_sigma.dot(sigma_prior.dot(mu_prior) + 1/(error_var / Ns) * x_expand_center.T.dot(y_train_center))
w_0 = y_train_center.mean() - x_expand_center.mean(axis=0).dot(w_mu)
y_predict = x_expand.dot(w_mu) + w_0
"""
预测y
p(y|D) ∝ p(w|D)p(y|w)
再次利用高斯线性系统公式得到
p(y|D) = N(y|x w_mu, error_var + x w_wigma x')
"""
x_test = np.linspace(-30, 30, 24)[:, np.newaxis]
x_test_expand = np.hstack((x_test, x_test **2))
y_test = (x_test - 4) ** 2
y_predict_mean = x_test_expand.dot(w_mu) +w_0
y_predict_std = np.sqrt(error_var + np.diag(x_test_expand.dot(w_sigma.dot(x_test_expand.T))))[:, np.newaxis]

figure = plt.figure(1)
plt.plot(x_test.ravel(), y_test.ravel(), label="True")
plt.plot(x_test.ravel(), y_predict_mean.ravel(), label="predict")
plt.fill_between(x_test.ravel(), (y_predict_mean - 2 * y_predict_std).ravel(),
                 (y_predict_mean + 2*y_predict_std).ravel(),  color="gray", alpha=0.2)
plt.scatter(x.ravel(), y_predict.ravel(), color="blue" ,label="train")
plt.legend(loc="best")
plt.xlabel("X")
plt.ylabel("y")
plt.title("EB for linear regression")
plt.show()