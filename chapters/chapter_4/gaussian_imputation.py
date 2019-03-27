# using utf-8

import numpy as np
import matplotlib.pyplot as plt

d = 20  # 维度
n = 5  # 样本数量

# 产生数据
np.random.seed(100)
mu = np.random.rand(d, 1)

# 计算协方差参数
simple_ = np.random.rand(100, 20)
sigma = np.dot(simple_.T, simple_) / len(simple_)
while np.all(np.linalg.eigvals(sigma)<=0):  # 检验正定
    simple_ = np.random.rand(100, 20)
    sigma = np.dot(simple_.T, simple_) / len(simple_)

Xfull = np.random.multivariate_normal(mu.ravel(), sigma, n)
missing = np.random.rand(n, d) < 0.5
Xmiss = Xfull.copy()
Xmiss[missing] = np.NAN
sigma_inv = np.linalg.pinv(sigma)
# 计算条件分布
# 绘图
figure = plt.figure(1,  figsize=(18, 18))
for i in range(n):
    hidNodes = np.where(np.isnan(Xmiss[i, :])==True)[0]
    visNodes = np.where(np.isnan(Xmiss[i, :])==False)[0]
    h_sigma = np.linalg.pinv(sigma_inv[:, hidNodes][hidNodes, :])  # 计算条件协方差矩阵
    h_mu = mu[hidNodes] - np.dot(np.dot(h_sigma, sigma_inv[hidNodes, :][:, visNodes]),
                                 Xmiss[i, visNodes].reshape(-1, 1) - mu[visNodes])
    error = np.sqrt(np.diag(h_sigma, 0))
    ax = plt.subplot(n, 1, i+1)
    ax.scatter(hidNodes.ravel(), Xfull[i, hidNodes].ravel(), label="truth", marker='+')
    ax.errorbar(x=hidNodes.ravel(), y=h_mu.ravel(), yerr=error.reshape(-1, 1), fmt="o",
                color="r", ecolor="gray", alpha=0.2, label="Imputed")
    ax.legend(loc="best")
plt.show()