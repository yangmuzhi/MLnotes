# using utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

np.random.seed(1)
D = 150  # 总样本数
obs = 10  # 可观测的数量
xs = np.linspace(0, 1, D)  # x坐标轴

perm = np.arange(D)
np.random.shuffle(perm)

obsNdx = perm[:obs]  # 观测样本索引
hidNdx = np.array(list(set(np.arange(D)) - set(obsNdx)))  # 缺失数据索引
xobs = np.random.randn(obs).reshape(-1, 1)  # 生成观测值

L = spdiags((np.ones(shape=(D, 1)) * np.array([-1, 2, -1])).T, [0, 1, 2], D-2, D).toarray()  # 生成L矩阵
lam = 30  # 控制先验的lambda
L = lam * L

L1 = L[:, hidNdx]
L2 = L[:, obsNdx]
lam11 = np.dot(L1.T, L1)
lam12 = np.dot(L1.T, L2)
posterior_Sigma = np.linalg.pinv(lam11)
posterior_mu = -np.dot(np.dot(np.linalg.pinv(lam11), lam12), xobs).reshape(-1, 1)

# plot
figure = plt.figure(1)
ax1 = plt.subplot(1, 2, 1)
ax1.plot(xs[hidNdx], posterior_mu.ravel(), linewidth=2)
ax1.plot(xs[obsNdx], xobs.ravel(), 'ro', markersize=6)
ax1.set_title(r"$\lambda=30$")

ax2 = plt.subplot(1, 2, 2)
xbar = np.zeros((D, 1))
sigma = np.zeros((D, 1))
xbar[obsNdx] = xobs
sigma[obsNdx] = 0
xbar[hidNdx] = posterior_mu
sigma[hidNdx] = np.sqrt(np.diag(posterior_Sigma, 0).reshape(-1, 1))
ax2.plot(xs[obsNdx], xobs.ravel(), 'ro', markersize=6)
ax2.plot(xs, xbar.ravel(), linewidth=2)
ax2.fill_between(xs, (xbar-2*sigma).ravel(), (xbar+2*sigma).ravel(), color="gray", alpha=0.2)
ax2.set_title(r"$\lambda=30$")
plt.show()