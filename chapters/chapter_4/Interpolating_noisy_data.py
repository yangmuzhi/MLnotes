# using utf-8

import numpy as np
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
np.random.seed(36)

# Interpolating noisy data
D = 150  # 总样本数
obs = 10  # 观测样本数
xs = np.linspace(0, 1, D)

index = np.arange(D)
np.random.shuffle(index)

obsNdx = index[:obs]  # 观测值索引
hidNdx = np.array(list(set(index) - set(obsNdx)))  # 隐藏值索引

y_var = 1  # 观测误差
y = np.sqrt(y_var) * np.random.randn(obs)  # 观测值
A = np.zeros(shape=(obs, D))  # A, projection matrix, select out the oberved value
for i in range(len(obsNdx)):
    A[i, obsNdx[i]] = 1

# Model is p(y|x) = N(Ax, obsNoiseVar * I)
L = spdiags((np.ones(shape=(D, 1)) * np.array([-1, 2, -1])).T, [0, 1, 2], D-2, D).toarray()
lam = 30
L = lam * L
mu_prior = np.zeros((D, 1))
sigma_prior = np.linalg.pinv(np.dot(L.T, L))

sigma_posterior = np.linalg.pinv(np.linalg.pinv(sigma_prior) +
                                 np.dot(np.dot(A.T, 1 / y_var * np.eye(obs, obs)), A))

mu_posterior = sigma_posterior.dot(np.dot(np.linalg.pinv(sigma_prior), mu_prior) +
                                          A.T.dot(1/y_var * np.eye(obs, obs)).dot(y.reshape(-1, 1))).ravel()

variance = np.sqrt(np.diag(sigma_posterior))

# plot
figure = plt.figure(1, figsize=(10, 10))
plt.scatter(xs[obsNdx], y,  marker='*', label="measure")
plt.scatter(xs[obsNdx], mu_posterior[obsNdx],  marker='3', label="sample")
plt.plot(xs, mu_posterior, linewidth=2)
plt.fill_between(xs, mu_posterior-2*variance, mu_posterior+2*variance,  color="gray", alpha=0.2)
plt.legend(loc="best")
plt.title(r"$\lambda=30$")
plt.show()
