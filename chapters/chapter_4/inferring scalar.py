# using utf-8

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(30)

# Inferring an unknow scalar from noisy measurements
# x generates y, y is x's noisy measurement
mu_x = 0  # mu_prior
variance_x = 3  # variance_prior
x = np.sqrt(variance_x) * np.random.randn(15, ) + mu_x
y = np.zeros(x.shape)
variance_y = 1
for i in range(len(x)):
    y[i] = np.sqrt(variance_y) * np.random.randn(1, ) + x[i]
posterior_sigma = (variance_x ** (-1) + len(x) * variance_y ** (-1)) ** -1  # 3/(15*3+1) = 0.06521
posterior_mu = posterior_sigma * (mu_x / variance_x + len(y)*(variance_y ** -1) *y.mean())

# plot
figure = plt.figure(1, figsize=(10, 10))
sample = np.sqrt(posterior_sigma) * np.random.randn(1000, ) + posterior_mu
sns.kdeplot(sample, shade=False, label="Posterior")
sample = np.sqrt(variance_x) * np.random.randn(1000, ) + mu_x
sns.kdeplot(sample, shade=False, label="Prior")
sample = np.sqrt(variance_y/len(y)) * np.random.randn(1000, ) + x.mean()
sns.kdeplot(sample, shade=False, label="lik")
plt.legend(loc="best")
plt.show()