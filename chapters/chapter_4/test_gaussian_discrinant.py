# using utf-8

import numpy as np
from gaussian_discriminant_analysis import Gaussian_discriminant_analysis

np.random.seed(50)
X_1 = np.random.multivariate_normal(mean=np.zeros(shape=(6,)), cov=np.eye(6), size=30)
y_1 = np.array([0]).repeat(30).reshape(-1, 1)
X_2 = np.random.multivariate_normal(mean=np.array([1, 0.8, -1, 1, -0.2, 0.61]),
                                    cov=np.eye(6)*0.8, size=27)
y_2 = np.array([1]).repeat(27).reshape(-1, 1)
np.random.seed(31)
X_3 = np.random.multivariate_normal(mean=np.random.random(size=(6, )),
                                    cov=np.eye(6)*1.2, size=33)
y_3 = np.array([2]).repeat(33).reshape(-1, 1)

X = np.vstack((X_1, X_2, X_3))
y = np.vstack((y_1, y_2, y_3))

index = np.arange(len(X))
np.random.shuffle(index)
X = X[index]
y = y[index]

gaussian_disrinant_clf = Gaussian_discriminant_analysis()
gaussian_disrinant_clf.fit(X, y)
predict_y, y_prob = gaussian_disrinant_clf.predict(X)
print(predict_y, y_prob)
print("准确率为", len(y[predict_y==y])/len(y))