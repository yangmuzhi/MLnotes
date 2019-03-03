from naive_bayes_clf import Naive_bayes
import numpy as np

X = np.random.randint(0,2,(100,5))
num_date, num_feature = X.shape
theta = np.array([0.5,0.5,0.3,0.3,1])

y = np.matrix(X) * np.matrix(theta).transpose()
y = np.array([int(i > np.mean(y)) for i in y])

nb = Naive_bayes(num_feature, 2)
theta_c, theta_x = nb.inference(X, y)
nb.theta_c
nb.theta_x
y_pred = nb.predict(X)

np.sum(y_pred == y) / len(y)
