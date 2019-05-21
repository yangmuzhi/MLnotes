import numpy as np
from algorithms.Clustering.k_means import Kmeans

X = np.random.randn(100,4)

kmeans = Kmeans()

kmeans.clustering(X,max_iters=100)

kmeans.is_convergence

np.abs(kmeans.eval - kmeans.new_eval) / kmeans.eval <= 1e-5

kmeans.center

from algorithms.Clustering.mean_shift import MeanShift
ms = MeanShift()
ms._guassian(X[0], X[1])

X = ms.shift(X)
ms.clustering(X)
