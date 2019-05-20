import numpy as np
from algorithms.Clustering.k_means import Kmeans

X = np.random.randn(1000,4)

kmeans = Kmeans()

kmeans.clustering(X,max_iters=100)

kmeans.is_convergence

np.abs(kmeans.eval - kmeans.new_eval) / kmeans.eval <= 1e-5

kmeans.center
