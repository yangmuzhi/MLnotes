import numpy as np

class MeanShift:

    def __init__(self, h=4):
        self.X = None
        self.h = h

    def clustering(self,X, iters=10):
        self.X = X
        x = X
        for i in range(iters):
            x_new = self.shift(x)
            if self.is_convergence(x, x_new):
                return x
            x = x_new
        return x

    def is_convergence(self, x, x_new):
        return (np.abs(x - x_new) / x <= 1e-4).all()

    def shift(self, x):
        weights = [list(map(self._guassian, x, np.ones_like(x) * x[i])) for i in range(x.shape[0])]
        shift = x
        for i, weight in enumerate(weights):
            shift[i] = np.mean(list(map(lambda x,w: x * w, x, weight)), axis=0)

        return shift

    def _guassian(self, x, c ):

        return 1 / np.sqrt(2 * np.pi) / self.h * np.exp(
        -0.5 * np.matmul( (x - c), np.transpose(x - c) ) / self.h ** 2 )
