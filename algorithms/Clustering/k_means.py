"""
K means 实现
主要是找点信心。。
"""
import numpy as np

class Kmeans():

    def __init__(self, K=3):
        self.K = K
        self.center = []
        self.C = None

    def step(self):
        return

    def eval_func(self, X, C):
        """类内距离， 越大说明不好
        Args:
            C 是0，1，2...
        """
        center = np.array([self.center[i] for i in C])
        return np.sum(list(map(self.dist_func, X, center)))

    def dist_func(self, x, center):
        # 使用二范数 可以override
        return np.sum(np.square(x - center))

    def choose_c(self, x):
        # 选择最小距离
        center = np.array(self.center)
        dist = [self.dist_func(x, c) for c in center]

        return np.argmin(dist)

    def compute_center(self,X,C):
        X_new = []
        center_new = []

        center = sorted(set(C))
        for c in center:
            X_new.append(X[c == C])
        for x in X_new:
            center_new.append(np.mean(x, axis=0))
        return center_new

    @property
    def is_convergence(self):
        return np.abs(self.eval - self.new_eval) / self.eval <= 1e-5



    def clustering(self, X, num_init_points=10, max_iters=10):
        """
        Args:
            X: data
            num_init_points: 初始数量
        """
        # 选择初始点
        # idx = np.random.choice(np.arange(X.shape[0]), num_init_points, replace=False)
        # init_point = X[idx]
        # # 选择一个eval_func 最好的初始点
        # eval_init = [self.eval_func(i) for i in init_point]
        # p = init_point[np.argmin(eval_init)]
        idx = np.random.choice(np.arange(X.shape[0]), self.K, replace=False)
        init_center = X[idx]
        self.center = init_center
        self.eval = np.inf
        # 按距离分类
        for i in range(max_iters):
            C = list(map(self.choose_c, X))
            center_new = self.compute_center(X, C)
            self.new_eval = self.eval_func(X, C)
            # print("eval",self.eval)
            # print("new_eval",self.new_eval)
            new_eval = self.eval_func(X, C)

            if self.is_convergence:
                break
            self.center = center_new
            self.C = C
            self.eval = new_eval
        return
