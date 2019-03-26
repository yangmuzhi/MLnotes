#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt
import math

def p(x):
 #standard normal
    mu=0
    sigma=1
    return 1/(math.pi*2)**0.5/sigma*np.exp(-(x-mu)**2/2/sigma**2)

class MC_rej_sampling:
    
    """
    给一个需要抽样的p函数
    使用均匀分布q 来抽样
    1. 可以使用矩阵运算的方式直接抽样，优点：速度快，但是可能因为伪随机数的原因会导致，结果不够好
    2. 可以使用循环一个一个抽样，速度慢
    """
    def __init__(self, p, a=-5,b=5, N=100):
        
        self.p = p
        self.a = a
        self.b = b
        self.N = N

    
    def q(self, x):
        return np.array([1/abs(self.b-self.a) for i in np.arange(x.shape[0])])
    
    def q_(self, x):
        return np.array(1/abs(self.b-self.a))
    
        
    def sampling_matrix(self, M):
        X = np.random.uniform(self.a, self.b, self.N)
        assert sum(self.p(X) > (M * self.q(X))) ==  0 
        u = np.random.uniform(0, 1, self.N)
            
        def _filter(x):
            return u < (self.p(x) / (M * self.q(x)))
        
        result = X[_filter(X)]
        self.ratio = result.shape[0] / X.shape[0]
        return result
    
    def sampling_one_by_one(self, M):
        
        i = 0
        result = []
        count = 0
        
        while i < self.N:
            u = np.random.uniform(0,1)
            x = np.array(np.random.uniform(-10,10))
            res = u < (self.p(x) / (M * self.q_(x)))
            if res:
                result.append(x)
                i += 1
            count += 1
        self.ratio = i / count
        
        return np.array(result)
            
    
    def plot(self, result):
                
        plt.hist(np.array(result),bins=100,normed=True)
        plt.title('Rejection Sampling')
        
        
# 1
mc = MC_rej_sampling(p, a=-5, b=5, N=100000)      
res = mc.sampling_matrix(4)
mc.ratio
mc.plot(res)

# 2
res_o = mc.sampling_one_by_one(4)
mc.ratio
mc.plot(res_o)


