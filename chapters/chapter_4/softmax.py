import numpy as np
"""
测试softmax
证明softmax输出每个class是1
"""

def softmax(X, beta=beta, gmma=gmma):
    assert len(beta) == len(gmma)
    s = []
    for i in range(len(beta)):
        s.append(np.exp(X * beta[i] + gmma[i]))
    s_sum = 0
    for i in range(len(s)):
        s_sum += s[i]
    for i in range(len(s)):
        s[i] = s[i] / s_sum
    s = np.array(s).squeeze()
    return s


if __name__ == '__main__':

    X = np.matrix(np.random.random([10,5]))
    num_class = 2
    beta = []
    gmma = []

    for i in range(num_class):
        beta.append(np.matrix(np.random.random([5,1])))
        gmma.append(np.matrix(np.random.random([1])))

    y_pred = softmax(X)
    print(y_pred.sum(axis=0))
