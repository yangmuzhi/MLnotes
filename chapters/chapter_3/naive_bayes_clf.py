# using utf-8
import numpy as np
import pandas as pd

class Naive_Bayes_clf():

    def __init__(self):
        self.N = None  # 样本数
        self.p = None  # 维度
        self.C = None  # 类别数
        self.category = None  # 种类名

    def fit(self, X_train, y_train, method="MLE"):

        try:
            X_train = np.array(X_train)
            y_train = np.array(y_train).reshape(-1, 1)
        except:
            print("请输入合适的数据类型，并保证y的维度为1")
            return None

        self.category = np.unique(y_train)
        self.C = len(np.unique(y_train))
        self.N, self.p = X_train.shape

        if self.C:
            category_params = np.zeros((self.C, 1))  # 存储类别参数
            feature_params = np.zeros((self.C, self.p))  # 存储特征参数
            for i in range(self.C):
                class_x = X_train[(y_train == self.category[i]).ravel()]
                category_params[i, 0] = len(class_x)
                for j in range(self.p):
                    feature_params[i, j] = len(class_x[class_x[:, j] == 1])
        else:
            return "请检查数据"
        print(feature_params)
        # 加一平滑
        if method == "MAP":
            self.feature_param = feature_params / category_params.reshape(self.C, 1)
            self.category_param = category_params / self.N
        elif method == "MLE":
            self.feature_param = (feature_params + 1) / (category_params.reshape(self.C, 1) + 2)
            self.category_param = (category_params + 1) / (self.N + self.C)
        else:
            return None
        print(self.feature_param)

        return self.feature_param, self.category_param

    def predict(self, X_test):
        num_test, num_dim = X_test.shape
        if num_dim != self.p:
            print("测试数据维度与训练数据维度不相同")
            return None
        numerator = np.dot(np.log(self.feature_param), X_test.T) + \
                    np.dot(np.log(1 - self.feature_param), 1 - X_test.T) + \
                    np.log(self.category_param)
        predict_category = self.category[numerator.argmax(axis=0)].reshape(self.N,)
        return predict_category
