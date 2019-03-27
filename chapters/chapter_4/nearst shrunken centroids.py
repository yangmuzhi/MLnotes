# using utf-8

import numpy as np
import pandas as pd

gen_data = pd.read_excel("SRBCT.xlsx")
print(gen_data.shape)
assert np.alltrue(~gen_data.isna()) == True

# 划分训练集与测试集
train_test_split = np.arange(len(gen_data))
np.random.shuffle(train_test_split)
X_train = gen_data.iloc[train_test_split[:63], :-1]
y_train = gen_data.iloc[train_test_split[:63], -1]
X_test = gen_data.iloc[train_test_split[63:len(gen_data)], :-1]
y_test = gen_data.iloc[train_test_split[63:len(gen_data)], -1]

C = len(np.unique(y_train.values))
category = np.unique(y_train.values)
N, D = X_train.shape

# 计算参数
# d = (x_category_mean - x_global_mean) / (mk * (sj + s0))
# mk = (1/Nk - 1/N) ** (1/2)
# sj = sum((x_kj - x_catrgory_k_mean_j) ** 2) / (N - C)
# s0 = median(sj)
x_global_mean = X_train.mean(axis=0).values.reshape(1, -1)
category_mean = np.zeros(shape=(C, D))
sse = np.zeros(shape=(1, D))
mk = np.zeros(shape=(C, 1))
for i in range(C):
    x_category = X_train.loc[y_train==category[i], :]
    Nclass = len(x_category)
    assert Nclass!=0
    centroid = x_category.mean(axis=0)
    category_mean[i, :] = centroid.values
    mk[i] = np.sqrt(1 / Nclass - 1 / N)
    sse = sse + ((x_category.values - centroid.values) ** 2).sum(axis=0)
sse = np.sqrt(sse / (N -C))
s0 = np.median(sse)

# 计算d和softthreshold
lam = 4.3
d = (category_mean - x_global_mean) / mk / (sse + s0)  # /mk 是列的广播运算, /(sse + s0)是行的广播运算
zero_index = np.where(np.abs(d)<=lam)
positive_index = np.where(d>lam)
negative_index = np.where(d<-lam)
d[zero_index] = 0
d[positive_index] = d[positive_index] - lam
d[negative_index] = d[negative_index] - lam

used_feature_index = ~np.all(d==0, 0)
d = d[:, used_feature_index]
print("使用的特征数量为", d.shape[1])
d = x_global_mean[:, used_feature_index] + sse[:, used_feature_index] * d * mk
sigma = sse[:, used_feature_index] ** 2

# predict
X_test = X_test.values[:, used_feature_index]
y_prob = np.zeros(shape=(C, len(X_test)))
for i in range(len(X_test)):
    stand_diff = (d - X_test[i, :]) ** 2
    y_prob[:, i] = -(stand_diff / 2 / sigma).sum(axis=1)
y_pred = category[y_prob.argmax(axis=0)]
print("预测精度为", len(y_pred[y_pred == y_test.values.ravel()])/len(y_pred))
print(y_pred)
print(y_test.values)
print(y_pred==y_test.values)