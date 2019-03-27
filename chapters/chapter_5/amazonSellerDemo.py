# using utf-8

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(23)
a_positive, a_negative = (90, 10)
b_positive, b_negative = (2, 0)
seller_a = np.random.beta(a_positive+1, a_negative+1, 5000)
seller_b = np.random.beta(b_positive+1, b_negative+1, 5000)
probability = len(seller_a[seller_a > seller_b]) / len(seller_a)
print("卖家A好评率大于B商家好评率的概率为", probability)

# plot
figure = plt.figure(1, figsize=(10, 10))
sns.kdeplot(seller_a, shade=False, label=r"$p(\theta_a | data)$")
sns.kdeplot(seller_b, shade=False, label=r"$p(\theta_b | data)$")
plt.title("Compare Seller A and Seller B")
plt.show()