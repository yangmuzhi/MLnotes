<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# 贝叶斯模型选择

$$
\mathcal{p}(m|\mathcal{D}) = \frac { \mathcal{p}(\mathcal{D}|m)\mathcal{p}(m)d\theta}{\mathcal{p}(D)}
$$

在$\mathcal{p}(m)​$是均匀分布的情况下，$\mathcal{p}(m|\mathcal{D}) \varpropto \mathcal{p}(\mathcal{D}|m) ​$因此计算$\mathcal{p}(\mathcal{D}|m)​$即可

> $\mathcal{p}(\mathcal{D|m})​$计算方式
>
> * $\mathcal{p}(\mathcal{D|m})=\int \mathcal{p}(\mathcal{D}|\theta)\mathcal{p}(\mathcal{\theta|m})d\theta​$当其中密度函数是共轭时，可以直接积分计算。
> * $p(\theta|\mathcal{D},m)=\frac{\mathcal{p}(\mathcal{D}|\theta)\mathcal{p}(\theta|m)}{p(\mathcal{D}|m)}​$当得知$p(\theta|\mathcal{D},m)​$和$\mathcal{p}(\mathcal{D}|\theta), \mathcal{p}(\theta|m)​$的形式时，即可推导出$p(\mathcal{D}|m)​$，$p158​$ $5.3.2​$例子使用的该方法。
>
> 上述两种方法在密度函数共轭时是一致的。

在计算evidence时，$\theta​$的先验比MAP时的要重要。MAP的数据量压过先验，而evidence计算的是先验对似然的加权平均。因为先验的重要性，所以引出了章节后对先验的讨论，如无信息的先验，经验贝叶斯。

# 混合共轭先验

$$\mathcal{p}(\theta)=\Sigma_{k}\, p(z=k)p(\theta|z=k)​$$
$$
\begin{equation}
\mathcal{p}(\theta)=\Sigma_{k}\, p(z=k)p(\theta|z=k)
\end{equation}
$$

$$
p(\theta|\mathcal{D})=\Sigma_{k}\, p(z=k|\mathcal{D})p(\theta|\mathcal{D}, z=k)
$$

对于$p(z=k|\mathcal{D})​$的计算如下

$p(z=k|\mathcal{D})=\frac{p(z=k)p(\mathcal{D}|z=k)}{p(\mathcal{D})}​$，其中$p(z=k)​$为先验，计算$p(\mathcal{D}|z=k)​$利用上面第二种方法，即：

$p(\theta|\mathcal{D},m)=\frac{\mathcal{p}(\mathcal{D}|\theta)\mathcal{p}(\theta|m)}{p(\mathcal{D}|m)}​$中已知道$p(\theta|\mathcal{D}, z=k)​$的后验分布，$p(\theta|z=k)​$的先验分布已知，$p(\mathcal{D}|\theta, z=k)​$的似然已知，则可计算evidence$p(\mathcal{D}|z=k)​$。

```python
from scipy.special import gamma
p1 = gamma(40) * gamma(40) * gamma(30) / gamma(70) /     gamma(20) / gamma(20)
p2 = gamma(40) * gamma(50) * gamma(20) / gamma(70) /     gamma(30) / gamma(10)
propor_1 = p1 / (p1 + p2)
propor_2 = p2 / (p1 + p2)
```

而例子$5.4.4.2​$，计算$p(N_{t}|Z_{t})​$用的是第一种方法，因为$p(\theta|Z_{t})​$先验为Dirichlet分布，为共轭先验，计算积分更容易。

# Hierarchical Bayes

上述的混合先验提出了一个隐藏变量$\, Z \,​$来决定$\, \theta \,​$的先验，而多层贝叶斯是为$\, \theta \,​$的先验分布的参数$\eta​$提供了一个先验，也就是先验参数的先验。

# Empirical Bayes

| 方法            | 描述                                                         |
| --------------- | ------------------------------------------------------------ |
| MLE             | $\hat{\theta}=\mathop{\arg\max}_{\theta} \,p(\mathcal{D}|\theta) $ |
| MAP             | $\hat{\theta}=\mathop{\arg\max}_{\theta} \,p(\mathcal{D}|\theta) \, p(\theta)$ |
| Empirical Bayes | $\hat{\eta}=\mathop{\arg\max}_{\eta}p(\mathcal{D}|\eta)$，不需要考虑先验参数$\, \eta \,​$的分布，通过计算$\int\, p(\mathcal{D}|\theta)p(\theta|\eta)d_{\theta}​$即可 |
| MAP-II          | $\hat{\eta}=\mathop{\arg\max}_{\eta}p(\mathcal{D}|\eta)p(\eta)$，相比empirical Bayes考虑了参数$\,\eta \,$的分布，对于$\,\eta\, $的分布可同样设置为共轭先验 |
| Full Bayes      | $p(\theta, \eta|\mathcal{D}) \varpropto p(mathcal{D}|\theta)p(\theta|\eta)p(\eta)$ |

empirical Bayes 违背了先验设置与数据无关的原则，其观测数据后再设置$\, \theta\, ​$先验分布的参数取值。

> 为癌症率建模$p171​$
>
> 观测为N个城市的癌症人数，设第$\, j \,​$个城市的癌症率为$\, \theta_{j}\, ​$ ，假设所有的$\, \theta_{j} \,​$的先验均为参数为$\, a\, ,b\,​$的Beta分布，则其中$\eta=(a,b)​$。
>
> 在之前第三章，先验通常被赋予uniform的形式，即$a=1, b=1​$，然而用empirical Bayes的方法，参数$\,  \theta \,​$先验分布的参数$a,b​$的选取方式如下：
>
> $$
>  \eta= \mathop{\arg\max}_{\eta}\int\, p(\mathcal{D}|\theta)p(\theta|\eta)d_{\theta} 
> $$
> 因为共轭，所以积分计算也较为简单

> 为学校学生成绩建模
>
> 观测为D个学校，共N个学生的成绩，设第$\, j \, $个学校学生的成绩服从的参数为$\theta_j, \sigma^2$的高斯分布，其中$\sigma^2$已知$\, \theta_j$的服从参数为$\, (\mu, \tau)$的高斯分布。
> $$
> p(\theta, \mathcal{D}|\eta, \sigma^2)=\prod_{j=1}^{D}\mathcal{N}(\theta_j|\mu, \tau)\prod_{i=1}^{N_j}\mathcal{N}(x_{ij}|\theta_j, \sigma^2)
> $$
> 利用第四章公式可以推知$p(\mathcal{D}|\mu, \tau, \sigma^2)​$，再对$\mu, \tau​$求极值即可
>
> 在得到参数$\, \eta \,​$的取值后即得知$\, \theta \,​$先验分布。再利用第四章公式计算参数$\theta​$的取值





























