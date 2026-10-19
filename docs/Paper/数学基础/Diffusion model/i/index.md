## 基本目标

实现生成的图像分布与现实世界的图像分布尽可能接近

如何衡量尽可能接近？

使用最大似然估计

假设训练模型产生的图像分布为$P_{\theta}(x)$，现实世界的图像分布为$P_{data}(x)$

从$P_{data}(x)$中采样$N$个样本，记为$x^1,x^2,\cdots,x^m$

则有

$$\begin{align}
   \theta^*=\arg\max_{\theta}\sum_{i=1}^m\log P_{\theta}(x^i)\\
    \approx \arg\max_{\theta}E_{x\sim P_{data}(x)}\log P_{\theta}(x)\\
    =\argmax_{\theta}\int_{x}p_{data}(x)\log P_{\theta}(x)dx\\
    =\argmax_{\theta}\int_{x}p_{data}(x)\log P_{\theta}(x)-\int_{x}p_{data}(x)\log P_{data}(x)dx\\
    = \arg\max_{\theta}\int_{x}p_{data}(x)\frac{\log P_{\theta}(x)}{\log P_{data}(x)}dx\\
    =\argmax KL(p_{\theta}(x)\Vert p_{data}(x))
\end{align}$$

## FDP(Forward Diffusion Process)

$x_t$可以直接根据$x_0$和$t$推导出来

$$x_t=\sqrt{\prod_{i=1}^{t}\alpha_i}x_0+\sqrt{1-\prod_{i=1}^{t}\alpha_i}\epsilon$$

## RDP(Reverse Diffusion Process)

从一个高斯分布进行采样，通过反转过程生成图像

在$x_0$**已知**的情况下，反向生成过程是一个**确定性的过程**

由贝叶斯公式$q(x_{t-1}|x_t)=\frac{q(x_{t}|x_{t-1})q(x_{t-1})}{q(x_t)}$

且已知$q(x_t|x_{t-1},x_0)$和$q(x_{t-1}|x_0)$

则有

$$q(x_{t-1}|x_t,x_0)=\frac{q(x_t|x_{t-1},x_0)q(x_{t-1}|x_0)}{q(x_t|x_0)}\sim N(x_{t-1},\tilde{\mu}(x_t,x_0),\tilde{\sigma}_t I)$$

在上述表达式中，使用了前向过程中的$t$步骤中随机采样的高斯噪声$\epsilon_t$，

![alt text](image.png)

我们无法通过$x_t$求得$x_{t-1}$，即$q(x_{t-1}|x_t)$

所以可以通过神经网络$p_{\theta}(x_{t-1}|x_t)$来进行拟合
