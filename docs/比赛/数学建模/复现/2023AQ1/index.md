## 初步构想

- 使用L-V模型构建种间竞争
- 对降雨、干旱的定义与量化
- 污染物和栖息地减少对该模型哪些参数有影响？
- 对一个物种而言，他自身包含了哪些特征参数（抗干旱、竞争能力）
- 对每个生物的环境容纳量的量化（影响因素：抗干旱能力、温度、降水量等。。。）
- 数据集？

## 模型假设
- 在一个封闭的生态系统内进行分析
- 植物种群之间仅存在共生与竞争关系
- 每个物种的内禀增长率仅受到温度、降水量的影响
- 不考虑生物的变异

## Lotka-Volterra模型

在一个群落当中，不同的生物之间存在着捕食、竞争、共生关系，在该题条件下，仅考虑竞争和共生关系，在n个种群构成的群落当中，第$i$个种群数量$x_i$满足如下关系式

$$\frac{dx_i}{dt}=x_i(b_i+\sum_{j=1}^{n}a_{ij}x_j)\quad i\in 1,2,\dots ,n$$

> 其中$b_i$表示种群$i$的内禀增长率，$a_ij$表示种群$i$对种群$i$的影响性质和强度

化为矩阵形式

$$\left(\begin{matrix}\frac{dx_1}{dt}\\\dots\\\frac{dx_n}{dt}\end{matrix}\right)=\left(\begin{matrix}x_1&\dots&x_1\\\dots&&\dots\\x_n&\dots&x_n\end{matrix}\right)\left[\left(\begin{matrix}a_{11}&\dots&a_{1n}\\\dots&&\dots\\a_{n1}&\dots&a_{nn}\end{matrix}\right)\left(\begin{matrix}x_1\\\dots\\x_n\end{matrix}\right)+\left(\begin{matrix}b_1\\\dots\\b_n\end{matrix}\right)\right]$$

## 参数b的量化

内禀增长率（b）指在给定的物理和生物的条件下，具有稳定的年龄组配的种群的最大瞬时增长率

仅考虑降雨量与气温对一个种群内禀增长率的影响，经分析可得温度与降水量均为`中间型指标`，即越接近于“最佳指标”，获得的评价越高。

根据中心极限定理，当采样量趋于无穷时，参数b的频率分布越趋近于正态分布，根据以上分析，构造以下公式

$$b=\lambda_1e^{-\lambda_2\left[(\frac{t-t_{op}}{t_{max}-t_{min}})^2+\left(\frac{p-p_{op}}{p_{max}-p_{min}}\right)^2\right]}$$

> 定义$t_{op}$为该种群的最适宜温度
> 
> $p_{op}$为该种群的最适宜年平均降水量
> 
> $t_{max}$与$t_{min}$分别为该种群的适宜温度的上界与下界，年平均降水量同理
> 
> $t_{op}=\frac{t_{max}+t_{min}}{2}$，$p_{op}$同理
> 
> $\lambda_1$与$\lambda_2$为常数，在本问题中我们将$\lambda_1$设定为$3$，$\lambda_2$的取值依据在下文给出

对于植物，定义$t_{max}$为内禀增长率不低于$1$的最高温度，$t_{min}$为内禀增长率不低于$1$的最低温度，得到

$$\lambda_2=2\ln 3$$

引入温带大陆性气候地区的三种植物，植物1,3和植物2,3为共生关系，植物1,2为竞争关系，初始化种群数量，随机生成天气状况，在一封闭生态系统中进行演化得到如下图像

![2023mcmq1-0](2023mcmq1-0.png)

在[https://earthdata.nasa.gov/](https://earthdata.nasa.gov/)查询到科罗拉多地区17-22年的气象数据，代入模型当中，得到如下图像

![2023mcmq1-1](2023mcmq1-1.png)