## 问题一

Develop and analyze a mathematical model that will assist negotiators to respond to a fixed set of water supply and demand conditions. Use the model to inform dam operations: When the water level in Lake Mead is M and the water level in Lake Powell is P, how much water should be drawn from each lake to meet stated demands? If no additional water is supplied (from rainfall, etc.), and considering the demands as fixed, how long will it take before the demands are not met? How much additional water must be supplied over time to ensure that these fixed demands are met?

## 模型假设

1. 米德湖和鲍威尔湖仅用于一般的用水和水电生产
2. 题中所述五州的所有水资源和电力均来自米德湖和鲍威尔湖
3. 在计算有关五个州的用水量时，将**公共用水**分为`住宅类`，包含灌溉、水产养殖、牲畜的`农业类`，包含火电、矿业的`工业类`；**水电**分为包含商业与住宅的`住宅类`，包含5%工业的`农业类`，包含其余部分工业以及交通运输的`工业类`
4. 运输过程中水电的损失与运输距离成正比
5. 把湖作为一个圆柱形的容器
6. 在从鲍威尔湖到米德湖的过程中，不考虑支流的影响(因为这条河的主流水比支流多得多)

## 资源分配网络构建

由于在水运和水电的过程中存在不可避免的自然损失和消耗，引入运输速率来衡量从大坝i到j州的传输效率，即成功传播的百分比

<figure markdown>
![2022](2022mcmb0.png){width=600}
<figcaption>供水和电力需求的分配网络</figcaption>
</figure>

令$v_{ij}^{(t)}$为t时刻大坝i可运输至j州的用于日常使用的水体积

令$\alpha_{ij}$为传输效率

令$d_{j}$为单位时间内j州的日常用水的需求量

根据上述定义，我们可以得到如下恒等式

$$\sum_{i=1}^{2}\alpha_{ij}v_{ij}^{(t)}=d_j^{water},j=1,2,3,4,5\quad(1)$$

对于用电同理，令$w_{ij}^{(t)}$表示t时刻大坝i可运输至j州的电能

令$\gamma_{ij}$为传输效率，我们可以得到如下恒等式

$$\sum_{i=1}^{2}\gamma_{ij}w_{ij}^{(t)}=d_{j}^{elec},j=1,2,3,4,5\quad(2)$$

进一步得到更新鲍威尔湖的t+1时刻储水量$V_1^{(t+1)}$的公式

$$V_1^{(t+1)}=V_1^{(t)}-\sum_{j=1}^{5}(v_{1j}^{(t)}+u_{1j}^{(t)})+v_f\Delta t\quad(3)$$

> $v_f$表示来自格伦峡谷上游的流入速率，$\Delta t$表示一定间隔的时间长度

对于米德湖t+1时刻的储水量$V_2^{(t+1)}$，我们可以同理可得迭代公式

$$V_2^{(t+1)}=V_2^{(t)}-\sum_{j=1}^{5}(v_{2j}^{(t)}+u_{2j}^{(t)})-v_f\Delta t\quad(4)$$

> 与鲍威尔湖迭代公式不同之处在于最后一项为“-”，因为米德湖有部分水流入鲍威尔湖

<figure markdown>
![2022](2022mcmb1.png){width=600}
<figcaption>双坝串联示意图</figcaption>
</figure>

为了简化模型，将两个大坝简化为两个圆柱体进行计算，可以得出t时刻水坝i的水位

$$h_i^{(t)}=\frac{V_i^{(t)}}{S_i},i=1,2\quad(5)$$

> $S_i$表示湖底部面积

接下来考虑电能的转化，定义$u_{ij}^{(t)}$为t时刻大坝i流向j州用于发电的水的体积，根据物理学中的机械能守恒定律，可以得到如下能量转化公式

$$w_{ij}^{(t)}=\beta_i\rho_{water}u_{ij}^{(t)}gh_i\quad i\in \{1,2\},j\in\{1,2,3,4,5\}\quad(6)$$

> $\beta_i$表示发电机的能量转化效率，$\rho_{water}$表示水的密度，$g$表示重力加速度

与此同时需要保证当水位高度$h_i$大于等于发电所需最低高度$h_i^{lowest}$，因为发电机功率与大坝水位高度成正比，定义比例系数$\eta_i$，有总发电功率

$$\sum_{j=1}^5w_{ij}^{(t)}≤\begin{cases}\eta_ih_i^{(t)}\Delta t&h_i≥h_i^{lowest}\\0&else\end{cases},i=1,2\quad(7)$$

为了尽量节省水资源，需要将$v_{ij}^{(t)}$与$u_{ij}^{(t)}$控制到最小

得到目标函数为

$$\min_{v_{ij}^{(t)},u_{ij}^{(t)}}\sum_{i=1}^{2}\sum_{j=1}^{5}v_{ij}^{(t)}+\sum_{i=1}^{2}\sum_{j=1}^{5}u_{ij}^{(t)}\quad(8)$$

## 水分配计算

给出如下伪码

$input:\{\alpha_{ij},\gamma{ij},v_f,S_i,\beta_i,\rho_{water},g,h_i^{power}\}$

$output:\{v_{ij}^{(t)},u_{ij}^{(t)},t_1,t_2\},for \quad i=1,2,j=1,2,3,4,5$

初始化:$\{V_{1}^{(0)},V_{2}^{(0)},h_1^{(0)},h_2^{(0)}\}$
