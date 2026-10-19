# 粒子群优化算法

## 背景

粒子群算法(PSO)是一种进化计算技术，在1995年提出，源于对鸟群的捕食行为研究，利用群体的个体信息共享使整个群体的运动在问题求解空间中产生从无序到有序的演化过程从而获得最优解。

## 基础知识

问：已知A为全局最优，B和C如何移动才能到达A点？

![2020AB1-0](2020AB1-0.png){width=400}

该过程如何用数学表达式描述？

1. 某个粒子的移动有大小，有方向（即位移矢量）
2. 每个点的位置都是一个坐标

$$(1,1)=(2,3)+\alpha\to\alpha=(1,1)-(2,3)=(-1,-2)$$

## 基本原理

假设在一个D维的目标搜索空间中，有N个粒子组成的一个群落，其中第i个粒子表示为一个D维向量

$$X_i=(x_{i1},x_{i2},\dots,x_{iD}),i=1,2,\dots,N$$

第i个粒子的"飞行"速度也是一个D维向量，记作：

$$V_i=(v_{i1},v_{i2},\dots,v_{iD}),i=1,2,\dots,N$$

在第t代的第i个粒子向t+1代进化时，根据以下式子更新：

$$v_{ij}(t+1)=wv_{ij}(t)+c_1r_1(t)[p_{ij}(t)-x_{ij}]+c_2r_2(t)[p_{gj}(t)-x_{ij}(t)]$$

$$x_{ij}(t+1)=x_{ij}(t)+v_{ij}(t+1)$$

> $wv_{ij}(t)$项表示粒子的“惯性”

> $c_1r_1(t)[p_{ij}(t)-x_{ij}]$项表示往个体最优解方向

> $c_2r_2(t)[p_{gj}(t)-x_{ij}(t)]$项表示往全局最优解的方向

> $w$为惯性权重，$c_1$表示个体学习系数，$c_2$表示全局学习系数，$r_1,r_2$为随机数

~~假如某粒子下一代适应度更差，则应该阻止该粒子本轮迭代，让其他粒子迭代后进行下一轮迭代？~~

## 基本流程

- 输入参数$w:0.5-0.8$；$c_1,c_2:0.1-2$；$v_{max},x_{max}:$取决于优化函数

1. 初始化种群x
2. 计算个体**适应度***
3. 更新粒子速度->更新粒子位置
4. 并计算新位置的适应度，若新位置适应度更高，将该粒子的位置进行更新，否则不更新
5. 判断是否满足终止条件，是则退出，否则继续迭代

>一般的，优化目标是**最小化**一个函数值，所以个体计算出的函数值越小，适应度越高

## 算法分析

优点：

   - 原理简单
   - 参数少
   - 实现容易

缺点：

   - 容易陷入局部最优
   - 迭代后期收敛速度慢

>在迭代前期应该让$w$较大，将整个搜索空间充分搜索，在迭代后期让$w$较小，充分“向其他粒子学习”

## 算法拓展

1. 实现参数自适应变化
2. 引入其他机制，比如随机扰动，后期压缩最大速度等
3. 结合其他智能算法：遗传算法，模拟退火算法等等

## 案例分析

求解$f(x)=xsin(x)cos(2x)-2xsin(3x)+3xsin(4x)$在[0,50]的最小值

![2020AB1-1](2020AB1-1.png)

```python
import numpy as np

N=20 # 种群个数
d=1 # 维度
cnt_max=2000 # 最大迭代次数
limit=[0,50] #位置界限
vlimit=[-3,3] # 速度界限
w=0.8 # 惯性系数
c1=0.5 # 个体学习参数
c2=0.5 # 全局学习系数

#目标函数
def func(x):
    return x*np.sin(x)*np.cos(2*x)-2*x*np.sin(3*x)+3*x*np.sin(4*x)

# 初始化种群
x=np.random.uniform(low=0, high=50, size=20)
# 初始化个体速度
v=np.random.uniform(low=-3, high=3, size=20)

# 个体历史最优
individual_best=np.copy(func(x))
# 个体历史最优位置
individual_bestx=np.copy(x)
#全局最优解
global_best=100
#全局历史最优位置
global_bestx=0

# 初始化全局最优
for index in range(x.size):
    if global_best<func(x[index]):
        global_best=func(x[index])
        global_bestx=x[index]

# 进行迭代
for cnt in range(cnt_max):
    r1=np.random.rand(1)
    r2=np.random.rand(1)
    for index in range(x.size):
        # 判断v是否在边界内
        v_temp=w*v[index] + c1*r1[0]*(individual_bestx[index]-x[index]) + c2*r2[0]*(global_bestx-x[index])
        if v_temp>=vlimit[0] and v_temp<=vlimit[1]:
            # 判断x是否在边界内
            v[index]=v_temp
        else:
            # 若超出界限，则取边界值
            if v_temp>vlimit[1]:
                v[index]=vlimit[1]
            else:
                v[index]=vlimit[0]
        if x[index]+v[index]>=limit[0] and x[index]+v[index]<=limit[1]:
            x[index]=x[index]+v[index]
            # 更新个体历史最优
            if individual_best[index]>func(x[index]):
                individual_best[index]=func(x[index])
                individual_bestx[index]=x[index]
            # 更新全局最优
            if global_best>func(x[index]):
                global_best=func(x[index])
                global_bestx=x[index]
        
# 输出最优解
print("最小值为:",global_best)
```