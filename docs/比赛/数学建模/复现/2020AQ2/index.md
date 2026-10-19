# Q2
## 使用二分法求解

根据题目所给的制程界限，需要满足：

- 温度斜率绝对值不大于3

$$|u_i-u_{i-1}|<3*0.5$$
    
- 温度上升过程中在150℃~190℃的时间T1

$$60≤T_1≤120$$

- 温度大于217℃时间T2

$$40≤T_2≤90$$

- 最大温度在240~250之间

$$ 240≤\max u≤250$$

考查斜率绝对值，T1、T2以及max u与过炉速度之间的关系

注意到斜率绝对值关于过炉速度单增；T1、T2以及max u关于过炉速度单减，均为单调函数。

<figure markdown>
![2020AQA2-0](2020AQ2-0.png){width=400}
<figcaption>传送带速度(cm/s)与温度斜率</figcaption>
</figure>

<figure markdown>
![2020AQA2-1](2020AQ2-1.png){width=400}
<figcaption>传送带速度(cm/s)与高于217度时间(s)</figcaption>
</figure>

<figure markdown>
![2020AQA2-2](2020AQ2-2.png){width=400}
<figcaption>传送带速度(cm/s)与位于150-190℃之间时间(s)</figcaption>
</figure>

<figure markdown>
![2020AQA2-3](2020AQ2-3.png){width=400}
<figcaption>传送带速度(cm/s)与峰值温度(℃)</figcaption>
</figure>

可由**二分法**求得各个参数对应的最大速度$v_i$，$\min(v_i),i\in \{1,2,3,4\}$即为最大过炉速度

最终求得最大过炉速度为$79.492cm/min$
