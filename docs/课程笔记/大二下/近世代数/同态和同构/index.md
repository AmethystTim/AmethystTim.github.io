## 同态

### 定义

集合M上有代数运算$\circ$，集合$\overline{M}$上有代数运算$\bar{\circ}$，若存在映射$\phi,\phi:M\to \overline{M}$，若满足：

$$\phi:a\to\bar{a}\quad b\to \bar{b}$$

$$\phi: a\circ b\to \bar{a}\bar{\circ}\bar{b}$$

即$\overline{a\circ b}=\bar{a}\bar{\circ}\bar{b}$ 或 $\phi(a\circ b)=\phi(a)\bar{\circ}\phi(b)$

则称$\phi$是代数系统$M$到$\bar{M}$的一个同态映射，若其为一个满射，则称代数系统$M$与$\bar{M}$同态，记作

$$M\sim\overline{M}$$

例：$\circ$是数域F上全体n阶方阵的集合$M$上的矩阵乘法代数运算, 是数域$F=\overline{M}$上的普通数乘代数运算

$\phi:\quad M\to\overline{M}\quad \phi:A\to|A|$

问$\phi$是不是一个同态映射？

$|AB|=|A|\cdot|B|\quad \phi(AB)=\phi(A)\cdot\phi(B)$

同时也是一个满射 

得$M\sim\overline{M}$（$M$与$\overline{M}$同态）

### 定理1

代数系统$M$和$\overline{M}$分别有代数运算$\circ、\bar{\circ}$，若$M$~$\overline{M}$，有以下结论成立

1. $\circ$满足结合律, 那么$\bar{\circ}$也满足结合律。
2. $\circ$满足交换律, 那么$\bar{\circ}$也满足交换律。

### 定理2

$\circ$和$\oplus$是代数系统$M$上的两个代数运算,$\bar{\circ}$和$\bar{\oplus}$是代数系统$\overline{M}$上的两个代数运算，$\phi$是$M$到$\overline{M}$的一个满射，且对 $\circ$和$\oplus$，$\bar{\circ}$和$\bar{\oplus}$同态；那么如果$\circ$关于$\oplus$满足左（右）分配律那么$\bar{\circ}$对$\bar{\oplus}$也将满足左（右）分配律

## 同构

### 定义

$\phi$是代数系统$M$到$\overline{M}$关于代数运算$\circ$和$\bar{\circ}$的同态满射，如果$\phi$同时一个单射，那么它就是$M$到$\overline{M}$的一个同构映射。称$M$与$\overline{M}$同构，记为：

$$M\cong \overline{M}$$

- 如果没有这种同构映射，则称$M$与$\bar{M}$不同构。
- $M$到其自身的同态映射叫做$M$的自同态映射，简称为$M$的自同态。
- $M$到其自身的同构映射叫做$M$的自同构映射，简称为$M$的自同构。

> 常见同构映射: $I(x)$恒等映射

例：$M=Z$，$\bar{M}$是偶数集，$\phi\to2n$，是否构成他们之间的同构映射？

> 一定要对运算来讲，仅讨论两个集合无意义

### 性质

- 自反性 $M\cong M$
- 对称性 $M_1\cong M_2\quad M_2\cong M_1$
- 传递性 $若M_1\cong M_2\quad M_2\cong M_3 \quad 则 M_1\cong M_3$

> 若$\phi_1$是$M_1\cong M_2$的同构映射，那么$\phi^{-1}$就是$M_2\cong M_1$的同构映射<br>
> 若$\phi_1$是$M_1\cong M_2$的同构映射，$\phi_2$是$M_2\cong M_3$的同构映射，那么$\phi^{3}=\phi_1\phi_2$就是$M_1\cong M_3$的同构映射

例：集合M上有代数运算$\circ$，集合$\overline{M}$上有代数运算$\bar{\circ}$，$M$与$\overline{M}$同态