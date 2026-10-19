## 级数

### 级数是否收敛的判别方法

#### 比值判别法

#### 莱布尼茨判别法

### 收敛的分类

1. 绝对收敛：若$\sum_{n=1}^{\infty}a_n$满足$\sum_{n=1}^{\infty}|a_n|$收敛，则称$\sum_{n=1}^{\infty}a_n$为绝对收敛
2. 条件收敛：若$\sum_{n=1}^{\infty}a_n$收敛，$\sum_{n=1}^{\infty}|a_n|$发散，则称$\sum_{n=1}^{\infty}a_n$为条件收敛，其中$C$为常数

### 幂级数

- 标准幂级数：$a_0+a_1x+a_2x^2+\cdots+a_nx^n$
- 一般幂级数：$a_0+a_1(x-x_0)+a_2(x-x_0)^2+\cdots+a_n(x-x_0)^n$

#### 定理1（收敛半径）

幂级数$\sum_{n=0}^{\infty}a_n(x-x_0)^n$的收敛域$J$为以下情形之一

1. $J=\{x_0\}$或$J=\{-\infty,+\infty\}$
2. $\exists R>0,s.t. (x_0-R,x_0+R)\subseteq J \subseteq [x_0-R,x_0+R]$

此时称$J$区间的长度的一半为**收敛半径**，记作$R$。

#### 定理2（收敛半径计算）

设$\sum_{n=0}^{\infty}a_n(x-x_0)^n$满足$\lim_{n\to\infty}\frac{|a_{n+1}|}{|a_n|}=\rho$，则该幂级数的收敛半径$R=\frac{1}{\rho}$。
