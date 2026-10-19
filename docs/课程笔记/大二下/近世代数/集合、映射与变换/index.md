## 基本概念

![img](cxds1-1.png){width=600}

## 应用领域

- 密码学
- 编码
- 现代物理
- 现代化学

## 集合

一般使用大写字母表示集合，小写字母表示元素

### 定义

具有某种特定性质的具体的或抽象的对象汇总而成的全体

### 性质

- 确定性
- 唯一性
- 无序性

### 集合与集合之间的关系 & 集合运算

1. 子 $A \subset B（真子集） \quad A \subseteq B（子集）\quad A \subsetneq B（真子集）$
2. 交 $A \cap B$
3. 并 $A \cup B$
4. 补 $A-B\quad \bar{X}$

### 集合性质

- 幂等性 $A\cap A=A\quad A\cup A=A$
- 结合律
- 分配率
- 德摩根律

> 分配律证明
> 
> ......

## 映射与变换

### 映射定义

$集合A,B,\forall x \in A ,\exists 唯一 y\in B与之对应$

定义映射为$\phi: x\to y\quad or \quad \phi(x)=y$

> 判定是否为映射
> 
> $A=\{1,2,3,\dots\},B=Q\quad \phi: x\to x^2$

### 映射类别

- 满射
    - $\phi: A\to B,\forall y\in B,\exists x\in A,st. \quad \phi(x)=y$
- 单射
    - $\phi: A\to B,\forall x_1,x_2\in A,x_1\neq x_2 \Rightarrow\phi(x_1)\neq\phi(x_2)$ 
- 双射
    - 同时满足满射和双射条件

#### 满射的充要条件

设$\phi$是集合A到集合B的一个映射，$A_1\subseteq A,B_1\subseteq B$。则

$$\phi(A_1)\subseteq B\quad \phi^{-1}(B_1)\subseteq A$$

分别称他们为：$A_1$在$\phi$下的像，$B_1$在$\phi$之下的逆像

> 对于“充分必要条件”、“$\Leftrightarrow$”
> 
> 从B推向A是在证A成立的**充分性**，从B推向A是在证B成立的**必要性**

### 逆映射

设$\phi$为从集合A到集合B的一个**双射**，且$\phi(x)=y\quad (x\in A,y\in B)$，则显然法则

$$\phi^{-1}:y\to x,即\phi^{-1}(y)=x$$

便是集合B到集合A的一个双射。称$\phi^{-1}$为$\phi$的**逆映射**

特别的，有

$$(\phi^{-1})^{-1}=\phi$$

#### *有限集*A、B*可以建立双射*的充要条件：

$$|A|=|B|$$

$\Leftrightarrow$ 若$\phi$为A、B两有限集之间的双射，则：

$$|\phi(A)|=|A|,\phi(A)=B$$

于是得：$|A|=|\phi(A)|=|B|$


### 定理

$$|A|=|B|<\infty,\quad\phi是满射\Leftrightarrow\phi是单射$$

#### 两个映射相等的概念

- 设$\phi,\tau$是集合A到集合B的两个映射
- 若$\forall x\in A$，都有

$$\phi(x)=\tau(x)$$

有$\phi=\tau$

### 变换定义

集合A到其**自身**的映射，叫做集合A的一个**变换**

- 满射 “满射变换”
- 单射 “单射变换”
- 双射 “双射变换”
- identity transform $I(x)=x$ “恒等变换”

<figure markdown>
![img](cxds1-2.png){width=400}
<figcaption>三元集上的六元置换</figcaption>
</figure>

### 定理

任意n元集共有n!个双射变换

> 变换个数=全排列数量=n!
> 
> 这种变换又称为“n元/阶置换”



