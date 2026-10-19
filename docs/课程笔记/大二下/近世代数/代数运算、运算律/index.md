## 代数运算

### 定义

集合$M$上的一个法则$\circ$，若对于集合上的每一组有序对$a,b\in M$，总存在唯一的$d\in M$，使得$a\circ b=d$

那么，这样一个法则$\circ$就被称为集合M上的**代数运算**

### T(M)

记$T(M)$为集合M上所有**变换**构成的集合，那么法则“变换乘法/变换合成”将会是这个集合上的一个**代数运算**

证明：

$$\sigma,\tau\in T(M),\forall x\in M\quad \sigma\tau(x)=\sigma(\tau(x))$$

$$\sigma\tau\in T(M)\quad \sigma\circ\tau=\sigma\tau$$

### 恒等变换

$$\sigma\epsilon(x)=\epsilon\sigma(x)=\sigma(x),\forall x \in M \quad for \quad \forall \sigma\in T(M)$$

$$\sigma\epsilon=\epsilon\sigma=\sigma$$

### 双射变换集 S(M)

$$S(M)\subseteq T(M),\forall \phi \in S(M),\phi是一个双射变换$$

> “变换乘法”也是这个集合$S(M)$上的一个**代数运算**

例：给出自然数集N上的两个不同双射变换$\sigma$，$\tau$，s.t. $\sigma\tau\neq\tau\sigma$

$$\sigma:\quad 1\to2,2\to1\quad x\to x$$

$$\tau:\quad 1\to3,3\to1\quad x\to x$$

## 运算律

### 结合律

集合M上的代数运算，对集合上任意三个元素$\forall a,b,c\in M$，都有以下等式成立:

$$(a\circ b)\circ c=a\circ(b\circ c)$$

> 并非所有集合上的代数运算都满足结合律
>
> 如：$a\circ b=ab+1$

例：变换乘法满足结合律

$\forall \sigma,\tau,\phi\in T(M),\forall x\in M$

$$[(\sigma\tau)\phi](x)=(\sigma\tau)(\phi(x))=\sigma[\tau(\phi(x))]$$

$$[\sigma(\tau\phi)](x)=\sigma[(\tau\phi)(x)]=\sigma[\tau(\phi(x))]$$

#### 定理1

集合M, 如果其上的代数运算$\circ$满足结合律，则对M上任意n(n≥3)个元素无论如何加括号, 其结果都一样

> 证明(数学归纳法)
>
> ......


### 交换律

集合M上的代数运算$\circ$，如果对于$\forall a,b\in M$，都有

$$a\circ b=b\circ a$$

#### 定理2

集合M上的代数运算$\circ$，若它既满足结合律又满足交换律，那么任意的n个集合中的元素任意的结合（加括号）和交换位置的前后顺序，其所得的结果都一样

> 证明(数学归纳法)
>
> ......

### 分配率

集合M上的两个代数运算$\circ$和$\oplus$，如果对$\forall a,b,c\in M$，总有

$$a\circ(b\oplus c)=(a\circ b)\oplus(a\circ c)$$

$$(b\oplus c)\circ a=(b\circ a)\oplus(c\circ a)$$

例：给出集合$M=\{1,2,3\}$上既满足结合律又满足交换律的一个代数运算；再给出其上满足交换律但不满足结合律的一个代数运算

$a\circ b=\max(a,b)$

满足交换律，不满足结合律

> 若代数运算满足交换律，则该代数运算“乘法表”对应的矩阵为**对称矩阵**