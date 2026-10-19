## PDA（下推自动机）

具备与上下文无关文法等价的定义语言的能力

可以看作ε-NFA和一个栈(stack)的结合

## 抽象装置

![img](xsyy10-1.png){width=400}

## 形式定义

P定义为七元组

$$P=(Q,\Sigma,\Gamma,\delta,q_0,Z_0,F)$$

1. $Q$：有穷状态集;
2. $Σ$: 有穷输入符号集 (input alphabet);
3. $Γ$: 有穷栈符号集(stack alphabet);
4. $δ$:$\quad Q×(Σ∪{ε})×Γ→ 2^{Q×Γ^∗}$:状态转移函数;
5. $q_0∈Q$：初始状态;
6. $Z_0∈Γ−Σ$:初始栈底符号(start stack symbol);
7. $F⊆Q$：接受状态集。

例：构造$\{0^n1^n|n≥1\}$的PDA

- 当PDA读入0，栈压入0
- 每读入一个1，从栈中弹出一个0
- 当扫描完整个字符串，如果栈顶刚好把初始符号$Z_0$露出来，那么可以通过空转移跳转到接受状态
- 状态转移
    - $\delta(q_0,0,Z_0)=\{(q_0,0Z_0)\}$
    - $\delta(q_0,0,0)=\{(q_0,00)\}$
    - $\delta(q_0,1,0)=\{(q_1,\epsilon)\}$
    - $\delta(q_1,1,0)=\{(q_1,\epsilon)\}$
    - $\delta(q_0,\epsilon,Z_0)=\{(q_2,Z_0)\}$

## 瞬时描述

为描述PDA瞬间的格局，定义$Q×Σ^∗×{Γ^∗}$中三元组$(q,w,γ)$为瞬时描述(**ID**)

- q是当前状态
- w是剩下的输入符号串(remaining input)
- γ是栈中的符号串(stack contents)（栈顶符号在前）


## 转移符号

在PDA的一个动作下，由$\text{ID} I$到$\text{ID} J$的变化，称为$\text{ID}$的转移。

例： $(q, aw, Zα)$ 到 $(p, w, βα)$ 表示状态q读取字符串aw，“消耗”字符$a$，使用$\beta$替换栈顶元素$Z$

$⊢P$,或者$⊢$（当所指PDA明确时） 记为

$$(q, aw, Zα)⊢(p, w, βα)$$

> 扩展$\vdash 到\vdash^*$，表示经过0个或多个动作进行转移

例：语言$L_{01}=\{0^n1^n|n≥1\}$的PDA，识别000111的ID序列

<img src="xsyy10-2.png" align=right>

$$\begin{split}
    (q_0,000111,Z_0)\vdash\\
    (q_0,00111,0Z_0)\vdash\\
    (q_0,0111,00Z_0)\vdash\\
    (q_0,111,000Z_0)\vdash\\
    (q_1,11,00Z_0)\vdash\\
    (q_1,1,0Z_0)\vdash\\
    (q_1,\epsilon,Z_0)\vdash\\
    (q_2,\epsilon,Z_0)
\end{split}$$

## 下推自动机接受的语言

PDA $P = (Q, Σ, Γ, δ, q_0, Z_0, F)$的语言一般通过终态方式接受来定义，记为$\text{L}(P)$，定义为

$$\text{L}(P)= \{w |(q_0, w,Z_0)⊢^∗(p, ε, γ), p ∈ F\}$$

即能够使PDA到达终态的符号串的集合

另外一种PDA定义的语言是以空栈方式接受的语言，记为$N(P)$，定义为

$$\text{N}(P)=\{w|(q_0,w,Z_0)\vdash* (p,\epsilon,\epsilon)\}$$

即能够使得PDA的栈变空的符号串集合

例：设PDA接受的语言示例L：识别$L_{wwr}=\{ww^R|w\in (0+1)^*\}$的PDA $P$

以终态方式接受：

<img src="xsyy10-3.png">

以空栈方式接受：

<img src="xsyy10-4.png">

## 从终态方式到空栈方式

### 定理

如果PDA $P_F$以终态方式接受语言$L$, 那么一定存在PDA $P_N$以空栈方式接受$L$

同样的，如果PDA $P_N$以空栈方式接受语言$L$, 那么一定存在PDA $P_F$以终态方式接受$L$

## 下推自动机与文法的等价性

### 由CFG到PDA

PDA模拟文法的最左派生，当输入的符号串消耗完时，处于按空栈接受的状态或者非接受的状态

给定CFG $G=(V,T,P,S)$构造PDA $(\{q\},T,V\cup T,\delta, q, S, \Phi)$

> S是初始栈底符号

- 对每个变元A：(读取空串，压入产生式)

$$\delta(q,\epsilon,A)=\{(q,\alpha)|A\to \alpha\in P\}$$

- 对终结符a：(读取字符a，弹出字符a)

$$\delta(q,a,a)=\{(q,\epsilon)\}$$

例：根据语言$L=\{0^n1^m|1≤m≤n\}$对应的如下文法，设计相应的PDA

文法产生式：$S\to AB,A\to 0A|\epsilon,B\to 0B1|01$

有PDA的转移函数

$$\begin{split}
    \delta(q,\epsilon,S)=\{(q,AB)\}\\
    \delta(q,\epsilon,A)=\{(q,0A),(q,\epsilon)\}\\
    \delta(q,\epsilon,B)=\{(q,0B1),(q,01)\}\\
    \delta(q,0,0)=\{(q,\epsilon)\},\delta(q,1,1)=\{(q,\epsilon)\}
\end{split}$$

例如识别00011：有$S\to AB \to 0AB \to 0B \to 00B1 \to 00011$

<img src="xsyy10-5.png">

> 这里只讨论正确的路径，由于不确定性必然会出现dead path

#### 下推自动机和文法的等价性

对于任何CFL $L$，一定存在PDA $P$，使得$L=\text{N}(P)$

#### 构造与GNF格式文法等价的PDA

回顾GNF：每个不带$\epsilon$的CFL都可以由这样的CFG $𝐺$定义,$𝐺$中每个产生式的形式都为

$$A\to a\alpha$$

> $A$是变元, $a$是终结符, $\alpha$是零或多个变元的串

如果GNF格式的CFG $G = (V, T, P, S)$, 那么构造 PDA $P_N= ({q}, T, V, δ, q, S, \Phi)$

为每个产生式，定义$\delta$为

$$\delta(q,a,A)=\{(q,\beta)|A\to a\beta \in P\}$$

例：文法$S\to aAA,A\to aS|bS|a$为GNF格式，构造与之等价的PDA

<img src="xsyy10-6.png" style="width:300px">

> 注意这里GNF的$\delta$函数是$\delta(q,a,A)=\{(q,\beta)|A\to a\beta \in P\}$

### 由PDA到CFG

如果 PDA $P_N$, 有$L= N(P)$, 那么$L$是上下文无关语言

构造：设PDA $P_N= (Q, Σ, Γ, δ, q_0, Z_0, ∅)$，那么构造CFG $G = (V, Σ, P, S)$,

其中$V=\{[qXp]|p,q\in Q,X\in \Gamma\}\cup\{S\}$，[qXp]表示从状态q出发，弹出栈符号串X，到达状态p

产生式集合$P$包括

1. $\forall p\in Q$，构造产生式$S\to[q_0,Z_0,p]$
2. 对$\forall(p,Y_1Y_2\dots Y_n)\in \delta(q,a,X)$构造$|Q|^n$个产生式$[qXr_n]\to a[pY_1r_1][r_1Y_2r_2]\dots[r_{n-1}Y_nr_n]$
3. 若$Y_1Y_2\dots Y_n=\epsilon$，则有产生式$[qXp]\to a$

> 其中$a ∈ Σ ∪ {ε}, X,Y_i∈ Γ, r_1,r_2,⋯,r_n$是$Q$中各种可能组合的n个状态，即$r_i$ 可以是$Q$中任意一个状态

例：将PDA $P=(\{q_0,q_1\},(0,1),\{X,Z_0\},\delta,q_0,Z_0)$转为CFG

<img src="xsyy10-7.png" style="width:300px">

初始符号产生式:

$$\begin{split}
    S\to [q_0Z_0q_0]\\
    S\to [q_0Z_0q_1]
\end{split}$$

$\delta(q_0,0,Z_0)=\{(q_0,XZ_0)\}$产生式

$$\begin{split}
    [q_0Z_0q_0]\to 0[q_0Xq_0][q_0Z_0q_0]\\
    [q_0Z_0q_1]\to 0[q_0Xq_0][q_0Xq_1]\\
    [q_0Xq_0]\to 0[q_0Xq_1][q_1Xq_0]\\
    [q_0Xq_1]\to 0[q_0Xq_1][q_1Xq_1]
\end{split}$$