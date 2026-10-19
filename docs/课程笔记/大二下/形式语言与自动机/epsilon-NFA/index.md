$\epsilon-NFA$：对NFA进行扩展，在不输入任何字符的情况下（空串$\epsilon$），可以**自发地**发生状态转移

- 可简化NFA构造
- $\epsilon-NFA$明确定义了至少一个空转移的NFA
- 在状态表里面多了对应于$\epsilon$的一列

## 定义

带空转移非确定有穷自动机($\epsilon-NFA$)A为五元组$A=(Q,\Sigma,\delta,q_0,F)$

1. $Q$: 有穷状态集；
2. $\Sigma$: 有穷输入符号集或字母表；
3. $\delta$: $Q\times (\Sigma\cup\epsilon)\to2^Q$ 状态转移函数；
4. $q_0\subseteq Q$: 初始状态
5. $F\subseteq Q$: 终结状态集或接受状态集

例：$L=\{w\in\{0,1\}^*|w倒数3个字符至少有一个是1\}$的$\epsilon-NFA$

![img](xsyy5-1.png){width=400}

## 状态的$\epsilon$-闭包

定义：状态q的$\epsilon$-闭包，记为$\text{ECLOSE}(q)$，表示从q经过$\epsilon\epsilon\dots\epsilon$序列可到达的全部状态集合（即经过0个或>0个空转移到达的状态集合）

递归定义：

$q\in\text{ECLOSE}(q)$

$\forall p\in \text{ECLOSE}(q),若r\in\delta(p,\epsilon),则r\in\text{ECLOSE}(q)$

![img](xsyy5-2.png){width=400}

> $\text{ECLOSE}(1)=\{1,2,5,3,4\}\quad\text{ECLOSE}(2)=\{2,3,4\}$

## 扩展状态转移函数

定义：扩展$\delta$到字符串，定义扩展状态转移函数$\hat{\delta}:Q\times\Sigma^*\to 2^Q$为：

$$\hat{\delta}(q,w)=\begin{cases}
\text{ECLOSE}(q)&w=\epsilon\\\text{ECLOSE}(\cup_{p\in\hat{\delta}(q,x)}\delta(p,a))&w=xa
\end{cases}$$

## 定理

如果语言L被$\epsilon-NFA$接受，当且仅当L被DFA接受

子集构造法（消除空转移）：如果$\epsilon-NFA\quad E=(Q_E,\Sigma,\delta_E,q_E,F_E)$，构造DFA

$$D=(Q_D,\Sigma,\delta_D,q_D,F_D)$$

