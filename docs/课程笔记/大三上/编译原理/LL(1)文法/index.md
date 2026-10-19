## 预测分析法工作流程

从文法开始符号出发，在每一步推导过程中根据当前句型的最左非终结符A和当前输入符号a，选择正确的A-产生式。为保证**分析的确定性**，选出的候选式必须是**唯一**的。

## 预测分析法实现步骤

1. 构造文法
2. 改造文法：消除二义性、消除左递归、消除回溯
3. 求每个变量的FIRST集和FOLLOW集
4. 检查是不是 LL(1) 文法。若是，构造预测分析表
5. 对于递归的预测分析，根据预测分析表为每一个非终结符**编写一个过程**；对于非递归的预测分析，**实现表驱动**的预测分析算法

## 串首终结符 FIRST集

串首第一个符号，并且是终结符，简称首终结符。

给定一个文法符号串α， α的串首终结符集$FIRST(α)$被定义为可以从α推导出的所有串首终结符构成的集合。如果$α\Rightarrow^* ε$，那么ε也在$FIRST(α)$中

$$\forall \alpha\in (V_T\cup V_N)^+, FIRST(\alpha)=\{\alpha|\alpha\Rightarrow^*\alpha\beta,\alpha\in V_T,\beta\in (V_T\cup V_N)^*\}$$

### 定理

对文法G的任意两个具有相同左部的产生式$A\to \alpha|\beta\quad(\alpha,\beta\neq \epsilon)$

若$FIRST(\alpha)\cap FIRST(\beta)=\Phi$，即同一非终结符的各个产生式**互不冲突**，那么可以对G进行自顶向下分析

> 简单理解为：语法分析树的子树不能存在共享的叶子结点

## 非终结符的后继符号集 FOLLOW集

对非终结符A，可能在某个句型中紧跟在A后边的终结符a的集合，记作$FOLLOW(A)$

$$FOLLOW(A)=\{\bold{a}|S\Rightarrow^*\alpha A \bold{a}\beta,\bold{a}\in V_T,\alpha,\beta\in (V_T\cup V_N)^*\}$$

如果A是某个句型的的最右符号，则将结束符“$”添加到FOLLOW(A)中

> 对G的任意两个具有相同左部的产生式$A\to \alpha|\beta$
>
> 若$\beta\Rightarrow^*\epsilon$，则$FIRST(\alpha)\cap FOLLOW(A)=\Phi$<br>
> 若$\alpha\Rightarrow^*\epsilon$，则$FIRST(\beta)\cap FOLLOW(A)=\Phi$

## LL(1)文法

文法G是LL(1)的，当且仅当G的任意两个具有相同左部的产生式A → α | β 满足下面的条件：

1. 不存在终结符a使得α和β都能够推导出以a开头的串（$FIRST(\alpha)\cap FIRST(\beta)=\Phi$）
2. α和β**至多**有一个能推导出ε

若$\beta\Rightarrow^*\epsilon$，则$FIRST(\alpha)\cap FOLLOW(A)=\Phi$<br>
若$\alpha\Rightarrow^*\epsilon$，则$FIRST(\beta)\cap FOLLOW(A)=\Phi$

