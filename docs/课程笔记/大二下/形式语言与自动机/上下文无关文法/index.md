## 自然语言的文法

- <名词短语><动词短语>
- <动词><名词短语>
- ……

## 回文

若字符串$w\in \Sigma^*$满足$w=w^R$，则称字符串$w$为回文

如果语言L的字符串都是回文，则称L为回文语言

$$L=\{w\in \Sigma^*|w=w^R\}$$

例：字母表Σ = {0, 1} 上的回文语言

如何定义？

使用嵌套定义表示这种递归的结构：令A为回文语言中的字符串

有$A\to \epsilon,A\to0,A\to 1$

$A\to0A0,A\to1A1$

## 上下文无关文法(CFG)

G是一个四元组$G=(V,T,P,S)$

- V:变元的有穷集, 变元也称为非终结符或语法范畴
- T:终结符的有穷集（即字母表），且$V\cap T=\phi$
- P:产生式的有穷集，每个产生式包括:
    - 一个变元，称为产生式的头(head)或左部；
    - 一个产生式符号→，读作定义为;
    - 一个$(V⋃T)^∗$中的符号串, 称为体(body)或右部（由变元和终结符组成的字符串）
- S:初始符号S∈V，表示文法开始的地方

> 产生式：$A\to \alpha，读作A定义为\alpha$，若有多个A的产生式，<br>
> 则可以简写为$A\to\alpha_1|\alpha_2|\dots|\alpha_n$<br>
> 文法中变元A的全体产生式，称为A产生式<br>

例：回文字符串

定义变元A，有产生式集$\{A\to 0|1|0A0|1A1|\}$

## 归约和派生

- 从字符串到文法变元的分析过程, 称为归约
- 从文法变元到字符串的分析过程, 称为派生

### 定义

若上下文无关文法 $G = (V, T, P, S)$, 设 $α, β, γ ∈ (V ∪ T)^∗, A ∈ V , A → γ ∈ P$, 那么称在$G$中由 $αAβ$ 可**派生**出 $αγβ$, 记为

$$\alpha A \beta\xRightarrow[G]{} \alpha \gamma \beta$$

相应地，称$\alpha\gamma\beta$可归约为$\alpha A \beta$

> 派生指的是用产生式$A\to \gamma$的右部分$\gamma$替换串$\alpha A\beta$中变元A得到串$\alpha\gamma\beta$的过程

例：使用算术表达式文法$G_{exp}$，将a*(a+b00)归约的过程

<img src="xsyy9-1.png" style="width:150px;margin-right:30%;" align=right>

1. I*(I+I00)
2. I*(I+I0)
3. I*(I+I)
4. E*(E+E)
5. E*(E)
6. E*E
7. E

## 最左派生与最右派生

为限制派生的随意性, 要求只替换符号串中最左边变元的派生过程, 称为最左派生；只替换最右的，称为最右派生，分别记为

$$\xRightarrow[lm]{}\quad \xRightarrow[rm]{}$$

> 任何派生都有等价的最左派生和最右派生

## 文法的语言

定义：上下文无关文法$G=(V,T,P,S)$的语言定义为

$$L(G)=\{w|w\in T^*,S\xRightarrow[G]{*}w\}$$

需要满足:

1. w仅由终结符组成
2. 初始符号S能派生出w

## 上下文无关语言(CFL)

如果语言$L$是某个 CFG $G$ 定义的语言, 即 $L = L(G)$, 则称L为**上下文无关语言**

> 之所以称之为“上下文无关”，是因为文法派生的每一步$\alpha A\beta \Rightarrow \alpha\gamma\beta$符号串$\gamma$仅根据A的产生式派生，而无需依赖A的上下文$\alpha$和$\beta$

## 文法的歧义性

如果CFG $G$使某些符号串有两棵不同的语法分析树，则称该文法是**歧义的**

例：算数表达式的文法$G_exp$中，对句型a+a∗a有下面两棵语法树

<img src="xsyy9-2.png" style="width:400px;">

### 固有歧义性

定义:定义同样的语言可以有多个文法，如果上下文无关语言L的所有文法都是有歧义的，那么称语言L是**固有歧义的**。

## 文法的化简

1. 消除无用符号
2. 消除$\epsilon$-产生式($A\to \epsilon$)
3. 消除单元产生式($A\to B$)

### 消除无用符号

定义:CFG$G = (V, T, P, S)$, 符号$X ∈ (V∪T)$:

- 如果$S\xRightarrow{*}αXβ$,称X是**可达的**
- 如果$\alpha X\beta\xRightarrow{*}w(w\in T^*)$，称X是**产生的**
- 如果X同时是产生的和可达的，那么称X是**有用的**，否则X是无用符号

> 注意:先寻找并消除全部非“产生的”符号，再寻找并消除全部非“可达的”符号，否则可能消除不完整

### 消除epsilon-产生式

- 假如$A\to \epsilon$，则A是可空的
- 如果$B\to \alpha$且$\alpha$中的每个符号都是可空的，那么B是可空的

例：消除CFG $G=(\{S,A,B\},\{a,b\},P,S)$的$\epsilon$-产生式。$S\to AB,A\to AaA|\epsilon,B\to BbB|\epsilon$

先确定全部可空的变元，再替换全部带有可空符号的产生式

## 格雷巴赫范式（GNF）

- 每个不带$\epsilon$的CFL都可以由这样的CFG G定义，G的每个产生式的形式都为

$$A\to a\alpha$$

> 其中 A 是变元, a 是终结符, α 是零或多个变元的串

- 特点
    - GNF的每个产生式都会引入一个终结符
    - 长度为n的串的派生正好是n步

例：将下列文法转换为GNF

$$\begin{split}
    S\to AB\\
    A\to aA|bB|b\\
    B\to b
\end{split}$$

转换为GNF后：

$$\begin{split}
    S\to aAB|bBB|bB\\
    A\to aA|bB|b\\
    B\to b
\end{split}$$