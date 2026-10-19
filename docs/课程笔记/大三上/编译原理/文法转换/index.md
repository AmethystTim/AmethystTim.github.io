## 问题1：文法的二义性

二义性通常指某语言L(G)中某个句子存在两棵以上分析树

文法G:$E\to E+E|E*E|(E)|id$存在二义性

改造为$\begin{cases}
    E\to E+T|T\\
    T\to T*F|F
    F\to (E)|id
\end{cases}$

## 问题2：同一非终结符的多个候选式存在共同前缀

多个候选式存在共同前缀导致回溯现象，可以使用提取左公因子来推迟决定

### 提取左公因子算法

对于每个非终结符A，找出它的两个或多个选项之间的最长公共前缀α。如果α ≠ ε，即存在一个非平凡的( nontrivial )公共前缀，那么将所有A-产生式

$$A\to \alpha \beta_1|\alpha\beta_2|...|\alpha\beta_n|\gamma_1|\gamma_2|...|\gamma_m$$

替换为

$$\begin{cases}A\to\alpha A'\gamma_1|\gamma_2|...|\gamma_m\\A'\to\beta_1|\beta_2|...|\beta_n\end{cases}$$

## 问题3：左递归文法

左递归文法会导致递归下降分析器陷入无限循环

如果一个文法中有一个非终结符A使得对某个串α存在一个推导$A\Rightarrow^+Aα$ ，则该文法就是**左递归**的

含有$A→Aα$形式产生式的文法称为是**直接左递归**

经过**两步或两步以上**推导产生的左递归称为是**间接左递归**

### 消除直接左递归

![](compile2-1.png)

### 消除间接左递归

![](compile2-2.png)