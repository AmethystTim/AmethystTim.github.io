## 代换

定义：两个字母表$\Sigma到\Gamma$的函数$s:\Sigma\to 2^{\Gamma^*}$称为**代换**

$\Sigma$中的一个字符a在s的作用下为$\Gamma$上的一个语言$L_a$，即

$$s(a)=L_{a}$$

|语言|封闭性|
|-|-|
|正则语言|交、并、补、连接、同态运算、反转|
|上下文无关语言|代换（对交、补不封闭）|

扩展s的定义到字符串$s(\epsilon)=\epsilon,s(xa)=s(x)s(a)$

再扩展到s语言，对$\forall L\subset\Sigma^*$

有$s(L)=\bigcup_{x\in L}s(x)$

## 代换的封闭性

定理:如果有Σ上的CFL L和代换s, 且每个$a ∈ Σ$的$s(a)$都是CFL, 那么$s(L)$也是$CFL$

## CFL和正则语言

定理：若L是CFL且R是正则语言，则$L\cap R$是CFL

设DFA $D=(Q_1,\Sigma,\delta_1,q_1,F_1)$且$\text{L}(D)=R$

设PDA $P=(Q_2,\Sigma,\Gamma,\delta_2,q_2,Z_0,F_2)$，且$\text{L}(P)=L$