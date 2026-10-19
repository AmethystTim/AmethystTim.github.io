## 正规子群

## 定义1

设N是群G的一个子群。如果对G中每个元素a都有

$$aN=Na,即aNa^{-1}=N$$

则称N是G的一个正规子群

> 正规子群的任何一个左陪集都是一个右陪集，因此简称为陪集。

- 任意一个群G都至少有其平凡子群$\{e\}$和G本身是其正规子群，称为G的**平凡正规子群**，G的其他正规子群若存在称为G的**非平凡正规子群**
- 任何一个群G的中心都是正规子群，有$C \trianglelefteq G$
- 若$N\trianglelefteq G 又有 N\neq G，记为 N \triangleleft G$

例：$N=\{(1),(123),(132)\}$是三元对称群$S_3$的一个正规子群，但是$S_3$的三个子群

$$H_1=\{(1),(12)\},H_2=\{(1),(13)\},H_3=\{(1),(23)\}$$

都不是$S_3$的正规子群

对$S_3$中任意元素$\sigma$有$\sigma N\sigma^{-1}=\{(1),\sigma (123)\sigma^{-1},\sigma(132)\sigma^{-1}\}=N$

### 定理1

设群是G，N≤G，则$N\trianglelefteq G\Leftrightarrow aNa^{-1}\subseteq N(\forall a \in G)$

必要性显然

充分性证明：设对G中任意元素a有$aNa^{-1}a\subseteq Na$，即$aN\subseteq Na$

又由于$a^{-1}Na\subseteq N$可得$Na\subseteq aN$因此$aN=Na$，即$N\trianglelefteq G$

例：n元交代群$A_n$是n元对称群$S_n$的一个正规子群

证明：任意的n元置换$\sigma$与其逆$\sigma^{-1}$具有相同的奇偶性从而易知：$\sigma A_n \sigma^{-1}\subseteq A_n$，故$A_n\trianglelefteq S_n$

> 正规子群的正规子群不一定是原群的正规子群。即，正规子群不具有有传递性

### 定理2

设$\phi$是群G到群$\bar{G}$的**同态满射**，则

1. $N为G正规子群\Rightarrow \phi(N)\trianglelefteq\bar{G}$
2. $\bar{N}为\bar{G}正规子群\Rightarrow \phi^{-1}\trianglelefteq (N)G$

### 定理3 

群G的一个正规子群与一个子群的乘积是一个子群；两个正规子群的乘积仍是一个正规子群

### 陪集乘法

设N是群G的一个正规子群，任取二陪集aN和bN，根据群中子集乘法有

$$(aN)(bN)=a(Nb)N=a(bN)N=(ab)NN=abN$$

即$(aN)(bN)=(ab)N$

- 群G关于N的陪集集合在陪集乘法下构成一个代数系统（也可以说陪集乘法是全体陪集的一个代数运算）

### 定理4

群G的正规子群N的全体陪集对于陪集乘法作成一个群，称为G关于N的**商群**，记为$G/N$

证明：首先陪集满足乘法结合律

其次，考虑单位元和逆元的存在问题，N为单位元显然

由于$(a^{-1}N)(aN)=a^{-1}aN=N$

所以$a^{-1}N$是$aN$的逆元

> $(aN)^m=a^mN(\forall m \in Z)$
> 
> $|G/N|=(G:N)$

### 定理5

设G是一个pn阶有限交换群，其中p是一个素数，则G有p阶元素，从而有p阶子群

> 实际上，当G不是交换群的时候仍然成立


## 定义2

每个子群都是正规子群的非交换群，称为哈密顿群

例： 四元数群$G=\{1,i,j,k,-1,-i,-j,-k\}$（四元数群：）

首先G是非交换群，其次G的真子群只有$<-1>,<i>,<j>,<k>$

$<-1>\trianglelefteq G$

令$N=<i>,x\in G=\{1,-1,i,-i\}$

有$\forall x\in G$，有$xN=Nx$，则$N\triangleleft G$

同理$<j>,<k>$也是G的正规子群，所以G为哈密顿群

## 定义3

阶大于1只有平凡正规子群的群，称为**单群**

### 定理6

有限交换群G为单群的充要条件是，|G|为素数