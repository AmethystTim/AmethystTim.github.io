## 陪集

### 定义1

设$H$是群$G$的一个子群，$a\in G$，则称群G的子集

$$aH=\{ax|x\in H\}$$

为群G关于子群H的一个左陪集，而称

$$Ha=\{xa|x\in H\}$$

为群G关于子群H的一个右陪集

> 左陪集和右陪集一般不相等，特别地，当群G为交换群的时候他们相等

#### 左陪集性质

1. $a\in aH$
2. $a\in H\Leftrightarrow aH = H$
3. $b\in aH\Leftrightarrow aH=bH$
4. $aH=bH\Leftrightarrow a^{-1}b\in H（或b^{-1}a\in H）$
5. 若$aH\cap bH\neq \Phi，则aH=bH$

> 根据性质5，不难得出群G的所有左陪集可以构成群G的一个**分类**，其两个元素属于同一个类当且仅当$a^{-1}b\in H$
>
> 若aH,bH,...为群G关于子群H的所有不同左陪集，则有
>
> $$G=aH\cup bH\cup cH\cup \dots$$
> 
> 称其为群G关于**子群H的左陪集分解**，称$\{a,b,c,\dots\}$是G关于子群H的一个**左陪集代表系**

### 定理1

设H是群G的一个子群，又令

$$L=\{aH|a\in G\}\quad R=\{Ha|a\in G\}$$

则L和R之间存在一个双射，从而左、右陪集的个数或者都为无限或者都有限且个数相等

存在一个双射$\phi:aH\to Ha^{-1}$

单射：若Ha=Hb，则$ba^{-1}\in H$，从而$(b^{-1})^{-1}a^{-1}\in H$，故有$a^{-1}H=b^{-1}H$

满射：对任意$Ha\in R$，总存在$a^{-1}\in G$是的$\phi(a^{-1}H)=Ha$

### 定义2

群G中关于子群H的互异的左（或右）陪集的个数叫做H在G里的**指数**，记作

$$(G:H)$$

> 例如$(S_3:H)=3$，当然，指数可能无限也有可能有限

## Lagrange定理

设H是G的一个子群，则

$$|G|=|H|(G:H),\quad 即(G:H)=\frac{|G|}{|H|}$$

> 任何子群的阶和指数都是群G的阶的因数

证明：令$(G:H)=s$，且

$$G=a_1H\cup a_2H\cup\dots\cup a_s H$$

是G关于子群H的左陪集分解。由于

$$\phi:a_1h\to a_jh(\forall h\in M)$$

==是左陪集$a_iH$到$a_jH$的一个双射??==，从而$|a_iH|=|a_jH|$

### 推论1

有限群中每个元素的的阶都整除群的阶

> 注：素数阶群必为循环群

### 定理3

设G是一个有限群，又K≤H≤G ，则

$$(G:K)(H:K)=(G:K)$$

证明：由Lagrange定理，$|G|=|H|(G:H)=|K|(G:K)$，而$|H|=|K|(H:K)$，可得结论成立

### 定理4

设H，K是群G的两个有限子群

$$|HK|=\frac{|H|\cdot|K|}{|H\cap K|}$$

证明：

设$H\cap K≤H，设\frac{|H|}{|H\cap K|}=m$，且

$$H=h_1(H\cap K)\cup h_2(H\cap K)\cup\dots \cup h_m(H\cap K)$$

右乘$K$有$HK=h_1K\cup h_2K\cup\dots \cup h_mK$

> 证明 $(H \cap K)K \subseteq K$： 
> 
> 任取 $x \in (H \cap K)K$，则存在 $h \in H \cap K$ 和 $k \in K$ 使得 $x = hk$。
> 
> 因为 $H \cap K$ 是 $H$ 和 $K$ 的交集，所以 $h \in K$。
> 
> 既然 $h$ 和 $k$ 都属于 $K$，而 $K$ 是一个子群，根据子群的封闭性质，$hk \in K$。因此，$x \in K$。这说明 $(H \cap K)K \subseteq K$。
> 
> 证明 $K \subseteq (H \cap K)K$： 任取 $k \in K$。
> 
> 注意到单位元 $e \in H \cap K$（因为 $e$ 是群 $G$ 的单位元，同时属于所有子群）。
> 
> 所以我们可以写 $k = ek$，其中 $e \in H \cap K$ 且 $k \in K$。
> 
> 这表明 $k \in (H \cap K)K$。因此，$K \subseteq (H \cap K)K$。

又因为$h_iK\cap h_jK=\Phi,i\neq j$

从而$|HK|=m|K|$

### 推论

设p,q是两个素数且p<q，则pq阶群G最多有一个q阶子群