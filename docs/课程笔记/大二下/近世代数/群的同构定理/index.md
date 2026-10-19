## 群的同构定理

### 定理1（第一同构定理）

设$\phi$是群G到群$\bar{G}$的一个同态满射，又$Ker \phi\subseteq N \trianglelefteq G,\bar{N}=\phi(N)$，则

$$G/N\cong \bar{G}/\bar{N}$$

<div style="display:flex;justify-content:center;">
    <img src="cxds12-1.png" style="width:400px;">
</div>

#### 推论

设H，N是群G的两个正规子群，且$N\subseteq H$，则

$$G/H\cong G/N/H/N$$

### 定理2（第二同构定理）

设G是群，又$H≤G,N\trianglelefteq G$，则$H\cap N\triangleleft H$，且

$$HN/N\cong H/(H\cap N)$$

证明：首先$HN≤G，$且$N\trianglelefteq HN$，易知

$$\phi: x\to xN(\forall x\in H)\quad (H\to HN/N)$$

是一个同态满射，且同态核为$H\cap N$，由群同态基本定理有$H\cap N \trianglelefteq H$且

$$HN/N\cong H/(H\cap N)$$

### 定理3（第三同构定理）

设G是群，又$N\trianglelefteq G$，$\bar{H}≤G/N$，则

1. 存在G的唯一子群$N\subseteq H$，且$\bar{H}=H/N$
2. 又当$\bar{H}\trianglelefteq G/N$时，有唯一的$H\trianglelefteq G$使得

$$\bar{H}=H/N 且 G/N\cong G/N/H/N$$


## 群的自同构群

### 定理1

设M是一个有代数运算的代数系统，则M的全体自同构群关于变换的乘法作成一个群G，称为M的自同构群

证明：封闭性

$\forall \sigma,\tau\in G$，$\forall a,b\in M$，有$\sigma\tau(ab)=\sigma[\tau(ab)]=\sigma[\tau(a)\tau(b)]=\sigma\tau(a)\cdot \sigma\tau(b)$

逆元存在性

$\forall x\in M$有$\sigma\sigma^{-1}(x)=\sigma^{-1}\sigma(x)=x$

#### 推论1

群G的全体自同构关于变换乘法作成一个群，这个群称为群G的自同构群，记为$AutG$