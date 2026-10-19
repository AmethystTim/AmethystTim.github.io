## RSA算法概述

MIT三位年青学者Rivest，Shamir和Adleman在1978年发现了一种用数论构造双钥体制的方法，称作MIT体制，后来被广泛称之为RSA体制

既可以用于加密，又可以用于数字签名

RSA算法的安全性基于数论中的**大整数分解的困难问题**

## RSA算法步骤

### 密钥生成

选择两个大素数$p$和$q$，计算它们的乘积$n=pq$

计算$n=pq,\varphi(n)=(p-1)(q-1)$

- 公钥$e$，随机选择$e$，使得$1<e<\varphi(n),gcd(e,\varphi(n))=1$
- 私钥$d$，计算$d$，使得$ed\equiv 1\ (\text{mod}\ \varphi(n))$

### 加密

明文$m$，公钥$e$，加密过程如下：

$$c=m^e\ (\text{mod}\ n)$$

### 解密

密文$c$，私钥$d$，解密过程如下：

$$m=c^d\ (\text{mod}\ n)$$

## 补充

### 快速指数运算

$$e=b_k 2^k+b_{k-1} 2^{k-1}+\cdots+b_1 2+b_0$$

$$m^e =(\dots((((m^{b_k})^2)\times m^{b_{k-1}})^2)\dots \times m^{b_1})^2\times m^{b_0}$$

RSA算法的加密和解密运算都用到幂运算和取模运算，根据取模运算的性质，我们可以使用快速幂取模的方式来进行快速且不溢出的运算

快速模幂算法示例：

```python
def quick_pow(m,e,n) 
    ans = 1
    while e:
        if e & 1:
            ans = (ans * m) % n
        m = m * m % n
        e >>= 1
    return ans
```

### RSA的解密正确性

假设$m$是明文，$c$是公钥加密得到的密文，$d$是私钥，$n$是公钥的模数，$e$是公钥的指数。

$$\begin{align*}
    C^d\mod n=(m^e)^d\mod n=m^{ed}\mod n\\
    =m^{k\varphi(n)+1}\mod n\\
    =m\mod n\\
    =m(m<n)
\end{align*}$$

> 由欧拉定理有$m^{k\varphi(n)}\equiv 1\mod n$