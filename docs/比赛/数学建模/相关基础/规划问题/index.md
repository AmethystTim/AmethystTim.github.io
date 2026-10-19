## 线性规划

线性规划求解需分为两部分：目标函数(max, min)和约束条件(s.t.)，求解前应该转化为标准形式$\min c^T x$

$$\begin{cases}
    Ax\leq b\\
    Aeq*x=beq
    lb\leq x\leq ub
\end{cases}$$

> lb与ub分别为x的上界与下界

### scipy库求解

```python
from scipy import optimize
import numpy as np
res = optimize.linprog(c, A, b, Aeq, beq, LB, UB, X0, OPTIONS)
```

## 整数规划

整体与线性规划相同，只是额外增加了部分变量为整数的约束

基本求解框架为**分支定界法**：去除整数约束得到“松弛模型”，利用线性规划的方法进行求解，若某个变量不是整数 ，在松弛模型上，分别添加约束$x\leq floor(A)$和$x\geq ceil(A)$，然后再分别求解，这个过程叫做分支，当节点求解结果中所有变量都是整数时，停止分支