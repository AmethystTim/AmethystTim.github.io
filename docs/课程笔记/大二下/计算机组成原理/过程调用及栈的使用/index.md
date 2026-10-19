## 过程调用的步骤

- 调用者caller vs. 被调用者callee
- Caller将参数放在过程可以访问的寄存器x10~x17/a0-a7
- Caller将控制权交给过程(Callee)
- (Callee)获取过程所需的存储资源
- (Callee)执行过程中的操作
- (Callee)将结果值放在调用程序可以访问到的寄存器s0~s11
- (Callee)返回调用位置(x1/ra中的地址)，控制权交还调用者

## 过程调用指令

- 过程调用
    - `jal x1, ProcedureAddress`
        - 讲下一条指令的地址保存在x1/ra中
        - 跳转到目标地址ProcedureAddress
- 过程返回
    - `jalr x0, 0(x1)`
        - 类似`jal`，但是跳转到0+x1中保存的地址
        - 把x0用作目的寄存器（实际x0不会被改变）
        - 同样可用于跳转到计算出的位置

## 过程调用相关的寄存器

- RISC-V实现过程调用的的相关寄存器
    - 返回地址寄存器 `x1(ra)`
    - 参数寄存器 `x10-x17(a0-a7)`
    - 返回值寄存器 `x10-x11(a0-a1)`
    - 保存寄存器 `x8,x9,x18-x27(s0,s1,s2-s11)`
    - 临时寄存器 `x5-x7,x28-x31(t0-t2,t3-t6)`
    - 栈指针 `x2(sp)`

## RISC-V过程调用（人为设置返回地址）

C语言函数调用

```c
int function(int a, int b){
    return (a+b);
}
```

```RISC-V
1000|mv a0, s0
1004|mv a1, s1
1008|addi ra, zero, 1016 # ra=1016
1012|j sum # jump to sum
1016|...  # 跳转返回下条指令
... |
2000|sum: add a0, a0, a1
2004|jr ra  # jalr x0, ra, 0
```
> 伪指令j跳转到给定的地址

例：过程嵌套问题

```c
int numSquare(){
    x=mult(x,x);
    ...;    //5000
}

int main(){
    ...;
    numSqaure(9,10);
    ...;    //4080
}
```

- main调用过程numSquare，寄存器ra保存返回地址4080
- 过程numSquare调用过程mult，寄存器ra需要保存返回地址5000，4080被覆盖
- 调用mult前需提前保存寄存器ra中的返回地址4080

RISC-V过程调用时对需要保护现场的寄存器有如下约定

- 保存 x1(ra), sp, gp, tp, x8-x9(s0-s1),x18-x27(s2-s11)
- 不保存 x10-x17(a0-a7), x5-x7(t0-t2), x28-x31(t3-t6)


## 栈的概念

- 客栈：货栈都指的是临时“保存”人或物的地方

- 在计算机中，是一种数据结构，满足LIFO后进先出原则，在调用函数的时候可以保存寄存器的旧值（主存中），在调用返回时并还原到寄存器中
- 具有**压栈**和**弹栈**两个操作

## 栈的使用

- x86指令集有压栈和出栈命令
- RISC-V没有对应指令
- `sp(x2)`是RISC-V中的栈指针寄存器，用来保存**栈顶地址**
- 按照惯例，RISC-V中的栈按照从高到低的地址顺序增长（栈底高地址，栈顶低地址）
    - 压栈，指针向下移动，sp减少
    - 弹栈，指针向上移动，sp增加
- 栈指针寄存器sp总是**指向栈中最后使用的空间**
- **首先将栈指针减少所需的空间量，然后用需要保存的信息填充相应的存储空间**

## 过程调用时栈的变化

- 压栈

```RISC-V
addi sp, sp, -24 //预留栈空间
sd x5, 16(sp)   //先压入x5保存的值
sd x6, 8(sp)    //先压入x6保存的值
sd x20,0(sp)
```

- 出栈

```RISC-V
ld x20,0(sp)
ld x6, 8(sp)
ld x5, 16(sp)
addi sp, sp, 24 //回复栈指针
```

例：过程嵌套

```c
int numSquare(int x,int y){
    return mult(x,x) + y;
}
```

```risc-v
numSquare:  # a0=x,a1=y
    addi sp,sp,-8   # 预留栈空间
    sw ra, 4(sp)    # ra寄存器入栈
    sw al, 0(sp)    # 参数寄存器al值y入栈

    mv al, a0       # al=a0=x，为mult(x, x)准备参数
    jal mult        # 调用mult计算x*x
    lw al, 0(sp)    #  出栈，恢复a1的值y

    add a0, a0, a1 # x*x+y

    lw ra, 4(sp) # 返回地址出栈
    addi sp, sp, 8 # 恢复栈指针
    jr ra
mult:…
```

## 用栈实现递归

```c
long long int fact (long long int n){
    if (n < 1) {
        return 1;
    }else {
        return n * fafact(n - 1);
    }
}
```

设定：参数n在`x10`中，结果在`x10`中

```risc-v
fact:
    addi sp, sp, -16
    sd x1, 8(sp) # 返回地址存在栈中
    sd x10, 0(sp)
    addi x5 , x10, -1

    bge x5, x0, L1 # 判断参数n-1是否≥0,如果是，跳转至L1
    addi x10, x0, 1 返回值1并且存入寄存器
    addi sp, sp, 16 # 释放2个双字空间
    jalr x0, 0(x1) # 返回fact调用者

L1:
    addi x10, x10, -1
    jal x1, fact    # 调用fact(n-1)，重复直到fact(0)

Calc: 
    addi t1, a0, 0 # 将fact()计算得到的值存入t1
    ld a0, 0(sp)    # 恢复调用者的参数
    ld ra, 8(sp)    # 恢复调用者的返回地址
    addi sp,sp ,16  # 释放两个双字栈空间
    mul a0, a0, t1
    jalr x0, 0(ra)  # 返回主程序

```

<!-- ![img](jz5-1.png){width=600} -->

## 实现sort过程（非叶过程）

```c
void sort(long long int v[], size_t n){
    size_t i,j;
    for (i = 1; i < n; i += 1) {
        for (j = i – 1; j >= 0 && v[j] > v[j + 1]; j - = 1){
            swap(v, j);
        }
    }
}
```

假设v保存在x10，n保存在x11，i保存在x19，j保存在x20

### 外层循环RV实现 

`for (i = 1; i < n; i += 1){...}`

```risc-v
li x19,0
for1tst:
    bge x19,x11,exit1
    ... # 外层循环包围的函数体
    addi x19,x19,1
    j for1st
exit1:
```

### 内层循环RV实现 

`for (j (j = i − 1; j >= 0 && v[j] > v[j + 1]; j − = 1) { ...}`

```risc-v
for2tst:
    blt x20,x0,exit2 # 如果j < 0，跳转到exit2
    slli x5,x20,3 # x5 = j * 8
    add x5,x10,x5 # x5 = v + (j * 8)
    ld x6,0(x5) # x6 = v[j][j]
    ld x7,8(x5) # x7 = v[j][j + 1]
    ble x6,x7,exit2 # 如果x6 ≤ x7，转exit2
    mv x21, x10 # 将参数x10复制到x21
    mv x22, x11 # 将参数x11复制到x22
    mv x10, x21 # swap的第一个参数是v
    mv x11, x20 # swap的第二个参数是j是j
```