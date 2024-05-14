指令(字)分成多个“字段”（一部分位）

- 每个字段有特定的含义和作用
- 理论上可以为每条指令定义不同的字段
- RISC-V定义了六种基本类型的指令格式：
    - R型指令 用于寄存器 —— 寄存器操作
    - I型指令 用于**短立即数**和访问内存的 load 操作
    - S型指令 用于访问内存的 store 操作
    - B型指令 用于条件跳转/分支/转移操作 
    - U型指令 用于长立即数操作
    - J型指令 用于无条件跳转操作

- 实际上用来区分某条指令种类的编码只有`inst[30],inst[14:12],inst[6:2]`

![img](https://github.com/DINOREXNB/DINOREXNB.github.io/blob/main/docs/images/jz7-1.png?raw=true)