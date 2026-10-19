# Java语言基础

因为 Java 语言和 C/C++ 在很多语法上相似甚至完全一致（例如各种控制语句），因此这里只列出 Java 与 C 家族不同的语言特性。

## 命名规范

1. 变量名、对象名、方法名、包名等标识符全部采用**小写字母**；如果标识符由多个单词组成，则其首字母小写、其后所有单词**仅首字母大写**，如:`getAge`
2. 不能与关键字、以及特殊值同名，如：`false`, `for`, `while`
3. **常量名全部大写**，单词间由下划线隔开。如：`MONTH_OF_YEAR`
4. 类名的每个单词**首字母大写**，如：`HelloWorldClass`

## 数据类型

- Java属于强类型语言，必须为每个变量声明一种类型
- Java按照数据类型可以将变量分为两类，==基本数据类型和引用数据类型==

### 基本数据类型

Java 一共有 8 种「基本数据类型」。

| 类型 | 空间 | 范围 | 备注 |
| --- | --- | --- | --- |
| byte | 1 字节 | -128 — 127 |  |
| short | 2 字节 | -2^15 — 2^15 - 1 |  |
| int | 4 字节 | -2^31 — 2^31 - 1 |  |
| long | 8 字节 | -2^63 — 2^63 - 1 | 声明常量要加「L」 |
| float | 4 字节 | —— | 声明常量要加「F」 |
| double | 8 字节 | —— | 浮点数的默认类型 |
| char | 16 字节 | UTF-16 |  |
| boolean | —— | false 或 true | 不能用 0 和非 0 代替 |

### 引用类型

类 `class`、接口 `interface` 和数组都是引用数据类型。

## 输入输出

### 输入

Java 的标准输入流是一个 `InputStream` 对象，有两种常用的读取方式：

- 用读字符流的方法将 `System.in` 读到一个 `BufferedReader` 中。
    
    ```java
    BufferedReader stdin = new BufferedReader(new InputStreamReader(System.in));
    stdin.readLine(); // <= 得到用户输入的字符串
    ```
    
- 用 `java.util.Scanner` 对 `System.in` 进行封装。
    
    ```java
    Scanner stdin = new Scanner(System.in);
    stdin.nextLine(); // <= 得到用户输入的字符串
    ```
    

### 输出

- `System.out.print()` 不换行输出。
- `System.out.println()` 换行输出。
- `System.out.printf()` 格式化输出，语法同 C 标准库的 `printf()`。

## 数组

Java 中的「数组」是确定长度的，一经创建就不能再加长了。

### 声明和创建

在创建数组时，要么指定长度，要么给出所有的项。

```java
int[] arr = new int[3]; // 指定长度
double[] anotherArray = new double[]{ 2.1, 4.5, 4.2 }; // 指定项
```

### 长度和索引

可以使用 `arr.length` 得到一个数组的长度。数组的下标从 0 开始，到 `length - 1` 结束。

### 数组是引用类型

所谓「引用类型」，可以理解为 C 家族中的「指针」。下面的代码解释了「引用」的本质。

```java
int[] arr = new int[]{ 1, 4, 2, 5 };
System.out.println(arr[2]);       // 打印 2
arr = new int[]{ 7, 3, 6, 9 };    // 现在 arr 指向了一个新的数组，旧的数组会被 JVM 回收
System.out.println(arr[2]);       // 打印 6
```

### 数组中的引用类型

思考下面代码的输出：

```java
String[] name = { "HIT", "SZ", "C++" };
String s = name[2];
name[2] = "Java";
System.out.println(name[2]);
System.out.println(s);
```

结果为

```
Java
C++
```

内存映射图：

![img](java2-1.png){width=400}

### 「多维」数组

没有所谓的「多维数组」，它本质上是「数组的数组」。例如

```java
int[][] arr = new int[5][];
```

是一个长度为 5 的「数组」数组，它内部的 5 个数组因为现在还没有初始化，所以长度可以不给出。我们当然也可以在定义给出所有的项，如下：

```java
int[][] arr = new int[][]{ { 1, 4, 2}, {2, 1}, {9, 1, 5, 6} };
```

显然，多维数组中的每一行不一定要一样长。

## 异常

![img](java2-3.png){width=400 align=right}

异常关键字:

- `try`：用于监听
- `catch`：用于捕获异常
- `finally`：finally语句块总会被执行。主要用于回收try块里面的物力资源（如数据库连接、网络连接和磁盘文件）

> 只有`finally`块执行完成后，才会回来执行`try`和`catch`块的`return`或者`throw`语句，如果`finally`中使用过`return`或者`throw`等终止方法的语句，则不会跳回执行，直接停止

- `throws`：==用在方法签名中==，用于声明该方法可能抛出的异常
- `throw`：用于抛出一个语句异常

Java 的异常捕获结构为 `try` - `catch` - `finally`。其中 `catch` 和 `finally` 可以只出现一个。

```java
try {
    // 做点啥
} catch (Exception e1|Exception e2|...) {
    // 出现异常时做的事
} finally {
    // 不管出现异常与否都要做的事
}
```

> `throw`与`throws`均为消极处理异常方式，仅负责抛出异常

## 类与对象

### 对象

1. 定义：客观存在的具体实体，具有明确定义的状态和行为
2. 特征：标识符、属性、操作

    1. 属性：与对象关联的变量，描述对象的静态特性
    2. 操作：与对相关联的函数，描述对象的动态特性

### 类

1. 类与对象

    1. 类是对象的抽象，是创建对象的模板
    2. 对象是类的具体事例
    3. 同一个类可以定义多个对象

2. 二者比较

    1. 类是静态的，在程序设计的时候就定义好了
    2. 对象是动态的，在程序执行的时候可以被创建、修改、删除

### 类的构造

类的访问权限：

![img](java2-2.png){width=400}

```java
// 见上表
public class Person {
    private String name;    // private: 只有本类能访问。
    protected int age;      // protected: 包外不能访问。但如果子类在包外，也能访问。
    int id;                 // friendly: 包外不能访问。如果子类在包外，也不能访问。
    public boolean gender;  // public: 谁都可以访问

    public Person(String name, int age) {
        this.name = name;
        this.age = age;
        id = 1;
    }

    public Person(String name) {
        this(name, 22);
    }
}
```

### 静态类

1. 静态成员属于类所有而不是某一具体对象所有
2. 加载时间

    1. 静态成员在类加载的时候被静态地分配地址空间和方法的入口地址
    2. 静态属性当且仅当在类初次加载的时候初始化

3. 实例

    1. 非静态对象在创建对象的时候初始化，存在多个实例副本，各个对象之间的副本不互相影响
    2. 静态对象被所有的对象共享，在内存中只有一个副本

4. `static` 属性是全局属性，可以直接通过类名访问，所以又叫类属性
5. 调用限制
    
    1. `static` 方法不可访问非 `static` 的方法：非 `static` 的对象必须要实例化才能访问
    2. 非 `static` 的方法可以访问 `static` 的方法：通过类名访问

6. 静态块
    
    1. 可以放在类的任何地方，类中可以有多个 `static` 块
    2. 语法：
        
        ```java
        class Person {
            static {
                totalNum = 10000;
                System.out.println("static block run!");
            }
        }
        ```
        
    3. 生命周期：在类加载的时候执行且只执行一次
    4. 用法：用于初始化静态变量和调用静态方法

7. 静态是否破坏了面向对象的特性？

    1. 静态属于类而非具体对象
    2. 具有一定程度上的全局性：初始化时加载到内存，并且所有对象有访问权限
    3. 保持类的封装性

## Java虚拟机与垃圾回收

### JVM

==虚拟机==：指的是通过软件**模拟**的具有完全硬件系统的运行在一个完全隔离环境中的计算机系统

JVM是通过软件来模拟`Java`字节码的指令集，是`Java`程序的运行环境

如今JVM不仅支持`Java`，还支持`Kotlin`，`Groovy`等语言，黄钻混过后的字节码文件都能通过Java虚拟机进行运行和处理

### JVM体系结构

![img](java2-4.png){width=400 align=right}

主要包含两个子系统和两个组件

- `Class Loader`（类装载器）和`Execution`（执行引擎）子系统
- `Runtime Data Area`（运行时数据区域）和`Native Interface`（本地接口）组件

### Java堆

- 为进行高效垃圾回收，虚拟机把堆内存逻辑上划分为三块区域
    - 新生代：新对象和没达到一定年龄的对象都在新生代
    - 老年代：被长时间使用的对象，老年代的内存空间应该比新生代更大
    - 元空间（JDK1.8之前叫做“永久代”）：像一些方法中的操作临时对象等，JDK1.8之前是占用JVM内存，之后直接使用物理内存

### JVM垃圾回收

Java的核心思想是面向对象，屏蔽了很多底层细节，让程序员更多地关注“对象”

Java使用隐式分配器，程序员只管创建对象使用堆内存，回收交给垃圾回收器

垃圾回收器的任务：

- 跟踪监控每一个Java对象，当某一对象处于不可达状态，回收该对象所占用的内存
- 清理内存分配，回收过程中产生的内存碎片。

#### 回收机制优点

1. 对开发者屏蔽内存管理的细节，提高开发效率
2. 开发者无权操作内存，降低内存泄漏风险

#### 回收机制劣势

1. 垃圾回收不收开发者控制，在一些对时间极其敏感的场景中，不受控制的垃圾回收会带来多余的时间开销

> 如果开发者正在开发一个高性能的实时数据处理系统，该系统需要在毫秒级别内响应用户请求并处理大量数据。在这种情况下，任何导致系统暂停或延迟的因素都可能对系统性能产生重大影响。在Java中，当发生垃圾回收时，虚拟机会暂停应用程序的执行，以便清理不再被引用的对象，并回收它们占用的内存。如果在系统的关键时刻发生了长时间的垃圾回收，可能会导致系统在此期间停顿，无法响应新的请求或处理数据，从而降低系统的实时性能

## JVM、JRE与JDK

- JVM：Java虚拟机
    - 所有的Java程序都运行在JVM上，是JRE的一部分
- JRE：Java运行环境（Java Runtime Environment）
    - JRE一般用于执行Java程序，JRE除了包括JVM之外还包括一些基础的JAVA API，JRE是JDK的一部分
- JDK：JAVA开发工具（Java Development Kit）
    - JDK提供了Java的开发环境和运行环境（JRE），开发环境主要包括一些常用工具，如常用的JAVAc编译工具，jar打包程序等等