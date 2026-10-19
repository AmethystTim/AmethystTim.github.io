## 对象和类

### 对象（object）

客观存在的具体实体，具有明确的状态和行为

特征：标识符（用于区别其他对象）、属性（状态）、操作（行为）

- 属性：与对象关联的变量，描述对象**静态特征**
- 操作：与对象关联的函数，描述对象**动态特征**

### 类（class）

对现实生活中一类具有共同属性和共同操作的对象的**抽象**

```java
Student ——类名
---------
name
gender  ——属性
major
---------
enrol() ——操作方法
take_course()
```

## 类的声明构造

```java
public class Person{
    String name;
    int age;

    setName(String name){...}
    getName(){...}
}
```

- 成员变量
    - 作用：表示类和对象的属性、状态
    - 类型：可以是**基本**类型和**引用类**类型
- 成员方法
    - 实质：实现某一功能的程序段，可以改变成员变量的属性和状态

> 访问控制符<br>
> `public`：可被公开访问<br>
> `private`：私有<br>
> `protected`：受保护<br>
> `default`：默认<br>

## 封装性

```java
//不使用封装
public class Person{
    String name;
    int age;
    public void tell(){
        System.out.println("姓名: "+name+","+"年龄: "+age);
    }
}

//使用封装
public class Person{
    private String name;
    private int age;
    public void tell(){
        System.out.println("姓名: "+name+","+"年龄: "+age);
    }
}
```

若直接运行下方代码，编译器会报错，因为外部无法直接调用类内部的属性

```java
Person per = new Person();
per.name = “张三” ;
per.age = -30;
per.tell();
```

### 通过`getter/setter`对属性进行访问

在Java开发标准规定中，只要是属性封装，设置和取得就要依靠`setter`和`getter`方法完成操作

如：`Setter: public void setName(String n){}`

`Getter: public String getName(){}`

## 类的构造方法

- 构造方法（constructor）是一种特殊方法
    - 用来构造并初始化（new）该类的一个**新的对象**
    - ==构造方法和类名**同名**==，而且**不返回数据类型**

```java
public class Person {
    private String name;
    private int age;
    //…
    Person( String n, int a ) {
        name = n;
        age = a;
    }
}
Person Li = new Person(“Li Lei”,19);
Person Han= new Person(“Han Meimei”); // error
Person Han= new Person(“Han Meimei”, 18); 
```

### this 关键字

```java
public Person(String name, int age){
    name = name;
    age = age;
}
```
> 构造函数的两个参数能够表达出含义，但是输入参数和Person类定义的私有变量名称一致，导致name=name，age=age。无法区分哪个是私有变量，哪个是输入变量。

采用`this`关键字，将上方代码修改为

```java
public class Person {
    private String name;
    private int age;
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
    // 编写setter、getter
    …
} 
```

### 默认构造方法

- 类都有一个至多个构造方法

```java
class Person {
    // …
    public Person() {
        System.out.println("一个新的对象");
    }
    public Person(String name) {
        this(); // 调用本类中的无参构造
        this.name = name;
    }
    public Person(String name, int age){
        this(name); // 调用有一个参数的构造
        this.age = age; }
    }
    // …
}
```

> 一个类中如果有多个构造方法，则需有一个不用`this`调用的构造方法作为出口

- 若没有定义任何构造方法，系统会自动产生一个构造方法，称作**默认构造方法**

```java
// 默认构造函数
class Person {
    public Person(){
        
    }
}
```

## 类的访问域

定义：本类及本类内部的成员（成员变量、成员方法、内部类）对
其他类的可见性，即这些内容是否允许其他类访问

- `public`：该类可以被其他类所访问
- `default`：该类只能被同一个包中的类访问
- `private`：无法被其他类所访问
- `protected`：可以被子类访问，以及子类的子类（作用于**继承**关系）

> 包(package)： 在java编程中，经常把功能相似或相关的类或者接口放在一个包里；可以避免名字冲突；也限定了访问权限。

## 抽象类与接口

### 抽象类

- 抽象类是不能实例化的
- **只能被继承**，保证子类实现其定义的抽象方法
- 可用于实现**多态**

### 接口

抽象类中，抽象方法本质上是定义接口规范，保证子类都有相同的接口实现

比抽象类还要抽象：没有字段，所有方法都是抽象方法

### 接口 VS 抽象类

||抽象类|接口|
|-|-|-|
|抽象方法|可以定义抽象方法|可以定义抽象方法|
|字段|可以定义字段|无字段|
|继承|只可以extends一个class|可以implement多个interface|

> 均不能使用`new`创建一个对象<br>
> 接口是特殊的抽象类：接口里面的所有方法都是抽象方法，接口里面的所有属性都是常量

```java
public class Student implements Interface1, Interface2{
    private String name;
    private int age;
    @Override
    public void getDuty(){
        System.out.println("好好学习");
    }
    @Override
    public void getMission(){
        System.out.println("为中华之崛起而读书");
    }
}
```

## 多态

- 多态（Polymorphism）是面向对象编程的三大重要特性之一
- 多态是同一个行为具有多种表现形态的能力
    - 不同情况下的不同处理方式
    - 程序中定义的变量和方法在编程时并不确定，而是在程序运行期间才确定

### 实现方式

- 重写（继承）
- 抽象类和抽象方法
- 接口

#### 重写

- 父类中的某些方法对子类并不一定适用，需要重写/覆盖(override)父类中的这个方法
- 重写：如果在子类中定义一个方法，其名称、参数、返回类型正好与父类中某个方法相同，那么可以说，子类的方法重写了父类的方法

#### 抽象类

```java
//定义一个抽象类
public abstract class Person{
    //抽象方法
    //没有方法体，用abstract做修饰
    public abstract void getTarget();
}

public class Engineer extends Person{
    @Override
    public void getTarget(){
        System.out.println("技术改变世界");
    }
}

public class Scientist extends Person{
    @Override
    public void getTarget(){
        System.out.println("勇攀科学高峰");
    }
}
```

#### 接口

```java
public interface GetTarget(){
    void getTarget();
}
```

```java
public class Talent extends Person implements GetTarget(){
    @Override
    public void getTarget(){
        System.out.println("打造国之重器");
    }
}

public class Entrepreneur extends Person implements GetTarget(){
    @Override
    public void getTarget(){
        System.out.println("促进共同富裕");
    }
}

public static void main(String[] args) {
    Person Zhang= new Talent();
    Zhang.getTarget();
    Person Ma = new Entrepreneur();
    Han.getTarget();
}
```

## 多继承

多继承存在的问题：

1. 若子类继承的父类中拥有相同的成员变量，子类在引用该变量时将无法判别使用哪个父类的成员变量。
2. 若一个子类继承的多个父类拥有相同方法，同时子类并未覆盖该方法（若覆盖，则直接使用子类中该方法），那么调用该方法时将无法确定调用哪个父类的方法。

- 实现方法
    - 内部类
    - 接口