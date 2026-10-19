## 含义

表明该属性、该方法是属于类的，称为静态属性或静态方法

```java
static 成员属性;
static 成员方法;
```

> 静态成员属于类所有，不属于某一具体对象私有<br>
> 静态成员随类加载时被静态地分配内存空间、方法的入口地址

## 应用场景

```java
public class Person {
    private String name;
    private int age;
    String country = "A城";
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
}
```

|栈内存|堆内存|
|-|-|
|per1|name="张三"<br>age=20<br>country="A城"|
|per2|name="李四"<br>age=21<br>country="A城"|
|per3|name="王五"<br>age=22<br>country="A城"|

每个对象占用自己的`country`属性，会造成**内存空间的浪费**，可以使用`static`将`country`属性设置成一个**公共属性**

```java
public class Person {
    private String name;
    private int age;
    static String country = "A城";
    public Person(String name, int age) {
        this.name = name;
        this.age = age;
    }
}
// 调用
Person.country; //类名.属性
```

> 由于全局属性拥有可以通过类名称直接访问的特点，所以这种属性又称为**类属性**

- 注意：
    - 使用`static`声明的方法，不能访问非`static`的操作（属性或方法）
    - 非`static`声明的方法，可以访问`static`声明的属性或方法
- 原因：
    - 如果一个类中的属性和方法都是非`static`类型的，一定要有实例化对象才可以调用
    - `static`声明的属性或方法可以通过类名访问，可以在没有实例化对象
的情况下调用 （先于对象而存在）

## 静态块

- 可以置于类中的任何地方，类中可以有多个`static`块
- 在类被加载的时候**执行且仅会被执行一次**,按照`static`块的顺序来执行每个`static`块
- 一般用来初始化静态属性和调用静态方法

```java
class Person{
    static long totalNum;
    String name;
    public Person(String name) { … }
    static { //静态块
        totalNum=10000
        System.out.println(“run static code block!");
    }
}
```

> 上述代码中，无论构建多少个对象，静态块中的代码都仅执行一次