## 设计模式

### 四个基本要素

1. 模式名
2. 问题
3. 解决方案
4. 效果

### 设计模式分类

1. **创建型**（工厂方法下的抽象工厂、单例模式）
2. **结构型**
3. **行为型**（如模板方法下的观察者模式、策略模式）

### 单例模式

实现单例模式的三种方法：

- 饿汉式
    - 类的构造方法变为**私有**
    - 在类初始化的时候就加载对象
    - 客户只能通过调用`getInstance`方法来访问类的唯一实例
    - 无需同步，执行效率高

```java
public class Singleton{
    private static singleton = new Singleton();
    private Singleton(){}
    public static Singleton getInstance(){
        return singleton;
    }
}
```

- 懒汉式
    - 类的构造方法变为**私有**
    - 在定义的`getInstance`方法中判断对象是否已经被创建

```java
public class Singleton{
    private static singleton;
    private Singleton(){}
    public static Singleton getInstance(){
        if(singleton == null){
            singleton = new Singleton();
        }
        return singleton;
    }
}
```

- 线程锁
    - 在多线程情况下，同时访问`getInstance`方法可能会导致创建多个实例
    - 使用`sychronized`修饰`getInstance`方法，强制线程访问该方法时，必须等待其他线程离开该方法后，才可以访问
    - 虽然保证了线程安全，但是执行效率较低
```java
public class Singleton{
    private static singleton;
    private Singleton(){}
    public static synchronized Singleton getInstance(){
        if(singleton == null){
            singleton = new Singleton();
        }
        return singleton;
    }
}
```

### 简单工厂模式

例：用程序模拟一家披萨店，在`orderPizza`方法中进行创建pizza，交付pizza的过程

```java
public class PizzaStore{
    public Pizza orderPizza(type){
        Pizza pizza;
        if(type.equal(pizza1)){
            pizza = new Pizza1();
        }else if(type.equal(pizza1)){
            pizza = new Pizza2();
        }else{
            ...
        }

        pizza.prepare();
        pizza.bake();
        pizza.box();

        return pizza;
    }
}
```

在上面的代码中，不难发现：如果在之后的运营中要上架新的pizza或者下架过气的pizza，需要对`if-else`语句进行修改，违反了OCP原则；同时，上面的代码将订购pizza和制作pizza的逻辑全部放在一个方法内实现，我们需要将创建pizza的逻辑从`orderPizza`内分离

定义一个新的类`PizzaFactory`去创建pizza

```java
public class PizzaFactory{
    public Pizza createPizza(type){
        Pizza pizza = null;
        if(type.equal(pizza1)){
            pizza = new Pizza1();
        }else if(type.equal(pizza1)){
            pizza = new Pizza2();
        }else{
            ...
        }

        return pizza;
    }
}
```

与此同时，修改PizzaStore类，使用factory的`createPizza`方法代替`new`来创建pizza对象

```java
public class PizzaStore{
    private PizzaFactory factory;
    public PizzaStore(PizzaFactory factory){
        this.factory = factory;
    }
    public Pizza orderPizza(type){
        Pizza pizza;
        
        pizza = factory.createPizza(type);

        pizza.prepare();
        pizza.bake();
        pizza.box();

        return pizza;
    }
}
```

**简单工厂模式并非是一个真正的设计模式**，因为它违背了**开闭原则**（如果有复杂的产品结构或者增加新产品，工厂类的产品创建的逻辑就不得不被修改），但是工厂方法模式是在此基础上建立的。

### 工厂方法模式

如何避免修改创建pizza方法的逻辑？

将创建pizza的方法放回`PizzaStore`类，将`PizzaStore`设置为抽象类，该方法设置为抽象方法，让每个不同的PizzaStore去实现对应的创建pizza的方法。**实际上就是把创建对象的行为放到子类中，让不同子类去实现不同的创建逻辑**

### 抽象工厂模式

有手机和路由器两种产品，有A和B两种品牌，两种品牌都可以生产手机和路由器

有两个产品，所以定义两个创建产品的接口

接着将工厂类进行抽象，其中包含了创建两个产品的方法，返回的是**产品的接口类**

客户端调用时，**直接用工厂接口类创建需要的工厂**，拿到对应的产品

![img](java5-1.png){width=500}

抽象工厂的**产品族**扩展十分容易（如加入新的品牌C），但是**产品等级**扩展十分困难（需要修改抽象工厂的代码，还要修改具体实现类里的代码）