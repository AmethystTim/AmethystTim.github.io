## 泛型

### 为何需要泛型？

```java
ArrayList list = new ArrayList();
list.add("str");
list.add(123);
list.add(true);
String name = (String)list.get(0);
```

1. 没有错误检查，可以向数组列表内添加**任何对象**
2. 获取一个值时必须进行**强制类型转换**

### 泛型类

具有一个或多个类型变量的类

- 语法格式：在类名之后声明泛型，泛型类先声明类型变量再使用。类型变量通常使用较短的大写，如 T, E, K, V等

**类型参数**（又称类型变量）作占位符，**指示分配的类型**

- E：(Element) 元素
- K：(Key) 键
- N：(Number) 数字
- T：(Type) 类型
- V：(Value) 值

```java
class Generic<T, E> {
    public T t;
    public E e;
    public void fun(T t, E e) {}
}
Generic<String, Integer> g3 = new Generic<>();
g3.t = "str";
g3.e = 100;
```

> 声明泛型时**不能使用基本数据类型**，比如不能使用`int`，需使用`Intager`
>
> 泛型方法可以定义在**普通类**中，也可以定义在**泛型类**中。
> 
> 泛型类中使用了泛型成员的方法不是泛型方法，==只有声明泛型的方法才是泛型方法==。

```java
// 非泛型类中定义泛型方法
class GenericFun{
    public <T,E> void fun1(E e){}
    public <T> T fun2(T t){
        return t;
    }
}

// 泛型类中定义泛型方法
class GenericFun<K>{
    public <T> T fun2(T t,K k){
        return null;
    }
}
```

类的**静态泛型方法**，不得使用泛型类中声明的泛型，**需要独立声明**

```java
class GenericStaticMethod<K>{
    private K k;
    public GenericStaticMethod(K k){
    this.k=k;
}
public static GenericStaticMethod <K> fun1(K k){
    return new GenericStaticMethod <K>(k)}
}//报错：无法从静态上下文中引用非静态类型变量K
```

### 泛型的通配符

考虑一个场景：为所有`List`抽象一个方法，不论给的参数是`List<Integer>`，`List<String>`，都可以接收并且打印`List`中的元素

- 严格的泛型类型系统一旦指定便无法改变
- 需要**允许类型参数发生变化**
- 同时需要控制可以指定的类型，而非不加限制

```java
public static void printAllObject(List<?> list){
    for(Object obj : list){
        System.out.println(obj);
    }
}
```

- 通配符
    - 允许类型参数发生变化
    - `<? extends ClassName>`：类型参数是ClassName的**子类**
    - `<? super ClassName>`：类型参数是ClassName的**超类**
    - `<?>`：无限定通配符

> 类型参数T通常表示一个确定的类型，一般用于泛型类和泛型方法的定义；`?`表示不确定的类型，一般用于泛型方法的调用代码或者形参


## 模板方法模式

模板方法模式（Template Method）
- 定义一个操作中的算法骨架
- 将一些步骤延迟到子类中
- 模板方法模式使得子类可以不改变一个算法的结构即可重新定义该算法的某些特定步骤

- 抽象类：抽象模板；定义并实现一个模板方法，给出顶层逻辑的骨架
- 具体类：实现父类定义的一个或多个抽象方法

模板模式把**不变的行为**搬到了**超类**，去除了子类中的重复代码（各子类中公共的行为应被提取出来并**集中到一个公共父类**中以避免**代码重复**）

> 模板方法模式还有一个应用是**控制子类拓展**，即模板方法只在特定点调用`hook`操作

例：
```java
public abstract class AbstractClass {
    // 模板方法，不允许子类重写
    public final void templateMethod() {
        stepOne();
        stepTwo();
        stepThree();
    }

    // 抽象方法，必须由子类实现
    protected abstract void stepOne();
    protected abstract void stepTwo();

    // 钩子方法，子类可以选择性地重写
    protected void stepThree() {
        // 默认实现
    }
}
```

在钩子方法中一般提供了一个空实现或者默认的实现，在具体类中可以选择性地覆盖实现，不影响整体结构

## 反射

### 为何需要反射？

什么时候需要反射？

- 在运行时获知任意一个对象**所属的类**。
- 在运行时构造任意一个**类的对象**。
- 在运行时获知任意一个类所具有的**成员变量**和**方法**。
- 在运行时调用任意一个对象的**方法和属性**。

```java
public class Test {
    interface X {
        public void test();
    }

    class A implements X {
        @Override
        public void test() {
            System.out.println("I am A");
        }
    }
    class B implements X {
        @Override
        public void test() {
            System.out.println("I am B");
        }
    }

    public static void main(String[] args) {
        X a = create1("A");
        a.test();
        X b = create1("B");
        b.test();
    }

    public static X create1(String name){
        if (name.equals("A")) {
            return new A();
        }
        else if(name.equals("B")) {
            return new B();
        }
        return null;
    }
}
```

> 如果有成百上千个不同的 X 的实现类需要创建，需要写上千个 if 语句来返回不同的 X 对象？

### 反射实现

```java
// 上述例子的static方法可以改写成
public static X create2(String name) {
    Class<?> class = Class.forName(name);//加载参数指定的类，并且初始化它
    X x = (X) class.newInstance();
    return x;
}
```

$$反射常用类\begin{cases}
    Field类 &提供有关类的域信息，以及对它的动态访问权限\\
    Constructor类 & 提供有关类的构造信息，以及对它的动态访问权限\\
    Method类 & 提供有关类的方法信息\\
    Class类 & 接受类的信息\\
    Object类 & 接收实例信息
\end{cases}$$

### Class类

- 正常方式
    - 引入包类名称
    - `new`实例化
    - 实例化对象
- 反射方式
    - 实例化对象
    - `getClass()`方法
    - 完整的包类名称

#### 获取Class对象的三种方法

- 对象.getClass()

```java
// Student类描述学生
Student Harry = new Student(“Harry Potter”,11);
// Class对象描述一个特定类的属性
System.out.println(Harry.getClass());
// 输出 class Student
System.out.println(Harry.getClass().getName());//Class.getName() 返回类的名字
// 输出 Student 
```

- Class.forName()
    - 加载参数指定的类，并且初始化它
- 类名.class
    - 通过类名的属性class获取

```java
Class cl1 = Random.class;
// int不是类，但int.class是一个Class类型的对象
Class cl2 = int.class;
```

#### 通过反射构造类的实例

- `Class.newInstance()`
    - `newInstance`方法调用默认的构造函数(**无参**)初始化新创建的对象
    - 如果这个类没有默认的构造函数，就会抛出一个异常

```java
Date date1 = new Date();
Class dateClass2 = date1.getClass();
Date date2 = (Date)dateClass2.newInstance(); 
```

- 使用`Constructor`的`newInstance()`
    - 通过反射先获取构造方法再调用
    - 先获取构造函数，再执行构造函数
    - 该方法可以**携带参数**

```java
Student Harry = new Student("Harry Potter", 11);
Class StudentClass = Harry.getClass();
Constructor con = StudentClass.getConstructor(String.class, int.class);
Student Ron = (Student)con.newInstance("Ron Weasley", 11);
System.out.println(Ron);
```

### 通过反射获取修改成员变量

获取和修改成员变量
- 获取所有公有的字段

`public Field[] getFields() { }`

- 获取所有的字段（包括私有、受保护、默认的）

`public Field[] getDeclaredFields() { }`

- 获取一个指定名称的公有的字段

`public Field getField(String name) { }`

- 获取一个指定名称的字段，可以是私有、受保护、默认的

`public Field getDeclaredField(String name) { }`

- 使用Field类中的get方法查看字段值
- 使用Field类中的set方法修改字段值

### 通过反射获取成员方法

- 获取所有"公有方法"（包含父类的方法，当然也包含 Object 类）

`public Method[] getMethods() { }`

- 获取所有的成员方法，包括私有的（不包括继承的）

`public Method[] getDeclaredMethods() { }`

- 获取一个指定方法名和参数类型的成员方法

`public Method getMethod(String name, Class<?>... parameterTypes)`

`Object invoke(Object obj, Object… args)`

- 第一个参数是哪个对象要来调用这个方法
- 第一个参数是调用方法时所传递的实参
- 对于**静态方法**，第一个参数可以忽略，即可设为`null`

### 分析

- 优点
    - 比较灵活，能够在运行时动态获取类的实例
- 缺点
    - 性能瓶颈：反射相当于一系列解析操作，通知JVM要做的事情，性能比直接的Java代码要慢很多
    - 安全问题：反射机制破坏了**封装性**，因为通过反射可以**获取并调用类的私有方法和字段**

### 利用反射破坏单例模式

- 单例模式

```java
public class Singleton {
    private static Singleton singleton = new Singleton();
    private Singleton(){}
    public static Singleton getInstance(){
        return singleton;
    }
}
```

- 利用反射破坏单例模式

```java
Class objectClass = Singleton.class;
Constructor constructor = objectClass.getDeclaredConstructor();
constructor.setAccessible(true); //覆盖java的访问控制，进而调用私有的构造函数
Singleton newInstance = (Singleton)constructor.newInstance();
Singleton instance = Singleton.getInstance();
System.out.println(instance == newInstance);
```

抵御反射破坏

```java
public class Singleton {
    private static Singleton singleton = new Singleton();
    private Singleton(){
        if(singleton != null) {
            throw new RuntimeException("单例构造器禁止通过反射调用");
        }
    }
    public static Singleton getInstance(){
        return singleton;
    }
}
```