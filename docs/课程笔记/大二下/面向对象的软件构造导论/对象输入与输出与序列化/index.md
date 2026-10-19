## 对象输入/输出与序列化

Java语言支持一种称为对象序列化(object serialization)的通用机制，它可以将任何对象写出到输出流中，并在之后将其读回

- 序列化与反序列化
    - 把对象转换为字节序列的过程称为对象的**序列化**
    - 把字节序列恢复为对象的过程称为对象的**反序列化**

## 对象序列化用途

- 把对象的字节序列永久地保存到硬盘上，通常存放在文件中
- 在网络上传送对象的字节序列
    - 无论是何种类型的数据，都会以二进制序列的形式在网络上传送。
    - 发送方需要把这个Java对象转换为字节序列，才能在网络上传送
    - 接收方需要将字节序列再恢复为Java对象

## JDK类库中的序列化API

- `Java.io.ObjectOutputStream`代表对象输出流，它的`writeObject(Object obj)`方法可对参数指定的`obj`对象进行序列化，把得到的**字节序列**写到一个目标输出流中
- `ObjectOutputStream(OutputStream out)`，创建一个`ObjectOutputStream`使得你可以将对象写出到指定的`OutputStream`
- `Void writeObject(Object obj)`，写出指定的对象到ObjectOutputStream，这个方法将存储指定对象的类、类的签名以及这个类及其超类中所有非静态和非瞬时的域的值

### 序列化步骤

1. 创建一个对象输出流，可以包装一个其他类型的目标输出流，比如文件输出流
2. 通过对象输出流的`writeObject()`方法写对象

## JDK类库中的反序列化API

- `Java.io.ObjectInputStream`代表对象输入流，它的`readObject()`方法可从一个源输入流中读取字节序列，再把它们反序列化为一个对象，并将其返回。
- `ObjectInputStream(InputStream in)`，创建一个`ObjectInputStream`用于从指定的`InputStream`中读回对象信息。

### 反序列化步骤

1. 创建一个对象输入流，它可以包装一个其他类型的源输入流，比如文件输入流`FileInputStream`
2. 通过对象输入流的`readObject`方法读取对象

例：利用`ObjectOutputStream`对象`writeObject`输出可序列化对象

```java
public class SeriDemo {
    public static void main(String[] args) {
        try{
            Person p1 = new Person("zhangsan",30)
            ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream("person.dat"));
            oos.writeObject(p1);
        }catch(Exception e){
            e.printStackTrace();
        }
    }
}
```

调用`readObject()`方法得到序列化的对象

```java
try{
    ObjectInputStream ois = new ObjectInputStream(new FileInputStream("person.dat"));
    System.out.println("readObject():)");
    Person person = (Person)ois.readObject();   //反序列化
}catch(Exception e){    
    e.printStackTrace();
}
```

> 被static修饰的字段不可以被序列化（因为被static修饰的字段属于类而不是对象，只有实例字段才会被序列化）