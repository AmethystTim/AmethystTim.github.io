## 定义

流是一组有序的数据序列，将数据从一个地方带到另一个地方

输入输出设备的抽象

- Java程序中，对于数据的输入/输出操作都是以“流”的方式进行。
- 设备可以是文件，网络，内存等

流具有方向性，至于是输入流还是输出流则是一个相对的概念，一般以程序为参考：

- 如果数据的流向是程序至设备，我们称之为输出流，
- 反之，我们称之为输入流。

## 流的分类

- 流的方向：输入流，输出流
- 按数据单位：字节流、字符流
    - 字节流：基本单位为字节
    - 字符流：基本单位为字符
- 按照数据封装类型：节点流、处理流
    - 节点流：可以从或向一个特定的地方（节点）读写数据，如FileReader
    - 处理流：对一个已存在的流的连接和封装，通过所封装的流的功能调用实现数据填写，如BufferReader

### IO流的分类

- 方向：input/reader,output/writer
- 数据：
    - 字节：表示数据类型（**文本、图片、声音**等等，例如word文档的读取，就是使用字节流）
    - 字符：仅限**纯文本**

所有的流位于`java.io`包中，并分别继承以下四种抽象流类型

||字节流|字符流|
|-|-|-|
|输入流|InputStream|Reader|
|输出流|OutputStream|Writer|

字节流与字符流是可转换的

- 字节流->字符流：`InputStreamReader(InputStream in)`
- 字符流->字节流：`OutputStreamWriter(OutputStream out)`

## 字节输入输出流

### FileInputStream

- 该流用于从文件读取数据，可以使用关键字`new`来创建对象

```java
InputStream f = new FileInputStream("~/java/demo")

// 也可以使用文件对象File创建
File f = new File("~/java/demo");
InputStream fOut = new FileInputStream(f);
```

### FileOutputStream

- 该类用来创建一个文件并向文件中写数据。
- 如果该流在打开文件进行输出前，目标文件不存在，那么该流
会创建该文件

```java
OutputStream f = new FileOutputStream("~/java/demo")
```

## 字符输入输出流

对于Unicode文本，可以使用抽象类`Reader`和`Writer`的子类

`BufferedReader`：用来提高效率，例如包装InputStreamReader

```java
BufferedReader in = new BufferedReader(new InputStreamReader(System.in));//System.in表示读取Java控制台收入
```

`InputStreamReader/OutputStreamReader`：可包装 `InputStream` ，从而将基于字节的输入流
转换为基于字符的 `Reader`，用于从文件中读取字符

## 控制台读取多字符输入

```java
import java.io.*

public class BRRead{
    public static void main(String[] args) throws IOException{
        char c;
        BufferReader br = new BufferReader(new InputStreamReader(System.in));
        System.out.println("输入字符，按'q'退出");
        do{
            c = (char) br.read();
            System.out.println(c);
        }while(c != 'q');
    }
}
```