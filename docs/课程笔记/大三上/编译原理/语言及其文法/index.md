与**形式语言与自动机**文法部分基本一致

以下是一些简单补充

## 表达式(expression)与语句(statement)的区别：

- 表达式进行计算并产生一个值作为结果
- 语句执行操作但不返回值得指令

参考文章[https://www.cnblogs.com/yuyoubei/p/15834708.html](https://www.cnblogs.com/yuyoubei/p/15834708.html)

所以在编写代码的过程中不能将语句赋值给另外一个变量

```rust
fn main() {
    let x = (let y = 6);
}
```

运行后报错信息如下：

```rust
> cargo run
   Compiling hello_world v0.1.0 (D:\projects\hello_world)
error: expected expression, found statement (`let`)
 --> src\main.rs:2:14
  |
2 |     let x = (let y = 6);
  |              ^^^^^^^^^
  |
  = note: variable declaration using `let` is a statement
```