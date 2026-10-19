## 语言概述

### SQL概览

1974，由Boyce与Chamber提出，是集DDl，DML，DCL于一体的数据库语言

1. DDL语句引导词：`CREATE`，`ALTER`，`DROP`
2. DML语句引导词：`INSERT`，`UPDATE`，`DELETE`，`SELECT`
3. DCL语句引导词：`GRANT`，`REVOKE`

## 简单的SQL-DDl/DML：创建数据库

### 创建数据库

```sql
CREATE DATABASE <dbname>;
```

### 创建表

```sql
CREATE TABLE <tbname> (<col_name data_type [PRIMARY KEY | UNIQUE] [NOT NULL] [, col_name_data_type]>, ...);
```

- `[ ]`：表示内部内容可以省略<br>
- `|`：表示隔开选项可以任选其一<br>
- `UNIQUE`：唯一性约束，即候选键<br>
- `NOT NULL`：指该列不允许出现空值，通常主键是不允许有空值

> 各个商用DBMS的数据类型或有差异

例：

```sql
CREATE TABLE COURSE(CNo char(3), CName char(12))
```

### 向表中追加元组 `INSERT INTO`

```sql
INSERT INTO <tbname> (<colname>[,<colname>]) VALUES (<val>[, <val>], ...);
```

## SQL-DML之查询SELECT

### 基本检索操作

```sql
SELECT <colname> 
FROM <tbname>  
 |WHERE <conditions>;
```

例：

```sql
SELECT SNo, Sname, Sage FROM Students WHERE Sage <= 19;
```

### 子查询

#### IN谓词

练习：求选修了001号课程的学生的学号和姓名

```sql
SELECT sno, sname FROM Student
WHERE sno IN (
    SELECT sno FROM SC WHERE cno == "001"
);
```

- 求既学过001号课程，又学过002号课程学生的学号

```sql
SELECT sno FROM SC
WHERE cno = "001" AND sno IN(SELECT sno FROM SC WHERE cno = "002");
```

- 列出**没有**学过李明老师讲授课程的所有同学的姓名

```sql
SELECT sname FROM Student
WHERE sno NOT IN(
    SELECT sno FROM SC ,Course C, Teacher T
    WHERE T.name = "李明"
    AND SC.cno = C.cno
    AND T.tno = C.tno
);
```

#### 相关子查询

内层查询有时候需要依靠外层查询的某些参量作为**限定条件**才能进行

外层向内层传递的参量需要使用**外层的表名或者表别名来限定**

例：求学过001号课程同学的姓名

```sql
SELECT sname
FROM Student Stud
WHERE sno IN(
    SELECT sno 
    FROM SC 
    WHERE sno = Stud.sno AND cno = "001"
);
```

#### $\theta$ SOME/ $\theta$ ALL谓词

例：找出**工资最低**的教师姓名

```sql
SELECT tname FROM Teacher
WHERE salary <= ALL(SELECT salary FROM Teacher);
```

例：找出所有课程**都不及格**的学生姓名

```sql
SELECT sname FROM Student
WHERE 60 > ALL(SELECT score FROM SC WHERE sno = Studnet.sno);
```

#### EXISTS谓词

基本语法：`[NOT] EXISTS (子查询)`

例：检索选修了李明老师主讲课程的**所有同学**的姓名

```sql
SELECT DISTINCT sname FROM Student
WHERE EXISTS (
    SELECT * FROM SC, Course, Teacher 
    WHERE SC.cno = Course.cno 
        AND SC.sno = Student.sno 
        AND Course.tno = Teacher.tno 
        AND tname = "李明"
);
```

Extra: 强大的`NOT EXISTS`

例：检索学过001号老师主讲的所有课程的所有同学的姓名

> 可以将其转化为“**不存在**有一门001号老师主讲的课程该同学**没学过**”<br>
> 本质是全称量词$\forall$与存在量词$\exists$的转换

```sql
SELECT sname FROM Student  
WHERE NOT EXISTS(   -- 不存在
    SELECT * FROM Course    -- 有一门001教师主讲课程
    WHERE Course.tno = "001" AND NOT EXISTS(    -- 该同学没学过
        SELECT * FROM SC 
        WHERE sno = Student.sno 
        AND cno = Course.cno
    )
);
```

例：已知SPJ(sno, pno, jno, qty)，其中sno为供应商号，pno为零件号，jno为工程号，qty为数量，列出至少用了供应商S1供应的全部零件的工程号

```sql
SELECT DISTINCT jno FROM SPJ spj1
WHERE NOT EXISTS(   -- 不存在
    SELECT * FROM SPJ spj2  -- 有一种S1的零件
    WHERE spj2.sno = "S1" AND NOT EXISTS(   -- 该工程没用过
        SELECT * FROM SPJ spj3
        WHERE spj3.pno = spj2.pno
        AND spj3.jno = spj1.jno
    )
)
```

### 分组查询

#### 聚集函数

`COUNT`、`SUM`、`AVG`

例：求数据库课程平均成绩

```sql
SELECT AVG(Score) FROM Course C, SC
WHERE C.cname = '数据库' AND C.cname = SC.cname
```

#### GROUP BY

为了解决若干个集合的聚集运算问题，引出了**分组**概念

SQL可以将检索到的元组按照某一条件进行分类，**具有相同条件值的元组划分到一个组或者一个集合中**

分组可以在基本的SELECT语句基础上引入分组子句来完成

例：求**每一个学生**的平均成绩

```sql
SELECT sno, AVG(score) FROM SC
GROUP BY sno;
```

例：求**每一门课程**的平均成绩

```sql
SELECT cno, AVG(score) FROM SC
GROUP BY cno;
```

#### HAVING

例：求不及格课程**超过两门**的同学的学号

```sql
-- WRONG!
SELECT sno FROM SC
WHERE score < 60 AND COUNT(*)>2 GROUP BY sno
```

> 聚集函数不可以用于WHERE子句中<br>
> WHERE子句对每个元组进行条件过滤，而不是对集和进行条件过滤

- 若要对集和进行条件过滤，可使用`HAVING`子句
- HAVING子句中，又称为分组过滤子句，需要有GROUP BY子句

修改后得到

```sql
SELECT sno FROM SC
WHERE score < 60 
GROUP BY sno HAVING COUNT(*)>2;
```

例：求有两门以上不及格课程的同学的学号的学号及其平均成绩

```sql
-- WRONG!
SELECT sno, AVG(score) FROm SC
WHERE score < 60
GROUP BY sno HAVING COUNT(*) > 2;
```

注意语义问题，上述sql查询所求平均值为“**不及格课程的平均值**”

修改为

```sql
SELECT sno, AVG(score) FROm SC
GROUP BY sno 
HAVING COUNT(*) > 2 AND score < 60;
```

### 并、交、差的处理

UNION、INTERSECT、EXCEPT

`子查询 {UNION[ALL] | INTERSECT[ALL] | EXCEPT[ALL] 子查询}`

例：求学过002号课程或者学过003号课程的同学学号

```sql
SELECT sno FROM SC WHERE cno = "002" UNION
SELECT sno FROM SC WHERE cno = "003"
```

例：假定所有学生都有选课，求没有学过002号课程的学生学号

上例不能写成`SELECT sno FROM SC WHERE cno <> "002"`，这样只会排除“只选了002号课程的学生”
 
```sql
SELECT sno FROM SC EXCEPT 
SELECT sno FROM SC WHERE cno = "002"
```

另解：

```sql
SELECT DISTINCT sno FROM SC SC1
WHERE NOT EXISTS(
    SELECT sno FROM SC SC2
    WHERE SC2.cno = "002" AND SC1.sno = SC2.sno
)
```

例：求没学过李明老师讲课的所有同学的姓名

```sql
SELECT sname FROM Student EXCEPT
SELECT sname FROM Student
JOIN SC ON SC.sno = Student.sno 
JOIN Course ON SC.tno = Course.tno AND Course.tname = "李明"
```
 
### 空值处理

- 除了is [not] null 之外，空值不满足任何查找条件
- 如果NULL参与算术运算，则算术表达式的值为NULL
- 如果NULL参与比较运算，则返回结果均为False
- 如果NULL参与聚集运算，则除了COUNT(*)之外的其他聚集函数都忽略NULL

### JOIN

```sql
SELECT <column_name> [[,列名]...]
FROM <table_name> [NATURAL] [INNER | {LEFT|RIGHT|FULL}[OUTER]] JOIN <column_name2>
{ON <conditions> | USING (colname1{,colname...})}
[WHERE <conidtions>]
```

## SQL视图

对应概念模式的数据在SQL中被称为**基本表**(table)，而对应外模式的数据被称为**视图**(view)

- 基本表是实际存储于存储文件中的表，**基本表**中的数据是**需要存储**的
- 视图在SQL中只存储其由基本表导出视图所需要的公式，即由基本表产生视图的映像信息，其**数据并不存储**，而是在运行过程中动态产生与维护的
- 对视图数据的更改最终要反映在**对基本表的更改**上

### 视图的定义与使用

#### 定义视图

```sql
CREATE VIEW view name [(<column_name>[,<column_name>])]
AS 子查询 [with check option]
```

> `with check option`指明当对视图进行insert，update，delete时，要检查进行insert/update/delete的元组是否满足视图定义中子查询中定义的条件表达式

例：定义一个视图CompStud为计算机系的学生，通过该视图可以将Student表中其他系的学生屏蔽掉

```sql
CREATE VIEW CompStud AS(
    SELECT * FROM Student
    WHERE dno IN (
        SELECT dno FROM Dept
        WHERE dname = "计算机"
    )
)
```

#### 使用视图

例：定义好CompStud之后，可以检索计算机系中年龄小于20的所有学生

```sql
SELECT * FROM CompStud
WHERE sage < 20;
```

### 视图的更新

SQL视图更新操作是一个比较复杂的问题，因视图不保存数据，对视图的更新最终要反映到对基本表的更新上，而有时，视图定义的映射不是可逆的

因此，SQL视图更新操作受到很大的约束，很多情况是不能进行视图更新的。

- 如果视图的select目标列包含**聚集函数**，则不能更新
- 如果视图的select子句使用了**UNIQUE**或**DISTINCT**，则不能更新
- 如果视图中包括了**GROUP BY**子句，则不能更新
- 如果视图中包括经**算术表达式**计算出来的列，则不能更新
- 如果视图是由单个表的列构成，但并**没有包括主键**，则不能更新

### 视图的撤消

已经定义的视图也可以撤消

```sql
DROP VIEW view_name
```

例：撤消视图CompStud `Drop View CompStud`