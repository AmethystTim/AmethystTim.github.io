## 数据库完整性

指的是DBMS应该保证DB在任何情况下的**正确性**、**有效性**和**一致性**

### 什么会引发数据库完整性的问题

- 不正当的数据库操作，比如输入错误、操作失误、程序处理失误等等

### 数据库完整性的作用

- 防止和避免数据库中不合理的数据出现
- DBMS应尽可能自动防止DB中语义的不合理现象
- DBMS不能自动防止，则需要应用程序员和用户进行在进行数据库操作时需要处处加以小心，每写一条SQL都需要考虑是否符合语义完整性
- 为了减轻工作负担，需要**尽可能多地让DBMS来承担**

### 如何自动保证完整性

- DBMS允许用户定义一些**完整性约束规则**（使用SQL-DDL来定义）
- 当有DB更新操作时，DBMS自动按照完整性约束条件进行检查

### 如何定义规则

完整性约束条件的一般形式

$$Integrity \quad Constraint = (O,P,A,R)$$

- O——数据集合：约束的对象
- P——谓词条件：什么样的约束
- A——触发条件：什么时候检查
- R——响应动作：不满足时该怎么办

#### 按照约束对象分类

- 域完整性约束条件
- 关系完整性约束条件

#### 按照约束来源

- 结构约束
    - 来自于模型的约束，例如对函数依赖的约束、主键约束、外键约束
    - 只关心数值相等与否，是否为空值等
- 内容约束
    - 来自于用户的约束，例如用户自定义完整性，关心元组或属性的取值范围。例如Student表的sage属性值在15岁到40岁之间

#### 按照约束状态

- 静态约束
    - 要求DB在任意时候都应该满足的约束，比如Sage在任何时候都应该满足大于0并且小于150
- 动态约束
    - 要求DB从一状态变为另一个状态应该满足的约束
    - 比如工资*只能升不能降*

### SQL支持的约束

- 静态约束
    - 列完整性
    - 表完整性
- 动态约束
    - 触发器

- 使用`Create Table`

```sql
CREATE TABLE tablename
    ((colname datatype [DEFAULT {default_constant | NULL}]
            [col_constr {col_constr...}]
            |,table_constr
        {,{colname datatype [DEFAULT {default_constant | NULL}]
            [col_constr {col_constr}]
            |,table_constr}
            })
    )
```

> 使用空格区分列约束，逗号区分表约束

#### Col_constr列约束

```sql
{
    NOT NULL|
    [CONSTRAINT constraintname]
    {
        UNIQUE
        | PRIMARY KEY
        | CHECK (search_condition)
        | REFERENCES tablename [(colname)]
            [ON DELETE CASCADE]
    }
}

-- 引用另一个表tablename的列colname的值，如果有ON DELETE CASCADE 或 ON DELETE SET NULL，那么删除被引用表的某列值v时，要将本表该列值为v的列值更新为null，缺省则无操作
```

#### Table_constr表约束

```sql
[CONSTRAINT constraintname]
    {
        UNIQUE (colname {, colname...})
        | PRIMARY KEY (colname{, colname})
        | CHECK (search_condition)

        | FOREIGN KEY (colname{, colname})
            REFERENCES tablename [(colname {,colname...})]
            [ON DELETE CASCADE]
    }
```

### 追加和撤销

CREATE Table中定义的表约束或列约束可以在以后根据需要进行撤销或者追加，语句是`ALTER Table`

```sql
ALTER Table SC
    DROP CONSTRAINT ctscore
```

```sql
ALTER Table SC
    ADD CONSTRAINT nctscore (Score >= 0 AND Score <100>)
```

### 视图控制

通过视图可以限制用户对关系中的某些数据项的存取

访问权限分为以下三种

1. SELECT：读
2. MODIFY：更新（包含插入、更新、删除）
3. CREATE：创建（更新、删除）

> 通过超级用户DBA创建用户，实现权利的传递

### 授权命令

```sql
GRANT {ALL PRIVILEGES | privilege {,privilege...}} ON  
    [TABLE] tablename | viewname
    TO {public | userid {,userid...}} 
    [WITH GRANT OPTION]
```

> 授予视图访问的权利，并不意味着授予基本表访问的权利

### 收回授权命令

```sql
REVOKE {ALL privileges | priv {,priv...}} ON tablename | viewname
FROM {public | userid {,userid...}};
```

例：

`REVOKE SELECT ON employee FROM UserB`