# 包含可信第三方的部署

**声明**：一切以[隐语官方文档](https://www.secretflow.org.cn/zh-CN/docs/scql/0.5.0b2/topics)为准

官方部署文档有的地方说的不太清楚，我把自己的一点个人理解和自己的部署流程写在下面作为参考（）

## scql系统基本架构

![image](xxaq1-1.png){width=400}

简单来说就是两个参与方想要用双方的数据进行一些“操作”，于是他们将自己的数据放进scql系统当中取进行计算从而得到他们想要的查询结果

在这个查询过程当中，`scql`系统中的`SCDB`负责转化用户的查询语句转化“执行图”(我把他理解为一种计算任务)；然后把计算任务下发给各个参与查询的`SCQLEngine`。而`SCQLEngine`则会执行下发的计算任务，在本机进行计算后将计算结果发送给`SCDB`

## 部署流程（多机部署）

这里的话大部分和[官方部署文档](https://www.secretflow.org.cn/zh-CN/docs/scql/0.5.0b2/topics/deployment/how-to-deploy-centralized-cluster#step-1-deploy-scqlengine)步骤大同小异，主要在配置本地的数据集，还有用户注册上做了一些说明

### 配置引擎

```
--listen_port=8080
--datasource_router=embed
--enable_driver_authorization=false
--server_enable_ssl=false
--driver_enable_ssl_as_client=false
--peer_engine_enable_ssl_as_client=false
--embed_router_conf={"datasources":[{"id":"ds001","name":"mysql db","kind":"MYSQL","connection_str":"db=alice;user=root;password=__MYSQL_ROOT_PASSWORD__;host=mysql;auto-reconnect=true"}],"rules":[{"db":"*","table":"*","datasource_id":"ds001"}]}
# party authentication flags
--enable_self_auth=true
--enable_peer_auth=true
--private_key_pem_path=/home/admin/engine/conf/ed25519key.pem
--authorized_profile_path=/home/admin/engine/conf/authorized_profile.json
```

如果要在使用本地mysql的数据集需要修改`embed_router_conf`项

```
"datasources": [
  {
    "id": "ds001",
    "name": "mysql db for scql",
    "kind": "MYSQL",
    "connection_str": "${connection_str}"
  }
],
"rules":[
  {
    "db": "*",
    "table": "*",
    "datasource_id": "ds001"
  }
]
```

- `connection_str`一项可以参考如下配置，其中`db`为你的数据库名称，`user`是你的数据库的用户名（一般为`root`），`password`是连接数据库的密码，`host`是主机名称/IP地址（一般为`localhost:3306`）

```
db=${db};user=${user};password=${password};host=${host}
```

### 身份验证文件

> scql 0.3.0b1版本重构去除了GRM，使用公私钥验证来保证安全，因此engine、scdb都需要依赖额外的公私钥

```bash
# 生成私钥
openssl genpkey -algorithm ed25519 -out ed25519key.pem

# 从私钥文件中提取公钥，并将其输出为DER格式，然后对DER格式的公钥进行Base64编码
openssl pkey -in ed25519key.pem  -pubout -outform DER | base64

# 链接疑似失效（）
wget https://raw.githubusercontent.com/secretflow/scql/main/examples/scdb-tutorial/engine/alice/conf/authorized_profile.json
```

官方文档在这里用[`openssl`](https://zh.wikipedia.org/wiki/OpenSSL)生成了一个私钥存储在`ed25519key.pem`文件里，文档的意思应该是要让你手动把非己方的公钥放到`authorized_profile.json`这个文件里面作为参与方身份验证；这个配置文件的`parties`数组内应该是可以添加多个参与方，如果有新用户(比如`carol`)进来，就在每个旧用户的这个配置文件后面添加新用户的公钥即可（新用户的配置文件内也要添加所有旧用户的公钥）

- alice方的`authorized_json.profile`内容如下

```
{
    "parties": [
        {
            "party_code": "bob",
            "public_key": "__BOB_PUBLIC_KEY__"
        }
    ]
}
```

- bob方的`authorized_json.profile`内容如下

```
{
    "parties": [
        {
            "party_code": "alice",
            "public_key": "__ALICE_PUBLIC_KEY__"
        }
    ]
}
```

### 客户端上的用户创建

这一部分是针对自定义客户端来写的，[官方文档快速开始部分](https://www.secretflow.org.cn/zh-CN/docs/scql/0.5.0b2/intro/tutorial#create-database-user-and-tables)有一创建用户的步骤

```bash
# ./scqltool genCreateUserStmt --user alice --passwd some_password --party alice --pem examples/scdb-tutorial/engine/alice/conf/ed25519key.pem
root> CREATE USER `alice` PARTY_CODE 'alice' IDENTIFIED BY 'some_password' WITH '2023-08-23T20:03:34.268353853+08:00' '/oWeDbslKFQaqM6aOumnQY56i6MQKNNz84v0nkdhniXS0eBNX/q3n4IYz2EkABgKD+nkIVFtBokQqx5fr29CBw==' 'MCowBQYDK2VwAyEAzvfiNl1c1TjcvaTQBAxpG93MzHRGwuUBrPI3qf5N2XQ='
```

这里使用了`scqltool`来生成创建用户的语句，查看创建用户的命令，不难得出需要指定用户的姓名、密码以及参与方`ed25519key.pem`所在路径

在官方教程里，使用的`ed25519key.pem`文件位于`docker`容器内部的`example`文件夹下方；如果想使用本地的`ed25519key.pem`，或者不想在docker里面输入那段命令再手动复制到客户端执行，或许可以自己编写一个生成“创建用户语句”的脚本

创建用户语句的结构大部分都比较清晰，只有最后的“两段”比较迷惑，第二段比较明显，就是从`ed25519.pem`内加载出的公钥，而第一段尚不清楚。

```bash
'/oWeDbslKFQaqM6aOumnQY56i6MQKNNz84v0nkdhniXS0eBNX/q3n4IYz2EkABgKD+nkIVFtBokQqx5fr29CBw==' 'MCowBQYDK2VwAyEAzvfiNl1c1TjcvaTQBAxpG93MzHRGwuUBrPI3qf5N2XQ='
```

去查看了下[项目有关sqlbuilder的测试代码](https://github.com/secretflow/scql/blob/main/pkg/util/sqlbuilder/sqlbuilder_test.go#L75)的部分，可以看到有下面这部分

```golang
func TestBuildCreateUserStmtWithPubkeyAuth(t *testing.T) {
	r := require.New(t)

	expectSql := "CREATE USER `alice` PARTY_CODE 'ALICE' IDENTIFIED BY 'some_passwd' WITH '2023-08-18T18:12:04.00507585+08:00' 'FKNla5+qiybxx0Tx5gGpn5bHX1+0NgKJPNshq1eCT00/Pogu3QAPXJneUdtYQlmaf7dW1Vr25t+oDLRV9+TiCA==' 'MCowBQYDK2VwAyEA8tjoIkf3mIyz/HGdjBD+p/SDxlzHiNDcaTmhF3dHjZY='"
	mockTime, err := time.Parse(time.RFC3339Nano, "2023-08-18T18:12:04.00507585+08:00")
	r.NoError(err)

	privPemData := []byte(`
-----BEGIN PRIVATE KEY-----
MC4CAQAwBQYDK2VwBCIEIBSXcCv5G1YpIZSD127ImyGnlqA9s9HCpk7jYbl7OQZ5
-----END PRIVATE KEY-----
`)
	block, _ := pem.Decode(privPemData)
	r.True(block != nil && block.Type == "PRIVATE KEY")

    // 这里导出的”？？？“和.pem文件内的公钥部分
	priv, err := x509.ParsePKCS8PrivateKey(block.Bytes)
	r.NoError(err)

	builder := NewCreateUserStmtBuilder()
	sql, err := builder.SetUser("alice").SetParty("ALICE").SetPassword("some_passwd").AuthByPubkeyWithPrivateKey(priv).MockTime(mockTime).ToSQL()

	r.NoError(err)
	r.Equal(expectSql, sql)
}
```

可以看出是将预期`sql`和scqltool生成`sql`进行对比测试，然后"那两段"貌似就是在`priv, err := x509.ParsePKCS8PrivateKey(block.Bytes)`生成的

一波百科后发现：这个`crypto/x509`实际是Go标准库中的一个包，提供对 X.509 标准的支持，该标准定义了公钥证书和私钥存储的格式；然后这个`ParsePKCS8PrivateKey`方法实际是用于解析 PKCS#8 编码的私钥。PKCS#8 是用于存储私钥信息的标准语法，它允许私钥使用密码进行加密；“第一段”应该是程序读取并解析了.pem的私钥文件，并使用`crypto/x509.ed25519.ParsePKCS8PrivateKey`函数将其解析为Ed25519私钥

大致实现如下：

```golang
package main

import (
    "crypto/x509"
    "crypto/ed25519"
    "encoding/pem"
    "encoding/base64"
    "fmt"
    "log"
)

func main() {
    pemData := []byte(`-----BEGIN PRIVATE KEY-----
MC4CAQAwBQYDK2VwBCIEIBSXcCv5G1YpIZSD127ImyGnlqA9s9HCpk7jYbl7OQZ5
-----END PRIVATE KEY-----`)

    block, _ := pem.Decode(pemData)
    if block == nil {
        log.Fatal("Failed to decode PEM data")
    }

    privKey, err := x509.ParsePKCS8PrivateKey(block.Bytes)
    if err != nil {
        log.Fatalf("Failed to parse private key: %v", err)
    }

    ed25519PrivKey, ok := privKey.(ed25519.PrivateKey)
    if !ok {
        log.Fatal("Invalid Ed25519 private key")
    }

    privKeyBytes := []byte(ed25519PrivKey)
    privKeyEncoded := base64.StdEncoding.EncodeToString(privKeyBytes)

    fmt.Println(privKeyEncoded)
}
```

至此，已经弄清了创建用户语句的所有结构
