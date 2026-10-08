# 生产部署说明

本配置默认只向主机开放 80/443，MySQL、Redis、Kafka、OpenSearch、gRPC、控制面和业务接口仅能在容器内部访问。匿名注册在生产环境关闭，用户由管理员创建。

生产容器挂载项目根目录中的 `data`、`logs`、`model_ckp` 和 `rag_knowledge_base`，实验 YAML 配置从 `configs` 目录只读挂载。上传前必须确认这些目录不含涉密数据；密码和访问密钥只允许写入服务器上的 `.env.production`，不得上传本机 `.env` 或历史配置备份。

> 项目涉及保密内容时，只能部署到单位批准的服务器、网络和大模型环境。不得自行上传公共云、公共代码仓库或第三方托管平台。

## 服务器要求

- Linux x86_64；当前训练依赖固定为 amd64。
- Docker Engine 与 Docker Compose v2.24 以上版本。
- 完整演示建议至少 4 核、8 GB 内存、70 GB 磁盘，并配置 8 GB 交换空间；真正执行训练时需按算法要求准备显卡和驱动。
- 域名已经解析到服务器；防火墙只放行管理入口以及 80/443。

## 首次上线

```bash
cp .env.production.example .env.production
chmod 600 .env.production
```

生成随机密钥，每执行一次得到一个不同值：

```bash
openssl rand -hex 32
```

把生成的值分别填写进 `.env.production`，设置首次登录的管理员账号和强密码，同时填写域名、运维邮箱和大模型配置，然后运行：

```bash
./deploy/production/deploy.sh
```

## 验证

```bash
docker compose --env-file .env.production -f docker-compose.yml -f compose.production.yml ps
curl -fsS "https://你的域名/health"
curl -fsS "https://你的域名/readyz"
```

上线后使用管理员账号创建普通用户。生产环境的 `/api/user/register` 会拒绝匿名注册。

## 更新

备份数据库和持久化数据后，重新执行：

```bash
./deploy/production/deploy.sh
```

## 回滚和备份

每次更新前至少保存 MySQL 逻辑备份，以及 `mysql_data`、`redis_data`、`kafka_data`、`opensearch_data` 和业务数据目录。不要把包含业务数据的备份上传到未经批准的位置。
