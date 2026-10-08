# Flask HTTP 中转替换记录

## 当前线上链路

```text
React
  -> Go Business API（登录、业务 CRUD、会话、评测、审批、SSE）
  -> Go Control Plane（Run、Action、租约、事件、Kafka）
  -> Python gRPC（LangGraph、RAG、工具执行、Celery 投递、图片转换）
```

`docker-compose.yml` 已不再启动常驻 Flask HTTP `api` 服务。数据库建表和 Golden Dataset 初始化由一次性的 `bootstrap` 服务完成；Python 算法进程只开放内部 gRPC。

## 已迁移的接口

- Agent 对话、断点续传、运行取消、事件快照；
- 会话创建、列表、详情和删除；
- Action 草案校验、列表、详情、批准、拒绝和执行结果核验；
- Golden Case、Golden Eval 创建、列表、详情和 Celery 投递；
- 监控摘要、安全审计、能力清单；
- 实验对比、指标分析、数据集质量检查；
- RAG 状态、混合检索、索引重建；
- `.npy` 数据图片转换。

## 第二阶段：移除 Python 内部 Flask 依赖

- 使用原生 SQLAlchemy `declarative_base`、`scoped_session` 和显式 Engine 替换 Flask-SQLAlchemy；
- 使用 ContextVar 作为 Session scope，保证 gRPC 请求与 LangGraph 并行 Tool Call 互不共用 Session；
- gRPC、Celery Worker、MCP Server 和数据库 bootstrap 均显式创建并清理数据库 Session；
- Python 镜像默认入口改为 gRPC，不再包含 Flask、Flask-Cors、Flask-Migrate、Flask-SQLAlchemy 和 Werkzeug 运行依赖；
- 旧 Flask HTTP 路由已归档到 `Trash/legacy-flask-http`，不进入镜像运行链路。

当前活动代码中不再创建 Flask application，也不再使用 application context。保留的 `db.Model`、`db.session` 和 `Model.query` 只是 `app/database.py` 提供的兼容外观，底层全部是原生 SQLAlchemy。

## 验证命令

```bash
cd services/go-business-api && GOCACHE=/private/tmp/knowledge-agent-go-cache go test ./...
cd services/go-control-plane && GOCACHE=/private/tmp/knowledge-agent-go-cache go test ./...
.venv/bin/python -m pytest -q
docker compose config --quiet
```
