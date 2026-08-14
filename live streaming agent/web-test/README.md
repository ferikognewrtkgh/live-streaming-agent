# Live Streaming Agent 直播项目

这是一个与 Dream Maker 现有服务完全分离的测试项目，包含：

- 基于 React、TypeScript 和 vinext 的控制台前端
- 独立 FastAPI 后端
- Elasticsearch 用户、会话、消息和知识库访问层
- 以用户名为入口的完整历史对话载入

## 项目结构

```text
web-test/
├─ app/                  前端页面与样式
├─ lib/                  前端 API 客户端
├─ backend/
│  ├─ app/               FastAPI 应用与 ES 存储层
│  ├─ tests/             后端基础测试
│  ├─ .env.example       后端环境变量示例
│  └─ pyproject.toml     后端独立依赖
└─ .env.example          前端环境变量示例
```

## 本地运行

前端：

```powershell
npm install
npm run dev
```

后端：

```powershell
# 在 web-test/backend 目录执行，环境固定创建在独立后端内
uv venv .venv
uv sync --project . --cache-dir .uv-cache

Copy-Item .env.example .env
uv run --project . --cache-dir .uv-cache uvicorn app.main:app --app-dir . --reload --host 0.0.0.0 --port 8001
```

后端只使用 `web-test/backend/.venv` 和 `backend/.env`，不会读取或修改
Dream Maker 主项目的虚拟环境、环境文件或 `conf.yaml`。如果需要运行后端测试：

```powershell
.\.venv\Scripts\python.exe -m pytest
```

页面只允许选择 Provider 和 Model，不接收或保存 API Base URL 与 API Key。
各厂商密钥只从 `backend/.env` 中对应的 `LIVE_STREAMING_AGENT_*_API_KEY` 读取；可选的
`LIVE_STREAMING_AGENT_LLM_API_KEY` 仅作为当前默认 Provider 的后端兜底密钥。任何密钥都不会
从 Dream Maker 主项目导入，也不会发送到浏览器。

如果所选厂商没有配置密钥，模型请求会失败，并在聊天界面明确显示缺少 API Key
以及应编辑的后端配置位置，不会生成本地替代回复。

后端日志策略：

- 终端：显示前端调用后端的访问日志，以及后端调用模型 API 的 HTTP 日志
- `backend/logs/backend.log`：完整的应用、Uvicorn 访问及错误日志
- `backend/logs/model_calls.jsonl`：模型、完整上下文、回复、耗时及失败原因，只写文件

模型调用日志不会记录 API Key，完整上下文也不会输出到终端。该文件含用户名和
完整对话内容，应按敏感数据管理。

默认地址：

- 前端：`http://localhost:3000`
- 后端：`http://localhost:8001`
- API 文档：`http://localhost:8001/docs`

局域网访问：

- 前端开发和生产启动命令默认监听 `0.0.0.0`
- 后端在 `backend` 目录运行 `uv run --project . --cache-dir .uv-cache uvicorn app.main:app --app-dir . --reload --host 0.0.0.0 --port 8001`
- 同一局域网设备访问 `http://192.168.3.52:3000`
- 未设置 `NEXT_PUBLIC_API_URL` 时，前端会自动请求当前页面主机的 `8001` 端口

## Elasticsearch 索引

后端首次连接 ES 时会建立：

- `live_streaming_agent_users`
- `live_streaming_agent_conversations`
- `live_streaming_agent_messages`

现有知识库索引通过 `LIVE_STREAMING_AGENT_KNOWLEDGE_INDEX` 指定，不会与历史对话索引混用。

每条消息单独存储，以支持完整保留、分页载入、全文检索以及后续的统计分析。
