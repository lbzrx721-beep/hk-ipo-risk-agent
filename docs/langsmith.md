# LangSmith 调用追踪

已接入现有项目，不需要更换模型、重写 LangGraph 或重新建立向量库。

## 开启

1. 登录 [LangSmith](https://smith.langchain.com)，在设置中创建 API Key。
2. 在项目根目录 `.env` 中添加以下配置。不要覆盖已有数据库配置，也不要把密钥发到聊天里。

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=你的LangSmith密钥
LANGSMITH_PROJECT=hk-ipo-risk-agent
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

这是 LangSmith 的独立密钥，不是火山方舟的模型密钥。
如果使用 EU 区域，Endpoint 改为 `https://eu.api.smith.langchain.com`；
多工作区账号必要时补充 `LANGSMITH_WORKSPACE_ID`。
进程环境变量优先于 `.env`，修改后需要重启后端。

3. 按平时方式运行项目，发起一次小规模分析。
4. 打开 LangSmith 的 Tracing，进入 `hk-ipo-risk-agent` 项目。

Docker 用户需要重新构建并启动 app 服务，Compose 已接入上述变量。
所需 SDK 已在项目依赖中声明，按 README 的依赖安装流程安装即可。

## 能看到什么

- `ipo.analyze_prospectus`：单份招股书分析入口。
- `ipo.multi_agent` 及其 LangGraph 子节点：风险发现、各角色执行与裁决。
- `hybrid.search`、`pgvector.search`、`bm25.search`、`semantic.search`：query、top_k、命中原文、页码、分数、排名。
- `json_response` → 供应商 `.chat`：提示词、模型回答、流式汇总、耗时、错误，以及接口返回的 Token 用量。
- `embeddings.create`：实际向量请求的输入、返回条数与接口用量；不上传向量数组。缓存命中不产生新的模型请求。

这里只观测已埋点的调用和 LangGraph 节点，不是 Python 每一行代码的调试器。
注入测试用替身客户端时，不自动套用真实 OpenAI SDK 适配器。
已有 SQLite 业务审计和模型用量统计保持不变。非 OpenAI 模型的金额以供应商账单/项目原有计费配置为准，不能假设 LangSmith 会自动算对。

## 数据与费用

开启追踪会向 LangSmith 云端发送提示词、模型回答、检索到的招股书原文等内容。
未公开或敏感材料请保持 `LANGSMITH_TRACING=false`。
若只排查耗时，可同时设置 `LANGSMITH_HIDE_INPUTS=true` 和 `LANGSMITH_HIDE_OUTPUTS=true`，但错误消息和元数据仍需注意敏感信息。
模型密钥、数据库连接对象不作为自定义追踪输入上传；不要主动把密钥写进提示词。

追踪本身不会额外调用大模型；LangSmith 记录有独立的平台用量限制。
关闭方法：设置 `LANGSMITH_TRACING=false` 并重启后端。

离线测试不等于云端追踪已连通；配置后应在自己的工作区确认运行记录是否入库。

官方说明：[LangGraph 追踪](https://docs.langchain.com/langsmith/trace-with-langgraph)。
