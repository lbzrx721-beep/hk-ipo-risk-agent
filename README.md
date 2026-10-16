<div align="center">

# HK IPO Risk Agent

### 面向港股招股书的证据驱动多智能体分析工作台

从公司画像、审查规划到原文查证与人工复核，让风险分析有依据、可追溯。

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/Frontend-React%2019-149ECA?logo=react&logoColor=white)
![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-1C3C3C)

[项目能力](#项目能力) · [分析流程](#分析流程) · [快速开始](#快速开始) · [开发与测试](#开发与测试) · [项目文档](#项目文档)

</div>

---

## 项目简介

HK IPO Risk Agent 将招股书解析、检索增强生成（RAG）、多角色分析和人机协同复核整合到同一工作台，辅助研究人员梳理公司经营特征、核查风险线索，并回到原文验证结论。

项目的核心不是让模型直接给出一份总结，而是建立可检查的分析过程：**先理解公司、形成待查问题，再由专业 Agent 独立查证，最后整合分歧并交由人工复核。**

适用于招股书研究、长文档证据检索与多 Agent 工作流的原型验证。系统输出是辅助研究材料，不构成投资建议。

## 项目能力

- **招股书结构化解析**：提取文本与表格，保留页码、切片编号及位置引用，关联原文截图，便于逐项核验。
- **可降级的 RAG 检索**：支持 BM25 词法检索与混合检索。当前默认使用 BM25，不依赖向量服务；配置语义通道后可启用混合检索。
- **公司画像与审查规划**：围绕业务、收入模式、渠道、供应链、核心资产等主题读取原文，形成带引用的公司画像和角色化审查问题，保留基础必查项。
- **多 Agent 专业核验**：财务、法律、业务和市场 Agent 根据各自问题重新检索全文，形成证据支持的专业提案，而非仅复述上游摘录。
- **质疑与定向补证**：决策 Agent 在存在实质疑点时发起补证，由相关角色查证后再次裁决；并非每次分析都会触发辩论。
- **人工复核与审计交付**：在工作台查看证据、确认或驳回结论、修改意见，并导出风险报告、完整审计材料与推理日志。
- **运行过程可观察**：展示角色状态、事件进度和结构化结果，记录检索与工具执行信息；可选接入 LangSmith 追踪。

## 分析流程

1. **文档准备**：上传招股书，完成解析、切片与检索准备。
2. **理解公司**：风险发现 Agent 提取有原文依据的公司画像，列出未知信息及专项审查问题。此阶段不提前认定风险。
3. **确认范围**：将基础检查与公司专项问题整理为审查计划，支持人工确认调整。
4. **专业查证**：财务、法律、业务与市场 Agent 分工检索支持材料和反证，形成专业结论。
5. **决策整合**：汇总提案、处理分歧；必要时开展定向补证与再次裁决。
6. **复核交付**：研究人员核验原文及表格，完成复核并导出报告和审计记录。

| 角色 | 主要职责 |
| --- | --- |
| 风险发现 Agent | 公司画像、基础检查覆盖、专项问题规划 |
| 财务 Agent | 现金流、负债、盈利质量与集中度核验 |
| 法律 Agent | 合规、诉讼、特殊权利及合同条款核验 |
| 业务 Agent | 商业模式、客户与供应链依赖、披露表达分析 |
| 市场 Agent | 发行结构、流动性及可用市场信息分析 |
| 决策 Agent | 整合证据、发起质疑、补证后裁决 |

六个角色可在启动分析前分别选择模型；运行中保持模型分配不变。工作流由程序编排，模型承担画像提取、检索问题规划和证据分析等任务。

## 技术栈

| 层次 | 实现 |
| --- | --- |
| 交互界面 | React 19、TypeScript、Vite、Tailwind CSS |
| API 与任务服务 | FastAPI、Pydantic、Python |
| Agent 编排 | LangGraph、结构化输出、角色提示词与审计事件 |
| 文档处理 | PyMuPDF、表格提取、OCR 相关组件 |
| 检索 | BM25、中文分词、可选向量检索与 pgvector |
| 数据与报告 | SQLite、本地文件存储、ReportLab |
| 部署与观察 | Docker Compose、可选 LangSmith |

项目也保留特征处理与模型推理接口。预训练模型、原始招股书和私有数据不随 Git 仓库分发；相关功能需要另外配置匹配的模型与数据。

## 快速开始

### 1. 获取项目

```bash
git clone https://github.com/lbzrx721-beep/hk-ipo-risk-agent.git
cd hk-ipo-risk-agent
```

私有仓库需要先获得访问权限。后续命令均在项目根目录执行，不依赖特定磁盘路径或中文父目录。

### 2. 配置模型服务

复制配置模板：

**Windows PowerShell**

```powershell
Copy-Item configs/providers.example.yaml configs/providers.local.yaml
```

**Linux / macOS**

```bash
cp configs/providers.example.yaml configs/providers.local.yaml
```

编辑 `configs/providers.local.yaml`，填写 `llm` 和 `specialist_llm` 的服务地址、密钥等配置。模板使用的服务与模型名称需要与你的账号实际权限匹配；前端显示的模型选项不代表服务商已为账号开通权限。

默认配置为：

```yaml
retrieval:
  mode: bm25
agents:
  discovery_strategy: company_review_plan
```

此模式无需向量 API 或向量数据库。若要使用混合检索，再配置向量服务、索引及数据库连接，并将 `retrieval.mode` 改为 `hybrid`。详见[公司画像与审查规划](docs/company_review_planning.md)。

### 3. 使用 Docker 启动

先安装 Docker 与 Docker Compose。Windows / macOS 使用已启动的 Docker Desktop；Windows 需运行 **Linux 容器**。Linux 使用 Docker Engine 与 Compose 插件。

```bash
docker compose up -d --build
```

启动后访问：

- 工作台：<http://localhost:8000>
- API 文档：<http://localhost:8000/docs>

首次构建需要下载基础镜像和依赖，耗时取决于网络及设备；后续构建可复用缓存。若公共镜像源访问受限，请配置可用的容器镜像源或网络代理。

Compose 同时准备 PostgreSQL / pgvector 服务；默认 BM25 流程不使用该服务。内置数据库配置仅适合本机开发，部署到共享环境前应更换凭证并完善访问控制。Web 端口默认仅绑定本机地址。

常用维护命令：

```bash
# 查看应用日志
docker compose logs -f app

# 代码更新后重新构建应用
docker compose up -d --build app

# 停止服务，保留命名卷中的运行数据
docker compose down
```

不要随意执行 `docker compose down -v`，它会删除命名卷中的运行数据。模型文件不会被自动打入镜像；需要模型推理时，应另行提供匹配文件并挂载至 `/app/artifacts/models`。

## 开发与测试

本地开发需要 Python 3.11+ 和 Node.js 20+。建议使用独立 Python 虚拟环境，并先按上文完成模型配置。

```bash
python -m venv .venv
```

Windows 使用 `.venv\Scripts\Activate.ps1` 激活；Linux / macOS 使用 `source .venv/bin/activate`。

```bash
python -m pip install -e ".[pdf,ocr,retrieval,data,modeling,api,reporting,dev]"
python scripts/run_web_api.py --no-browser
```

另开终端启动前端开发服务：

```bash
cd frontend
npm ci
npm run dev
```

开发界面位于 <http://localhost:5173>，API 请求代理至本机 `8000` 端口。构建前端后，后端也可以直接提供 `frontend/dist` 中的静态界面。

```bash
# 项目根目录：运行后端单元测试
python -m pytest tests/unit -q

# frontend 目录：检查前端并构建
npm run build
node --test tests/company-review-plan.test.mjs
```

离线测试使用模拟模型验证流程与边界，不替代真实模型服务、真实招股书的端到端验证。

## 配置与数据

| 位置 | 用途 |
| --- | --- |
| `configs/base.yaml` | 检索模式、工作流、模型路由及运行参数 |
| `configs/providers.local.yaml` | 本地模型服务与凭证，不进入版本控制 |
| `configs/risks.yaml` | 基础风险检查目录 |
| `prompts/` | 画像、专业分析与决策提示词 |
| `data/` | 本地数据库、原始材料与处理数据 |
| `artifacts/`、`output/` | 分析证据、运行产物及报告 |

本地启动使用项目目录存储数据；Docker 中的数据库与分析产物通过命名卷保存。请自行备份重要运行结果。

API 密钥、原始招股书、模型文件、交付压缩包与本地生成结果均不作为源码上传。使用外部模型服务时，选中的原文片段会发送至所配置的服务商，上传材料前应确认其使用授权及数据处理要求。LangSmith 默认关闭，启用前请阅读[追踪配置说明](docs/langsmith.md)。

## 项目结构

```text
hk-ipo-risk-agent/
├── configs/             # 运行配置与风险检查目录
├── docs/                # 设计说明与配置文档
├── frontend/            # 分析、预测与复核工作台
├── prompts/             # 角色提示词与结构化输出约束
├── scripts/             # 启动与辅助脚本
├── src/ipo_risk/
│   ├── agents/          # 画像规划、专业分析、决策与补证
│   ├── api/             # API 与批量任务服务
│   ├── parser/          # 文本、表格与位置解析
│   ├── retrieval/       # 词法与语义检索
│   ├── modeling/        # 特征处理与模型推理接口
│   ├── reporting/       # 报告与审计导出
│   └── storage/         # 数据库与记录存储
├── tests/               # 自动化测试
├── Dockerfile
└── docker-compose.yml
```

## 项目文档

- [公司画像与审查规划](docs/company_review_planning.md)：风险发现 Agent 的输入、检索流程、问题分派与降级行为。
- [LangSmith 追踪配置](docs/langsmith.md)：运行追踪与敏感数据注意事项。
- [数据契约](docs/数据契约.md)：主要数据对象与字段约定。

## 使用边界

本项目是面向研究与验证的原型系统，不是证券投资决策工具。原文可定位不等于结论必然正确，表格提取、模型概括和风险归因仍需人工复核。BM25 模式侧重词法匹配，不能等同于语义检索；没有触发补证也不意味着所有风险已经排除。

当前部署面向本机或受控环境，不应在缺少身份认证、访问控制与数据保护措施时直接暴露到公网。
