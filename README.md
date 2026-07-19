# 港股 IPO 多智能体风险预警系统

面向港股 IPO 招股书的证据级解析、多角色尽调、上市后风险预测与人机复核原型。

## 当前状态

第一阶段的可运行基线已经完成。当前可以使用一份公开招股书跑通：

```text
PDF -> 页面与坐标 -> OCR兜底 -> 段落切片 -> BM25召回
    -> Evidence -> RiskFinding -> 高亮截图 -> AgentTrace -> 风险JSON
```

现有英矽智能样本可稳定输出8条待人工复核风险，每条均包含PDF物理页码、
印刷页码、原文、坐标、证据截图和置信度。当前风险结论由确定性规则与BM25生成，
尚未接入向量检索、真实大模型、多Agent裁决和上市后预测模型。

## 2026-07-19 首次交付里程碑

本次完成了项目底座和单公司证据闭环基线：

- 冻结 `Evidence`、`RiskFinding`、`AgentTrace`、`Prediction` 及资料目录契约。
- 完成公司、文档、数据集、分析任务、证据、风险和轨迹的SQLite登记与追溯。
- 跑通英矽智能672页招股书解析，生成6,091个带页码和坐标的检索切片。
- 通过8条启用风险规则生成8条候选风险、8条原文证据和8张证据截图。
- 输出版本化风险JSON，并将同一证据链及17次工具调用记录同步写入SQLite。
- 全量验证结果为42项测试通过，Ruff代码检查与mypy类型检查通过。

当前交付边界：全部风险仍为 `pending_verification`；语义向量检索、真实LLM多Agent、
人工复核界面、批量特征与标签、上市后风险预测、API和演示界面尚未完成。

## 核心原则

- 每条风险结论必须绑定原文、PDF 页码和坐标。
- 数值结论必须保留输入、单位、期间和公式。
- Agent 只通过结构化对象协作，不靠自由聊天传递关键事实。
- 预测特征必须满足 Point-in-Time 约束，禁止使用上市后信息。
- 原始数据只读，中间数据可重建，最终产物可追溯。

## 主要入口

- `项目启动方案.md`：总体方案与四阶段路线。
- `docs/目录说明.md`：完整目录职责。
- `docs/数据契约.md`：Evidence、RiskFinding、AgentTrace、Prediction 结构。
- `docs/资料登记与版本管理.md`：增加公司、PDF和结构化数据的标准操作。
- `configs/base.yaml`：运行配置。
- `configs/risks.yaml`：首版风险字典。
- `src/ipo_risk/`：正式源代码。
- `tests/`：单元、集成和防泄漏测试。

## Conda 环境

项目统一使用名为 `hk-ipo-risk` 的 Python 3.11 Conda 环境：

```powershell
conda env create -f environment.yml
conda activate hk-ipo-risk
python -m pytest -q
```

Conda 管理 Python 解释器和环境隔离，Python 包由 `pyproject.toml` 分组维护。不要向 `base` 环境安装项目依赖，也不要在 Conda 环境中再次创建 `.venv`。

## 运行单份招股书分析

先幂等登记公司全部已声明资料：

```powershell
conda activate hk-ipo-risk
python scripts/register_company_data.py `
  --company-folder "03696_英矽智能" `
  --project-root "D:\东吴证券"
```

再运行指定公司最新的正式招股书：

```powershell
conda activate hk-ipo-risk
python scripts/analyze_prospectus.py `
  --company-folder "03696_英矽智能" `
  --project-root "D:\东吴证券"
```

主要产物：

```text
data/interim/documents/<公司目录>/<document_id>/pages.jsonl
data/interim/chunks/<公司目录>/<document_id>/chunks.jsonl
artifacts/evidence/<公司目录>/<document_id>/*.png
artifacts/analysis/<公司目录>/<document_id>/risk_analysis.json
data/app.db
```

`risk_analysis.json` 中的风险状态统一为 `pending_verification`，不能未经人工确认
直接作为正式投研结论。同一份报告中的 Evidence、RiskFinding 和 AgentTrace
也会同步写入SQLite。

## 模型供应商固定分工

- 阿里云百炼 DashScope 的 `qwen3.7-text-embedding` 只负责招股书向量化和语义检索。
- DeepSeek 的 `deepseek-v4-flash` 负责多 Agent 分析、证据核验、冲突查证与最终裁决。
- 程序不读取 Windows 用户环境变量中的密钥，固定读取
  `configs/providers.local.yaml`。该文件在本机保存真实密钥，并被 `.gitignore`
  排除；字段示例见 `configs/providers.example.yaml`。

## 数据放置

官方资料到手后按“数据类型/公司”放入：

```text
data/raw/prospectus/<股票代码_公司名称>/
data/raw/company_info/<股票代码_公司名称>/
data/raw/market/<股票代码_公司名称>/
```

同一公司的跨目录文件由 `company_info/<股票代码_公司名称>/公司信息.json` 统一关联。每家公司在主数据区只维护 `公司信息.json` 和 `日线行情.json` 两个结构化文件；不要把原始数据直接写入 `src/`，也不要修改原始 PDF 内容。

每份文档根据“公司、文档类型、公开日期、SHA-256”生成不可变 `document_id`；
后续增加第二家公司、新版招股书、配发公告或其他结构化数据时，只需增加文件和
资料清单条目，再重新运行登记命令，不需要修改代码。
