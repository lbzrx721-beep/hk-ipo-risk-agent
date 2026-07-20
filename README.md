# 港股 IPO 多智能体风险预警系统

面向港股 IPO 招股书的证据级解析、多角色尽调、上市后风险预测与人机复核原型。

## 当前状态

第一阶段的可运行基线已经完成。当前可以使用一份公开招股书跑通：

```text
PDF -> 页面与坐标 -> OCR兜底 -> 段落切片
    -> 财务/法务/业务/市场Agent自主生成语义搜索问题
    -> DashScope向量召回原文候选 -> 专业Agent证据约束分析
    -> DeepSeek决策Agent裁决与分歧记录
    -> Evidence -> RiskFinding/Calculation -> 高亮截图 -> 风险JSON
```

现有英矽智能历史基线已输出8条待人工复核风险，每条均包含PDF物理页码、
印刷页码、原文、坐标、证据截图和置信度。项目配置有效密钥时，新分析会使用真实语义
向量检索、DeepSeek分角色搜索与分析和DeepSeek决策裁决。专业Agent会先按照角色职责为每类
风险生成语义搜索问题，再读取向量召回的候选原文；搜索规划或向量服务失败时才使用固定
多查询BM25兜底，模型重试耗尽后才退回精确规则证据。任何模式都不会生成无证据结论。上市后预测
模型仍需等待比赛行情与标签数据。

候选风险现可通过命令行或Streamlit工作台执行确定性核验、人工确认/驳回/编辑和最终报告
导出。核验与人工决定均写入SQLite审计记录；任何失败核验都不能被标记为 `verified`。
Streamlit工作台现在还支持直接拖入一份招股说明书PDF，完成文件校验、幂等登记、解析、
多Agent分析、决策裁决、报告生成并跳转到人工复核。

## 2026-07-19 首次交付里程碑

本次完成了项目底座和单公司证据闭环基线：

- 冻结 `Evidence`、`RiskFinding`、`AgentTrace`、`Prediction` 及资料目录契约。
- 完成公司、文档、数据集、分析任务、证据、风险和轨迹的SQLite登记与追溯。
- 跑通英矽智能672页招股书解析，生成6,091个带页码和坐标的检索切片。
- 通过8条启用风险规则生成8条候选风险、8条原文证据和8张证据截图。
- 输出版本化风险JSON，并将同一证据链及17次工具调用记录同步写入SQLite。
- 全量验证结果为42项测试通过，Ruff代码检查与mypy类型检查通过。

当前交付边界：全部模型风险仍为 `pending_verification`；多Agent分角色裁决、批量特征与
标签、上市后风险预测、完整API和比赛演示看板尚未完成。

其中“多Agent分角色裁决”已在2026-07-20里程碑完成；本段保留首次交付时的历史边界。

## 2026-07-19 风险核验与人工复核闭环

本次新增了可审计的复核状态机：

- 对 Evidence 复核源文件存在性、SHA-256、PDF物理页、坐标、原文和证据截图。
- 对 RiskFinding 复核证据引用、公司归属及数值公式；公式仅允许安全算术表达式。
- 数值风险通过规则声明字段路径，从公司资料自动取值并记录输入来源；生成与复核共用同一
  安全算术执行器。
- 支持 `confirmed`、`rejected`、`edited` 三种人工决定，并原子更新风险状态与复核记录。
- 已完成SQLite `1.0 -> 1.1` 幂等迁移，旧数据库会自动补充完整复核载荷和最终报告路径。
- 只有全部风险进入 `verified` 或 `rejected` 后才能导出 `verified_risk_analysis.json`；
  最终报告只包含 `verified` 风险及其引用证据。
- 新增 `list`、`verify`、`review`、`export` 四个命令行操作，以及可视化人工复核工作台。
- 人工复核进度可导出为独立版本的标注快照，保存当前风险、证据、核验结果和完整审计记录。
- 全量60项测试、Ruff和mypy通过。

现有英矽智能样本的基石配售比例已按
`37,204,500 / 94,690,500 * 100 = 39.290636%` 自动生成 `Calculation`；重新分析后的
8条候选风险已全部通过只读确定性核验。系统不会代替投研人员自动确认风险。

## 2026-07-19 混合检索与DeepSeek证据分析

- 新增真实语义向量检索，使用OpenAI兼容Embedding接口批量生成向量并按余弦相似度召回。
- 使用倒数排名融合（RRF）合并BM25与语义候选，审计记录保留两路分数和排名。
- 文档向量按来源哈希、切片文本、模型和维度生成指纹，缓存到文档级模型目录；相同版本
  再次分析不会重复向量化。
- DeepSeek通过JSON Output返回结构化判断，只能引用本次提供的候选 `chunk_id`；未知证据、
  重复证据、超出规则数量或不完整结构会被拒绝。
- Embedding批次和DeepSeek证据分析节点均支持指数退避重试；DeepSeek的JSON解析与证据
  白名单校验也属于同一重试节点。重试耗尽后才回退到满足全部锚定关键词的BM25规则证据。
- `AgentTrace` 记录实际尝试次数、重试次数、检索/分析模式与最终降级原因。
- 已完成真实百炼1024维向量连接、真实DeepSeek V4 Flash JSON连接，以及公开招股书现金流
  证据的受约束分析验证。
- 全量74项测试、Ruff和mypy通过。

默认重试参数位于 `configs/base.yaml` 的 `agents` 区段：首次失败后等待1秒、2秒，单次
等待上限8秒，`max_retries: 2` 表示首次调用之外最多再重试2次。普通鉴权或参数类4xx错误
不会重复请求；连接、超时、429、服务端错误及可重新生成修复的模型JSON/证据校验错误会重试。

## 2026-07-20 多角色分析与证据裁决

- 风险规则按 `owner` 分发给财务、法务、业务和市场 Agent，共享同一文档检索索引，避免
  重复向量化。
- 每条规则支持主查询和多组 `query_variants`；向量不可用时，多查询BM25仍会把宽召回候选
  交给DeepSeek，`required_keywords`只用于模型失败后的高确定性规则兜底。
- 已启用关联交易、赎回/对赌、诉讼处罚和单一产品依赖等非标风险，目前风险字典覆盖12类。
- DeepSeek决策Agent只能引用专业Agent已选中的提案ID和证据ID；它可以接受、调整或驳回，
  严重程度分歧和“是否成立”分歧均会写入裁决工具调用。
- 决策节点与证据分析节点使用相同的指数退避重试策略；裁决重试耗尽后使用证据约束的
  确定性裁决，不会丢失已完成的专业Agent结果。
- `risk_analysis.json` 保存总控、专业角色和决策节点的父子 `AgentTrace`。Streamlit复核页新增
  “Agent轨迹”页签，可查看角色分工、输入输出、模型模式、尝试次数、降级原因和冲突处理。
- 这些改进建立了可评测链路，但在比赛数据及人工金标准到位前，不能宣称已经达到80%准确率
  或85%证据召回率。

## 2026-07-20 Agent自主证据检索

- 财务、法务、业务和市场Agent不再以固定关键词查询作为主检索入口；每个Agent先根据自己的
  风险职责、风险定义和证据字段，自主生成3至6个互补语义问题。
- Agent生成的问题直接查询同一份招股书的DashScope向量索引，召回原文切片后再由该角色的
  DeepSeek证据分析节点判断风险是否成立；固定查询与BM25仅作为故障兜底。
- 搜索规划、JSON结构校验和风险覆盖校验使用与其他模型节点一致的指数退避重试策略。
- `AgentTrace` 新增 `agent_search_planning` 工具调用，记录角色、风险类型、实际问题、模型、
  提示词版本、尝试次数及兜底原因，实现搜索到裁决的完整可追踪。
- Streamlit运行台会实时显示每个Agent的搜索规划流、检索阶段、分析流和完成状态；发生兜底
  时会显示“检索兜底”，不会把算法结果伪装成模型检索结果。
- 当前能力、交付边界和后续工作统一维护在 `docs/项目状态.md`。

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
data/interim/embeddings/<公司目录>/<document_id>/<模型>/metadata.json
data/interim/embeddings/<公司目录>/<document_id>/<模型>/vectors.npy
artifacts/evidence/<公司目录>/<document_id>/*.png
artifacts/analysis/<公司目录>/<document_id>/risk_analysis.json
data/app.db
```

`risk_analysis.json` 中的风险状态统一为 `pending_verification`，不能未经人工确认
直接作为正式投研结论。同一份报告中的 Evidence、RiskFinding 和 AgentTrace
也会同步写入SQLite。

## 核验、人工复核与最终报告

列出一个分析任务的全部候选风险：

```powershell
python scripts/review_findings.py list `
  --run-id "RUN-03696-..." `
  --project-root "D:\东吴证券"
```

先执行不改变状态的确定性核验，再由投研人员作出决定：

```powershell
python scripts/review_findings.py verify `
  --run-id "RUN-03696-..." `
  --finding-id "F-03696-0001" `
  --project-root "D:\东吴证券"

python scripts/review_findings.py review `
  --run-id "RUN-03696-..." `
  --finding-id "F-03696-0001" `
  --decision confirmed `
  --reviewer "投研复核人" `
  --comment "原文、期间和风险表述一致" `
  --project-root "D:\东吴证券"
```

`rejected` 和 `edited` 必须填写复核意见；`edited` 还必须通过
`--edited-finding` 提供完整的 RiskFinding JSON。全部风险完成复核后导出最终报告：

```powershell
python scripts/review_findings.py export `
  --run-id "RUN-03696-..." `
  --project-root "D:\东吴证券"
```

仍有 `proposed` 或 `pending_verification` 风险时，最终报告导出会被拒绝。

## 人工复核工作台

在项目根目录启动Streamlit页面：

```powershell
conda activate hk-ipo-risk
streamlit run src/ipo_risk/ui/app.py
```

页面默认进入“新建分析”：拖入PDF后自动识别股票代码、公司名称和招股书日期，必要时可在
页面校正识别结果，再点击“开始分析”。
系统会按阶段显示PDF解析页数与切片数、专业Agent运行、决策Agent裁决和证据报告生成进度；
完成后可以直接点击“进入人工复核”。同一公司同一PDF通过SHA-256识别，只登记一次且不会
重复复制原始文件；不同内容或不同日期会生成独立的不可变文档版本。

当前上传页面一次处理一家公司的一份招股书；批量公司队列是后续扩展，不影响现有文档和
运行之间的隔离设计。单个文件上限250MB，损坏、空白、无页面或带密码的PDF会在分析前拒绝。

页面支持选择分析任务、查看复核进度、风险结论、严重程度、置信度、PDF原文、物理页码、
印刷页码、坐标、高亮截图、公式、全部确定性核验项和多Agent推理轨迹。轨迹页会展示每个
角色的工具调用、模型模式、重试降级与裁决分歧。填写复核人及意见后，可以确认、驳回或编辑
风险；操作会调用与命令行相同的 `ReviewService`，并写入SQLite不可变审计记录。

每次点击“导出版本化标注”都会在
`data/annotations/<company_id>/<run_id>/review_annotations_<version>.json` 生成独立快照。
全部风险复核完成后，“导出正式报告”按钮才会启用。

## 模型供应商固定分工

- 阿里云百炼 DashScope 的 `qwen3.7-text-embedding` 只负责招股书向量化和语义检索。
- DeepSeek 的 `deepseek-v4-flash` 负责财务、法务、业务、市场Agent的语义问题规划和候选
  原文分析，以及决策Agent的冲突查证与最终裁决。
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
