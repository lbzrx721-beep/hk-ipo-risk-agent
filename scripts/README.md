# 操作脚本

## 新版网页

```powershell
python scripts/run_web_api.py
```

当 `frontend/dist` 已构建时，访问 `http://127.0.0.1:8000` 即可使用
React批量分析工作台；开发模式说明见 `frontend/README.md`。

当前已经提供：

- `analyze_prospectus.py`：输入标准公司目录，生成页面、切片、证据截图和风险JSON。
- `check_competition_data.py`：只读扫描赛事招股书、主表和1GB级行情CSV，输出完整性、
  主表截断及1/5/20/60日覆盖率。
- `import_competition_data.py`：把赛事交付安全放入项目，生成565家公司资料、按公司拆分的
  官方日行情、市场标签、时间切分和质量报告，并可登记到SQLite。
- `build_modeling_dataset.py`：扫描全市场赛事行情并生成每日市场情绪缓存、554家公司
  Point-in-Time特征矩阵和逐特征可用时间/来源审计。
- `build_structured_fundamental_dataset.py`：复用既有表格和全文切片缓存，确定性提取发行价、
  发行规模、募集资金、市值、发行比例、收入增长、毛利率和经营现金流等30个结构化特征；
  不调用Agent或向量模型，完成后自动刷新基础特征表和Agent融合表。
- `train_prediction_baseline.py`：按2020—2023/2024/2025时间切分训练招股书结构、市场环境
  和融合三组五日显著下跌基线，输出指标、概率、解释和业务质量门禁。
- `train_agent_feature_comparison.py`：在相同公司样本上完成“64维/145维 × 逻辑回归/
  LightGBM”四组对照；固定2020—2022训练、2023验证与早停、2024测试，2025年仅作外部
  案例；强制阻断全部`target_*`、`label_*`和10个低信息字段，并输出LightGBM的Gain及
  验证集Permutation特征重要性。
- `train_stable_agent_model.py`：先用固定业务白名单排除文件大小、页数、重复发行上下限和
  Agent运行过程统计，再使用2022和2023滚动验证筛选跨年份稳定特征；分别训练“基础特征”
  和“基础＋Agent风险语义”两组模型，将2024完整保留为独立测试集，并输出Agent增量价值、
  特征稳定性、逐公司预测和两个可部署模型包。
- `run_llm_prediction.py`：与LightGBM并列的LLM直接预测入口。把可读基础字段、Agent原文
  证据和上市前市场环境交给多头/空头Agent并行分析，再由决策Agent分别输出“招股书风险
  严重度”和“五日显著下跌概率”。概率50分表示事件更可能发生，业务预警线独立校准为25分。
  增加`--prediction-mode single`后，同一份输入只调用一次单一决策Agent；输出契约不变，
  但`method`会标记为`llm_single_judge`，且不伪造多头/空头评分。
  20家公司对照未证明历史案例RAG带来增量，因此正式运行默认`--case-retrieval none`；仍可用
  `--case-retrieval hybrid`复现实验，或用`numeric`降级检索。先用
  `python scripts/run_llm_prediction.py --company-id 00300 --dry-run` 免费检查输入；去掉
  `--dry-run` 才会调用模型并产生费用；委员会模式调用3次，single模式调用1次。
- `build_historical_case_index.py`：从Agent增强宽表构建独立的历史IPO案例库。向量文本只含
  上市前发行、财务、市场、行业和Agent风险信息；五日真实表现仅作为返回元数据，不参与
  向量生成。先执行`--dry-run`核对数量，再去掉参数写入pgvector。
- `backtest_llm_case_rag.py`：对同一批历史公司执行严格成对对照。A组不提供历史案例，
  B组使用pgvector混合案例检索；两组共用模型、提示词、温度和阈值，并统计准确率、
  召回率、精确率、F1、PR-AUC、Brier、Token与耗时。先运行
  `python scripts/backtest_llm_case_rag.py --sample-size 5` 做小样本链路验证；结果逐条写入
  `artifacts/evaluations/llm_case_rag/<时间>/backtest.json`。支持`--arm`只跑一组、
  `--prediction-mode single`测试单模型、`--alert-threshold`重分类和`--exclude-backtest`
  构造不重叠留出集；小样本不代表正式效果。
- `run_prediction_baseline.py`：按顺序完成上述特征构建和三组模型训练；首次扫描大行情后会
  复用缓存。
- `run_agent_feature_experiment.py`：按文档哈希无标签抽样历史招股书，顺序执行向量化、
  动态风险发现、多Agent分析和Agent特征生成；逐公司落盘并支持失败隔离、断点续跑和
  基础/Agent特征矩阵融合。
- `run_unattended_agent_experiment.py`：从指定数据分组选择尚未成功分析且当前未运行的公司，
  自动跳过人工风险范围确认，受控并发生成逐公司报告、Agent特征和统一训练JSON；使用同一
  实验名称可断点续跑。
- `build_agent_augmented_dataset.py`：先离线读取现有 `risk_analysis.json`，将动态
  `open_xxx` 风险归一为稳定的一级风险族和细分主题，生成逐风险语义映射审计和公司级
  数值特征；再按公司保留最新run并校正时间分组，合并上市前基础特征及首日、5日、20日、
  60日表现标签。全程不会重新解析PDF、调用向量模型或Agent。
- `vectorize_competition_with_timeout.py`：批量生成和校验本地 `vectors.npy` 备份，支持
  PDF解析超时隔离、Embedding批次断点、失败重试和断点续跑；持久化导入由pgvector脚本负责。
- `init_pgvector.py`：在PostgreSQL中创建 `vector` 扩展、项目专用schema、1024维招股书
  向量表和独立历史IPO案例表；全量导入完成后可选择创建HNSW索引。
- `import_embeddings_to_pgvector.py`：只读取现有切片和 `vectors.npy`，以公司为事务并发迁移到
  pgvector；支持连接退避重试、断点续传、重复运行快速跳过完整文档和全量完整性核验，
  不会重新调用向量模型。
- `verify_pgvector_retrieval.py`：对指定公司执行一次真实查询向量生成和pgvector文档内召回，
  用于确认数据库、模型维度、文档隔离和证据元数据均可正常使用。
- `register_company_data.py`：把一家公司或全部公司的文档、数据版本登记到
  `data/app.db`；重复运行不会产生重复记录，并会提示未写入资料清单的文件。
- `review_findings.py`：提供 `list`、`verify`、`review`、`export` 四个子命令；
  确定性核验证据和公式，记录人工确认/驳回/编辑，并只导出已核验风险。

后续计划提供：

- 基于批量实验结果学习动态风险语义簇，并完成结构/市场/Agent融合消融。
- 抽取指标自动评测和比赛提交格式。
- `run_demo`：启动API和演示界面。
