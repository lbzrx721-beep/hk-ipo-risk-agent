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
- `train_prediction_baseline.py`：按2020—2023/2024/2025时间切分训练招股书结构、市场环境
  和融合三组五日显著下跌基线，输出指标、概率、解释和业务质量门禁。
- `run_prediction_baseline.py`：按顺序完成上述特征构建和三组模型训练；首次扫描大行情后会
  复用缓存。
- `run_agent_feature_experiment.py`：按文档哈希无标签抽样历史招股书，顺序执行向量化、
  动态风险发现、多Agent分析和Agent特征生成；逐公司落盘并支持失败隔离、断点续跑和
  基础/Agent特征矩阵融合。
- `run_unattended_agent_experiment.py`：从指定数据分组选择尚未成功分析且当前未运行的公司，
  自动跳过人工风险范围确认，受控并发生成逐公司报告、Agent特征和统一训练JSON；使用同一
  实验名称可断点续跑。
- `vectorize_competition_with_timeout.py`：批量生成和校验本地 `vectors.npy` 备份，支持
  PDF解析超时隔离、Embedding批次断点、失败重试和断点续跑；持久化导入由pgvector脚本负责。
- `init_pgvector.py`：在PostgreSQL中创建 `vector` 扩展、项目专用schema和1024维招股书
  向量表；全量导入完成后可选择创建HNSW索引。
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
