# 操作脚本

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
- `register_company_data.py`：把一家公司或全部公司的文档、数据版本登记到
  `data/app.db`；重复运行不会产生重复记录，并会提示未写入资料清单的文件。
- `review_findings.py`：提供 `list`、`verify`、`review`、`export` 四个子命令；
  确定性核验证据和公式，记录人工确认/驳回/编辑，并只导出已核验风险。

后续计划提供：

- 基于批量实验结果学习动态风险语义簇，并完成结构/市场/Agent融合消融。
- 抽取指标自动评测和比赛提交格式。
- `run_demo`：启动API和演示界面。
