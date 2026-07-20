# 操作脚本

当前已经提供：

- `analyze_prospectus.py`：输入标准公司目录，生成页面、切片、证据截图和风险JSON。
- `register_company_data.py`：把一家公司或全部公司的文档、数据版本登记到
  `data/app.db`；重复运行不会产生重复记录，并会提示未写入资料清单的文件。
- `review_findings.py`：提供 `list`、`verify`、`review`、`export` 四个子命令；
  确定性核验证据和公式，记录人工确认/驳回/编辑，并只导出已核验风险。

后续计划提供：

- `check_data`：检查官方数据字段和文件完整性。
- `build_labels`：生成1/5/20/60日标签。
- `train_model`：按时间切分训练模型。
- `evaluate`：生成抽取与预测指标。
- `run_demo`：启动API和演示界面。
