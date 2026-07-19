# 运行产物

保存证据截图、风险报告、预测表、Agent轨迹、模型文件和评测指标。全部应可由代码重建。

当前第一阶段产物：

```text
artifacts/
├── evidence/<公司目录>/<document_id>/<evidence_id>.png
└── analysis/<公司目录>/<document_id>/risk_analysis.json
```

风险JSON同时包含 Evidence、RiskFinding、AgentTrace 和简体展示文本；原文保持PDF官方语言。
报告生成后，同一证据链还会写入 `data/app.db`，供后续复核界面查询。
