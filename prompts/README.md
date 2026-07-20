# Agent提示词

每个角色单独保存系统提示词和输出契约。提示词不得包含真实密钥、测试标签或上市后信息。

建议每次调整记录版本、修改原因、使用的数据集合及评测变化。

当前真实模型入口：

- `search/role_search_planning_v1.md`：专业Agent根据职责自主生成招股书语义检索问题。
- `decision/evidence_analysis_v1.md`：财务、法务、业务和市场Agent的证据约束分析。
- `decision/decision_adjudication_v1.md`：决策Agent的提案复核、调整、驳回和冲突裁决。

分析任务会把版本分别写入 `agent_search_planning`、`evidence_analysis` 和
`risk_adjudication` 工具调用；修改内容时
必须新建版本文件，不得覆盖旧版本含义。
