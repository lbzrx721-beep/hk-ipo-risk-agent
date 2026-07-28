# 测试结构

- `unit/parser`：页码、坐标、OCR触发和截图。
- `unit/reviewing`：证据复现、公式复算、人工复核状态机、Streamlit页面操作、审计记录和
  正式报告门禁。
- `unit/ui` 与 `unit/test_uploads.py`：PDF拖拽页面、公司字段、上传登记、SHA-256去重、
  可见分析流程和完成后进入人工复核。
- `unit/retrieval`：繁简中文BM25、真实向量接口适配、余弦语义检索、RRF融合、索引缓存与
  Embedding批次指数退避重试。
- `unit/agents`：DeepSeek JSON证据约束、未知证据拦截、多查询召回、角色分发、决策白名单、
  严重程度/成立性分歧、完整节点重试与规则降级。
- `unit/skills`：财务与市场计算。
- `unit/modeling`：标签、时间切分和泄漏检查。
- `integration`：从PDF到风险JSON，以及混合检索、专业Agent分析、决策Agent裁决和父子
  AgentTrace的完整链路。
- `fixtures`：公开、脱敏、体积小的测试样本。

当前真实样本集成测试会验证：672页解析、5000条以上切片、至少5条风险、
每条风险的原文证据与截图、基石配售比例的元数据驱动复算，以及最终JSON的跨对象引用
完整性。公开PDF不存在时自动跳过。
