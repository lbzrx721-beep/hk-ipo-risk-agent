"""可测试、可复用的确定性计算工具。

每个 skill 模块从 Agent 分析结果（RiskFinding / Calculation）和/或
原始结构化数据中提取数值化特征，输出标准化的 FeatureValue 字典，
供下游 modeling/features.py 消费。

所有函数均为纯函数（无 LLM 调用），可单元测试。
"""

from ipo_risk.skills.cash_burn import extract_cash_burn_features
from ipo_risk.skills.concentration import extract_concentration_features
from ipo_risk.skills.market_features import extract_market_features
from ipo_risk.skills.related_parties import extract_related_party_features
from ipo_risk.skills.valuation import extract_valuation_features

__all__ = [
    "extract_cash_burn_features",
    "extract_concentration_features",
    "extract_market_features",
    "extract_related_party_features",
    "extract_valuation_features",
]
