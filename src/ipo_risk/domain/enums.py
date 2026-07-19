"""跨模块共享的有限状态与业务枚举。"""

from enum import IntEnum, StrEnum


class EvidenceType(StrEnum):
    """证据在原始文档中的主要载体。"""

    TEXT = "text"
    TABLE = "table"
    OCR_TEXT = "ocr_text"
    IMAGE = "image"


class RiskSeverity(StrEnum):
    """风险结论的业务严重度。"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VerificationStatus(StrEnum):
    """风险结论在证据核验流程中的状态。"""

    PROPOSED = "proposed"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    REJECTED = "rejected"


class AgentRole(StrEnum):
    """可出现在 AgentTrace 和 Evidence 中的标准角色名。"""

    DOCUMENT_PARSER = "document_parser"
    FINANCE = "finance"
    LEGAL = "legal"
    BUSINESS = "business"
    MARKET = "market"
    VERIFIER = "verifier"
    DECISION = "decision"
    REPORTER = "reporter"
    ORCHESTRATOR = "orchestrator"


class ToolCallStatus(StrEnum):
    """一次确定性工具调用的结果状态。"""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class TraceStatus(StrEnum):
    """一次 Agent 节点执行的最终状态。"""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class PredictionHorizon(IntEnum):
    """赛题要求的上市后交易日预测时间窗。"""

    DAY_1 = 1
    DAY_5 = 5
    DAY_20 = 20
    DAY_60 = 60


class PredictionTarget(StrEnum):
    """预测或回测标签的业务含义。"""

    SIGNIFICANT_DROP = "significant_drop"
    BREAK_ISSUE_PRICE = "break_issue_price"
    NEGATIVE_RETURN = "negative_return"
    MAX_DRAWDOWN = "max_drawdown"


class RiskLevel(StrEnum):
    """预测概率映射后的展示等级。"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
