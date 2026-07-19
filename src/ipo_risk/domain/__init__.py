"""跨模块共享的业务对象与枚举；其他模块只从这里导入契约。"""

from .catalog import CompanyRecord, DatasetRecord, DocumentRecord, VectorIndexRecord
from .enums import (
    AgentRole,
    EvidenceType,
    PredictionHorizon,
    PredictionTarget,
    RiskLevel,
    RiskSeverity,
    ToolCallStatus,
    TraceStatus,
    VerificationStatus,
)
from .models import (
    AgentTrace,
    BoundingBox,
    Calculation,
    Evidence,
    FeatureValue,
    Prediction,
    RiskFinding,
    ToolCall,
)

__all__ = [
    "AgentRole",
    "AgentTrace",
    "BoundingBox",
    "Calculation",
    "CompanyRecord",
    "DatasetRecord",
    "DocumentRecord",
    "Evidence",
    "EvidenceType",
    "FeatureValue",
    "Prediction",
    "PredictionHorizon",
    "PredictionTarget",
    "RiskFinding",
    "RiskLevel",
    "RiskSeverity",
    "ToolCall",
    "ToolCallStatus",
    "TraceStatus",
    "VerificationStatus",
    "VectorIndexRecord",
]
