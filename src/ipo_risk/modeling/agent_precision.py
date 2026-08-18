"""从既有 Agent 结论中生成少量、精确且可审计的衍生特征。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
import re

import pandas as pd

from ipo_risk.domain import Evidence, FeatureValue, RiskFinding
from ipo_risk.modeling.risk_semantics import classify_risk_finding


AGENT_PRECISION_FEATURE_NAMES = (
    "agent_cash_runway_months",
    "agent_redemption_strictness_score",
    "agent_pipeline_stage_score",
    "agent_related_party_ratio_pct",
)

_PRECISION_ELIGIBLE_STATUSES = {None, "supported", "duplicate_merged"}

_PIPELINE_STAGE_PATTERNS = (
    (
        6.0,
        (
            "已获批",
            "已獲批",
            "获批上市",
            "獲批上市",
            "批准上市",
            "已商业化",
            "已商業化",
            "已实现商业化",
            "已實現商業化",
            "已进入商业化",
            "已進入商業化",
            "commercialized",
            "approved for marketing",
        ),
    ),
    (5.0, ("nda", "bla", "上市申请", "申报上市", "申報上市", "受理")),
    (4.0, ("iii期", "iiia期", "iiib期", "三期", "phase iii", "phase 3")),
    (3.0, ("ii期", "iia期", "iib期", "二期", "phase ii", "phase 2")),
    (2.0, ("i期", "ia期", "ib期", "一期", "phase i", "phase 1")),
    (1.0, ("临床前", "臨床前", "preclinical")),
)

_STAGE_NOT_REACHED_PREFIX = re.compile(
    r"(?:尚未|仍未|未能|未曾|并未|並未|未|计划|計劃|预计|預計|预期|預期|"
    r"目标|目標|拟|擬|有望|能否|是否).{0,14}$",
    re.IGNORECASE,
)
_COMMERCIAL_STAGE_NON_STATUS_SUFFIX = re.compile(
    r"^.{0,10}?(?:公司资格|公司資格|回报|回報|目标|目標|计划|計劃|前景)",
    re.IGNORECASE,
)
_CORE_PIPELINE = re.compile(
    r"(?:核心|主要|主导|領先|领先)(?:候选|候選)?(?:药物|藥物|产品|產品|管线|管線|项目|項目)"
    r"|(?:lead|core|key)\s+(?:candidate|asset|product|pipeline)",
    re.IGNORECASE,
)
_STAGE_INVALID_STATUS = re.compile(
    r"(?:失败|失敗|终止|終止|中止|暂停|暫停|撤回|撤销|撤銷|淘汰|不再推进|不再推進)"
    r"|(?:failed|terminated|discontinued|suspended|withdrawn)",
    re.IGNORECASE,
)

_LISTING_TRIGGER = re.compile(
    r"(?:未能|未于|未於|未在|若.{0,12}?未|如.{0,12}?未|无法).{0,30}?"
    r"(?:上市|合资格上市|合資格上市|首次公开发售|首次公開發售|ipo)",
    re.IGNORECASE,
)
_CLAUSE_DEADLINE = re.compile(
    r"(?:赎回|贖回|回购|回購|上市|ipo).{0,50}?"
    r"20\d{2}\s*[年/.-]\s*\d{1,2}(?:\s*[月/.-]\s*\d{1,2})?"
    r"|20\d{2}\s*[年/.-]\s*\d{1,2}(?:\s*[月/.-]\s*\d{1,2})?"
    r".{0,50}?(?:赎回|贖回|回购|回購|上市|ipo)",
    re.IGNORECASE,
)
_RETURN_RATE = re.compile(
    r"(?:\d+(?:\.\d+)?\s*%\s*(?:年利率|每年|年回报|年回報|单利|單利|复利|複利))"
    r"|(?:(?:年利率|单利|單利|复利|複利|内部回报率|內部回報率|年回报|年回報)"
    r".{0,12}?\d+(?:\.\d+)?\s*%)",
    re.IGNORECASE,
)
_MANDATORY = re.compile(
    r"(?:必须|必須|须|須|有义务|有義務|强制|強制|应当|應當|shall|must).{0,30}?"
    r"(?:赎回|贖回|回购|回購|偿还|償還|支付)",
    re.IGNORECASE,
)
_TERMINATED_RIGHT = re.compile(
    r"(?:已|已经|已經|自动|自動).{0,12}?(?:终止|終止|失效|取消|撤销|撤銷)"
    r"|(?:获|獲).{0,8}?豁免",
    re.IGNORECASE,
)
_RIGHT_REMAINS_ACTIVE = re.compile(
    r"(?:仍然|仍|继续|繼續).{0,10}?(?:有效|生效|适用|適用)"
    r"|(?:恢复|恢復|复效|復效).{0,10}?(?:赎回|贖回|回购|回購|权|權)",
    re.IGNORECASE,
)
_RELATED_PARTY_RATIO_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?:关联|關聯|关连|關連)(?:方)?(?:交易|销售|銷售|收入|服务|服務)?"
        r".{0,40}?(?:占|占比|比例).{0,16}?(?:收入|收益|销售额|銷售額)"
        r".{0,12}?(\d+(?:\.\d+)?)\s*%",
        r"(?:收入|收益|销售额|銷售額).{0,24}?(?:来自|來自|源自|归因于|歸因於)"
        r".{0,20}?(?:关联|關聯|关连|關連)(?:方)?.{0,12}?(\d+(?:\.\d+)?)\s*%",
        r"(\d+(?:\.\d+)?)\s*%.{0,20}?(?:收入|收益|销售|銷售)"
        r".{0,20}?(?:来自|來自|源自).{0,20}?(?:关联|關聯|关连|關連)(?:方)?",
    )
)


def build_precision_agent_feature_snapshot(
    findings: Sequence[RiskFinding],
    evidences: Sequence[Evidence],
    *,
    available_at: date,
    source_ref: str,
    redemption_assessed: bool = False,
) -> dict[str, FeatureValue]:
    """只读取现有风险、证据和计算对象，不调用模型或重新解析 PDF。"""
    evidence_by_id = {evidence.evidence_id: evidence for evidence in evidences}
    by_subtype: dict[str, list[RiskFinding]] = {}
    for finding in findings:
        if finding.specialist_status not in _PRECISION_ELIGIBLE_STATUSES:
            continue
        classification = classify_risk_finding(
            finding,
            evidence_by_id=evidence_by_id,
        )
        subtypes = {
            classification.primary_subtype,
            *classification.secondary_subtypes,
        }
        for subtype in subtypes:
            by_subtype.setdefault(subtype, []).append(finding)

    features: dict[str, FeatureValue] = {}

    redemption_findings = _unique_findings(by_subtype.get("redemption_vam", ()))
    redemption_result = _redemption_strictness_score(
        redemption_findings,
        evidence_by_id=evidence_by_id,
    )
    redemption_score = redemption_result[1] if redemption_result is not None else 0
    if redemption_score > 0 or redemption_assessed:
        features["agent_redemption_strictness_score"] = FeatureValue(
            value=redemption_score,
            available_at=available_at,
            source_ref=(
                f"finding:{redemption_result[0].finding_id}:redemption_strictness"
                if redemption_result is not None
                else f"{source_ref}:redemption_not_detected_after_complete_scan"
            ),
        )

    runway = _cash_runway_months(
        _unique_findings(
            (*by_subtype.get("cash_burn_liquidity", ()), *by_subtype.get("persistent_loss", ()))
        )
    )
    if runway is not None:
        finding, value = runway
        features["agent_cash_runway_months"] = FeatureValue(
            value=value,
            available_at=available_at,
            source_ref=f"finding:{finding.finding_id}:calculation:cash_runway",
        )

    pipeline_findings = _unique_findings(by_subtype.get("commercialization_pipeline", ()))
    pipeline_result = _pipeline_stage_score(
        pipeline_findings,
        evidence_by_id=evidence_by_id,
    )
    if pipeline_result is not None:
        pipeline_finding, pipeline_score = pipeline_result
        features["agent_pipeline_stage_score"] = FeatureValue(
            value=pipeline_score,
            available_at=available_at,
            source_ref=f"finding:{pipeline_finding.finding_id}:pipeline_stage",
        )

    related_findings = _unique_findings(by_subtype.get("related_party_transaction", ()))
    related_ratio = _related_party_ratio_pct(
        related_findings,
        evidence_by_id=evidence_by_id,
    )
    if related_ratio is not None:
        finding, value = related_ratio
        features["agent_related_party_ratio_pct"] = FeatureValue(
            value=value,
            available_at=available_at,
            source_ref=f"finding:{finding.finding_id}:related_party_ratio",
        )
    return features


def reconcile_precision_agent_features(frame: pd.DataFrame) -> pd.DataFrame:
    """统一精细 Agent 字段的数值类型与有效范围，不使用 RAG 值回填。"""
    result = frame.copy()

    def numeric(name: str) -> pd.Series:
        if name not in result:
            return pd.Series(float("nan"), index=result.index, dtype=float)
        return pd.to_numeric(result[name], errors="coerce")

    result["agent_cash_runway_months"] = numeric("agent_cash_runway_months")
    result["agent_pipeline_stage_score"] = numeric("agent_pipeline_stage_score")
    result["agent_redemption_strictness_score"] = numeric("agent_redemption_strictness_score")
    result["agent_related_party_ratio_pct"] = numeric("agent_related_party_ratio_pct")

    ranges = {
        "agent_cash_runway_months": (0.0, 1_200.0),
        "agent_redemption_strictness_score": (0.0, 3.0),
        "agent_pipeline_stage_score": (1.0, 6.0),
        "agent_related_party_ratio_pct": (0.0, 100.0),
    }
    for name, (lower, upper) in ranges.items():
        values = pd.to_numeric(result[name], errors="coerce")
        result[name] = values.where(values.between(lower, upper))
    return result


def _finding_text(
    finding: RiskFinding,
    evidence_by_id: Mapping[str, Evidence],
) -> str:
    parts = [finding.title, finding.claim, finding.uncertainty or ""]
    parts.extend(
        evidence_by_id[evidence_id].quote
        for evidence_id in finding.evidence_ids
        if evidence_id in evidence_by_id
    )
    return " ".join(parts).casefold()


def _redemption_strictness_score(
    findings: Sequence[RiskFinding],
    *,
    evidence_by_id: Mapping[str, Evidence],
) -> tuple[RiskFinding, int] | None:
    if not findings:
        return None
    candidates: list[tuple[int, float, RiskFinding]] = []
    for finding in findings:
        text = _finding_text(finding, evidence_by_id)
        if _TERMINATED_RIGHT.search(text) and not _RIGHT_REMAINS_ACTIVE.search(text):
            continue
        score = 1
        trigger_or_deadline = bool(_LISTING_TRIGGER.search(text) or _CLAUSE_DEADLINE.search(text))
        return_or_mandatory = bool(_RETURN_RATE.search(text) or _MANDATORY.search(text))
        if trigger_or_deadline or return_or_mandatory:
            score = 2
        if trigger_or_deadline and return_or_mandatory:
            score = 3
        candidates.append((score, finding.confidence, finding))
    if not candidates:
        return None
    score, _, finding = max(candidates, key=lambda item: (item[0], item[1]))
    return finding, score


def _cash_runway_months(
    findings: Sequence[RiskFinding],
) -> tuple[RiskFinding, float] | None:
    candidates: list[tuple[tuple[int, int, int], float, RiskFinding, float]] = []
    for finding in findings:
        calculation = finding.calculation
        if calculation is None:
            continue
        value = _safe_float(calculation.result)
        if value is None or value < 0:
            continue
        calculation_text = " ".join(
            (
                calculation.tool,
                calculation.formula,
                *map(str, calculation.inputs.keys()),
            )
        ).casefold()
        has_cash_balance = any(
            marker in calculation_text
            for marker in (
                "cash",
                "现金",
                "現金",
                "银行结余",
                "銀行結餘",
                "cash_equivalent",
            )
        )
        has_cash_burn = any(
            marker in calculation_text
            for marker in (
                "burn",
                "outflow",
                "现金消耗",
                "現金消耗",
                "净流出",
                "淨流出",
                "operating_cash_flow",
                "经营现金流",
                "經營現金流",
            )
        )
        has_division = any(
            marker in calculation.formula.casefold() for marker in ("/", "÷", "divid")
        )
        if not (has_cash_balance and has_cash_burn and has_division):
            continue
        unit = calculation.unit.casefold()
        if any(marker in unit for marker in ("month", "个月", "個月", "月")):
            months = value
        elif any(marker in unit for marker in ("year", "年")):
            months = value * 12
        else:
            continue
        if months <= 1_200:
            candidates.append(
                (
                    _period_sort_key(calculation.period),
                    finding.confidence,
                    finding,
                    round(months, 6),
                )
            )
    if not candidates:
        return None
    _, _, finding, value = max(candidates, key=lambda item: (item[0], item[1]))
    return finding, value


def _pipeline_stage_score(
    findings: Sequence[RiskFinding],
    *,
    evidence_by_id: Mapping[str, Evidence],
) -> tuple[RiskFinding, float] | None:
    candidates: list[tuple[float, float, RiskFinding]] = []
    for finding in findings:
        text = _finding_text(finding, evidence_by_id)
        if not _CORE_PIPELINE.search(text):
            continue
        for score, markers in _PIPELINE_STAGE_PATTERNS:
            if any(_contains_achieved_stage(text, marker, score=score) for marker in markers):
                candidates.append((score, finding.confidence, finding))
                break
    if not candidates:
        return None
    score, _, finding = max(candidates, key=lambda item: (item[0], item[1]))
    return finding, score


def _contains_achieved_stage(text: str, marker: str, *, score: float) -> bool:
    """只接受已经达到的阶段，排除“尚未、计划、预计”等前瞻语境。"""
    expression = (
        re.compile(rf"\b{re.escape(marker)}\b", re.IGNORECASE)
        if marker.isascii()
        else re.compile(re.escape(marker), re.IGNORECASE)
    )
    for match in expression.finditer(text):
        prefix = text[max(0, match.start() - 18) : match.start()]
        prefix = re.split(r"[，。；;,.]", prefix)[-1]
        suffix = text[match.end() : match.end() + 14]
        clause = text[max(0, match.start() - 40) : match.end() + 40]
        if _STAGE_NOT_REACHED_PREFIX.search(prefix):
            continue
        if _STAGE_INVALID_STATUS.search(clause):
            continue
        if score == 6.0 and _COMMERCIAL_STAGE_NON_STATUS_SUFFIX.search(suffix):
            continue
        return True
    return False


def _related_party_ratio_pct(
    findings: Sequence[RiskFinding],
    *,
    evidence_by_id: Mapping[str, Evidence],
) -> tuple[RiskFinding, float] | None:
    candidates: list[tuple[tuple[int, int, int], float, RiskFinding, float]] = []
    for finding in findings:
        calculation = finding.calculation
        if calculation is None:
            continue
        value = _safe_float(calculation.result)
        if value is None:
            continue
        unit = calculation.unit.casefold()
        formula = calculation.formula.casefold().replace(" ", "")
        ratio_context = " ".join((finding.title, finding.claim, formula)).casefold()
        has_related_numerator = any(
            marker in ratio_context for marker in ("关联", "關聯", "关连", "關連", "relatedparty")
        )
        has_revenue_denominator = any(
            marker in ratio_context
            for marker in (
                "收入",
                "收益",
                "销售",
                "銷售",
                "revenue",
                "sales",
            )
        )
        if not (has_related_numerator and has_revenue_denominator):
            continue
        if any(marker in unit for marker in ("percent", "%", "百分比")):
            ratio = value
        elif any(marker in unit for marker in ("ratio", "比例")) and 0 <= value <= 1:
            ratio = value * 100
        elif "/" in formula and ("*100" in formula or "×100" in formula):
            ratio = value
        else:
            continue
        if 0 <= ratio <= 100:
            candidates.append(
                (
                    _period_sort_key(calculation.period),
                    finding.confidence,
                    finding,
                    round(ratio, 6),
                )
            )
            continue

        # 不猜测金额占比；仅接受同一条关联交易结论/证据中唯一明确披露的百分比。
    for finding in findings:
        text = _finding_text(finding, evidence_by_id)
        if not any(
            marker in text
            for marker in (
                "关联交易",
                "关联方",
                "关联采购",
                "关联销售",
                "关连交易",
                "關連交易",
                "關聯方",
                "related party",
            )
        ):
            continue
        percentages = {
            round(float(match.group(1)), 6)
            for pattern in _RELATED_PARTY_RATIO_PATTERNS
            for match in pattern.finditer(text)
            if 0 <= float(match.group(1)) <= 100
        }
        if len(percentages) == 1:
            candidates.append(((0, 0, 0), finding.confidence, finding, percentages.pop()))
    if not candidates:
        return None
    _, _, finding, value = max(candidates, key=lambda item: (item[0], item[1]))
    return finding, value


def _period_sort_key(period: str) -> tuple[int, int, int]:
    """从常见报告期文本中提取可比较日期；无法识别时排在最旧。"""
    match = re.search(
        r"((?:19|20)\d{2})(?:\D{0,4}(\d{1,2}))?(?:\D{0,4}(\d{1,2}))?",
        period,
    )
    if match is None:
        return (0, 0, 0)
    year = int(match.group(1))
    month = int(match.group(2) or 0)
    day = int(match.group(3) or 0)
    if not 0 <= month <= 12 or not 0 <= day <= 31:
        return (year, 0, 0)
    return (year, month, day)


def _unique_findings(findings: Sequence[RiskFinding]) -> list[RiskFinding]:
    result: dict[str, RiskFinding] = {}
    for finding in findings:
        result.setdefault(finding.finding_id, finding)
    return list(result.values())


def _safe_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        if result == result and result not in (float("inf"), float("-inf")):
            return result
    return None


__all__ = [
    "AGENT_PRECISION_FEATURE_NAMES",
    "build_precision_agent_feature_snapshot",
    "reconcile_precision_agent_features",
]
