"""无未来泄漏的历史IPO案例向量库与混合检索。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
import math
import re
from threading import RLock
from typing import Any

import numpy as np
import pandas as pd

from ipo_risk.modeling.industry import NormalizedIndustry, normalize_industry
from ipo_risk.retrieval.semantic import EmbeddingProvider


CASE_FEATURE_SCHEMA_VERSION = "historical-ipo-case-v1"

# 这些字段全部在招股书日可获得；target_*永远不得进入此清单。
CASE_NUMERIC_FEATURES: tuple[str, ...] = (
    "offering_offer_price_mid_hkd",
    "offering_gross_proceeds_mid_hkd",
    "offering_net_proceeds_hkd",
    "offering_market_cap_mid_hkd",
    "offering_share_ratio_pct",
    "cornerstone_ratio_pct",
    "cornerstone_lockup_months",
    "controlling_shareholder_lockup_months",
    "fundamental_revenue_latest_hkd_mn",
    "fundamental_revenue_cagr_3y_pct",
    "fundamental_gross_margin_latest_pct",
    "fundamental_operating_cash_flow_latest_hkd_mn",
    "fundamental_operating_cash_flow_to_revenue_latest_pct",
    "fundamental_net_profit_latest_hkd_mn",
    "fundamental_ending_cash_latest_hkd_mn",
    "fundamental_research_development_latest_hkd_mn",
    "customer_top5_ratio_pct",
    "supplier_top5_ratio_pct",
    "pipeline_stage_score",
    "market_return_5d_pct",
    "market_return_20d_pct",
    "market_return_60d_pct",
    "market_volatility_20d_pct",
    "recent_ipo_5d_drop_rate_180d",
    "recent_ipo_5d_mean_return_180d_pct",
    "recent_ipo_count_30d",
    "agent_finding_count",
    "agent_evidence_confidence_mean",
    "agent_max_severity_score",
    "agent_severity_critical_count",
    "agent_severity_high_count",
    "agent_severity_medium_count",
    "agent_semantic_supported_finding_share",
    "agent_cash_runway_months",
    "agent_pipeline_stage_score",
    "agent_redemption_strictness_score",
    "agent_related_party_ratio_pct",
)

_AGENT_RISK_FLAGS: tuple[tuple[str, str], ...] = (
    ("agent_has_risk_cash_burn", "现金消耗与流动性"),
    ("agent_has_risk_persistent_loss", "持续亏损"),
    ("agent_has_risk_customer_concentration", "客户集中"),
    ("agent_has_risk_supplier_concentration", "供应商集中"),
    ("agent_has_risk_redemption_or_vam", "对赌或赎回"),
    ("agent_has_risk_related_party_transaction", "关联交易"),
    ("agent_has_risk_litigation_or_penalty", "诉讼与处罚"),
    ("agent_has_risk_license_dependency", "牌照依赖"),
    ("agent_has_risk_single_product_dependency", "单一产品依赖"),
    ("agent_has_risk_commercialization_uncertainty", "商业化不确定"),
    ("agent_has_risk_pipeline_uncertainty", "研发管线不确定"),
    ("agent_has_risk_issuance_structure", "发行结构与流动性"),
)

_RETURN_5D_ALIASES = (
    "target_return_vs_offer_pct_5d",
    "target_return_vs_offer_5d_pct",
)
_OBSERVATION_5D_ALIASES = (
    "target_observation_date_5d",
    "target_observation_date",
)
_SAFE_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


@dataclass(frozen=True)
class HistoricalCaseIndexRecord:
    case_id: str
    company_id: str
    company_name: str
    prospectus_date: date
    outcome_available_at: date
    industry: NormalizedIndustry
    feature_payload: dict[str, float]
    case_summary_text: str
    significant_drop_5d: bool
    return_vs_offer_5d_pct: float | None


@dataclass(frozen=True)
class HistoricalCaseMatch:
    record: HistoricalCaseIndexRecord
    semantic_similarity: float


def build_case_index_records(frame: pd.DataFrame) -> tuple[HistoricalCaseIndexRecord, ...]:
    """从训练宽表构建一家公司一条的案例记录。"""
    required = {"company_id", "prospectus_date", "target_significant_drop_5d"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"feature matrix misses required columns: {sorted(missing)}")
    observation_name = _first_existing(frame.columns, _OBSERVATION_5D_ALIASES)
    if observation_name is None:
        raise ValueError("feature matrix misses 5-day outcome observation date")
    return_name = _first_existing(frame.columns, _RETURN_5D_ALIASES)
    working = frame.copy()
    working["_prospectus_date"] = pd.to_datetime(working["prospectus_date"], errors="coerce")
    working["_outcome_available_at"] = pd.to_datetime(working[observation_name], errors="coerce")
    working["_target"] = pd.to_numeric(working["target_significant_drop_5d"], errors="coerce")
    working["_company_id"] = working["company_id"].astype(str).str.zfill(5)
    working = working.loc[
        working["_prospectus_date"].notna()
        & working["_outcome_available_at"].notna()
        & working["_target"].isin([0, 1])
    ].sort_values(["_company_id", "_prospectus_date"])
    working = working.drop_duplicates("_company_id", keep="last")
    records: list[HistoricalCaseIndexRecord] = []
    for _, row in working.iterrows():
        industry = normalize_industry(row.get("industry_primary"))
        prospectus_date = row["_prospectus_date"].date()
        company_id = row["_company_id"]
        payload = case_feature_payload(row)
        summary = build_case_summary(row, industry=industry, feature_payload=payload)
        _assert_summary_is_prelisting_only(summary)
        records.append(
            HistoricalCaseIndexRecord(
                case_id=f"{company_id}-{prospectus_date:%Y%m%d}",
                company_id=company_id,
                company_name=_text(row.get("company_name")) or company_id,
                prospectus_date=prospectus_date,
                outcome_available_at=row["_outcome_available_at"].date(),
                industry=industry,
                feature_payload=payload,
                case_summary_text=summary,
                significant_drop_5d=bool(int(row["_target"])),
                return_vs_offer_5d_pct=(
                    _number(row.get(return_name)) if return_name is not None else None
                ),
            )
        )
    return tuple(records)


def case_feature_payload(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: value
        for name in CASE_NUMERIC_FEATURES
        if (value := _number(row.get(name))) is not None
    }


def build_case_summary(
    row: Mapping[str, Any],
    *,
    industry: NormalizedIndustry | None = None,
    feature_payload: Mapping[str, float] | None = None,
) -> str:
    """生成只含上市前信息的语义摘要，不放公司身份和真实涨跌结果。"""
    normalized = industry or normalize_industry(row.get("industry_primary"))
    features = dict(feature_payload or case_feature_payload(row))
    display_features = dict(features)
    for name in (
        "offering_gross_proceeds_mid_hkd",
        "offering_net_proceeds_hkd",
        "offering_market_cap_mid_hkd",
    ):
        if name in display_features:
            display_features[name] = display_features[name] / 1_000_000.0
    sections = [f"行业：{normalized.level_1}/{normalized.level_2}。"]
    _append_values(
        sections,
        "发行",
        display_features,
        (
            ("offering_offer_price_mid_hkd", "发行价中位数", "港元/股"),
            ("offering_gross_proceeds_mid_hkd", "募资总额", "百万港元"),
            ("offering_net_proceeds_hkd", "净募资额", "百万港元"),
            ("offering_market_cap_mid_hkd", "预计市值", "百万港元"),
            ("offering_share_ratio_pct", "发行比例", "%"),
            ("cornerstone_ratio_pct", "基石比例", "%"),
        ),
    )
    _append_values(
        sections,
        "基本面",
        features,
        (
            ("fundamental_revenue_latest_hkd_mn", "最新收入", "百万港元"),
            ("fundamental_revenue_cagr_3y_pct", "三年营收CAGR", "%"),
            ("fundamental_gross_margin_latest_pct", "毛利率", "%"),
            ("fundamental_operating_cash_flow_latest_hkd_mn", "经营现金流", "百万港元"),
            ("fundamental_net_profit_latest_hkd_mn", "净利润", "百万港元"),
            ("fundamental_ending_cash_latest_hkd_mn", "期末现金", "百万港元"),
            ("customer_top5_ratio_pct", "前五大客户占比", "%"),
            ("supplier_top5_ratio_pct", "前五大供应商占比", "%"),
        ),
    )
    _append_values(
        sections,
        "上市前市场",
        features,
        (
            ("market_return_20d_pct", "市场20日收益", "%"),
            ("market_volatility_20d_pct", "市场20日波动率", "%"),
            ("recent_ipo_5d_drop_rate_180d", "近期IPO五日显著下跌率", ""),
            ("recent_ipo_5d_mean_return_180d_pct", "近期IPO五日平均收益", "%"),
            ("recent_ipo_count_30d", "近30日IPO数量", "家"),
        ),
    )
    risk_names = [label for field, label in _AGENT_RISK_FLAGS if _truthy(row.get(field))]
    agent_values: list[str] = []
    for field, label, unit in (
        ("agent_finding_count", "风险数量", "条"),
        ("agent_severity_high_count", "高风险", "条"),
        ("agent_cash_runway_months", "资金续航", "个月"),
        ("agent_redemption_strictness_score", "赎回严苛度", "/3"),
        ("agent_pipeline_stage_score", "管线阶段", "/5"),
    ):
        if field in features:
            agent_values.append(f"{label}{_format_number(features[field])}{unit}")
    if risk_names:
        agent_values.append("主要风险=" + "、".join(risk_names))
    if agent_values:
        sections.append("Agent风险：" + "；".join(agent_values) + "。")
    return "".join(sections)


class PgVectorHistoricalCaseStore:
    """历史案例专用pgvector表；真实结果仅作为返回元数据。"""

    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        dsn: str,
        schema: str = "ipo_risk",
        connect_timeout_seconds: float = 10.0,
        connection: Any | None = None,
    ) -> None:
        if provider.dimensions != 1024:
            raise ValueError("历史案例向量库固定使用1024维向量")
        if not _SAFE_IDENTIFIER.fullmatch(schema):
            raise ValueError("postgres_schema must be a lowercase SQL identifier")
        try:
            import psycopg
            from pgvector.psycopg import register_vector
            from psycopg import sql
            from psycopg.types.json import Jsonb
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("pgvector后端未安装") from error
        self._provider = provider
        self._sql = sql
        self._jsonb = Jsonb
        self._schema = schema
        self._connection = connection or psycopg.connect(
            dsn,
            connect_timeout=max(1, int(connect_timeout_seconds)),
            application_name="hk-ipo-historical-cases",
        )
        register_vector(self._connection)
        self._connection.autocommit = True
        self._owns_connection = connection is None
        self._lock = RLock()

    def _table(self) -> Any:
        return self._sql.SQL("{}.{}").format(
            self._sql.Identifier(self._schema),
            self._sql.Identifier("historical_ipo_cases"),
        )

    def close(self) -> None:
        if self._owns_connection and not self._connection.closed:
            self._connection.close()

    @property
    def count(self) -> int:
        statement = self._sql.SQL(
            "SELECT count(*) FROM {} WHERE embedding_model = %s AND feature_schema_version = %s"
        ).format(self._table())
        with self._lock, self._connection.cursor() as cursor:
            cursor.execute(statement, (self._provider.model_name, CASE_FEATURE_SCHEMA_VERSION))
            row = cursor.fetchone()
            return int(row[0]) if row is not None else 0

    def ensure_schema(self) -> None:
        statement = self._sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {} (
                case_id text NOT NULL,
                embedding_model text NOT NULL,
                embedding_dimensions integer NOT NULL CHECK (embedding_dimensions = 1024),
                feature_schema_version text NOT NULL,
                company_id text NOT NULL,
                company_name text NOT NULL,
                prospectus_date date NOT NULL,
                outcome_available_at date NOT NULL,
                industry_raw text,
                industry_level_1 text NOT NULL,
                industry_level_2 text NOT NULL,
                feature_payload jsonb NOT NULL,
                case_summary_text text NOT NULL,
                significant_drop_5d boolean NOT NULL,
                return_vs_offer_5d_pct double precision,
                embedding vector(1024) NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now(),
                updated_at timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (case_id, embedding_model)
            )
            """
        ).format(self._table())
        indexes = (
            self._sql.SQL(
                "CREATE INDEX IF NOT EXISTS historical_cases_date_idx ON {} (outcome_available_at, prospectus_date)"
            ).format(self._table()),
            self._sql.SQL(
                "CREATE INDEX IF NOT EXISTS historical_cases_industry_idx ON {} (industry_level_1, industry_level_2)"
            ).format(self._table()),
            self._sql.SQL(
                "CREATE INDEX IF NOT EXISTS historical_cases_embedding_hnsw_idx ON {} USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 100)"
            ).format(self._table()),
        )
        with self._lock, self._connection.cursor() as cursor:
            cursor.execute(statement)
            for index in indexes:
                cursor.execute(index)

    def upsert(self, records: Sequence[HistoricalCaseIndexRecord]) -> int:
        if not records:
            return 0
        summaries = [record.case_summary_text for record in records]
        vectors = np.asarray(self._provider.embed(summaries), dtype=np.float32)
        if vectors.shape != (len(records), self._provider.dimensions):
            raise ValueError("historical case vectors have unexpected dimensions")
        statement = self._sql.SQL(
            """
            INSERT INTO {} (
                case_id, embedding_model, embedding_dimensions, feature_schema_version,
                company_id, company_name, prospectus_date, outcome_available_at,
                industry_raw, industry_level_1, industry_level_2, feature_payload,
                case_summary_text, significant_drop_5d, return_vs_offer_5d_pct, embedding
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (case_id, embedding_model) DO UPDATE SET
                embedding_dimensions = EXCLUDED.embedding_dimensions,
                feature_schema_version = EXCLUDED.feature_schema_version,
                company_id = EXCLUDED.company_id,
                company_name = EXCLUDED.company_name,
                prospectus_date = EXCLUDED.prospectus_date,
                outcome_available_at = EXCLUDED.outcome_available_at,
                industry_raw = EXCLUDED.industry_raw,
                industry_level_1 = EXCLUDED.industry_level_1,
                industry_level_2 = EXCLUDED.industry_level_2,
                feature_payload = EXCLUDED.feature_payload,
                case_summary_text = EXCLUDED.case_summary_text,
                significant_drop_5d = EXCLUDED.significant_drop_5d,
                return_vs_offer_5d_pct = EXCLUDED.return_vs_offer_5d_pct,
                embedding = EXCLUDED.embedding,
                updated_at = now()
            """
        ).format(self._table())
        rows = []
        for index, record in enumerate(records):
            rows.append(
                (
                    record.case_id,
                    self._provider.model_name,
                    self._provider.dimensions,
                    CASE_FEATURE_SCHEMA_VERSION,
                    record.company_id,
                    record.company_name,
                    record.prospectus_date,
                    record.outcome_available_at,
                    record.industry.raw,
                    record.industry.level_1,
                    record.industry.level_2,
                    self._jsonb(record.feature_payload),
                    record.case_summary_text,
                    record.significant_drop_5d,
                    record.return_vs_offer_5d_pct,
                    vectors[index],
                )
            )
        with self._lock, self._connection.transaction():
            with self._connection.cursor() as cursor:
                cursor.executemany(statement, rows)
        return len(records)

    def search_semantic(
        self,
        *,
        case_summary_text: str,
        current_company_id: str,
        current_prospectus_date: date,
        pool_size: int = 200,
    ) -> tuple[HistoricalCaseMatch, ...]:
        """只按无标签摘要生成查询向量，再返回历史结果元数据。"""
        if pool_size < 1:
            raise ValueError("pool_size must be positive")
        _assert_summary_is_prelisting_only(case_summary_text)
        vectors = self._provider.embed([case_summary_text])
        vector = np.asarray(vectors[0], dtype=np.float32)
        if vector.shape != (self._provider.dimensions,):
            raise ValueError("query vector has unexpected dimensions")
        statement = self._sql.SQL(
            """
            WITH query_vector AS (SELECT %s::vector AS value)
            SELECT case_id, company_id, company_name, prospectus_date,
                   outcome_available_at, industry_raw, industry_level_1,
                   industry_level_2, feature_payload, case_summary_text,
                   significant_drop_5d, return_vs_offer_5d_pct,
                   1 - (embedding <=> q.value) AS semantic_similarity
            FROM {} CROSS JOIN query_vector q
            WHERE embedding_model = %s
              AND feature_schema_version = %s
              AND company_id <> %s
              AND prospectus_date < %s
              AND outcome_available_at < %s
            ORDER BY embedding <=> q.value, prospectus_date DESC
            LIMIT %s
            """
        ).format(self._table())
        with self._lock, self._connection.transaction():
            with self._connection.cursor() as cursor:
                cursor.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
                cursor.execute("SET LOCAL hnsw.ef_search = 160")
                cursor.execute(
                    statement,
                    (
                        vector,
                        self._provider.model_name,
                        CASE_FEATURE_SCHEMA_VERSION,
                        str(current_company_id).zfill(5),
                        current_prospectus_date,
                        current_prospectus_date,
                        pool_size,
                    ),
                )
                rows = cursor.fetchall()
        matches: list[HistoricalCaseMatch] = []
        for row in rows:
            record = HistoricalCaseIndexRecord(
                case_id=row[0],
                company_id=row[1],
                company_name=row[2],
                prospectus_date=row[3],
                outcome_available_at=row[4],
                industry=NormalizedIndustry(raw=row[5], level_1=row[6], level_2=row[7]),
                feature_payload={key: float(value) for key, value in dict(row[8]).items()},
                case_summary_text=row[9],
                significant_drop_5d=bool(row[10]),
                return_vs_offer_5d_pct=_number(row[11]),
            )
            matches.append(
                HistoricalCaseMatch(
                    record=record,
                    semantic_similarity=float(np.clip(float(row[12]), 0.0, 1.0)),
                )
            )
        return tuple(matches)


def robust_numeric_similarity(
    current: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    scales: Mapping[str, float],
) -> float:
    shared = sorted(set(current) & set(candidate) & set(scales))
    if not shared:
        return 0.0
    squared = []
    for name in shared:
        scale = max(abs(float(scales[name])), 1e-9)
        delta = float(np.clip((current[name] - candidate[name]) / scale, -8.0, 8.0))
        squared.append(delta * delta)
    distance = math.sqrt(sum(squared) / len(squared))
    coverage = len(shared) / max(len(set(current) | set(candidate)), 1)
    return float(np.clip((1.0 / (1.0 + distance)) * (0.75 + 0.25 * coverage), 0, 1))


def robust_feature_scales(frame: pd.DataFrame, *, before: date) -> dict[str, float]:
    dates = pd.to_datetime(frame["prospectus_date"], errors="coerce")
    history = frame.loc[dates < pd.Timestamp(before)]
    scales: dict[str, float] = {}
    for name in CASE_NUMERIC_FEATURES:
        if name not in history:
            continue
        values = pd.to_numeric(history[name], errors="coerce")
        scale = values.quantile(0.75) - values.quantile(0.25)
        if not math.isfinite(float(scale)) or abs(float(scale)) < 1e-9:
            scale = values.std()
        if math.isfinite(float(scale)) and abs(float(scale)) >= 1e-9:
            scales[name] = float(scale)
    return scales


def industry_similarity(left: NormalizedIndustry, right: NormalizedIndustry) -> float:
    if left.level_2 not in {"其他", "未分类"} and left.level_2 == right.level_2:
        return 1.0
    if left.level_1 == right.level_1:
        return 0.55
    return 0.0


def _append_values(
    sections: list[str],
    title: str,
    features: Mapping[str, float],
    fields: Sequence[tuple[str, str, str]],
) -> None:
    values = [
        f"{label}{_format_number(features[field])}{unit}"
        for field, label, unit in fields
        if field in features
    ]
    if values:
        sections.append(title + "：" + "；".join(values) + "。")


def _format_number(value: float) -> str:
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none", "null"} else None


def _truthy(value: Any) -> bool:
    number = _number(value)
    if number is not None:
        return number > 0
    return str(value).strip().lower() in {"true", "yes", "是", "有"}


def _first_existing(columns: Sequence[str], names: Sequence[str]) -> str | None:
    return next((name for name in names if name in columns), None)


def _assert_summary_is_prelisting_only(text: str) -> None:
    lowered = text.lower()
    forbidden = (
        "target_",
        "上市后5日",
        "五日真实",
        "显著下跌标签",
        "return_vs_offer",
    )
    if any(token in lowered for token in forbidden):
        raise ValueError("post-listing outcome leaked into historical case embedding text")


__all__ = [
    "CASE_FEATURE_SCHEMA_VERSION",
    "CASE_NUMERIC_FEATURES",
    "HistoricalCaseIndexRecord",
    "HistoricalCaseMatch",
    "PgVectorHistoricalCaseStore",
    "build_case_index_records",
    "build_case_summary",
    "case_feature_payload",
    "industry_similarity",
    "robust_feature_scales",
    "robust_numeric_similarity",
]
