"""从既有招股书解析缓存提取统一发行与财务特征。

本模块不调用大模型，也不重新解析或向量化 PDF。文本切片负责提供发行条款，
结构化表格负责提供历史财务数值；所有派生指标均由确定性公式计算并保留来源。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd

from ipo_risk.domain import FeatureValue
from ipo_risk.storage.files import write_json, write_jsonl


STRUCTURED_FEATURE_NAMES = (
    "offering_offer_price_low_hkd",
    "offering_offer_price_high_hkd",
    "offering_offer_price_mid_hkd",
    "offering_share_count",
    "offering_gross_proceeds_low_hkd",
    "offering_gross_proceeds_high_hkd",
    "offering_gross_proceeds_mid_hkd",
    "offering_net_proceeds_hkd",
    "offering_market_cap_low_hkd",
    "offering_market_cap_high_hkd",
    "offering_market_cap_mid_hkd",
    "offering_share_ratio_pct",
    "offering_terms_available",
    "offering_net_proceeds_available",
    "offering_market_cap_available",
    "fundamental_reporting_currency",
    "fundamental_revenue_latest_native",
    "fundamental_revenue_growth_latest_pct",
    "fundamental_revenue_cagr_3y_pct",
    "fundamental_revenue_year_count",
    "fundamental_revenue_available",
    "fundamental_gross_margin_latest_pct",
    "fundamental_gross_margin_avg_3y_pct",
    "fundamental_gross_margin_change_3y_pct_points",
    "fundamental_gross_margin_available",
    "fundamental_operating_cash_flow_latest_native",
    "fundamental_operating_cash_flow_sum_3y_native",
    "fundamental_operating_cash_flow_to_revenue_latest_pct",
    "fundamental_negative_operating_cash_flow_years_3y",
    "fundamental_operating_cash_flow_available",
)

_INCOME_KEYWORDS = (
    "收入",
    "收益",
    "營業收入",
    "营业收入",
    "毛利",
    "gross profit",
    "revenue",
)
_CASH_FLOW_KEYWORDS = (
    "經營活動",
    "经营活动",
    "經營現金流",
    "经营现金流",
    "operating activities",
)
_OFFERING_KEYWORDS = (
    "發售股份數目",
    "发售股份数目",
    "發售價範圍",
    "发售价范围",
    "上市市值",
    "股份市值",
    "發售完成後的市值",
    "发售完成后的市值",
    "市值",
    "market capitalisation",
    "發售股份將相當於",
    "发售股份将相当于",
    "發售股份將佔",
    "发售股份将占",
    "募集資金淨額",
    "募集资金净额",
    "所得款項淨額",
    "所得款项净额",
)


@dataclass(frozen=True)
class StructuredFundamentalDatasetResult:
    """结构化特征产物。"""

    feature_matrix_path: Path
    audit_path: Path
    summary_path: Path
    company_count: int
    feature_columns: tuple[str, ...]


@dataclass(frozen=True)
class StructuredFundamentalSnapshot:
    """单份招股书的结构化发行与财务候选特征。"""

    features: dict[str, FeatureValue]
    audit: dict[str, Any]
    missing_reasons: tuple[str, ...]


@dataclass(frozen=True)
class _NumericFact:
    value: float
    year: int | None
    currency: str | None
    source_ref: str
    page: int
    table_id: str | None = None


@dataclass(frozen=True)
class _SeriesCandidate:
    values: dict[int, _NumericFact]
    score: int
    table_id: str
    page: int
    title: str


def build_structured_fundamental_dataset(
    project_root: Path | str,
    *,
    manifest_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    company_ids: Iterable[str] = (),
) -> StructuredFundamentalDatasetResult:
    """批量提取发行条款和历史财务特征，并输出逐字段审计。"""
    root = Path(project_root).resolve()
    manifest = (
        Path(manifest_path).resolve()
        if manifest_path is not None
        else root / "data" / "processed" / "competition" / "company_manifest.jsonl"
    )
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / "data" / "processed" / "modeling"
    )
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    destination.mkdir(parents=True, exist_ok=True)

    selected_ids = {_company_id(value) for value in company_ids}
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    missing_cache_counts = {"tables": 0, "chunks": 0}
    manifest_rows = _read_jsonl(manifest)
    for manifest_row in manifest_rows:
        company_id = _company_id(manifest_row.get("company_id"))
        if selected_ids and company_id not in selected_ids:
            continue
        available_at = _date_value(manifest_row.get("prospectus_date"))
        company_folder = str(manifest_row.get("company_folder") or "").strip()
        if available_at is None or not company_folder:
            continue

        table_path = _cache_file(root, "tables", manifest_row, "tables.jsonl")
        chunk_path = _cache_file(root, "chunks", manifest_row, "chunks.jsonl")
        if table_path is None:
            missing_cache_counts["tables"] += 1
        if chunk_path is None:
            missing_cache_counts["chunks"] += 1

        extracted, raw_audit, missing_reasons = _extract_company_features(
            root,
            manifest_row=manifest_row,
            available_at=available_at,
            table_path=table_path,
            chunk_path=chunk_path,
        )
        features = _complete_feature_set(extracted, available_at=available_at)
        rows.append(
            {
                "company_id": company_id,
                "company_name": manifest_row.get("company_name"),
                "company_folder": company_folder,
                "prospectus_date": available_at.isoformat(),
                "split": manifest_row.get("split"),
                **{name: features[name].value for name in STRUCTURED_FEATURE_NAMES},
            }
        )
        audits.append(
            {
                "schema_version": 1,
                "company_id": company_id,
                "company_name": manifest_row.get("company_name"),
                "company_folder": company_folder,
                "prospectus_date": available_at.isoformat(),
                "source_hash": manifest_row.get("source_hash"),
                "table_cache": _portable(table_path, root),
                "chunk_cache": _portable(chunk_path, root),
                "features": {
                    name: features[name].model_dump(mode="json")
                    for name in STRUCTURED_FEATURE_NAMES
                },
                "raw_extraction": raw_audit,
                "missing_reasons": missing_reasons,
            }
        )

    if not rows:
        raise ValueError("没有可提取结构化特征的公司")
    frame = pd.DataFrame(rows)
    frame.sort_values(["split", "prospectus_date", "company_id"], inplace=True)
    feature_matrix_path = destination / "structured_fundamental_feature_matrix.csv"
    audit_path = destination / "structured_fundamental_feature_audit.jsonl"
    summary_path = destination / "structured_fundamental_feature_summary.json"
    frame.to_csv(feature_matrix_path, index=False, encoding="utf-8-sig")

    audit_by_company = {row["company_id"]: row for row in audits}
    write_jsonl(
        [audit_by_company[str(company_id)] for company_id in frame["company_id"]],
        audit_path,
    )
    coverage = {
        name: int(frame[name].notna().sum())
        for name in STRUCTURED_FEATURE_NAMES
        if not name.endswith("_available")
    }
    availability = {
        name: int(pd.to_numeric(frame[name], errors="coerce").fillna(0).sum())
        for name in STRUCTURED_FEATURE_NAMES
        if name.endswith("_available")
    }
    write_json(
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "extraction_policy": "deterministic_cached_tables_and_chunks_only",
            "model_or_embedding_invoked": False,
            "company_count": len(frame),
            "feature_count": len(STRUCTURED_FEATURE_NAMES),
            "feature_columns": list(STRUCTURED_FEATURE_NAMES),
            "coverage_non_null": coverage,
            "availability_counts": availability,
            "missing_cache_counts": missing_cache_counts,
            "feature_matrix_path": _portable(feature_matrix_path, root),
            "audit_path": _portable(audit_path, root),
        },
        summary_path,
    )
    return StructuredFundamentalDatasetResult(
        feature_matrix_path=feature_matrix_path,
        audit_path=audit_path,
        summary_path=summary_path,
        company_count=len(frame),
        feature_columns=STRUCTURED_FEATURE_NAMES,
    )


def load_structured_feature_snapshots(
    audit_path: Path,
) -> dict[str, dict[str, FeatureValue]]:
    """加载可审计结构化特征，供基础数据集按公司合并。"""
    if not audit_path.is_file():
        return {}
    snapshots: dict[str, dict[str, FeatureValue]] = {}
    for row in _read_jsonl(audit_path):
        company_id = _company_id(row.get("company_id"))
        payload = row.get("features")
        if not isinstance(payload, dict):
            continue
        snapshots[company_id] = {
            str(name): FeatureValue.model_validate(feature)
            for name, feature in payload.items()
            if name in STRUCTURED_FEATURE_NAMES and isinstance(feature, dict)
        }
    return snapshots


def extract_structured_fundamental_snapshot(
    project_root: Path | str,
    *,
    available_at: date,
    table_path: Path | None,
    chunk_path: Path | None,
) -> StructuredFundamentalSnapshot:
    """复用批量训练口径，为单份在线分析提取结构化候选特征。

    该接口只读取已经落盘的文本切片和表格缓存，不重新解析PDF，
    也不调用大模型。训练与在线运行因此共享同一套提取实现。
    """

    root = Path(project_root).resolve()
    resolved_table = (
        table_path.resolve() if table_path is not None and table_path.is_file() else None
    )
    resolved_chunks = (
        chunk_path.resolve() if chunk_path is not None and chunk_path.is_file() else None
    )
    features, audit, missing = _extract_company_features(
        root,
        manifest_row={},
        available_at=available_at,
        table_path=resolved_table,
        chunk_path=resolved_chunks,
    )
    return StructuredFundamentalSnapshot(
        features=features,
        audit=audit,
        missing_reasons=tuple(missing),
    )


def _extract_company_features(
    root: Path,
    *,
    manifest_row: dict[str, Any],
    available_at: date,
    table_path: Path | None,
    chunk_path: Path | None,
) -> tuple[dict[str, FeatureValue], dict[str, Any], list[str]]:
    features: dict[str, FeatureValue] = {}
    audit: dict[str, Any] = {"offering": {}, "financials": {}}
    missing: list[str] = []

    if chunk_path is not None:
        offering_features, offering_audit = _extract_offering_features(
            chunk_path, root=root, available_at=available_at
        )
        features.update(offering_features)
        audit["offering"] = offering_audit
    else:
        missing.append("missing_chunk_cache")

    if table_path is not None:
        financial_features, financial_audit = _extract_financial_features(
            table_path, root=root, available_at=available_at
        )
        features.update(financial_features)
        audit["financials"] = financial_audit
    else:
        missing.append("missing_table_cache")

    for availability_name, required_names in {
        "offering_terms_available": (
            "offering_offer_price_mid_hkd",
            "offering_share_count",
        ),
        "offering_net_proceeds_available": ("offering_net_proceeds_hkd",),
        "offering_market_cap_available": ("offering_market_cap_mid_hkd",),
        "fundamental_revenue_available": ("fundamental_revenue_latest_native",),
        "fundamental_gross_margin_available": ("fundamental_gross_margin_latest_pct",),
        "fundamental_operating_cash_flow_available": (
            "fundamental_operating_cash_flow_latest_native",
        ),
    }.items():
        available = int(all(name in features for name in required_names))
        features[availability_name] = FeatureValue(
            value=available,
            available_at=available_at,
            source_ref="structured_extraction:availability",
        )
        if not available:
            missing.append(availability_name.removesuffix("_available"))
    return features, audit, sorted(set(missing))


def _extract_financial_features(
    table_path: Path,
    *,
    root: Path,
    available_at: date,
) -> tuple[dict[str, FeatureValue], dict[str, Any]]:
    tables = list(_iter_relevant_tables(table_path))
    income_candidates: list[tuple[_SeriesCandidate, _SeriesCandidate | None]] = []
    cash_candidates: list[_SeriesCandidate] = []
    for table in tables:
        revenue = _series_from_table(table, kind="revenue", source_path=table_path, root=root)
        gross_profit = _series_from_table(
            table, kind="gross_profit", source_path=table_path, root=root
        )
        cash_flow = _series_from_table(
            table, kind="operating_cash_flow", source_path=table_path, root=root
        )
        if revenue is not None:
            combined_score = revenue.score + (gross_profit.score + 80 if gross_profit else 0)
            income_candidates.append(
                (
                    _SeriesCandidate(
                        values=revenue.values,
                        score=combined_score,
                        table_id=revenue.table_id,
                        page=revenue.page,
                        title=revenue.title,
                    ),
                    gross_profit,
                )
            )
        if cash_flow is not None:
            cash_candidates.append(cash_flow)

    income = max(income_candidates, key=lambda item: item[0].score, default=None)
    cash = max(cash_candidates, key=lambda item: item.score, default=None)
    revenue_values = income[0].values if income else {}
    gross_values = income[1].values if income and income[1] else {}
    cash_values = cash.values if cash else {}

    features: dict[str, FeatureValue] = {}
    currencies = [
        fact.currency
        for series in (revenue_values, gross_values, cash_values)
        for fact in series.values()
        if fact.currency
    ]
    reporting_currency = max(set(currencies), key=currencies.count) if currencies else None
    if reporting_currency:
        reporting_source = (
            next(iter(revenue_values.values()), None)
            or next(iter(gross_values.values()), None)
            or next(iter(cash_values.values()), None)
        )
        features["fundamental_reporting_currency"] = _feature(
            reporting_currency,
            available_at,
            reporting_source.source_ref,
        )

    revenue_years = sorted(revenue_values)[-3:]
    if revenue_years:
        latest = revenue_values[revenue_years[-1]]
        features["fundamental_revenue_latest_native"] = _feature(
            latest.value, available_at, latest.source_ref
        )
        features["fundamental_revenue_year_count"] = _feature(
            len(revenue_years), available_at, latest.source_ref
        )
    if len(revenue_years) >= 2:
        previous = revenue_values[revenue_years[-2]]
        latest = revenue_values[revenue_years[-1]]
        if previous.value > 0:
            growth = 100.0 * (latest.value / previous.value - 1.0)
            if -100.0 <= growth <= 10_000.0:
                features["fundamental_revenue_growth_latest_pct"] = _feature(
                    round(growth, 8), available_at, _combine_refs(previous, latest)
                )
    if len(revenue_years) == 3:
        earliest = revenue_values[revenue_years[0]]
        latest = revenue_values[revenue_years[-1]]
        intervals = revenue_years[-1] - revenue_years[0]
        if earliest.value > 0 and latest.value > 0 and intervals > 0:
            cagr = 100.0 * ((latest.value / earliest.value) ** (1.0 / intervals) - 1.0)
            if -100.0 <= cagr <= 10_000.0:
                features["fundamental_revenue_cagr_3y_pct"] = _feature(
                    round(cagr, 8), available_at, _combine_refs(earliest, latest)
                )

    margin_by_year: dict[int, tuple[float, str]] = {}
    for year in sorted(set(revenue_values).intersection(gross_values)):
        revenue = revenue_values[year]
        gross = gross_values[year]
        if revenue.value > 0 and _same_currency(revenue, gross):
            margin = 100.0 * gross.value / revenue.value
            if -100.0 <= margin <= 100.0:
                margin_by_year[year] = (margin, _combine_refs(revenue, gross))
    margin_years = sorted(margin_by_year)[-3:]
    if margin_years:
        latest_margin, latest_ref = margin_by_year[margin_years[-1]]
        features["fundamental_gross_margin_latest_pct"] = _feature(
            round(latest_margin, 8), available_at, latest_ref
        )
        features["fundamental_gross_margin_avg_3y_pct"] = _feature(
            round(sum(margin_by_year[year][0] for year in margin_years) / len(margin_years), 8),
            available_at,
            latest_ref,
        )
    if len(margin_years) == 3:
        change = margin_by_year[margin_years[-1]][0] - margin_by_year[margin_years[0]][0]
        features["fundamental_gross_margin_change_3y_pct_points"] = _feature(
            round(change, 8), available_at, margin_by_year[margin_years[-1]][1]
        )

    cash_years = sorted(cash_values)[-3:]
    if cash_years:
        latest_cash = cash_values[cash_years[-1]]
        features["fundamental_operating_cash_flow_latest_native"] = _feature(
            latest_cash.value, available_at, latest_cash.source_ref
        )
        features["fundamental_operating_cash_flow_sum_3y_native"] = _feature(
            sum(cash_values[year].value for year in cash_years),
            available_at,
            latest_cash.source_ref,
        )
        features["fundamental_negative_operating_cash_flow_years_3y"] = _feature(
            sum(cash_values[year].value < 0 for year in cash_years),
            available_at,
            latest_cash.source_ref,
        )
        latest_revenue = revenue_values.get(cash_years[-1])
        if (
            latest_revenue is not None
            and latest_revenue.value > 0
            and _same_currency(latest_cash, latest_revenue)
        ):
            ratio = 100.0 * latest_cash.value / latest_revenue.value
            if -10_000.0 <= ratio <= 10_000.0:
                features["fundamental_operating_cash_flow_to_revenue_latest_pct"] = _feature(
                    round(ratio, 8), available_at, _combine_refs(latest_cash, latest_revenue)
                )

    audit = {
        "income_table": _candidate_audit(income[0]) if income else None,
        "gross_profit_table": _candidate_audit(income[1]) if income and income[1] else None,
        "cash_flow_table": _candidate_audit(cash) if cash else None,
        "revenue": _series_audit(revenue_values),
        "gross_profit": _series_audit(gross_values),
        "operating_cash_flow": _series_audit(cash_values),
    }
    return features, audit


def _extract_offering_features(
    chunk_path: Path,
    *,
    root: Path,
    available_at: date,
) -> tuple[dict[str, FeatureValue], dict[str, Any]]:
    chunks = list(_iter_offering_chunks(chunk_path))
    values: dict[str, tuple[float, str, int]] = {}

    for chunk in chunks:
        text = _normalized_text(str(chunk.get("text") or ""))
        source_ref = f"{_portable(chunk_path, root)}#{chunk.get('chunk_id')}"
        page = int(chunk.get("page") or 0)
        if "發售價" in text or "发售价" in text:
            price_range = _price_range(text)
            existing_low = _value(values, "price_low")
            existing_high = _value(values, "price_high")
            existing_is_single = (
                existing_low is not None
                and existing_high is not None
                and existing_low == existing_high
            )
            if price_range is not None and (
                existing_low is None or (existing_is_single and price_range[0] < price_range[1])
            ):
                values["price_low"] = (price_range[0], source_ref, page)
                values["price_high"] = (price_range[1], source_ref, page)
        if "發售股份數目" in text or "发售股份数目" in text:
            share_count = _offering_share_count(text)
            if share_count is not None and "share_count" not in values:
                values["share_count"] = (share_count, source_ref, page)
        if _looks_like_market_cap_block(text) and not re.search(
            r"H\s*股市值", text, flags=re.IGNORECASE
        ):
            market_cap = _market_cap_range(text)
            if market_cap is not None and "market_cap_low" not in values:
                values["market_cap_low"] = (market_cap[0], source_ref, page)
                values["market_cap_high"] = (market_cap[1], source_ref, page)
        share_ratio = _offering_share_ratio(text)
        if share_ratio is not None and "share_ratio" not in values:
            values["share_ratio"] = (share_ratio, source_ref, page)
        if any(
            keyword in text
            for keyword in ("募集資金淨額", "募集资金净额", "所得款項淨額", "所得款项净额")
        ):
            net_proceeds = _net_proceeds(text)
            if net_proceeds is not None and "net_proceeds" not in values:
                values["net_proceeds"] = (net_proceeds, source_ref, page)

    features: dict[str, FeatureValue] = {}
    low_price = _value(values, "price_low")
    high_price = _value(values, "price_high")
    share_count = _value(values, "share_count")
    market_cap_low = _value(values, "market_cap_low")
    market_cap_high = _value(values, "market_cap_high")
    net_proceeds = _value(values, "net_proceeds")
    direct_share_ratio = _value(values, "share_ratio")

    if low_price is not None and high_price is not None:
        mid_price = (low_price + high_price) / 2.0
        ref = values["price_low"][1]
        features.update(
            {
                "offering_offer_price_low_hkd": _feature(low_price, available_at, ref),
                "offering_offer_price_high_hkd": _feature(high_price, available_at, ref),
                "offering_offer_price_mid_hkd": _feature(mid_price, available_at, ref),
            }
        )
    else:
        mid_price = None
    if share_count is not None:
        features["offering_share_count"] = _feature(
            share_count, available_at, values["share_count"][1]
        )
    if share_count is not None and low_price is not None and high_price is not None:
        ref = f"{values['share_count'][1]};{values['price_low'][1]}"
        gross_low = share_count * low_price
        gross_high = share_count * high_price
        features["offering_gross_proceeds_low_hkd"] = _feature(gross_low, available_at, ref)
        features["offering_gross_proceeds_high_hkd"] = _feature(gross_high, available_at, ref)
        features["offering_gross_proceeds_mid_hkd"] = _feature(
            (gross_low + gross_high) / 2.0, available_at, ref
        )
    if net_proceeds is not None:
        features["offering_net_proceeds_hkd"] = _feature(
            net_proceeds, available_at, values["net_proceeds"][1]
        )
    if market_cap_low is not None and market_cap_high is not None:
        ref = values["market_cap_low"][1]
        market_cap_mid = (market_cap_low + market_cap_high) / 2.0
        features["offering_market_cap_low_hkd"] = _feature(market_cap_low, available_at, ref)
        features["offering_market_cap_high_hkd"] = _feature(market_cap_high, available_at, ref)
        features["offering_market_cap_mid_hkd"] = _feature(market_cap_mid, available_at, ref)
        if share_count is not None and mid_price and market_cap_mid > 0:
            post_offer_shares = market_cap_mid / mid_price
            ratio = 100.0 * share_count / post_offer_shares
            if 0 < ratio <= 50:
                features["offering_share_ratio_pct"] = _feature(
                    round(ratio, 8),
                    available_at,
                    f"{values['share_count'][1]};{ref}",
                )
    if direct_share_ratio is not None:
        features["offering_share_ratio_pct"] = _feature(
            direct_share_ratio, available_at, values["share_ratio"][1]
        )

    audit = {
        name: {"value": value, "source_ref": source_ref, "page": page}
        for name, (value, source_ref, page) in values.items()
    }
    return features, audit


def _iter_relevant_tables(path: Path) -> Iterable[dict[str, Any]]:
    keywords = tuple(keyword.lower() for keyword in (*_INCOME_KEYWORDS, *_CASH_FLOW_KEYWORDS))
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            lowered = line.lower()
            if not any(keyword in lowered for keyword in keywords):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _iter_offering_chunks(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not any(keyword in line for keyword in _OFFERING_KEYWORDS):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def _series_from_table(
    table: dict[str, Any],
    *,
    kind: str,
    source_path: Path,
    root: Path,
) -> _SeriesCandidate | None:
    rows = table.get("rows")
    headers = table.get("headers")
    if not isinstance(rows, list) or not isinstance(headers, list):
        return None
    matched_row: list[Any] | None = None
    for row in rows:
        if not isinstance(row, list) or not row:
            continue
        if _matches_financial_row(str(row[0]), kind=kind):
            matched_row = row
            break
    if matched_row is None:
        return None

    table_id = str(table.get("table_id") or "unknown")
    page = int(table.get("page") or 0)
    title = str(table.get("title") or "")
    default_currency = _normalize_currency(table.get("currency"))
    table_multiplier = _unit_multiplier(
        " ".join(str(value or "") for value in headers), table.get("unit")
    )
    by_year: dict[int, list[_NumericFact]] = {}
    for index in range(1, min(len(headers), len(matched_row))):
        header = str(headers[index] or "")
        year = _annual_year(header)
        if year is None:
            continue
        value = _parse_number(matched_row[index])
        if value is None:
            continue
        column_multiplier = _unit_multiplier(header, None)
        multiplier = column_multiplier if column_multiplier != 1.0 else table_multiplier
        currency = _normalize_currency(header) or default_currency
        source_ref = (
            f"{_portable(source_path, root)}#{table_id}:R{rows.index(matched_row) + 1}:C{index + 1}"
        )
        by_year.setdefault(year, []).append(
            _NumericFact(
                value=value * multiplier,
                year=year,
                currency=currency,
                source_ref=source_ref,
                page=page,
                table_id=table_id,
            )
        )
    # 同一年重复列通常依次为“金额、占比”；金额列在前，不能用最大值选择，
    # 否则收入小于100个报告单位时会误把100%当成收入。
    selected = {year: facts[0] for year, facts in by_year.items() if facts}
    if not selected:
        return None
    title_score = 0
    normalized_title = _normalized_text(title).lower()
    if any(
        token in normalized_title
        for token in ("財務資料概要", "财务资料概要", "歷史財務", "历史财务")
    ):
        title_score += 40
    if any(
        token in normalized_title for token in ("損益", "损益", "現金流量", "现金流量", "cash flow")
    ):
        title_score += 25
    score = len(selected) * 20 + title_score + (20 if page <= 80 else 0)
    return _SeriesCandidate(
        values=selected,
        score=score,
        table_id=table_id,
        page=page,
        title=title,
    )


def _matches_financial_row(value: str, *, kind: str) -> bool:
    normalized = re.sub(r"[\s：:（）()\[\]]", "", value).lower()
    if kind == "revenue":
        return normalized in {
            "收入",
            "收益",
            "營業收入",
            "营业收入",
            "總收入",
            "总收入",
            "revenue",
            "revenues",
            "totalrevenue",
        }
    if kind == "gross_profit":
        return normalized in {"毛利", "毛利潤", "毛利润", "grossprofit"}
    if kind == "operating_cash_flow":
        return bool(
            re.search(r"(?:經營|经营).*?(?:活動|活动).*?(?:現金|现金).*?(?:淨額|净额)", normalized)
            or "netcashgeneratedfromoperatingactivities" in normalized
            or "netcashusedinoperatingactivities" in normalized
        )
    raise ValueError(f"unknown financial row kind: {kind}")


def _annual_year(header: str) -> int | None:
    match = re.search(r"(20\d{2})", header)
    if match is None:
        return None
    normalized = header.lower()
    if re.search(r"(?:個月|个月|months?)", normalized):
        return None
    if any(token in normalized for token in ("年度", "全年", "year ended", "financial year")):
        return int(match.group(1))
    if "12月31日" in normalized or "december 31" in normalized:
        return int(match.group(1))
    if re.fullmatch(r"\s*20\d{2}\s*(?:年)?\s*", normalized):
        return int(match.group(1))
    return None


def _parse_number(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text in {"-", "—", "–", "不適用", "不适用", "n/a", "N/A"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    cleaned = re.sub(r"[^0-9.\-]", "", text.replace(",", ""))
    if cleaned in {"", "-", "."}:
        return None
    try:
        number = float(cleaned)
    except ValueError:
        return None
    if negative and number > 0:
        number = -number
    return number if math.isfinite(number) else None


def _unit_multiplier(header: str, table_unit: Any) -> float:
    text = f"{header} {table_unit or ''}".lower()
    if "十億" in text or "十亿" in text or "billion" in text:
        return 1_000_000_000.0
    if "億" in text or "亿" in text:
        return 100_000_000.0
    if "百萬" in text or "百万" in text or "million" in text:
        return 1_000_000.0
    if "千" in text or "thousand" in text or re.search(r"\b000s\b", text):
        return 1_000.0
    return 1.0


def _normalize_currency(value: Any) -> str | None:
    text = str(value or "").upper()
    if any(token in text for token in ("RMB", "人民幣", "人民币", "CNY")):
        return "RMB"
    if any(token in text for token in ("HKD", "港元", "港幣", "港币")):
        return "HKD"
    if any(token in text for token in ("USD", "美元", "US$")):
        return "USD"
    return None


def _price_range(text: str) -> tuple[float, float] | None:
    patterns = (
        r"(?:發售價範圍|发售价范围).{0,80}?([\d,.]+)\s*港元\s*(?:至|到|－|-|–).{0,30}?([\d,.]+)\s*港元",
        r"每股(?:H股|發售股份|发售股份|股份)?.{0,20}?([\d,.]+)\s*港元\s*(?:至|到|－|-|–).{0,30}?([\d,.]+)\s*港元",
        r"不(?:低於|低于).{0,30}?([\d,.]+)\s*港元.{0,80}?不(?:超過|超过).{0,30}?([\d,.]+)\s*港元",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is None:
            continue
        first = _parse_number(match.group(1))
        second = _parse_number(match.group(2))
        if first is None or second is None:
            continue
        low, high = sorted((first, second))
        if 0 < low <= high <= 10_000:
            return low, high
    single = re.search(r"最高發售價.{0,40}?([\d,.]+)\s*港元", text)
    if single is not None:
        value = _parse_number(single.group(1))
        if value is not None and 0 < value <= 10_000:
            return value, value
    return None


def _offering_share_count(text: str) -> float | None:
    match = re.search(
        r"(?:全球發售的?)?發售股份數目\s*[:：]?\s*([\d,]+)\s*股",
        text,
    ) or re.search(
        r"(?:全球发售的?)?发售股份数目\s*[:：]?\s*([\d,]+)\s*股",
        text,
    )
    value = _parse_number(match.group(1)) if match else None
    return value if value is not None and value >= 10_000 else None


def _offering_share_ratio(text: str) -> float | None:
    """提取发售股份占紧随发售完成后已发行股份的比例。"""
    if re.search(r"已發行H股總數|已发行H股总数", text, flags=re.IGNORECASE):
        return None
    patterns = (
        r"發售股份(?:將)?(?:相當於|佔).{0,80}?緊隨.{0,180}?"
        r"(?:已發行|經擴大|股本總額).{0,80}?約?\s*([\d.]+)%",
        r"发售股份(?:将)?(?:相当于|占).{0,80}?紧随.{0,180}?"
        r"(?:已发行|经扩大|股本总额).{0,80}?约?\s*([\d.]+)%",
        r"represent.{0,80}?approximately\s*([\d.]+)%\s+of.{0,80}?issued shares",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is None:
            continue
        value = _parse_number(match.group(1))
        if value is not None and 0 < value <= 100:
            return value
    return None


def _looks_like_market_cap_block(text: str) -> bool:
    """排除上市规则门槛叙述，只接受带实际数值的市值统计块。"""
    normalized = _normalized_text(text)
    strong_label = bool(
        re.search(
            r"(?:股份|H股|上市|發售完成後|发售完成后|緊隨全球發售後|紧随全球发售后)"
            r".{0,12}?市值",
            normalized,
            flags=re.IGNORECASE,
        )
        or re.search(r"(?:^|\n)\s*市值(?:\(\d+\))?", text, flags=re.IGNORECASE)
    )
    return strong_label and bool(_money_values(text))


def _market_cap_range(text: str) -> tuple[float, float] | None:
    positions = [match.start() for match in re.finditer("市值", text)]
    candidates: list[float] = []
    for position in positions:
        local = text[position : position + 300]
        local_values = [value for value in _money_values(local) if value >= 50_000_000]
        if local_values:
            candidates = local_values
            break
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0], candidates[0]
    return min(candidates[:3]), max(candidates[:3])


def _money_range_after_keyword(text: str, keyword: str) -> tuple[float, float] | None:
    position = text.find(keyword)
    if position < 0:
        return None
    values = [
        value for value in _money_values(text[position : position + 420]) if value >= 1_000_000
    ]
    if not values:
        return None
    if len(values) == 1:
        return values[0], values[0]
    low, high = sorted(values[:2])
    return low, high


def _net_proceeds(text: str) -> float | None:
    keywords = ("募集資金淨額", "募集资金净额", "所得款項淨額", "所得款项净额")
    for keyword in keywords:
        position = text.find(keyword)
        if position < 0:
            continue
        local = text[max(0, position - 140) : position + 180]
        values = [value for value in _money_values(local) if value > 0]
        if values:
            # 同一语句可能同时出现每股价格与募集资金；净额应是其中的金额主体。
            return max(values)
    return None


def _money_values(text: str) -> list[float]:
    values: list[float] = []
    for match in re.finditer(
        r"([\d,]+(?:\.\d+)?)\s*(十億|十亿|億|亿|百萬|百万|千)?\s*港元",
        text,
    ):
        number = _parse_number(match.group(1))
        if number is None:
            continue
        unit = match.group(2) or ""
        multiplier = _unit_multiplier(unit, None)
        values.append(number * multiplier)
    return values


def _complete_feature_set(
    extracted: dict[str, FeatureValue],
    *,
    available_at: date,
) -> dict[str, FeatureValue]:
    return {
        name: extracted.get(name)
        or FeatureValue(
            value=None,
            available_at=available_at,
            source_ref="structured_extraction:not_found",
        )
        for name in STRUCTURED_FEATURE_NAMES
    }


def _feature(value: Any, available_at: date, source_ref: str) -> FeatureValue:
    return FeatureValue(value=value, available_at=available_at, source_ref=source_ref)


def _same_currency(first: _NumericFact, second: _NumericFact) -> bool:
    return not first.currency or not second.currency or first.currency == second.currency


def _combine_refs(first: _NumericFact, second: _NumericFact) -> str:
    return f"{first.source_ref};{second.source_ref}"


def _candidate_audit(candidate: _SeriesCandidate | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "table_id": candidate.table_id,
        "page": candidate.page,
        "title": candidate.title,
        "score": candidate.score,
    }


def _series_audit(values: dict[int, _NumericFact]) -> list[dict[str, Any]]:
    return [
        {
            "year": year,
            "value": fact.value,
            "currency": fact.currency,
            "page": fact.page,
            "table_id": fact.table_id,
            "source_ref": fact.source_ref,
        }
        for year, fact in sorted(values.items())
    ]


def _value(values: dict[str, tuple[float, str, int]], name: str) -> float | None:
    payload = values.get(name)
    return payload[0] if payload else None


def _cache_file(
    root: Path,
    cache_kind: str,
    manifest_row: dict[str, Any],
    file_name: str,
) -> Path | None:
    company_folder = str(manifest_row.get("company_folder") or "")
    company_id = _company_id(manifest_row.get("company_id"))
    prospectus_date = str(manifest_row.get("prospectus_date") or "")[:10]
    source_hash = str(manifest_row.get("source_hash") or "")[:8]
    expected = (
        root
        / "data"
        / "interim"
        / cache_kind
        / company_folder
        / f"doc_{company_id}_prospectus_{prospectus_date}_{source_hash}"
        / file_name
    )
    if expected.is_file():
        return expected
    company_dir = root / "data" / "interim" / cache_kind / company_folder
    matches = sorted(company_dir.glob(f"*/{file_name}")) if company_dir.is_dir() else []
    return matches[0] if len(matches) == 1 else None


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _company_id(value: Any) -> str:
    return str(value or "").strip().split(".", maxsplit=1)[0].zfill(5)


def _date_value(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _portable(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


__all__ = [
    "STRUCTURED_FEATURE_NAMES",
    "StructuredFundamentalDatasetResult",
    "StructuredFundamentalSnapshot",
    "build_structured_fundamental_dataset",
    "extract_structured_fundamental_snapshot",
    "load_structured_feature_snapshots",
]
