"""导入人工/外部模型核验后的基础与 RAG 特征补充数据。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from ipo_risk.modeling.rag_fundamentals import (
    RAG_FUNDAMENTAL_FEATURE_NAMES,
    build_rag_fundamental_dataset,
)
from ipo_risk.modeling.structured_fundamentals import STRUCTURED_FEATURE_NAMES
from ipo_risk.storage.files import read_jsonl, write_json, write_jsonl


SUPPLEMENT_FILENAME = "2025_fundamental_supplement_verified.json"
_PDF_DATE = re.compile(r"_(\d{2})-(\d{2})-(\d{4})_")
_OBJECT_START = re.compile(r"\{\s*\"pdf_file\"\s*:")


@dataclass(frozen=True)
class FundamentalSupplementResult:
    supplement_path: Path
    company_count: int
    structured_matrix_path: Path
    rag_matrix_path: Path


def read_supplement_records(path: Path | str) -> list[dict[str, Any]]:
    """读取标准 JSON，或从拼接/局部损坏的文本中恢复完整公司对象。"""
    source = Path(path)
    text = source.read_text(encoding="utf-8-sig")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        payload = [payload]
    if isinstance(payload, list):
        return [dict(row) for row in payload if _is_company_record(row)]

    starts = [match.start() for match in _OBJECT_START.finditer(text)]
    recovered: list[dict[str, Any]] = []
    for index, start in enumerate(starts):
        boundary = starts[index + 1] if index + 1 < len(starts) else len(text)
        object_text = _balanced_object(text[start:boundary])
        if object_text is None:
            continue
        try:
            row = json.loads(object_text)
        except json.JSONDecodeError:
            continue
        if _is_company_record(row):
            recovered.append(dict(row))
    return recovered


def merge_fundamental_supplements(
    project_root: Path | str,
    *,
    source_paths: Sequence[Path | str],
    year: int = 2025,
    require_complete_year: bool = True,
) -> FundamentalSupplementResult:
    """按来源顺序合并补充数据；后一个来源覆盖前一个来源。"""
    root = Path(project_root).resolve()
    destination = root / "data" / "processed" / "modeling"
    manifest_path = root / "data" / "processed" / "competition" / "company_manifest.jsonl"
    manifest_rows = read_jsonl(manifest_path)
    all_manifest_by_company = {
        _company_id(row.get("company_id")): row for row in manifest_rows
    }
    manifest_by_company = {
        _company_id(row.get("company_id")): row
        for row in manifest_rows
        if int(row.get("prospectus_year") or 0) == year
    }

    merged: dict[str, dict[str, Any]] = {}
    for source_path in source_paths:
        for raw in read_supplement_records(source_path):
            company_id = _company_id(raw.get("company_id"))
            if company_id not in manifest_by_company:
                continue
            row = dict(raw)
            row["company_id"] = company_id
            row["prospectus_date"] = (
                str(row.get("prospectus_date") or "").strip()
                or _date_from_pdf_name(str(row.get("pdf_file") or ""))
                or str(manifest_by_company[company_id].get("prospectus_date") or "")[:10]
            )
            merged[company_id] = row

    if require_complete_year:
        missing = sorted(set(manifest_by_company) - set(merged))
        if missing:
            raise ValueError(f"{year}年补充数据仍缺少{len(missing)}家公司：{', '.join(missing)}")

    ordered = [merged[key] for key in sorted(merged)]
    supplement_path = destination / SUPPLEMENT_FILENAME
    write_json(ordered, supplement_path)

    structured_matrix_path = _merge_structured_audit(
        root,
        records=ordered,
        manifest_by_company=all_manifest_by_company,
        source_path=supplement_path,
    )
    rag_matrix_path = _merge_rag_audit(
        root,
        records=ordered,
        manifest_rows=manifest_rows,
        source_path=supplement_path,
    )
    return FundamentalSupplementResult(
        supplement_path=supplement_path,
        company_count=len(ordered),
        structured_matrix_path=structured_matrix_path,
        rag_matrix_path=rag_matrix_path,
    )


def _merge_structured_audit(
    root: Path,
    *,
    records: Sequence[Mapping[str, Any]],
    manifest_by_company: Mapping[str, Mapping[str, Any]],
    source_path: Path,
) -> Path:
    destination = root / "data" / "processed" / "modeling"
    audit_path = destination / "structured_fundamental_feature_audit.jsonl"
    audits = {
        _company_id(row.get("company_id")): dict(row)
        for row in read_jsonl(audit_path)
    }
    source_ref = source_path.relative_to(root).as_posix()
    for record in records:
        company_id = _company_id(record.get("company_id"))
        manifest = manifest_by_company[company_id]
        available_at = str(record["prospectus_date"])[:10]
        row = audits.get(company_id) or {
            "schema_version": 1,
            "company_id": company_id,
            "company_name": record.get("company_name") or manifest.get("company_name"),
            "company_folder": manifest.get("company_folder"),
            "prospectus_date": available_at,
            "source_hash": manifest.get("source_hash"),
            "features": {},
        }
        features = dict(row.get("features") or {})
        values = _structured_values(record)
        for name, value in values.items():
            if name in STRUCTURED_FEATURE_NAMES:
                features[name] = _feature(value, available_at, source_ref, company_id)
        row["features"] = features
        row["verified_supplement"] = _supplement_lineage(record, source_ref)
        audits[company_id] = row

    ordered = sorted(
        audits.values(),
        key=lambda row: (str(row.get("prospectus_date") or ""), _company_id(row.get("company_id"))),
    )
    write_jsonl(ordered, audit_path)
    matrix_path = destination / "structured_fundamental_feature_matrix.csv"
    matrix_rows: list[dict[str, Any]] = []
    for row in ordered:
        company_id = _company_id(row.get("company_id"))
        features = row.get("features") or {}
        matrix_rows.append(
            {
                "company_id": company_id,
                "company_name": row.get("company_name"),
                "company_folder": row.get("company_folder"),
                "prospectus_date": row.get("prospectus_date"),
                "split": (manifest_by_company.get(company_id) or {}).get("split"),
                **{
                    name: (features.get(name) or {}).get("value")
                    for name in STRUCTURED_FEATURE_NAMES
                },
            }
        )
    frame = pd.DataFrame(matrix_rows)
    frame.to_csv(matrix_path, index=False, encoding="utf-8-sig")
    _write_summary(
        destination / "structured_fundamental_feature_summary.json",
        frame=frame,
        feature_names=STRUCTURED_FEATURE_NAMES,
        policy="deterministic_cache_plus_verified_supplement",
    )
    return matrix_path


def _merge_rag_audit(
    root: Path,
    *,
    records: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
    source_path: Path,
) -> Path:
    destination = root / "data" / "processed" / "modeling"
    audit_path = destination / "rag_fundamental_feature_audit.jsonl"
    audits = {
        _company_id(row.get("company_id")): dict(row)
        for row in read_jsonl(audit_path)
    }
    source_ref = source_path.relative_to(root).as_posix()
    for record in records:
        company_id = _company_id(record.get("company_id"))
        available_at = str(record["prospectus_date"])[:10]
        row = audits.get(company_id) or {
            "schema_version": 1,
            "company_id": company_id,
            "company_name": record.get("company_name"),
            "prospectus_date": available_at,
            "features": {},
        }
        features = dict(row.get("features") or {})
        for name in RAG_FUNDAMENTAL_FEATURE_NAMES:
            if name in record:
                features[name] = _feature(
                    record.get(name), available_at, source_ref, company_id
                )
        row["features"] = features
        row["verified_supplement"] = _supplement_lineage(record, source_ref)
        audits[company_id] = row

    result = build_rag_fundamental_dataset(
        root,
        manifest_rows=manifest_rows,
        records=list(audits.values()),
    )
    return result.feature_matrix_path


def _structured_values(record: Mapping[str, Any]) -> dict[str, Any]:
    values = {
        name: record.get(name)
        for name in STRUCTURED_FEATURE_NAMES
        if name in record
    }
    for low, high, mid in (
        ("offering_offer_price_low_hkd", "offering_offer_price_high_hkd", "offering_offer_price_mid_hkd"),
        ("offering_gross_proceeds_low_hkd", "offering_gross_proceeds_high_hkd", "offering_gross_proceeds_mid_hkd"),
        ("offering_market_cap_low_hkd", "offering_market_cap_high_hkd", "offering_market_cap_mid_hkd"),
    ):
        values[mid] = _midpoint(record.get(low), record.get(high))
    values.update(
        {
            "offering_terms_available": int(
                values.get("offering_offer_price_mid_hkd") is not None
                and record.get("offering_share_count") is not None
            ),
            "offering_net_proceeds_available": int(record.get("offering_net_proceeds_hkd") is not None),
            "offering_market_cap_available": int(values.get("offering_market_cap_mid_hkd") is not None),
            "fundamental_revenue_available": int(record.get("fundamental_revenue_latest_native") is not None),
            "fundamental_gross_margin_available": int(record.get("fundamental_gross_margin_latest_pct") is not None),
            "fundamental_operating_cash_flow_available": int(
                record.get("fundamental_operating_cash_flow_latest_native") is not None
            ),
        }
    )
    return values


def _feature(value: Any, available_at: str, source_ref: str, company_id: str) -> dict[str, Any]:
    return {
        "value": value,
        "available_at": available_at,
        "source_ref": f"{source_ref}#{company_id}",
    }


def _supplement_lineage(record: Mapping[str, Any], source_ref: str) -> dict[str, Any]:
    return {
        "source_ref": source_ref,
        "pdf_file": record.get("pdf_file"),
        "financial_period_used": record.get("financial_period_used"),
        "evidence_pages": record.get("evidence_pages") or {},
        "warnings": record.get("warnings") or [],
    }


def _write_summary(
    path: Path,
    *,
    frame: pd.DataFrame,
    feature_names: Iterable[str],
    policy: str,
) -> None:
    names = tuple(feature_names)
    write_json(
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "extraction_policy": policy,
            "company_count": len(frame),
            "feature_count": len(names),
            "feature_columns": list(names),
            "coverage_non_null": {name: int(frame[name].notna().sum()) for name in names},
        },
        path,
    )


def _balanced_object(fragment: str) -> str | None:
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(fragment):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return fragment[: index + 1]
    return None


def _is_company_record(value: Any) -> bool:
    return isinstance(value, dict) and bool(str(value.get("company_id") or "").strip())


def _company_id(value: Any) -> str:
    return str(value or "").strip().removesuffix(".0").zfill(5)


def _date_from_pdf_name(name: str) -> str | None:
    match = _PDF_DATE.search(name)
    if match is None:
        return None
    day, month, year = match.groups()
    return f"{year}-{month}-{day}"


def _midpoint(low: Any, high: Any) -> float | None:
    if low is None and high is None:
        return None
    if low is None:
        return float(high)
    if high is None:
        return float(low)
    return (float(low) + float(high)) / 2.0


__all__ = [
    "FundamentalSupplementResult",
    "SUPPLEMENT_FILENAME",
    "merge_fundamental_supplements",
    "read_supplement_records",
]
