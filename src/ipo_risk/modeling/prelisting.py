"""上市前候选特征收集、模型契约投影与就绪校验。

分析阶段尽可能多收集可审计候选特征；预测阶段只按冻结模型契约取值。
训练和在线运行共用这一条 seam，避免从历史公司宽表回填新公司的输入。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ipo_risk.domain import FeatureValue
from ipo_risk.modeling.dataset import (
    DEFAULT_MARKET_WINDOWS,
    build_listing_pre_market_feature_snapshot,
    build_recent_ipo_feature_snapshot,
)
from ipo_risk.modeling.features import build_prelisting_feature_snapshot
from ipo_risk.modeling.stable_agent_model import add_derived_prediction_features
from ipo_risk.modeling.structured_fundamentals import (
    extract_structured_fundamental_snapshot,
)
from ipo_risk.reporting import RiskAnalysisReport
from ipo_risk.skills.valuation import extract_valuation_features


CANDIDATE_SCHEMA_VERSION = "2.0"
CONTRACT_SCHEMA_VERSION = "1.0"
EXTRACTOR_VERSION = "prelisting-candidates-v1"
_ALLOWED_DTYPES = frozenset({"number"})
_ALLOWED_MISSING_POLICIES = frozenset({"model_impute", "required"})

FeatureCollectionProgress = Callable[[int, int, str], None]


@dataclass(frozen=True)
class CandidateFeatureSnapshot:
    """本次分析得到的宽候选特征池及其治理信息。"""

    features: dict[str, FeatureValue]
    feature_groups: dict[str, str]
    feature_units: dict[str, str]
    feature_formula_versions: dict[str, str]
    prediction_as_of: date
    collection_warnings: tuple[str, ...] = ()

    def to_payload(self, *, run_id: str, company_id: str) -> dict[str, Any]:
        available = sum(feature.value is not None for feature in self.features.values())
        return {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "run_id": run_id,
            "company_id": company_id,
            "prediction_as_of": self.prediction_as_of.isoformat(),
            "usage_scope": "pre_listing_candidate_pool",
            "candidate_feature_count": len(self.features),
            "available_feature_count": available,
            "feature_completeness": (
                available / len(self.features) if self.features else 0.0
            ),
            "extractor_version": EXTRACTOR_VERSION,
            "collection_warnings": list(self.collection_warnings),
            "features": {
                name: {
                    **feature.model_dump(mode="json"),
                    "group": self.feature_groups[name],
                    "unit": self.feature_units[name],
                    "formula_version": self.feature_formula_versions[name],
                    "status": "available" if feature.value is not None else "missing",
                    "extractor_version": EXTRACTOR_VERSION,
                }
                for name, feature in sorted(self.features.items())
            },
        }


@dataclass(frozen=True)
class ModelFeatureField:
    """冻结模型对单个输入字段的要求。"""

    name: str
    group: str
    unit: str
    dtype: str = "number"
    required: bool = False
    missing_policy: str = "model_impute"
    formula_version: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | str) -> ModelFeatureField:
        if isinstance(payload, str):
            return cls(
                name=payload,
                group=feature_group(payload),
                unit=feature_unit(payload),
                formula_version=feature_formula_version(payload),
            )
        name = str(payload["name"])
        return cls(
            name=name,
            group=str(payload.get("group") or feature_group(name)),
            unit=str(payload.get("unit") or feature_unit(name)),
            dtype=str(payload.get("dtype") or "number"),
            required=bool(payload.get("required", False)),
            missing_policy=str(payload.get("missing_policy") or "model_impute"),
            formula_version=(
                str(payload["formula_version"])
                if payload.get("formula_version") is not None
                else None
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "unit": self.unit,
            "dtype": self.dtype,
            "required": self.required,
            "missing_policy": self.missing_policy,
            "formula_version": self.formula_version,
        }


@dataclass(frozen=True)
class ModelFeatureContract:
    """模型发布时冻结的输入 Interface。"""

    model_name: str
    fields: tuple[ModelFeatureField, ...]
    prediction_as_of: str = "listing_date"
    minimum_feature_completeness: float = 0.60
    review_below_completeness: float = 0.85
    review_group_completeness: tuple[tuple[str, float], ...] = (
        ("agent", 0.60),
        ("market", 1.00),
    )
    required_group_completeness: tuple[tuple[str, float], ...] = ()
    schema_version: str = CONTRACT_SCHEMA_VERSION

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ModelFeatureContract:
        raw_fields = payload.get("features")
        if not isinstance(raw_fields, list) or not raw_fields:
            raise ValueError("feature contract must declare features")
        raw_groups = payload.get("review_group_completeness", {})
        groups = (
            tuple((str(name), float(value)) for name, value in raw_groups.items())
            if isinstance(raw_groups, Mapping)
            else ()
        )
        raw_required_groups = payload.get("required_group_completeness", {})
        required_groups = (
            tuple(
                (str(name), float(value))
                for name, value in raw_required_groups.items()
            )
            if isinstance(raw_required_groups, Mapping)
            else ()
        )
        contract = cls(
            model_name=str(payload.get("model_name") or "unknown"),
            fields=tuple(ModelFeatureField.from_payload(item) for item in raw_fields),
            prediction_as_of=str(payload.get("prediction_as_of") or "listing_date"),
            minimum_feature_completeness=float(
                payload.get("minimum_feature_completeness", 0.60)
            ),
            review_below_completeness=float(
                payload.get("review_below_completeness", 0.85)
            ),
            review_group_completeness=groups or cls.review_group_completeness,
            required_group_completeness=required_groups,
            schema_version=str(payload.get("schema_version") or CONTRACT_SCHEMA_VERSION),
        )
        contract._validate()
        expected_fingerprint = payload.get("contract_fingerprint")
        if expected_fingerprint is not None and str(expected_fingerprint) != contract.fingerprint:
            raise ValueError("feature contract fingerprint mismatch")
        return contract

    @classmethod
    def from_bundle(cls, bundle: Mapping[str, Any]) -> ModelFeatureContract:
        embedded = bundle.get("feature_contract")
        if isinstance(embedded, Mapping):
            return cls.from_payload(embedded)
        names = bundle.get("features") or bundle.get("feature_columns")
        if not isinstance(names, list) or not names:
            raise ValueError("model bundle does not declare features")
        return cls(
            model_name=str(bundle.get("model_name") or "unknown"),
            fields=tuple(ModelFeatureField.from_payload(str(name)) for name in names),
            prediction_as_of=str(bundle.get("prediction_as_of") or "prospectus_date"),
            review_group_completeness=(("agent", 0.60),),
            required_group_completeness=(
                (("market", 1.00),)
                if any(str(name).startswith("market_") for name in names)
                else ()
            ),
        )

    def _core_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "prediction_as_of": self.prediction_as_of,
            "minimum_feature_completeness": self.minimum_feature_completeness,
            "review_below_completeness": self.review_below_completeness,
            "review_group_completeness": dict(self.review_group_completeness),
            "required_group_completeness": dict(self.required_group_completeness),
            "features": [field.to_payload() for field in self.fields],
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self._core_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_payload(self) -> dict[str, Any]:
        return {**self._core_payload(), "contract_fingerprint": self.fingerprint}

    def _validate(self) -> None:
        names = self.feature_names
        if len(names) != len(set(names)):
            raise ValueError("feature contract contains duplicate feature names")
        if any(field.name.startswith("target_") for field in self.fields):
            raise ValueError("target columns are forbidden in model contract")
        for field in self.fields:
            if field.dtype not in _ALLOWED_DTYPES:
                raise ValueError(f"unsupported dtype for {field.name}: {field.dtype}")
            if field.missing_policy not in _ALLOWED_MISSING_POLICIES:
                raise ValueError(
                    f"unsupported missing policy for {field.name}: {field.missing_policy}"
                )


@dataclass(frozen=True)
class ModelInputProjection:
    """候选池按模型契约投影后的固定输入与校验结论。"""

    values: dict[str, Any]
    status: str
    completeness: float
    available_feature_count: int
    missing_features: tuple[str, ...]
    issues: tuple[str, ...]
    group_completeness: dict[str, float]


def collect_prelisting_candidates(
    report: RiskAnalysisReport,
    *,
    project_root: Path,
    company_info: dict[str, object],
    prediction_as_of: date,
    tables_path: Path | None = None,
    chunks_path: Path | None = None,
    listing_date: date | None = None,
    supplemental_features: Mapping[str, FeatureValue] | None = None,
    progress_callback: FeatureCollectionProgress | None = None,
) -> CandidateFeatureSnapshot:
    """收集本次公司可用的宽候选池，不提前限制为模型的18个字段。"""

    features = build_prelisting_feature_snapshot(
        report,
        company_info=company_info,
        as_of_date=prediction_as_of,
    )
    _notify_feature_progress(progress_callback, 1, "Agent风险与公司基础特征已汇总")
    warnings: list[str] = []
    source_available_at = report.source_document.available_at

    try:
        structured = extract_structured_fundamental_snapshot(
            project_root,
            available_at=source_available_at,
            table_path=tables_path,
            chunk_path=chunks_path,
        )
        _merge(features, structured.features)
        warnings.extend(f"structured:{reason}" for reason in structured.missing_reasons)
    except Exception as exc:  # noqa: BLE001 - 单类候选失败不能丢掉整份快照
        warnings.append(f"structured_extraction_failed:{exc}")
    if supplemental_features:
        _merge_supplemental(features, supplemental_features, warnings)
    _notify_feature_progress(progress_callback, 2, "发行与财务结构化字段已汇总")

    _merge(
        features,
        _document_features(
            report,
            company_info=company_info,
            available_at=source_available_at,
            tables_path=tables_path,
        ),
    )
    _notify_feature_progress(progress_callback, 3, "文档结构与证据统计已汇总")

    if listing_date is not None:
        modeling_dir = project_root / "data" / "processed" / "modeling"
        try:
            _merge(
                features,
                build_listing_pre_market_feature_snapshot(
                    modeling_dir / "daily_market_sentiment.csv",
                    listing_date=listing_date,
                    windows=DEFAULT_MARKET_WINDOWS,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"market_context_failed:{exc}")
        try:
            _merge(
                features,
                build_recent_ipo_feature_snapshot(
                    project_root
                    / "data"
                    / "processed"
                    / "competition"
                    / "market_labels.csv",
                    listing_date=listing_date,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"recent_ipo_context_failed:{exc}")
    else:
        warnings.append("listing_date_missing:market_context_not_collected")
    _notify_feature_progress(progress_callback, 4, "上市前市场与近期IPO环境已汇总")

    _add_derived_candidates(features, prediction_as_of=prediction_as_of)
    _notify_feature_progress(progress_callback, 5, "确定性衍生特征已计算")
    _add_quality_candidates(features, prediction_as_of=prediction_as_of)
    _notify_feature_progress(progress_callback, 6, "数据质量与可用性特征已完成")
    future = sorted(
        name for name, feature in features.items() if feature.available_at > prediction_as_of
    )
    if future:
        raise ValueError(f"features available after prediction_as_of: {', '.join(future)}")
    return CandidateFeatureSnapshot(
        features=features,
        feature_groups={name: feature_group(name) for name in features},
        feature_units={name: feature_unit(name) for name in features},
        feature_formula_versions={
            name: feature_formula_version(name) for name in features
        },
        prediction_as_of=prediction_as_of,
        collection_warnings=tuple(dict.fromkeys(warnings)),
    )


def collect_prelisting_base_candidates(
    *,
    project_root: Path,
    company_info: dict[str, object],
    company_id: str,
    source_ref: str,
    source_available_at: date,
    page_count: int,
    chunk_count: int,
    prediction_as_of: date,
    tables_path: Path | None = None,
    chunks_path: Path | None = None,
    listing_date: date | None = None,
) -> CandidateFeatureSnapshot:
    """不依赖Agent结论，生成可立即用于基础模型的上市前候选特征。"""

    del company_id  # 保留显式参数，便于审计调用方传入正确公司上下文。
    features = extract_valuation_features(dict(company_info))
    warnings: list[str] = []
    try:
        structured = extract_structured_fundamental_snapshot(
            project_root,
            available_at=source_available_at,
            table_path=tables_path,
            chunk_path=chunks_path,
        )
        _merge(features, structured.features)
        warnings.extend(f"structured:{reason}" for reason in structured.missing_reasons)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"structured_extraction_failed:{exc}")

    _merge(
        features,
        _document_features_from_values(
            source_ref=source_ref,
            page_count=page_count,
            chunk_count=chunk_count,
            company_info=company_info,
            available_at=source_available_at,
            tables_path=tables_path,
        ),
    )
    if listing_date is not None:
        modeling_dir = project_root / "data" / "processed" / "modeling"
        try:
            _merge(
                features,
                build_listing_pre_market_feature_snapshot(
                    modeling_dir / "daily_market_sentiment.csv",
                    listing_date=listing_date,
                    windows=DEFAULT_MARKET_WINDOWS,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"market_context_failed:{exc}")
        try:
            _merge(
                features,
                build_recent_ipo_feature_snapshot(
                    project_root
                    / "data"
                    / "processed"
                    / "competition"
                    / "market_labels.csv",
                    listing_date=listing_date,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"recent_ipo_context_failed:{exc}")
    else:
        warnings.append("listing_date_missing:market_context_not_collected")

    _add_derived_candidates(features, prediction_as_of=prediction_as_of)
    _add_quality_candidates(features, prediction_as_of=prediction_as_of)
    future = sorted(
        name for name, feature in features.items() if feature.available_at > prediction_as_of
    )
    if future:
        raise ValueError(f"features available after prediction_as_of: {', '.join(future)}")
    return CandidateFeatureSnapshot(
        features=features,
        feature_groups={name: feature_group(name) for name in features},
        feature_units={name: feature_unit(name) for name in features},
        feature_formula_versions={
            name: feature_formula_version(name) for name in features
        },
        prediction_as_of=prediction_as_of,
        collection_warnings=tuple(dict.fromkeys(warnings)),
    )


def _notify_feature_progress(
    callback: FeatureCollectionProgress | None,
    current: int,
    detail: str,
) -> None:
    if callback is not None:
        callback(current, 6, detail)


def load_model_feature_contract(
    bundle: Mapping[str, Any], *, sidecar_path: Path | None = None
) -> ModelFeatureContract:
    """优先读取模型旁的显式契约，旧模型则从冻结字段生成兼容契约。"""

    if sidecar_path is not None and sidecar_path.is_file():
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("feature contract root must be an object")
        contract = ModelFeatureContract.from_payload(payload)
    else:
        contract = ModelFeatureContract.from_bundle(bundle)
    bundle_model_name = str(bundle.get("model_name") or "unknown")
    bundle_names = bundle.get("features") or bundle.get("feature_columns")
    if contract.model_name != bundle_model_name:
        raise ValueError(
            "feature contract model_name mismatch: "
            f"{contract.model_name} != {bundle_model_name}"
        )
    if isinstance(bundle_names, list) and tuple(str(name) for name in bundle_names) != contract.feature_names:
        raise ValueError("feature contract fields do not match model bundle order")
    embedded = bundle.get("feature_contract")
    if isinstance(embedded, Mapping):
        embedded_contract = ModelFeatureContract.from_payload(embedded)
        if embedded_contract.fingerprint != contract.fingerprint:
            raise ValueError("sidecar feature contract does not match embedded contract")
    return contract


def project_model_inputs(
    features: Mapping[str, Any],
    contract: ModelFeatureContract,
    *,
    prediction_as_of: date | None = None,
    feature_units: Mapping[str, str] | None = None,
    feature_formula_versions: Mapping[str, str] | None = None,
) -> ModelInputProjection:
    """只按冻结契约投影输入，并识别缺失、未来数据和低完整度。"""

    values: dict[str, Any] = {}
    missing: list[str] = []
    required_missing: list[str] = []
    future: list[str] = []
    metadata_mismatches: list[str] = []
    dtype_mismatches: list[str] = []
    groups: dict[str, list[bool]] = {}
    for field in contract.fields:
        if field.name.startswith("target_"):
            raise ValueError(f"target column is forbidden in model contract: {field.name}")
        raw = features.get(field.name)
        value, available_at = _value_and_date(raw)
        actual_unit = (feature_units or {}).get(field.name, feature_unit(field.name))
        actual_formula = (feature_formula_versions or {}).get(
            field.name, feature_formula_version(field.name)
        )
        if actual_unit != field.unit:
            metadata_mismatches.append(
                f"{field.name}.unit={actual_unit}, expected={field.unit}"
            )
        if field.formula_version is not None and actual_formula != field.formula_version:
            metadata_mismatches.append(
                f"{field.name}.formula={actual_formula}, expected={field.formula_version}"
            )
        if prediction_as_of is not None and available_at is not None and available_at > prediction_as_of:
            value = None
            future.append(field.name)
        value = _json_scalar(value)
        if value is not None and field.dtype == "number" and not isinstance(
            value, (int, float, bool)
        ):
            dtype_mismatches.append(field.name)
            value = None
        is_available = value is not None
        values[field.name] = value
        groups.setdefault(field.group, []).append(is_available)
        if not is_available:
            missing.append(field.name)
            if field.required:
                required_missing.append(field.name)

    available = len(contract.fields) - len(missing)
    completeness = available / len(contract.fields) if contract.fields else 0.0
    group_completeness = {
        group: sum(flags) / len(flags) if flags else 0.0
        for group, flags in groups.items()
    }
    issues: list[str] = []
    status = "ready"
    if contract.prediction_as_of == "listing_date" and prediction_as_of is None:
        issues.append("当前模型要求listing_date，缺少上市日期，不能计算上市日前市场特征。")
        status = "blocked"
    if metadata_mismatches:
        issues.append("特征契约元数据不一致：" + "; ".join(metadata_mismatches))
        status = "blocked"
    if dtype_mismatches:
        issues.append("特征类型不是数值：" + ", ".join(dtype_mismatches))
        status = "blocked"
    if future:
        issues.append(f"检测到预测时点之后才可用的字段：{', '.join(future)}")
        status = "blocked"
    if required_missing:
        issues.append(f"缺少必需字段：{', '.join(required_missing)}")
        status = "blocked"
    if completeness < contract.minimum_feature_completeness:
        status = "blocked"
    elif completeness < contract.review_below_completeness and status == "ready":
        status = "review_required"
    for group, minimum in contract.required_group_completeness:
        required_actual = group_completeness.get(group, 0.0)
        if required_actual < minimum:
            status = "blocked"
    for group, minimum in contract.review_group_completeness:
        review_actual = group_completeness.get(group)
        if review_actual is not None and review_actual < minimum:
            if status == "ready":
                status = "review_required"
    return ModelInputProjection(
        values=values,
        status=status,
        completeness=completeness,
        available_feature_count=available,
        missing_features=tuple(missing),
        issues=tuple(issues),
        group_completeness=group_completeness,
    )


def feature_group(name: str) -> str:
    if name.startswith("agent_"):
        return "agent"
    if name.startswith("offering_") or name in {
        "offer_price_hkd",
        "listing_market_cap_hkd",
        "gross_proceeds_hkd",
        "net_proceeds_hkd",
        "listing_expense_ratio",
        "net_proceeds_ratio",
        "market_cap_to_net_proceeds",
    }:
        return "offering"
    if name.startswith("fundamental_") or name.startswith("cash_"):
        return "fundamental"
    if name.startswith("market_"):
        return "market"
    if name.startswith("recent_ipo_"):
        return "recent_ipo"
    if name.startswith("document_") or name.startswith("calendar_"):
        return "document"
    if name.startswith("data_quality_") or name.startswith("missing_"):
        return "data_quality"
    if name.startswith("engineered_"):
        return "engineered"
    return "other"


def feature_unit(name: str) -> str:
    if name == "fundamental_reporting_currency":
        return "currency_code"
    if name.startswith("agent_semantic_has_") or name.startswith("agent_has_"):
        return "boolean_0_1"
    if name == "agent_severity_weighted_mean" or name.endswith("_max_severity"):
        return "score_1_4"
    if name.endswith("_pct") or name.endswith("_pct_points"):
        return "percent"
    if (
        name.endswith("_ratio")
        or "_ratio_" in name
        or name.endswith("_share")
        or "confidence" in name
    ):
        return "ratio"
    if name in {
        "offering_net_proceeds_hkd",
        "offering_market_cap_low_hkd",
        "offering_market_cap_high_hkd",
        "offering_market_cap_mid_hkd",
        "offering_gross_proceeds_low_hkd",
        "offering_gross_proceeds_high_hkd",
        "offering_gross_proceeds_mid_hkd",
    }:
        return "HKD"
    if name.endswith("_hkd_mn"):
        return "HKD_million"
    if name.endswith("_native") or name.endswith("_native_mn"):
        return "reporting_currency_million"
    if name.endswith("_hkd") or name == "offer_price_hkd":
        return "HKD"
    if name.endswith("_months"):
        return "month"
    if name.endswith("_count") or name.endswith("_years_3y"):
        return "count"
    return "scalar"


def feature_formula_version(name: str) -> str:
    if name in {"agent_severity_weighted_mean", "agent_role_market_share"}:
        return "stable_agent_model_v1"
    if name.startswith("agent_semantic_"):
        return "risk_semantics_v1"
    if name.startswith("agent_"):
        return "agent_features_v1"
    if name.startswith("offering_") or name.startswith("fundamental_"):
        return "structured_fundamentals_v1"
    if name.startswith("market_"):
        return "listing_pre_market_v1"
    if name.startswith("recent_ipo_"):
        return "recent_ipo_context_v1"
    if name.startswith("engineered_"):
        return "deterministic_derived_v1"
    if name.startswith("data_quality_"):
        return "data_quality_v1"
    if name.startswith("document_") or name.startswith("calendar_"):
        return "document_features_v1"
    return "prelisting_base_v1"


def _document_features(
    report: RiskAnalysisReport,
    *,
    company_info: Mapping[str, object],
    available_at: date,
    tables_path: Path | None,
) -> dict[str, FeatureValue]:
    return _document_features_from_values(
        source_ref=report.source_document.source_file,
        page_count=report.source_document.page_count,
        chunk_count=report.source_document.chunk_count,
        company_info=company_info,
        available_at=available_at,
        tables_path=tables_path,
    )


def _document_features_from_values(
    *,
    source_ref: str,
    page_count: int,
    chunk_count: int,
    company_info: Mapping[str, object],
    available_at: date,
    tables_path: Path | None,
) -> dict[str, FeatureValue]:
    competition_dataset = company_info.get("competition_dataset")
    offering_description = str(
        competition_dataset.get("offering_description", "")
        if isinstance(competition_dataset, Mapping)
        else ""
    )
    month_angle = 2.0 * math.pi * (available_at.month - 1) / 12.0
    table_count: int | None = None
    if tables_path is not None and tables_path.is_file():
        with tables_path.open("r", encoding="utf-8") as handle:
            table_count = sum(1 for line in handle if line.strip())
    values: dict[str, Any] = {
        "document_page_count": page_count,
        "document_chunk_count": chunk_count,
        "document_table_count": table_count,
        "document_is_global_offering": int("全球" in offering_description),
        "calendar_prospectus_year": available_at.year,
        "calendar_prospectus_month_sin": round(math.sin(month_angle), 8),
        "calendar_prospectus_month_cos": round(math.cos(month_angle), 8),
    }
    return {
        name: FeatureValue(value=value, available_at=available_at, source_ref=source_ref)
        for name, value in values.items()
    }


def _add_derived_candidates(
    features: dict[str, FeatureValue], *, prediction_as_of: date
) -> None:
    raw = {name: feature.value for name, feature in features.items()}
    derived = add_derived_prediction_features(pd.DataFrame([raw])).iloc[0].to_dict()
    for name, value in derived.items():
        if name in features:
            continue
        features[name] = FeatureValue(
            value=_json_scalar(value),
            available_at=prediction_as_of,
            source_ref="derived:stable_agent_model_v1",
        )


def _add_quality_candidates(
    features: dict[str, FeatureValue], *, prediction_as_of: date
) -> None:
    grouped: dict[str, list[bool]] = {}
    for name, feature in features.items():
        grouped.setdefault(feature_group(name), []).append(feature.value is not None)
    available = sum(feature.value is not None for feature in features.values())
    total = len(features)
    values: dict[str, Any] = {
        "data_quality_candidate_feature_count": total,
        "data_quality_available_feature_count": available,
        "data_quality_feature_completeness": available / total if total else 0.0,
    }
    for group, flags in grouped.items():
        values[f"data_quality_group_{group}_completeness"] = (
            sum(flags) / len(flags) if flags else 0.0
        )
    for name, value in values.items():
        features[name] = FeatureValue(
            value=_json_scalar(value),
            available_at=prediction_as_of,
            source_ref=f"{EXTRACTOR_VERSION}:quality",
        )


def _merge(
    destination: dict[str, FeatureValue], incoming: Mapping[str, FeatureValue]
) -> None:
    for name, feature in incoming.items():
        existing = destination.get(name)
        if existing is None:
            destination[name] = feature
            continue
        if existing.value != feature.value:
            raise ValueError(f"conflicting candidate feature values for {name}")


def _merge_supplemental(
    destination: dict[str, FeatureValue],
    incoming: Mapping[str, FeatureValue],
    warnings: list[str],
) -> None:
    """RAG只补足当前文档缺失值；表格与公式提取结果始终优先。"""
    for name, feature in incoming.items():
        if feature.value is None:
            continue
        existing = destination.get(name)
        if existing is None or existing.value is None:
            destination[name] = feature
            continue
        if existing.value != feature.value:
            warnings.append(f"rag_conflict_kept_deterministic:{name}")


def _value_and_date(raw: Any) -> tuple[Any, date | None]:
    if isinstance(raw, FeatureValue):
        return raw.value, raw.available_at
    if isinstance(raw, Mapping) and "value" in raw:
        available_at = raw.get("available_at")
        parsed = None
        if isinstance(available_at, date):
            parsed = available_at
        elif isinstance(available_at, str):
            try:
                parsed = date.fromisoformat(available_at)
            except ValueError:
                parsed = None
        return raw.get("value"), parsed
    return raw, None


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if pd.isna(value):
        return None
    return value


__all__ = [
    "CANDIDATE_SCHEMA_VERSION",
    "CONTRACT_SCHEMA_VERSION",
    "CandidateFeatureSnapshot",
    "ModelFeatureContract",
    "ModelFeatureField",
    "ModelInputProjection",
    "collect_prelisting_candidates",
    "feature_group",
    "feature_formula_version",
    "feature_unit",
    "load_model_feature_contract",
    "project_model_inputs",
]
