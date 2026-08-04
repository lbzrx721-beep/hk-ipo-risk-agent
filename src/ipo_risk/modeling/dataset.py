"""赛事数据的 Point-in-Time 特征矩阵构建。

预测时点固定为招股书发布日期。上市日期、发行价代理和上市后行情只用于标签，
不会进入特征列。每个特征同时写出可用日期和来源，便于自动检查数据泄漏。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from ipo_risk.domain import FeatureValue
from ipo_risk.storage.files import write_json, write_jsonl


DEFAULT_MARKET_WINDOWS = (5, 20, 60)
IDENTITY_COLUMNS = (
    "company_id",
    "ticker",
    "company_name",
    "prospectus_date",
    "as_of_date",
    "split",
)
TARGET_COLUMNS = (
    "target_significant_drop_5d",
    "target_return_vs_offer_5d_pct",
    "target_observation_date",
    "label_offer_price_source",
)


@dataclass(frozen=True)
class ModelingDatasetResult:
    """批量特征构建产物。"""

    feature_matrix_path: Path
    feature_audit_path: Path
    daily_market_path: Path
    summary_path: Path
    row_count: int
    feature_columns: tuple[str, ...]


def build_daily_market_sentiment(
    eod_path: Path,
    output_path: Path,
    *,
    temp_directory: Path | None = None,
    force: bool = False,
) -> Path:
    """把 1GB 级全市场行情压缩为每日情绪指标。

    DuckDB 直接流式扫描赛事 CSV，避免把四百多万行同时载入 Python 内存。
    """
    source = eod_path.resolve()
    destination = output_path.resolve()
    if destination.is_file() and not force:
        return destination
    if not source.is_file():
        raise FileNotFoundError(f"全市场行情不存在：{source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = (temp_directory or destination.parent / ".duckdb_tmp").resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    escaped_source = source.as_posix().replace("'", "''")
    escaped_scratch = scratch.as_posix().replace("'", "''")
    query = f"""
        WITH source_rows AS (
            SELECT
                TRY_STRPTIME(TRADE_DT, '%Y%m%d')::DATE AS trade_date,
                S_INFO_WINDCODE AS wind_code,
                TRY_CAST(S_DQ_HIGH AS DOUBLE) AS high_price,
                TRY_CAST(S_DQ_LOW AS DOUBLE) AS low_price,
                TRY_CAST(S_DQ_CLOSE AS DOUBLE) AS close_price,
                TRY_CAST(S_DQ_PRECLOSE AS DOUBLE) AS preclose_price,
                TRY_CAST(S_DQ_AMOUNT AS DOUBLE) * 1000.0 AS turnover_hkd
            FROM read_csv_auto(
                '{escaped_source}',
                header = true,
                all_varchar = true,
                ignore_errors = true
            )
        ),
        valid_rows AS (
            SELECT
                trade_date,
                wind_code,
                100.0 * (close_price / preclose_price - 1.0) AS return_pct,
                100.0 * (high_price - low_price) / preclose_price AS amplitude_pct,
                turnover_hkd
            FROM source_rows
            WHERE trade_date IS NOT NULL
              AND close_price > 0
              AND preclose_price > 0
        )
        SELECT
            trade_date,
            MEDIAN(return_pct) AS median_return_pct,
            AVG(CASE WHEN return_pct > 0 THEN 1.0 ELSE 0.0 END) AS advance_ratio,
            MEDIAN(amplitude_pct) AS median_amplitude_pct,
            SUM(COALESCE(turnover_hkd, 0.0)) AS market_turnover_hkd,
            MEDIAN(COALESCE(turnover_hkd, 0.0)) AS median_security_turnover_hkd,
            COUNT(DISTINCT wind_code) AS active_securities
        FROM valid_rows
        GROUP BY trade_date
        ORDER BY trade_date
    """
    connection = duckdb.connect(database=":memory:")
    try:
        connection.execute(f"SET temp_directory = '{escaped_scratch}'")
        daily = connection.execute(query).fetch_df()
    finally:
        connection.close()
    daily.to_csv(destination, index=False, encoding="utf-8")
    return destination


def build_competition_modeling_dataset(
    project_root: Path,
    *,
    manifest_path: Path | None = None,
    labels_path: Path | None = None,
    eod_path: Path | None = None,
    output_dir: Path | None = None,
    market_windows: tuple[int, ...] = DEFAULT_MARKET_WINDOWS,
    recent_ipo_lookback_days: int = 180,
    force_market_rebuild: bool = False,
) -> ModelingDatasetResult:
    """为有五日标签的公司生成上市前特征、目标和逐特征审计记录。"""
    if not market_windows or any(window < 2 for window in market_windows):
        raise ValueError("market_windows must contain values greater than one")
    if recent_ipo_lookback_days < 30:
        raise ValueError("recent_ipo_lookback_days must be at least 30")

    root = project_root.resolve()
    competition_processed = root / "data" / "processed" / "competition"
    delivery = root / "data" / "raw" / "competition" / "dongwu_hk_ipo_2020_2025"
    manifest = (manifest_path or competition_processed / "company_manifest.jsonl").resolve()
    labels = (labels_path or competition_processed / "market_labels.csv").resolve()
    eod = (eod_path or delivery / "hkshareeodprices.csv").resolve()
    destination = (output_dir or root / "data" / "processed" / "modeling").resolve()
    destination.mkdir(parents=True, exist_ok=True)

    manifest_rows = _read_jsonl(manifest)
    label_frame = pd.read_csv(labels, dtype={"company_id": str})
    prepared_labels = _prepare_labels(label_frame)
    target_labels = prepared_labels.loc[
        (prepared_labels["horizon"] == 5)
        & (prepared_labels["label_status"] == "complete")
        & prepared_labels["significant_drop"].notna()
    ].copy()
    target_by_company = {
        str(row["company_id"]).zfill(5): row for _, row in target_labels.iterrows()
    }

    daily_market_path = build_daily_market_sentiment(
        eod,
        destination / "daily_market_sentiment.csv",
        temp_directory=root / "tmp" / "modeling_duckdb",
        force=force_market_rebuild,
    )
    daily_market = pd.read_csv(daily_market_path)
    daily_market["trade_date"] = pd.to_datetime(daily_market["trade_date"]).dt.date

    matrix_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    all_feature_names: set[str] = set()
    for manifest_row in manifest_rows:
        company_id = str(manifest_row.get("company_id") or "").zfill(5)
        target = target_by_company.get(company_id)
        prospectus_date = _date_value(manifest_row.get("prospectus_date"))
        if target is None or prospectus_date is None:
            continue
        as_of_date = prospectus_date
        features: dict[str, FeatureValue] = {}
        features.update(_document_features(root, manifest_row, as_of_date))
        features.update(
            _market_features(
                daily_market,
                as_of_date,
                windows=market_windows,
                source_ref=daily_market_path.relative_to(root).as_posix(),
            )
        )
        features.update(
            _recent_ipo_features(
                prepared_labels,
                as_of_date,
                lookback_days=recent_ipo_lookback_days,
                source_ref=labels.relative_to(root).as_posix(),
            )
        )
        _assert_point_in_time(features, as_of_date=as_of_date, company_id=company_id)
        all_feature_names.update(features)

        row: dict[str, Any] = {
            "company_id": company_id,
            "ticker": manifest_row.get("ticker"),
            "company_name": manifest_row.get("company_name"),
            "prospectus_date": as_of_date.isoformat(),
            "as_of_date": as_of_date.isoformat(),
            "split": manifest_row.get("split"),
            "target_significant_drop_5d": int(bool(target["significant_drop"])),
            "target_return_vs_offer_5d_pct": _optional_float(
                target["return_vs_offer_pct"]
            ),
            "target_observation_date": _iso_date(target["observation_date"]),
            "label_offer_price_source": target.get("offer_price_source"),
        }
        row.update({name: feature.value for name, feature in features.items()})
        matrix_rows.append(row)
        audit_rows.append(
            {
                "company_id": company_id,
                "as_of_date": as_of_date.isoformat(),
                "features": {
                    name: feature.model_dump(mode="json")
                    for name, feature in sorted(features.items())
                },
                "target": {
                    "name": "significant_drop_5d",
                    "value": int(bool(target["significant_drop"])),
                    "available_at": _iso_date(target["observation_date"]),
                    "source_ref": labels.relative_to(root).as_posix(),
                    "prediction_input_eligible": False,
                },
            }
        )

    if not matrix_rows:
        raise ValueError("没有找到同时具备招股书日期和完整五日标签的公司")
    feature_columns = tuple(sorted(all_feature_names))
    ordered_columns = [*IDENTITY_COLUMNS, *TARGET_COLUMNS, *feature_columns]
    matrix = pd.DataFrame(matrix_rows).reindex(columns=ordered_columns)
    matrix.sort_values(["prospectus_date", "company_id"], inplace=True)
    feature_matrix_path = destination / "prelisting_feature_matrix.csv"
    matrix.to_csv(feature_matrix_path, index=False, encoding="utf-8-sig")

    audit_by_company = {row["company_id"]: row for row in audit_rows}
    ordered_audits = [
        audit_by_company[company_id] for company_id in matrix["company_id"].astype(str)
    ]
    feature_audit_path = destination / "prelisting_feature_audit.jsonl"
    write_jsonl(ordered_audits, feature_audit_path)

    split_counts = {
        str(key): int(value)
        for key, value in matrix["split"].value_counts(dropna=False).items()
    }
    positive_counts = {
        str(key): int(value)
        for key, value in matrix.groupby("split")[
            "target_significant_drop_5d"
        ].sum().items()
    }
    summary = {
        "schema_version": 1,
        "prediction_as_of": "prospectus_date",
        "row_count": len(matrix),
        "feature_count": len(feature_columns),
        "feature_columns": list(feature_columns),
        "identity_columns": list(IDENTITY_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "split_counts": split_counts,
        "significant_drop_positive_counts": positive_counts,
        "market_windows": list(market_windows),
        "recent_ipo_lookback_days": recent_ipo_lookback_days,
        "leakage_controls": {
            "company_post_listing_market_used_as_feature": False,
            "offer_price_proxy_used_as_feature": False,
            "listing_date_used_as_feature": False,
            "feature_availability_checked": True,
            "target_columns_excluded_from_features": True,
        },
        "sources": {
            "manifest": manifest.relative_to(root).as_posix(),
            "labels": labels.relative_to(root).as_posix(),
            "full_market_eod": eod.relative_to(root).as_posix(),
            "daily_market_cache": daily_market_path.relative_to(root).as_posix(),
        },
    }
    summary_path = destination / "feature_matrix_summary.json"
    write_json(summary, summary_path)
    return ModelingDatasetResult(
        feature_matrix_path=feature_matrix_path,
        feature_audit_path=feature_audit_path,
        daily_market_path=daily_market_path,
        summary_path=summary_path,
        row_count=len(matrix),
        feature_columns=feature_columns,
    )


def _document_features(
    project_root: Path,
    manifest_row: dict[str, Any],
    as_of_date: date,
) -> dict[str, FeatureValue]:
    source_ref = str(manifest_row.get("prospectus_file") or "company_manifest")
    relative_file = manifest_row.get("prospectus_file")
    file_size_mb: float | None = None
    if relative_file:
        source_path = project_root / str(relative_file)
        if source_path.is_file():
            file_size_mb = round(source_path.stat().st_size / (1024 * 1024), 6)
    month_angle = 2.0 * math.pi * (as_of_date.month - 1) / 12.0
    offering = str(manifest_row.get("offering_description") or "未提供").strip()
    values: dict[str, Any] = {
        "document_page_count": _optional_float(manifest_row.get("page_count")),
        "document_file_size_mb": file_size_mb,
        "document_offering_description": offering,
        "document_is_global_offering": int("全球" in offering),
        "calendar_prospectus_year": as_of_date.year,
        "calendar_prospectus_month_sin": round(math.sin(month_angle), 8),
        "calendar_prospectus_month_cos": round(math.cos(month_angle), 8),
    }
    return {
        name: FeatureValue(
            value=value,
            available_at=as_of_date,
            source_ref=source_ref,
        )
        for name, value in values.items()
    }


def _market_features(
    daily_market: pd.DataFrame,
    as_of_date: date,
    *,
    windows: tuple[int, ...],
    source_ref: str,
) -> dict[str, FeatureValue]:
    eligible = daily_market.loc[daily_market["trade_date"] < as_of_date]
    features: dict[str, FeatureValue] = {}
    for window in windows:
        sample = eligible.tail(window)
        available_at = (
            max(sample["trade_date"]) if not sample.empty else as_of_date
        )
        daily_returns = pd.to_numeric(sample["median_return_pct"], errors="coerce")
        compounded = (
            (float(np.prod(1.0 + daily_returns.dropna().to_numpy() / 100.0)) - 1.0)
            * 100.0
            if daily_returns.notna().any()
            else None
        )
        volatility = (
            float(daily_returns.std(ddof=0) * math.sqrt(252))
            if daily_returns.notna().sum() >= 2
            else None
        )
        half = max(1, len(sample) // 2)
        earlier_turnover = pd.to_numeric(
            sample.iloc[:-half]["market_turnover_hkd"], errors="coerce"
        ).mean()
        recent_turnover = pd.to_numeric(
            sample.iloc[-half:]["market_turnover_hkd"], errors="coerce"
        ).mean()
        turnover_trend = (
            float(recent_turnover / earlier_turnover)
            if pd.notna(earlier_turnover)
            and pd.notna(recent_turnover)
            and float(earlier_turnover) > 0
            else None
        )
        values: dict[str, Any] = {
            f"market_return_{window}d_pct": _finite_or_none(compounded),
            f"market_median_daily_return_{window}d_pct": _series_mean(daily_returns),
            f"market_advance_ratio_{window}d": _series_mean(sample["advance_ratio"]),
            f"market_volatility_{window}d_pct": _finite_or_none(volatility),
            f"market_amplitude_{window}d_pct": _series_mean(
                sample["median_amplitude_pct"]
            ),
            f"market_turnover_trend_{window}d": _finite_or_none(turnover_trend),
            f"market_active_securities_{window}d": _series_mean(
                sample["active_securities"]
            ),
        }
        lineage = (
            f"{source_ref}#trade_date<{as_of_date.isoformat()};"
            f"last_{window}_trading_days"
        )
        for name, value in values.items():
            features[name] = FeatureValue(
                value=value,
                available_at=available_at,
                source_ref=lineage,
            )
    return features


def _recent_ipo_features(
    labels: pd.DataFrame,
    as_of_date: date,
    *,
    lookback_days: int,
    source_ref: str,
) -> dict[str, FeatureValue]:
    observed = labels.loc[
        labels["observation_date"].notna()
        & (labels["observation_date"] < as_of_date)
        & labels["listing_date"].notna()
        & (labels["listing_date"] < as_of_date)
    ]
    horizon_5 = observed.loc[
        (observed["horizon"] == 5)
        & (observed["label_status"] == "complete")
        & observed["return_vs_offer_pct"].notna()
    ]
    horizon_1 = observed.loc[
        (observed["horizon"] == 1)
        & (observed["label_status"] == "complete")
        & observed["return_vs_offer_pct"].notna()
    ]
    recent_5 = horizon_5.loc[
        horizon_5["listing_date"] >= as_of_date - timedelta(days=lookback_days)
    ]
    recent_1 = horizon_1.loc[
        horizon_1["listing_date"] >= as_of_date - timedelta(days=90)
    ]
    count_30 = int(
        horizon_1.loc[
            horizon_1["listing_date"] >= as_of_date - timedelta(days=30)
        ]["company_id"].nunique()
    )
    count_90 = int(recent_1["company_id"].nunique())
    values: dict[str, Any] = {
        "recent_ipo_count_30d": count_30,
        "recent_ipo_count_90d": count_90,
        f"recent_ipo_5d_observed_count_{lookback_days}d": int(
            recent_5["company_id"].nunique()
        ),
        f"recent_ipo_5d_mean_return_{lookback_days}d_pct": _series_mean(
            recent_5["return_vs_offer_pct"]
        ),
        f"recent_ipo_5d_drop_rate_{lookback_days}d": _series_mean(
            recent_5["significant_drop"].astype(float)
        ),
        "recent_ipo_1d_mean_return_90d_pct": _series_mean(
            recent_1["return_vs_offer_pct"]
        ),
    }
    selected = pd.concat([recent_5, recent_1], ignore_index=True)
    available_at = (
        max(selected["observation_date"]) if not selected.empty else as_of_date
    )
    lineage = f"{source_ref}#observation_date<{as_of_date.isoformat()}"
    return {
        name: FeatureValue(
            value=value,
            available_at=available_at,
            source_ref=lineage,
        )
        for name, value in values.items()
    }


def _prepare_labels(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["company_id"] = prepared["company_id"].astype(str).str.zfill(5)
    for column in ("prospectus_date", "listing_date", "observation_date"):
        prepared[column] = pd.to_datetime(prepared[column], errors="coerce").dt.date
    prepared["horizon"] = pd.to_numeric(prepared["horizon"], errors="coerce")
    prepared["return_vs_offer_pct"] = pd.to_numeric(
        prepared["return_vs_offer_pct"], errors="coerce"
    )
    prepared["significant_drop"] = prepared["significant_drop"].map(
        _optional_bool
    )
    return prepared


def _assert_point_in_time(
    features: dict[str, FeatureValue],
    *,
    as_of_date: date,
    company_id: str,
) -> None:
    future = sorted(
        name for name, feature in features.items() if feature.available_at > as_of_date
    )
    if future:
        raise ValueError(
            f"{company_id} contains features available after {as_of_date}: "
            + ", ".join(future)
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _date_value(value: object) -> date | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _iso_date(value: object) -> str | None:
    parsed = _date_value(value)
    return parsed.isoformat() if parsed else None


def _optional_bool(value: object) -> bool | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def _optional_float(value: object) -> float | None:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _finite_or_none(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def _series_mean(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce")
    if not numeric.notna().any():
        return None
    result = float(numeric.mean())
    return result if math.isfinite(result) else None


__all__ = [
    "DEFAULT_MARKET_WINDOWS",
    "IDENTITY_COLUMNS",
    "TARGET_COLUMNS",
    "ModelingDatasetResult",
    "build_competition_modeling_dataset",
    "build_daily_market_sentiment",
]
