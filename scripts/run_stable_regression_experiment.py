"""新建模范式实验：回归目标 + 跨期稳定性筛选 + LightGBM 原生单调约束。

与 optimize_high_risk_recall_model.py 使用完全相同的评估协议（3 个开发折 +
2025-12-01 硬留出），模型选择只看开发折，留出集仅用于最终评估，保证与
既有实验公平对比。本脚本不覆盖任何线上模型。

五个建模臂（arm）：
  A. clf_baseline      分类 + CORE_STABLE_FEATURES（现有最好简单模型，锚点）
  B. clf_stable        分类 + 稳定性筛选特征
  C. clf_stable_mono   分类 + 稳定性筛选特征 + 单调约束
  D. reg_stable        回归（winsorized 5 日收益率）+ 稳定性筛选特征
  E. reg_stable_mono   回归 + 稳定性筛选特征 + 单调约束（本实验主角）

运行（必须显式使用项目环境，PATH 默认 python 没有 lightgbm）：
  D:/AnacondaDate/envs/hk-ipo-risk/python.exe scripts/run_stable_regression_experiment.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
import typer
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from optimize_high_risk_recall_model import (  # noqa: E402
    CORE_STABLE_FEATURES,
    DEFAULT_HOLDOUT_START,
    ISSUANCE_SEVERITY_FEATURE,
    TARGET,
    _choose_recall_threshold,
    _current_model_holdout_metrics,
    _load_combined_frame,
    _metrics,
    _numeric_frame,
    _percentile_score,
    _rank_capacity_metrics,
    _sample_weights,
)
from normalize_fundamental_currency import (  # noqa: E402
    _MANUAL_CURRENCY_BY_COMPANY,
    _VERIFIED_NATIVE_DIVISORS,
    _load_audited_native_values,
    _load_ecb_rates,
    _rate_to_hkd,
    _to_million,
)
from ipo_risk.modeling.stable_agent_model import (  # noqa: E402
    AGENT_SEMANTIC_FEATURES,
    BUSINESS_BASE_FEATURES,
    add_derived_prediction_features,
)

REG_TARGET = "regression_return_5d_pct"
SEEDS = (42, 137, 271)

# 与生产模型同族的强正则 LightGBM 配置（小样本防过拟合）。
_LGB_COMMON: dict[str, Any] = {
    "n_estimators": 220,
    "learning_rate": 0.025,
    "num_leaves": 7,
    "max_depth": 3,
    "min_child_samples": 30,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "n_jobs": 1,
    "verbosity": -1,
    "deterministic": True,
    "force_row_wise": True,
}


# ---------------------------------------------------------------------------
# 阶段 1：数据装配
# ---------------------------------------------------------------------------


def _repair_currency_and_hkd_mn(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    """为缺失 hkd_mn 列的行（2025 实验矩阵）补齐汇率与百万港元换算。

    复用 normalize_fundamental_currency.py 的完整单位识别逻辑（审计值 +
    市值参照），保证与主矩阵既有 hkd_mn 值口径一致。只填缺失行，不重算
    主矩阵已有值。
    """
    result = frame.copy()
    normalized = (
        "fundamental_fx_to_hkd",
        "fundamental_revenue_latest_hkd_mn",
        "fundamental_operating_cash_flow_latest_hkd_mn",
        "fundamental_operating_cash_flow_sum_3y_hkd_mn",
    )
    for column in normalized:
        if column not in result:
            result[column] = np.nan
        result[column] = pd.to_numeric(result[column], errors="coerce")

    needs = result["fundamental_revenue_latest_hkd_mn"].isna() & result[
        "fundamental_revenue_latest_native"
    ].notna()
    if not needs.any():
        return result

    rates = _load_ecb_rates(root / "tmp" / "fx_rates", 2020, 2025)
    audited = _load_audited_native_values(
        root
        / "data"
        / "processed"
        / "modeling"
        / "structured_fundamental_feature_audit.jsonl"
    )

    # 主矩阵 CSV 没有持久化 fx 列，但 add_derived_prediction_features 重算
    # engineered_cash_runway_months 时要用 fx；先为所有缺 fx 的行补齐。
    fx_missing = result["fundamental_fx_to_hkd"].isna()
    for idx in result.index[fx_missing]:
        row = result.loc[idx]
        company_id = str(row["company_id"]).zfill(5)
        currency = _MANUAL_CURRENCY_BY_COMPANY.get(
            company_id,
            str(row.get("fundamental_reporting_currency") or "").strip().upper(),
        )
        if not currency:
            continue
        try:
            _, rate = _rate_to_hkd(currency, row["prospectus_date"].date(), rates)
        except (ValueError, KeyError, AttributeError):
            continue
        result.at[idx, "fundamental_fx_to_hkd"] = rate

    def convert(
        row: pd.Series,
        company_id: str,
        field: str,
        rate: float,
        **kwargs: Any,
    ) -> float:
        value = row.get(field)
        if value is None or pd.isna(value) or not str(value).strip():
            return np.nan
        # _to_million 内部用 `value or ""` 判空，数值 0 会被误判为空；
        # 统一转成字符串（对齐原脚本从 CSV 读入的输入形态）再传入。
        million_text = _to_million(
            str(value),
            audited.get((company_id, field)),
            field=field,
            reference_hkd_mn=row.get("offering_market_cap_mid_hkd"),
            rate=rate,
            forced_divisor=_VERIFIED_NATIVE_DIVISORS.get((company_id, field)),
            **kwargs,
        )
        if not str(million_text).strip():
            return np.nan
        return float(million_text) * rate

    def clean_related(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        return str(value)

    repaired = 0
    for idx in result.index[needs]:
        row = result.loc[idx]
        company_id = str(row["company_id"]).zfill(5)
        currency = _MANUAL_CURRENCY_BY_COMPANY.get(
            company_id,
            str(row.get("fundamental_reporting_currency") or "").strip().upper(),
        )
        if not currency:
            continue
        rate = result.at[idx, "fundamental_fx_to_hkd"]
        if pd.isna(rate):
            continue
        revenue_native = row.get("fundamental_revenue_latest_native")
        result.at[idx, "fundamental_revenue_latest_hkd_mn"] = convert(
            row, company_id, "fundamental_revenue_latest_native", rate
        )
        result.at[idx, "fundamental_operating_cash_flow_latest_hkd_mn"] = convert(
            row,
            company_id,
            "fundamental_operating_cash_flow_latest_native",
            rate,
            related_value=clean_related(revenue_native),
            related_audited_value=audited.get(
                (company_id, "fundamental_revenue_latest_native")
            ),
            expected_ratio_pct=row.get(
                "fundamental_operating_cash_flow_to_revenue_latest_pct"
            ),
        )
        result.at[idx, "fundamental_operating_cash_flow_sum_3y_hkd_mn"] = convert(
            row,
            company_id,
            "fundamental_operating_cash_flow_sum_3y_native",
            rate,
            related_value=clean_related(revenue_native),
            related_audited_value=audited.get(
                (company_id, "fundamental_revenue_latest_native")
            ),
        )
        repaired += 1
    print(f"汇率与 hkd_mn 补齐：{repaired} 行（2025 实验矩阵缺口修复）", flush=True)
    return result


def _attach_regression_target(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    """从 market_labels.csv join 5 日连续收益率作为回归目标。"""
    labels = pd.read_csv(
        root / "data" / "processed" / "competition" / "market_labels.csv",
        dtype={"company_id": str},
    )
    labels["company_id"] = labels["company_id"].astype(str).str.zfill(5)
    sub = labels.loc[
        labels["horizon"].eq(5) & labels["label_status"].eq("complete"),
        ["company_id", "return_vs_offer_pct"],
    ].rename(columns={"return_vs_offer_pct": REG_TARGET})
    result = frame.merge(sub, on="company_id", how="left")
    coverage = result[REG_TARGET].notna().mean()
    assert coverage >= 0.98, f"回归目标覆盖率过低：{coverage:.1%}"
    result = result.dropna(subset=[REG_TARGET]).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# 阶段 2：特征稳定性筛选
# ---------------------------------------------------------------------------


def _year_blocks(frame: pd.DataFrame, holdout_start: pd.Timestamp) -> dict[str, pd.DataFrame]:
    return {
        "2020": frame.loc[frame["modeling_year"].eq(2020)],
        "2021": frame.loc[frame["modeling_year"].eq(2021)],
        "2022": frame.loc[frame["modeling_year"].eq(2022)],
        "2023": frame.loc[frame["modeling_year"].eq(2023)],
        "2024": frame.loc[frame["modeling_year"].eq(2024)],
        "2025_h1": frame.loc[
            frame["prospectus_date"].between("2025-01-01", "2025-07-01", inclusive="left")
        ],
        "2025_h2": frame.loc[
            frame["prospectus_date"].between("2025-07-01", holdout_start, inclusive="left")
        ],
    }


def _block_auc(rows: pd.DataFrame, feature: str) -> float | None:
    values = pd.to_numeric(rows[feature], errors="coerce")
    labels = rows[TARGET]
    valid = values.notna() & labels.notna()
    if valid.sum() < 20 or values.loc[valid].nunique() < 2:
        return None
    present = labels.loc[valid].astype(int)
    if present.nunique() < 2 or present.sum() < 3:
        return None
    return float(roc_auc_score(present, values.loc[valid]))


def _screen_stable_features(
    frame: pd.DataFrame,
    *,
    holdout_start: pd.Timestamp,
    min_blocks: int = 4,
    min_agreement: float = 0.80,
    min_strength: float = 0.53,
) -> tuple[list[str], dict[str, int], pd.DataFrame]:
    """只用开发期（留出日之前）数据做跨期方向稳定性筛选，冻结方向。"""
    development = frame.loc[frame["prospectus_date"].lt(holdout_start)]
    historical = development.loc[development["modeling_year"].le(2024)]
    recent = development.loc[development["modeling_year"].eq(2025)]

    pool = []
    for feature in (*BUSINESS_BASE_FEATURES, *AGENT_SEMANTIC_FEATURES):
        if feature not in frame.columns or feature == ISSUANCE_SEVERITY_FEATURE:
            continue
        if pd.to_numeric(historical[feature], errors="coerce").notna().sum() < 100:
            continue
        if pd.to_numeric(recent[feature], errors="coerce").notna().sum() < 30:
            continue
        pool.append(feature)

    blocks = _year_blocks(development, holdout_start)
    rows: list[dict[str, Any]] = []
    for feature in pool:
        aucs = {
            name: _block_auc(block, feature) for name, block in blocks.items()
        }
        valid = {name: auc for name, auc in aucs.items() if auc is not None}
        if len(valid) < min_blocks:
            rows.append({"feature": feature, **aucs, "verdict": "insufficient_blocks"})
            continue
        signs = [np.sign(auc - 0.5) for auc in valid.values()]
        majority = 1.0 if np.mean(signs) >= 0 else -1.0
        agreement = float(np.mean([sign == majority for sign in signs]))
        strength = float(np.mean([abs(auc - 0.5) + 0.5 for auc in valid.values()]))
        verdict = (
            "stable"
            if agreement >= min_agreement and strength >= min_strength
            else "unstable_or_weak"
        )
        rows.append(
            {
                "feature": feature,
                **aucs,
                "direction_agreement": agreement,
                "mean_auc_strength": strength,
                "verdict": verdict,
                "frozen_direction": "up" if majority > 0 else "down",
            }
        )
    table = pd.DataFrame(rows)
    stable = table.loc[table["verdict"].eq("stable"), "feature"].tolist()
    directions = {
        row["feature"]: (1 if row["frozen_direction"] == "up" else -1)
        for _, row in table.loc[table["verdict"].eq("stable")].iterrows()
    }
    return stable, directions, table


# ---------------------------------------------------------------------------
# 阶段 3：五个建模臂
# ---------------------------------------------------------------------------


def _build_classifier(seed: int, monotone: list[int] | None) -> Pipeline:
    params = dict(_LGB_COMMON, class_weight="balanced")
    if monotone is not None:
        params["monotone_constraints"] = monotone
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", lgb.LGBMClassifier(objective="binary", random_state=seed, **params)),
        ]
    )


def _build_regressor(seed: int, monotone: list[int] | None) -> Pipeline:
    params = dict(_LGB_COMMON)
    if monotone is not None:
        params["monotone_constraints"] = monotone
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", lgb.LGBMRegressor(objective="regression", random_state=seed, **params)),
        ]
    )


def _dev_folds(
    frame: pd.DataFrame, holdout_start: pd.Timestamp
) -> list[tuple[pd.DataFrame, pd.DataFrame, str]]:
    """与 optimize_high_risk_recall_model._development_predictions 相同的 3 折。"""
    return [
        (
            frame.loc[frame["modeling_year"].between(2020, 2023)],
            frame.loc[frame["modeling_year"].eq(2024)],
            "2024",
        ),
        (
            frame.loc[frame["modeling_year"].between(2020, 2024)],
            frame.loc[
                frame["prospectus_date"].between(
                    "2025-01-01", "2025-07-01", inclusive="left"
                )
            ],
            "2025_h1",
        ),
        (
            frame.loc[frame["prospectus_date"].lt("2025-07-01")],
            frame.loc[
                frame["prospectus_date"].between(
                    "2025-07-01", holdout_start, inclusive="left"
                )
            ],
            "2025_h2",
        ),
    ]


def _fit_predict_classifier(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: tuple[str, ...],
    monotone: list[int] | None,
) -> np.ndarray:
    weights = _sample_weights(
        train, scheme="half_life_1y", as_of=validation["prospectus_date"].min()
    )
    train_matrix = _numeric_frame(train, features)
    validation_matrix = _numeric_frame(validation, features)
    probabilities: list[np.ndarray] = []
    for seed in SEEDS:
        model = _build_classifier(seed, monotone)
        model.fit(
            train_matrix,
            train[TARGET].astype(int),
            model__sample_weight=weights,
        )
        probability = model.predict_proba(validation_matrix)[:, 1]
        reference = model.predict_proba(train_matrix)[:, 1]
        probabilities.append(_percentile_score(reference, probability))
    return np.mean(probabilities, axis=0)


def _fit_predict_regressor(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: tuple[str, ...],
    monotone: list[int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """回归臂：winsorized 收益率训练，风险分 = 1 - percentile(预测收益率)。"""
    y_train = train[REG_TARGET].astype(float)
    lower, upper = np.nanpercentile(y_train, (2.0, 98.0))
    y_clipped = y_train.clip(lower, upper)
    weights = _sample_weights(
        train, scheme="half_life_1y", as_of=validation["prospectus_date"].min()
    )
    train_matrix = _numeric_frame(train, features)
    validation_matrix = _numeric_frame(validation, features)
    risks: list[np.ndarray] = []
    raw_predictions: list[np.ndarray] = []
    for seed in SEEDS:
        model = _build_regressor(seed, monotone)
        model.fit(train_matrix, y_clipped, model__sample_weight=weights)
        predicted = model.predict(validation_matrix)
        reference = model.predict(train_matrix)
        risks.append(1.0 - _percentile_score(reference, predicted))
        raw_predictions.append(predicted)
    return np.mean(risks, axis=0), np.mean(raw_predictions, axis=0)


@dataclass
class ArmResult:
    name: str
    kind: str
    features: tuple[str, ...]
    monotone: list[int] | None
    dev_probability: np.ndarray
    dev_labels: np.ndarray
    threshold: float
    dev_metrics: dict[str, Any]
    selection_score: float
    final_models: list[dict[str, Any]]
    clip_bounds: tuple[float, float] | None


def _train_arm(
    name: str,
    kind: str,
    frame: pd.DataFrame,
    features: tuple[str, ...],
    directions: dict[str, int],
    holdout_start: pd.Timestamp,
) -> ArmResult:
    monotone: list[int] | None = None
    if name.endswith("_mono"):
        # 分类臂：风险方向 up=+1；回归臂预测收益率，风险方向要取反。
        sign = 1 if kind == "clf" else -1
        monotone = [sign * directions[feature] for feature in features]

    all_probability: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    fold_metrics: list[dict[str, Any]] = []
    for train, validation, fold_name in _dev_folds(frame, holdout_start):
        assert train["modeling_year"].max() <= validation["modeling_year"].min(), (
            f"折 {fold_name} 训练集年份不早于验证集"
        )
        if kind == "clf":
            probability = _fit_predict_classifier(
                train, validation, features, monotone
            )
        else:
            probability, _ = _fit_predict_regressor(
                train, validation, features, monotone
            )
        labels = validation[TARGET].astype(int).to_numpy()
        all_probability.append(probability)
        all_labels.append(labels)
        fold_metrics.append(
            {
                "fold": fold_name,
                "sample_count": int(len(labels)),
                "positive_count": int(labels.sum()),
                "roc_auc": float(roc_auc_score(labels, probability)),
                "pr_auc": float(average_precision_score(labels, probability)),
                "rank_recall_at_40pct": _rank_capacity_metrics(
                    labels, probability, 0.40
                )["recall"],
            }
        )
    dev_probability = np.concatenate(all_probability)
    dev_labels = np.concatenate(all_labels)
    threshold = _choose_recall_threshold(dev_labels, dev_probability)
    dev_metrics = _metrics(dev_labels, dev_probability, threshold)
    dev_metrics["rank_recall_at_40pct"] = _rank_capacity_metrics(
        dev_labels, dev_probability, 0.40
    )["recall"]
    dev_metrics["worst_fold_pr_auc"] = min(fold["pr_auc"] for fold in fold_metrics)
    dev_metrics["folds"] = fold_metrics
    selection_score = (dev_metrics["worst_fold_pr_auc"] + dev_metrics["pr_auc"]) / 2.0

    # 用全部开发期数据训练最终模型（用于留出集 / 2025 全年评估）。
    train_final = frame.loc[frame["prospectus_date"].lt(holdout_start)]
    weights = _sample_weights(train_final, scheme="half_life_1y", as_of=holdout_start)
    final_matrix = _numeric_frame(train_final, features)
    final_models: list[dict[str, Any]] = []
    clip_bounds: tuple[float, float] | None = None
    if kind == "clf":
        for seed in SEEDS:
            model = _build_classifier(seed, monotone)
            model.fit(
                final_matrix, train_final[TARGET].astype(int),
                model__sample_weight=weights,
            )
            reference = np.sort(model.predict_proba(final_matrix)[:, 1])
            final_models.append({"model": model, "reference": reference})
    else:
        y_final = train_final[REG_TARGET].astype(float)
        lower, upper = np.nanpercentile(y_final, (2.0, 98.0))
        clip_bounds = (float(lower), float(upper))
        for seed in SEEDS:
            model = _build_regressor(seed, monotone)
            model.fit(
                final_matrix, y_final.clip(lower, upper),
                model__sample_weight=weights,
            )
            reference = np.sort(model.predict(final_matrix))
            final_models.append({"model": model, "reference": reference})

    return ArmResult(
        name=name,
        kind=kind,
        features=features,
        monotone=monotone,
        dev_probability=dev_probability,
        dev_labels=dev_labels,
        threshold=threshold,
        dev_metrics=dev_metrics,
        selection_score=selection_score,
        final_models=final_models,
        clip_bounds=clip_bounds,
    )


def _predict_arm(
    arm: ArmResult, rows: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray | None]:
    matrix = _numeric_frame(rows, arm.features)
    scores: list[np.ndarray] = []
    raw: list[np.ndarray] = []
    for payload in arm.final_models:
        model = payload["model"]
        if arm.kind == "clf":
            probability = model.predict_proba(matrix)[:, 1]
            scores.append(_percentile_score(payload["reference"], probability))
        else:
            predicted = model.predict(matrix)
            raw.append(predicted)
            scores.append(1.0 - _percentile_score(payload["reference"], predicted))
    return np.mean(scores, axis=0), (np.mean(raw, axis=0) if raw else None)


# ---------------------------------------------------------------------------
# 阶段 4：比赛口径评估
# ---------------------------------------------------------------------------


def _lift_table(labels: np.ndarray, risk: np.ndarray) -> list[dict[str, Any]]:
    """风险分层 lift 表：高档（风险最高 20%）/ 中档（30%）/ 低档（50%）。"""
    order = np.argsort(-risk)
    n = len(labels)
    base_rate = float(labels.mean())
    rows: list[dict[str, Any]] = []
    boundaries = (("high_top20pct", 0.0, 0.20), ("mid_30pct", 0.20, 0.50), ("low_50pct", 0.50, 1.0))
    for name, start_frac, end_frac in boundaries:
        segment = labels[order[int(n * start_frac): int(n * end_frac)]]
        rate = float(segment.mean()) if len(segment) else 0.0
        rows.append(
            {
                "tier": name,
                "sample_count": int(len(segment)),
                "positive_count": int(segment.sum()),
                "actual_drop_rate": rate,
                "lift_vs_base": float(rate / base_rate) if base_rate > 0 else None,
            }
        )
    return rows


def _evaluate_cohort(
    arm: ArmResult,
    rows: pd.DataFrame,
    production: pd.DataFrame | None,
) -> dict[str, Any]:
    risk, raw_return = _predict_arm(arm, rows)
    labels = rows[TARGET].astype(int).to_numpy()
    result: dict[str, Any] = {
        "cohort_size": int(len(rows)),
        "metrics": _metrics(labels, risk, arm.threshold),
        "rank_capacity": {
            str(fraction): _rank_capacity_metrics(labels, risk, fraction)
            for fraction in (0.20, 0.30, 0.40, 0.50, 0.60)
        },
        "lift_table": _lift_table(labels, risk),
    }
    if raw_return is not None:
        actual = rows[REG_TARGET].astype(float).to_numpy()
        result["spearman_ic"] = float(spearmanr(raw_return, actual).statistic)
    if production is not None:
        joined = rows[["company_id", TARGET]].merge(
            production[["company_id", "probability", "threshold"]],
            on="company_id",
            how="inner",
        )
        if not joined.empty:
            result["production_model_same_cohort"] = _metrics(
                joined[TARGET].astype(int).to_numpy(),
                joined["probability"].to_numpy(),
                float(joined["threshold"].iloc[0]),
            )
            result["production_model_cohort_size"] = int(len(joined))
    return result


def _summary_markdown(
    arms: list[ArmResult],
    holdout_results: dict[str, dict[str, Any]],
    oot_2025_results: dict[str, dict[str, Any]],
    stable_features: list[str],
    stability_table: pd.DataFrame,
) -> str:
    lines = [
        "# 新建模范式实验报告：回归目标 + 稳定性筛选 + 单调约束",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 稳定特征（{len(stable_features)} 个）：{', '.join(stable_features)}",
        "",
        "## 开发折对比（模型选择只看这里）",
        "",
        "| 臂 | PR-AUC | 最差折PR-AUC | ROC-AUC | 召回@阈值 | 预警比例 | 召回@40%容量 | 选型分 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm in arms:
        m = arm.dev_metrics
        lines.append(
            f"| {arm.name} | {m['pr_auc']:.3f} | {m['worst_fold_pr_auc']:.3f} "
            f"| {m['roc_auc']:.3f} | {m['recall']:.3f} | {m['alert_fraction']:.2f} "
            f"| {m['rank_recall_at_40pct']:.3f} | {arm.selection_score:.3f} |"
        )
    lines += ["", "## 留出集（2025-12-01 起，仅最终评估）", ""]
    for arm in arms:
        hold = holdout_results[arm.name]
        m = hold["metrics"]
        ic = f", IC={hold['spearman_ic']:.3f}" if "spearman_ic" in hold else ""
        lines.append(
            f"- **{arm.name}**：PR-AUC {m['pr_auc']:.3f}，ROC-AUC {m['roc_auc']:.3f}，"
            f"召回 {m['recall']:.2f}，精度 {m['precision']:.2f}，"
            f"预警 {m['alert_count']}/{m['sample_count']}{ic}"
        )
        for tier in hold["lift_table"]:
            lift = tier["lift_vs_base"]
            lines.append(
                f"  - {tier['tier']}：实际下跌率 {tier['actual_drop_rate']:.1%}"
                + (f"，lift {lift:.2f}" if lift is not None else "")
            )
        prod = hold.get("production_model_same_cohort")
        if prod:
            lines.append(
                f"  - 生产模型同批样本：PR-AUC {prod['pr_auc']:.3f}，"
                f"ROC-AUC {prod['roc_auc']:.3f}，召回 {prod['recall']:.2f}"
            )
    lines += ["", "## 2025 全年 OOT", ""]
    for arm in arms:
        oot = oot_2025_results[arm.name]
        m = oot["metrics"]
        lines.append(
            f"- **{arm.name}**：PR-AUC {m['pr_auc']:.3f}，ROC-AUC {m['roc_auc']:.3f}"
        )
        prod = oot.get("production_model_same_cohort")
        if prod:
            lines.append(
                f"  - 生产模型同批样本：PR-AUC {prod['pr_auc']:.3f}，ROC-AUC {prod['roc_auc']:.3f}"
            )
    lines += [
        "",
        "## 已知限制",
        "",
        "- 留出集仅 25 家（8 正例），AUC 差异 ±0.1 以内视为噪声，选型权重在开发折。",
        "- 2025 年数据曾被此前多轮实验观察，存在自适应过拟合；本实验筛选与选型均未使用留出期数据。",
        "",
        "## 稳定性筛选明细（前 20）",
        "",
        "| 特征 | 判定 | 冻结方向 | 方向一致率 | 平均强度 |",
        "|---|---|---|---|---|",
    ]
    top = stability_table.loc[
        stability_table["verdict"].eq("stable")
    ].sort_values("mean_auc_strength", ascending=False)
    for _, row in top.head(20).iterrows():
        lines.append(
            f"| {row['feature']} | {row['verdict']} | {row['frozen_direction']} "
            f"| {row['direction_agreement']:.2f} | {row['mean_auc_strength']:.3f} |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main(
    output_dir: Path = Path("artifacts/experiments/stable_regression_v1"),
    holdout_start: str = "2025-12-01",
) -> None:
    root = ROOT
    destination = output_dir if output_dir.is_absolute() else root / output_dir
    destination.mkdir(parents=True, exist_ok=True)
    holdout_timestamp = pd.Timestamp(holdout_start)

    print("阶段 1：数据装配", flush=True)
    frame = _load_combined_frame(root)
    frame = _repair_currency_and_hkd_mn(frame, root)
    frame = add_derived_prediction_features(frame)
    frame = _attach_regression_target(frame, root)
    assert 540 <= len(frame) <= 560, f"样本数异常：{len(frame)}（预期约 545）"
    print(f"样本：{len(frame)} 家，正例率 {frame[TARGET].mean():.1%}", flush=True)

    print("阶段 2：特征稳定性筛选（仅开发期数据）", flush=True)
    stable_features, directions, stability_table = _screen_stable_features(
        frame, holdout_start=holdout_timestamp
    )
    stability_table.to_csv(
        destination / "feature_stability.csv", index=False, encoding="utf-8-sig"
    )
    print(
        f"稳定特征 {len(stable_features)} 个：" + ", ".join(stable_features),
        flush=True,
    )
    if len(stable_features) < 5:
        raise ValueError(
            f"稳定特征过少（{len(stable_features)} 个），请检查筛选阈值或数据装配"
        )

    print("阶段 3：五个建模臂训练与开发折评估", flush=True)
    stable_tuple = tuple(stable_features)
    arm_specs = (
        ("A_clf_baseline", "clf", tuple(CORE_STABLE_FEATURES)),
        ("B_clf_stable", "clf", stable_tuple),
        ("C_clf_stable_mono", "clf", stable_tuple),
        ("D_reg_stable", "reg", stable_tuple),
        ("E_reg_stable_mono", "reg", stable_tuple),
    )
    arms: list[ArmResult] = []
    for name, kind, features in arm_specs:
        arm = _train_arm(name, kind, frame, features, directions, holdout_timestamp)
        arms.append(arm)
        m = arm.dev_metrics
        print(
            f"  {name}: dev PR-AUC={m['pr_auc']:.3f} "
            f"worst={m['worst_fold_pr_auc']:.3f} ROC={m['roc_auc']:.3f} "
            f"recall={m['recall']:.3f} alert={m['alert_fraction']:.2f} "
            f"选型分={arm.selection_score:.3f}",
            flush=True,
        )
        if arm.monotone is not None:
            expected = [directions[f] for f in features]
            actual = [sign * arm.monotone[i] for i, sign in enumerate([1 if kind == "clf" else -1] * len(features))]
            assert actual == expected, f"{name} 单调约束方向与冻结方向不一致"

    print("阶段 4：留出集与 2025 全年 OOT 评估", flush=True)
    production_path = (
        root
        / "artifacts"
        / "experiments"
        / "agent_2025_full_validation_20260826"
        / "predictions.csv"
    )
    production = None
    if production_path.exists():
        production = pd.read_csv(production_path, dtype={"company_id": str})
        production["company_id"] = production["company_id"].astype(str).str.zfill(5)

    holdout = frame.loc[frame["prospectus_date"].ge(holdout_timestamp)]
    oot_2025 = frame.loc[frame["modeling_year"].eq(2025)]
    holdout_results: dict[str, dict[str, Any]] = {}
    oot_2025_results: dict[str, dict[str, Any]] = {}
    prediction_frames: list[pd.DataFrame] = []
    lift_rows: list[dict[str, Any]] = []
    for arm in arms:
        holdout_results[arm.name] = _evaluate_cohort(arm, holdout, production)
        oot_2025_results[arm.name] = _evaluate_cohort(arm, oot_2025, production)
        risk, _ = _predict_arm(arm, holdout)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "company_id": holdout["company_id"].to_numpy(),
                    "arm": arm.name,
                    "risk_score": risk,
                    "alert": (risk >= arm.threshold).astype(int),
                    TARGET: holdout[TARGET].to_numpy(),
                    REG_TARGET: holdout[REG_TARGET].to_numpy(),
                }
            )
        )
        for tier in holdout_results[arm.name]["lift_table"]:
            lift_rows.append({"arm": arm.name, "cohort": "holdout", **tier})
        for tier in oot_2025_results[arm.name]["lift_table"]:
            lift_rows.append({"arm": arm.name, "cohort": "oot_2025", **tier})
        m = holdout_results[arm.name]["metrics"]
        print(
            f"  留出集 {arm.name}: PR-AUC={m['pr_auc']:.3f} ROC={m['roc_auc']:.3f} "
            f"recall={m['recall']:.2f} precision={m['precision']:.2f}",
            flush=True,
        )

    pd.concat(prediction_frames, ignore_index=True).to_csv(
        destination / "predictions_holdout.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(lift_rows).to_csv(
        destination / "lift_tables.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        [
            {
                "arm": arm.name,
                "kind": arm.kind,
                "feature_count": len(arm.features),
                "monotone": arm.monotone is not None,
                "threshold": arm.threshold,
                "selection_score": arm.selection_score,
                **{
                    key: value
                    for key, value in arm.dev_metrics.items()
                    if key != "folds"
                },
            }
            for arm in arms
        ]
    ).sort_values("selection_score", ascending=False).to_csv(
        destination / "candidate_metrics.csv", index=False, encoding="utf-8-sig"
    )

    best = max(arms, key=lambda arm: arm.selection_score)
    payload = {
        "schema_version": 1,
        "experiment": "stable_regression_paradigm",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target": TARGET,
        "regression_target": REG_TARGET,
        "holdout_policy": {
            "development_end_exclusive": holdout_timestamp.date().isoformat(),
            "test_start": holdout_timestamp.date().isoformat(),
            "test_used_for_candidate_selection": False,
        },
        "cohort_counts": {
            "all": int(len(frame)),
            "development": int(frame["prospectus_date"].lt(holdout_timestamp).sum()),
            "holdout": int(len(holdout)),
            "holdout_positive": int(holdout[TARGET].sum()),
            "oot_2025": int(len(oot_2025)),
        },
        "stable_features": stable_features,
        "frozen_directions": directions,
        "arms": {
            arm.name: {
                "kind": arm.kind,
                "features": list(arm.features),
                "monotone_constraints": arm.monotone,
                "threshold": arm.threshold,
                "selection_score": arm.selection_score,
                "development_metrics": arm.dev_metrics,
                "holdout": holdout_results[arm.name],
                "oot_2025": oot_2025_results[arm.name],
            }
            for arm in arms
        },
        "selected_by_development_folds": best.name,
        "current_model_holdout": _current_model_holdout_metrics(root, holdout),
    }
    (destination / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (destination / "summary.md").write_text(
        _summary_markdown(
            arms, holdout_results, oot_2025_results, stable_features, stability_table
        ),
        encoding="utf-8",
    )
    print(f"\n开发折选型最佳：{best.name}（选型分 {best.selection_score:.3f}）", flush=True)
    print(f"结果目录：{destination}", flush=True)


if __name__ == "__main__":
    typer.run(main)
