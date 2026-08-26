"""基于当前统一矩阵重建 v5 兼容建模矩阵。

基础：data/processed/modeling/agent_augmented_feature_matrix.csv
（先由建模数据构建、Agent汇总和币种归一化流程刷新）
重建：add_derived_prediction_features 生成 engineered_*/agent_role_*_share/agent_interaction_* 派生层
输出：data/processed/modeling/agent_augmented_feature_matrix_v5_compatible.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ipo_risk.modeling.stable_agent_model import (  # noqa: E402
    AGENT_SEMANTIC_FEATURES,
    BUSINESS_BASE_FEATURES,
    add_derived_prediction_features,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = (
    ROOT
    / "data"
    / "processed"
    / "modeling"
    / "agent_augmented_feature_matrix.csv"
)
OUT = ROOT / "data" / "processed" / "modeling" / "agent_augmented_feature_matrix_v5_compatible.csv"
NORMALIZED_FALLBACK = (
    ROOT
    / "data"
    / "processed"
    / "modeling"
    / "agent_augmented_feature_matrix_rebuilt.csv"
)
NORMALIZED_FACT_COLUMNS = (
    "fundamental_reporting_currency",
    "fundamental_fx_to_hkd",
    "fundamental_fx_rate_date",
    "fundamental_revenue_latest_hkd_mn",
    "fundamental_operating_cash_flow_latest_hkd_mn",
    "fundamental_operating_cash_flow_sum_3y_hkd_mn",
    "offering_gross_proceeds_low_hkd",
    "offering_gross_proceeds_high_hkd",
    "offering_gross_proceeds_mid_hkd",
    "offering_net_proceeds_hkd",
    "offering_market_cap_low_hkd",
    "offering_market_cap_high_hkd",
    "offering_market_cap_mid_hkd",
)

frame = pd.read_csv(SRC, dtype={"company_id": str})
frame["company_id"] = frame["company_id"].astype(str).str.zfill(5)
print(f"读入: {len(frame)} 行, {len(frame.columns)} 列")

# 已核验的币种换算不因预测时点改变。当前抽取表若缺币种，复用上次审计后的
# 港元金额快照；只更新本次需要重算的上市日前市场特征。
if NORMALIZED_FALLBACK.is_file():
    normalized = pd.read_csv(
        NORMALIZED_FALLBACK,
        dtype={"company_id": str},
        low_memory=False,
    )
    normalized["company_id"] = normalized["company_id"].astype(str).str.zfill(5)
    available = [
        column for column in NORMALIZED_FACT_COLUMNS if column in normalized.columns
    ]
    normalized = normalized[["company_id", *available]].drop_duplicates("company_id")
    frame = frame.drop(columns=[column for column in available if column in frame])
    frame = frame.merge(normalized, on="company_id", how="left", validate="one_to_one")
    print(f"复用已核验港元金额: {len(available)} 列")

# 派生层重建
frame = add_derived_prediction_features(frame)
print(f"派生后: {len(frame.columns)} 列")

# 补齐原始就为空的特征列（agent_cash_runway_months 在历史数据中非空率即 0）
for feature in AGENT_SEMANTIC_FEATURES:
    if feature not in frame.columns:
        frame[feature] = pd.NA
        print(f"补齐空列: {feature}")

# 校验白名单
for name, whitelist in (("base", BUSINESS_BASE_FEATURES), ("agent", AGENT_SEMANTIC_FEATURES)):
    missing = sorted(set(whitelist).difference(frame.columns))
    print(f"{name} 白名单 {len(whitelist)} 个, 缺失 {len(missing)} 个")
    if missing:
        print(f"  仍缺失: {missing}")

frame.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"\n已写出: {OUT}")
