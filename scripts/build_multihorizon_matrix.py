"""把 1/20/60 日显著下跌标签合并进 v5_compatible 矩阵，生成多窗口建模矩阵。

输出：data/processed/modeling/agent_augmented_feature_matrix_multihorizon.csv
（在 v5_compatible 基础上新增 target_significant_drop_1d / 20d / 60d）
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ipo_risk.modeling.labels import PRIMARY_TARGET  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "data" / "processed" / "modeling" / "agent_augmented_feature_matrix_v5_compatible.csv"
LABELS = ROOT / "data" / "processed" / "competition" / "market_labels.csv"
OUT = ROOT / "data" / "processed" / "modeling" / "agent_augmented_feature_matrix_multihorizon.csv"

frame = pd.read_csv(MATRIX, dtype={"company_id": str})
frame["company_id"] = frame["company_id"].astype(str).str.zfill(5)
print(f"矩阵: {len(frame)} 家")

labels = pd.read_csv(LABELS, dtype={"company_id": str})
labels["company_id"] = labels["company_id"].astype(str).str.zfill(5)
labels["sd"] = labels["significant_drop"].map({True: 1, False: 0})
labels = labels[labels["label_status"].eq("complete") & labels["sd"].notna()]

for horizon in (1, 20, 60):
    target_column = f"target_significant_drop_{horizon}d"
    if target_column in frame:
        frame = frame.drop(columns=[target_column])
    sub = labels.loc[labels["horizon"] == horizon, ["company_id", "sd"]].copy()
    sub = sub.rename(columns={"sd": target_column})
    frame = frame.merge(sub, on="company_id", how="left")

# 校验
print(f"合并后: {len(frame)} 家, {len(frame.columns)} 列")
for h in (1, 5, 20, 60):
    col = PRIMARY_TARGET if h == 5 else f"target_significant_drop_{h}d"
    print(f"  {col}: 非空 {frame[col].notna().mean():.1%}")
frame.to_csv(OUT, index=False, encoding="utf-8-sig")
print(f"已写出: {OUT}")
