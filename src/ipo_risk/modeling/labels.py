"""预测目标定义。"""

from __future__ import annotations

import pandas as pd


PRIMARY_TARGET = "target_significant_drop_5d"


def validate_primary_target(frame: pd.DataFrame) -> None:
    if PRIMARY_TARGET not in frame:
        raise ValueError(f"feature matrix missing target: {PRIMARY_TARGET}")
    values = set(
        pd.to_numeric(frame[PRIMARY_TARGET], errors="raise").astype(int).unique()
    )
    if not values.issubset({0, 1}) or len(values) < 2:
        raise ValueError("primary target must contain both binary classes")


__all__ = ["PRIMARY_TARGET", "validate_primary_target"]
