"""时间切分约束与验证。"""

from __future__ import annotations

import pandas as pd


SPLIT_ORDER = ("train", "validation", "test")


def validate_time_ordered_splits(frame: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """校验公司唯一性以及训练、验证、测试的时间边界互不倒置。"""
    required = {"company_id", "prospectus_date", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"feature matrix missing columns: {', '.join(sorted(missing))}")
    duplicates = frame["company_id"].astype(str).duplicated(keep=False)
    if duplicates.any():
        companies = sorted(frame.loc[duplicates, "company_id"].astype(str).unique())
        raise ValueError(f"companies occur in multiple rows: {', '.join(companies[:10])}")

    dated = frame.copy()
    dated["prospectus_date"] = pd.to_datetime(
        dated["prospectus_date"], errors="raise"
    )
    unknown = sorted(set(dated["split"].dropna().astype(str)).difference(SPLIT_ORDER))
    if unknown:
        raise ValueError(f"unknown split values: {', '.join(unknown)}")

    boundaries: dict[str, tuple[str, str]] = {}
    previous_max: pd.Timestamp | None = None
    for split in SPLIT_ORDER:
        rows = dated.loc[dated["split"] == split]
        if rows.empty:
            raise ValueError(f"time split is empty: {split}")
        current_min = rows["prospectus_date"].min()
        current_max = rows["prospectus_date"].max()
        if previous_max is not None and current_min <= previous_max:
            raise ValueError(
                f"time split overlaps: {split} begins {current_min.date()} "
                f"before previous split ends {previous_max.date()}"
            )
        boundaries[split] = (
            current_min.date().isoformat(),
            current_max.date().isoformat(),
        )
        previous_max = current_max
    return boundaries


__all__ = ["SPLIT_ORDER", "validate_time_ordered_splits"]
