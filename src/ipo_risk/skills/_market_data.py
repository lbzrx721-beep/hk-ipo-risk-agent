"""行情 skill 共享的数据质量过滤。"""

from __future__ import annotations

from typing import Any


def sorted_market_rows(market_data: dict[str, Any]) -> list[dict[str, Any]]:
    """保留原始交易日序列并按日期排序。"""
    rows = market_data.get("rows", [])
    if not isinstance(rows, list):
        return []
    valid = [
        row
        for row in rows
        if isinstance(row, dict)
        and str(row.get("date", "")).strip()
    ]
    return sorted(valid, key=lambda row: str(row["date"]))


def market_anomaly_dates(market_data: dict[str, Any]) -> set[str]:
    """返回被数据源明确标记为不可靠的交易日期。"""
    data_quality = market_data.get("data_quality", {})
    values = data_quality.get("anomaly_dates", []) if isinstance(data_quality, dict) else []
    return {
        str(value).strip()
        for value in values
        if isinstance(value, str) and value.strip()
    }


def require_positive_window(window_days: int) -> None:
    if window_days < 1:
        raise ValueError("window_days must be positive")


__all__ = ["market_anomaly_dates", "require_positive_window", "sorted_market_rows"]
