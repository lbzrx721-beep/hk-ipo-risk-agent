"""同行估值与发行估值比对。

从结构化公司信息（发行价、市值、融资额）和日线行情（上市后表现）中
提取估值相关特征，计算发行溢价/折价率。

所有函数均为纯函数，不调用 LLM，可单元测试。
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

from ipo_risk.domain import FeatureValue
from ipo_risk.skills._market_data import (
    market_anomaly_dates,
    require_positive_window,
    sorted_market_rows,
)


def extract_valuation_features(
    company_info: dict[str, Any],
    *,
    market_data: dict[str, Any] | None = None,
    window_days: int = 5,
) -> dict[str, FeatureValue]:
    """从公司信息和市场数据中提取估值特征。

    Parameters
    ----------
    company_info:
        结构化公司信息 JSON（data/raw/company_info/…/公司信息.json）。
    market_data:
        可选的日线行情 JSON，用于计算上市后溢价/折价。
    window_days:
        上市后观察窗口（交易日数），默认 5。

    Returns
    -------
    dict[str, FeatureValue]
        特征名 → FeatureValue。缺失字段优雅跳过。
    """
    require_positive_window(window_days)
    features: dict[str, FeatureValue] = {}

    as_of = _parse_date(company_info.get("as_of_date"))
    if as_of is None:
        return features

    offering = company_info.get("offering", {})
    if not isinstance(offering, dict):
        offering = {}

    # ---- 发行估值指标 ----
    offer_price = _safe_float(offering.get("offer_price_hkd"))
    market_cap = _safe_float(offering.get("listing_market_cap_hkd"))
    gross_proceeds = _safe_float(offering.get("gross_proceeds_hkd"))
    net_proceeds = _safe_float(offering.get("net_proceeds_hkd"))
    listing_expenses = _safe_float(offering.get("estimated_listing_expenses_hkd"))
    if offer_price is not None:
        features["offer_price_hkd"] = FeatureValue(
            value=offer_price,
            available_at=as_of,
            source_ref="company_info:offering.offer_price_hkd",
        )

    if market_cap is not None:
        features["listing_market_cap_hkd"] = FeatureValue(
            value=market_cap,
            available_at=as_of,
            source_ref="company_info:offering.listing_market_cap_hkd",
        )

    if gross_proceeds is not None:
        features["gross_proceeds_hkd"] = FeatureValue(
            value=gross_proceeds,
            available_at=as_of,
            source_ref="company_info:offering.gross_proceeds_hkd",
        )

    if net_proceeds is not None:
        features["net_proceeds_hkd"] = FeatureValue(
            value=net_proceeds,
            available_at=as_of,
            source_ref="company_info:offering.net_proceeds_hkd",
        )

    # 发行费用率 = 上市费用 / 募资总额
    if listing_expenses is not None and gross_proceeds and gross_proceeds > 0:
        expense_ratio = listing_expenses / gross_proceeds
        features["listing_expense_ratio"] = FeatureValue(
            value=round(expense_ratio, 6),
            available_at=as_of,
            source_ref="company_info:offering.listing_expense_ratio",
        )

    # 净募资率 = 净募资 / 总募资
    if net_proceeds is not None and gross_proceeds and gross_proceeds > 0:
        net_ratio = net_proceeds / gross_proceeds
        features["net_proceeds_ratio"] = FeatureValue(
            value=round(net_ratio, 6),
            available_at=as_of,
            source_ref="company_info:offering.net_proceeds_ratio",
        )

    # 发行市值 / 净募资 = 估值倍数代理
    if market_cap is not None and net_proceeds and net_proceeds > 0:
        cap_to_proceeds = market_cap / net_proceeds
        features["market_cap_to_net_proceeds"] = FeatureValue(
            value=round(cap_to_proceeds, 4),
            available_at=as_of,
            source_ref="company_info:offering.market_cap_to_net_proceeds",
        )

    # ---- 上市后溢价/折价（需要行情数据） ----
    if market_data and offer_price is not None and offer_price > 0:
        _extract_post_listing_valuation(
            features, market_data, offer_price, window_days,
        )

    return features


def _extract_post_listing_valuation(
    features: dict[str, FeatureValue],
    market_data: dict[str, Any],
    offer_price: float,
    window_days: int,
) -> None:
    """从行情数据计算上市后估值溢价/折价。"""
    sorted_rows = sorted_market_rows(market_data)
    anomaly_dates = market_anomaly_dates(market_data)
    if not sorted_rows:
        return
    window = sorted_rows[:window_days]
    if not window:
        return

    first_date = _parse_date(window[0].get("date"))
    window_end_date = _parse_date(window[-1].get("date"))
    if first_date is None or window_end_date is None:
        return

    # 首日收盘溢价
    first_close = _safe_float(window[0].get("close_hkd"))
    first_is_valid = str(window[0].get("date", "")) not in anomaly_dates
    if first_is_valid and first_close is not None:
        first_day_premium = (first_close / offer_price - 1) * 100
        features["first_day_premium_pct"] = FeatureValue(
            value=round(first_day_premium, 4),
            available_at=first_date,
            source_ref="market:first_day:premium_vs_offer",
        )

    if len(window) < window_days or any(
        str(row.get("date", "")) in anomaly_dates for row in window
    ):
        return

    # 窗口期最高溢价
    highs = [_safe_float(r.get("high_hkd")) for r in window]
    valid_highs = [h for h in highs if h is not None]
    if valid_highs:
        max_premium = (max(valid_highs) / offer_price - 1) * 100
        features[f"first_{window_days}d_max_premium_pct"] = FeatureValue(
            value=round(max_premium, 4),
            available_at=window_end_date,
            source_ref=f"market:first_{window_days}d:max_premium_vs_offer",
        )

    # 窗口期最低折价
    lows = [_safe_float(r.get("low_hkd")) for r in window]
    valid_lows = [lo for lo in lows if lo is not None]
    if valid_lows:
        min_discount = (min(valid_lows) / offer_price - 1) * 100
        features[f"first_{window_days}d_min_discount_pct"] = FeatureValue(
            value=round(min_discount, 4),
            available_at=window_end_date,
            source_ref=f"market:first_{window_days}d:min_discount_vs_offer",
        )

    # 窗口期末收盘溢价
    last_close = _safe_float(window[-1].get("close_hkd"))
    if last_close is not None:
        window_end_premium = (last_close / offer_price - 1) * 100
        features[f"first_{window_days}d_close_premium_pct"] = FeatureValue(
            value=round(window_end_premium, 4),
            available_at=window_end_date,
            source_ref=f"market:first_{window_days}d:close_premium_vs_offer",
        )


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _safe_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        if math.isfinite(result):
            return result
    return None


def _parse_date(value: object) -> date | None:
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            return None
    if isinstance(value, date):
        return value
    return None


__all__ = ["extract_valuation_features"]
