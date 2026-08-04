"""上市前指数、板块、流动性和同期新股特征。

从日线行情 JSON（data/raw/market/…/日线行情.json）和结构化公司信息中
提取上市后窗口期的市场特征，供建模层消费。

注意：日线行情标记为 post_listing_label_and_backtest_only，
仅用于标签计算和回测，不作为上市前预测特征。
本模块提取的是上市后市场表现特征（用于标签和回测验证）。

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


def extract_market_features(
    market_data: dict[str, Any],
    *,
    company_info: dict[str, Any] | None = None,
    window_days: int = 5,
) -> dict[str, FeatureValue]:
    """从日线行情和公司信息中提取市场窗口特征。

    Parameters
    ----------
    market_data:
        日线行情 JSON（含 rows 数组，每行有 date/open_hkd/close_hkd 等）。
    company_info:
        可选的结构化公司信息 JSON。
    window_days:
        上市后观察窗口（交易日数），默认 5。

    Returns
    -------
    dict[str, FeatureValue]
        特征名 → FeatureValue。数据不足时优雅跳过。
    """
    require_positive_window(window_days)
    features: dict[str, FeatureValue] = {}

    sorted_rows = sorted_market_rows(market_data)
    anomaly_dates = market_anomaly_dates(market_data)
    if not sorted_rows:
        return features

    # ---- 首日特征 ----
    first = sorted_rows[0]
    first_date = _parse_date(first.get("date"))
    if first_date is None:
        return features

    first_change = _safe_float(first.get("change_pct"))
    first_turnover = _safe_float(first.get("turnover_hkd"))
    first_turnover_rate = _safe_float(first.get("turnover_rate_pct"))
    first_amplitude = _safe_float(first.get("amplitude_pct"))
    first_volume = _safe_float(first.get("volume_shares"))

    first_is_valid = str(first.get("date", "")) not in anomaly_dates
    if first_is_valid and first_change is not None:
        features["first_day_return_pct"] = FeatureValue(
            value=first_change,
            available_at=first_date,
            source_ref="market:first_day:change_pct",
        )
    if first_is_valid and first_turnover is not None:
        features["first_day_turnover_hkd"] = FeatureValue(
            value=first_turnover,
            available_at=first_date,
            source_ref="market:first_day:turnover_hkd",
        )
    if first_is_valid and first_turnover_rate is not None:
        features["first_day_turnover_rate_pct"] = FeatureValue(
            value=first_turnover_rate,
            available_at=first_date,
            source_ref="market:first_day:turnover_rate_pct",
        )
    if first_is_valid and first_amplitude is not None:
        features["first_day_amplitude_pct"] = FeatureValue(
            value=first_amplitude,
            available_at=first_date,
            source_ref="market:first_day:amplitude_pct",
        )
    if first_is_valid and first_volume is not None:
        features["first_day_volume_shares"] = FeatureValue(
            value=first_volume,
            available_at=first_date,
            source_ref="market:first_day:volume_shares",
        )

    # ---- 窗口期特征（前 N 个交易日） ----
    window = sorted_rows[:window_days]
    window_complete = len(window) == window_days and not any(
        str(row.get("date", "")) in anomaly_dates for row in window
    )
    window_end_date = _parse_date(window[-1].get("date")) or first_date

    # 窗口期是否破发（需要发行价）—— 只需 ≥1 行即可判断
    offer_price = _get_offer_price(company_info)
    lows = [_safe_float(r.get("low_hkd")) for r in window]
    valid_lows = [lo for lo in lows if lo is not None]
    highs = [_safe_float(r.get("high_hkd")) for r in window]
    valid_highs = [h for h in highs if h is not None]

    if window_complete and offer_price is not None and offer_price > 0 and valid_lows:
        broke_issue = min(valid_lows) < offer_price
        features[f"first_{window_days}d_broke_issue_price"] = FeatureValue(
            value=broke_issue,
            available_at=window_end_date,
            source_ref=f"market:first_{window_days}d:broke_issue",
        )
        # 最大折价率
        max_discount = (min(valid_lows) / offer_price - 1) * 100
        features[f"first_{window_days}d_max_discount_pct"] = FeatureValue(
            value=round(max_discount, 4),
            available_at=window_end_date,
            source_ref=f"market:first_{window_days}d:max_discount",
        )
        # 最大溢价率
        if valid_highs:
            max_premium = (max(valid_highs) / offer_price - 1) * 100
            features[f"first_{window_days}d_max_premium_pct"] = FeatureValue(
                value=round(max_premium, 4),
                available_at=window_end_date,
                source_ref=f"market:first_{window_days}d:max_premium",
            )

    if window_complete and len(window) >= 2:
        # 累计收益率
        closes = [
            _safe_float(r.get("close_hkd")) for r in window
        ]
        valid_closes = [c for c in closes if c is not None and c > 0]
        if len(valid_closes) >= 2:
            cum_return = (valid_closes[-1] / valid_closes[0] - 1) * 100
            features[f"first_{window_days}d_cumulative_return_pct"] = FeatureValue(
                value=round(cum_return, 4),
                available_at=window_end_date,
                source_ref=f"market:first_{window_days}d:cumulative_return",
            )

        # 日收益率序列 → 波动率（标准差）
        daily_returns = [
            _safe_float(r.get("change_pct")) for r in window
        ]
        valid_returns = [r for r in daily_returns if r is not None]
        if len(valid_returns) >= 2:
            mean_ret = sum(valid_returns) / len(valid_returns)
            variance = sum(
                (r - mean_ret) ** 2 for r in valid_returns
            ) / (len(valid_returns) - 1)
            volatility = math.sqrt(variance)
            features[f"first_{window_days}d_volatility_pct"] = FeatureValue(
                value=round(volatility, 4),
                available_at=window_end_date,
                source_ref=f"market:first_{window_days}d:volatility",
            )
            features[f"first_{window_days}d_mean_daily_return_pct"] = FeatureValue(
                value=round(mean_ret, 4),
                available_at=window_end_date,
                source_ref=f"market:first_{window_days}d:mean_return",
            )

        # 平均成交额
        turnovers = [
            _safe_float(r.get("turnover_hkd")) for r in window
        ]
        valid_turnovers = [t for t in turnovers if t is not None]
        if valid_turnovers:
            avg_turnover = sum(valid_turnovers) / len(valid_turnovers)
            features[f"first_{window_days}d_avg_turnover_hkd"] = FeatureValue(
                value=round(avg_turnover, 2),
                available_at=window_end_date,
                source_ref=f"market:first_{window_days}d:avg_turnover",
            )

        # 平均振幅
        amplitudes = [
            _safe_float(r.get("amplitude_pct")) for r in window
        ]
        valid_amps = [a for a in amplitudes if a is not None]
        if valid_amps:
            avg_amp = sum(valid_amps) / len(valid_amps)
            features[f"first_{window_days}d_avg_amplitude_pct"] = FeatureValue(
                value=round(avg_amp, 4),
                available_at=window_end_date,
                source_ref=f"market:first_{window_days}d:avg_amplitude",
            )

        # 窗口期最高价 / 最低价
        if valid_highs:
            features[f"first_{window_days}d_max_price_hkd"] = FeatureValue(
                value=max(valid_highs),
                available_at=window_end_date,
                source_ref=f"market:first_{window_days}d:max_price",
            )
        if valid_lows:
            features[f"first_{window_days}d_min_price_hkd"] = FeatureValue(
                value=min(valid_lows),
                available_at=window_end_date,
                source_ref=f"market:first_{window_days}d:min_price",
            )

        # 窗口内最大回撤，不读取第 N 日之后的未来行情
        drawdown_closes = [_safe_float(row.get("close_hkd")) for row in window]
        valid_drawdown_closes = [
            value for value in drawdown_closes if value is not None and value > 0
        ]
        if len(valid_drawdown_closes) >= 2:
            peak = valid_drawdown_closes[0]
            max_dd = 0.0
            for price in valid_drawdown_closes:
                if price > peak:
                    peak = price
                dd = (price - peak) / peak * 100
                if dd < max_dd:
                    max_dd = dd
            features[f"first_{window_days}d_max_drawdown_pct"] = FeatureValue(
                value=round(max_dd, 4),
                available_at=window_end_date,
                source_ref=f"market:first_{window_days}d:max_drawdown",
            )

    # ---- 从 company_info 提取发行结构特征 ----
    if company_info:
        _extract_offering_features(features, company_info)

    return features


def _extract_offering_features(
    features: dict[str, FeatureValue],
    company_info: dict[str, Any],
) -> None:
    """从公司信息中提取发行结构特征。"""
    as_of = _parse_date(company_info.get("as_of_date"))
    if as_of is None:
        return

    offering = company_info.get("offering", {})
    if not isinstance(offering, dict):
        return

    # 基石配售占比
    cornerstone_ratio = _safe_float(
        offering.get("cornerstone_ratio_of_offer")
    )
    if cornerstone_ratio is not None:
        features["cornerstone_ratio_of_offer"] = FeatureValue(
            value=cornerstone_ratio,
            available_at=as_of,
            source_ref="company_info:offering.cornerstone_ratio_of_offer",
        )

    # 发行市值
    market_cap = _safe_float(offering.get("listing_market_cap_hkd"))
    if market_cap is not None:
        features["listing_market_cap_hkd"] = FeatureValue(
            value=market_cap,
            available_at=as_of,
            source_ref="company_info:offering.listing_market_cap_hkd",
        )

    # 超额配股比例
    overallocated = _safe_float(offering.get("overallocated_shares"))
    offer_shares = _safe_float(offering.get("offer_shares"))
    if overallocated is not None and offer_shares and offer_shares > 0:
        overallotment_ratio = overallocated / offer_shares
        features["overallotment_ratio"] = FeatureValue(
            value=round(overallotment_ratio, 6),
            available_at=as_of,
            source_ref="company_info:offering.overallotment_ratio",
        )


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _get_offer_price(company_info: dict[str, Any] | None) -> float | None:
    """从公司信息中获取发行价。"""
    if not company_info:
        return None
    offering = company_info.get("offering", {})
    if not isinstance(offering, dict):
        return None
    return _safe_float(offering.get("offer_price_hkd"))


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


__all__ = ["extract_market_features"]
