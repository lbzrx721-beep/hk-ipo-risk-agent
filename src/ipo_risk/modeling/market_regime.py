"""市场环境调节器（Market Regime Detector）。

在预测一家新公司前，先检查当前大盘所处的状态：
- 牛市/情绪高涨 → 调低预警阈值（好行情下还敢跌的是真的差）
- 熊市/恐慌 → 调高预警阈值（大盘都崩了，个股破发不完全怪基本面）
- 震荡 → 保持原阈值

输入：上市前 N 日大盘累计收益与波动率
输出：regime 标签 + 概率调节系数
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd


MarketRegime = Literal["bull", "bear", "neutral"]


@dataclass(frozen=True)
class MarketRegimeResult:
    regime: MarketRegime
    label: str
    adjustment_factor: float
    market_return_60d_pct: float | None
    market_volatility_60d_pct: float | None
    market_return_20d_pct: float | None
    description: str


def detect_market_regime(
    daily_market_path: str,
    listing_date: date,
    *,
    bull_return_threshold_pct: float = 5.0,
    bear_return_threshold_pct: float = -5.0,
    panic_volatility_pct: float = 25.0,
) -> MarketRegimeResult:
    """按大盘 60 日累计收益与该段 60 日波动率判定三档市场环境。

    - 牛市（bull）：60日累计收益 >= +5% 且波动 < 25%
    - 熊市（bear）：60日累计收益 <= -5% 或 波动 >= 25%（恐慌）
    - 震荡（neutral）：两者之间
    """

    daily = pd.read_csv(daily_market_path)
    daily["trade_date"] = pd.to_datetime(daily["trade_date"], errors="coerce").dt.date
    eligible = daily.loc[daily["trade_date"] < listing_date]

    if eligible.empty:
        return MarketRegimeResult(
            regime="neutral",
            label="市场数据缺失",
            adjustment_factor=1.0,
            market_return_60d_pct=None,
            market_volatility_60d_pct=None,
            market_return_20d_pct=None,
            description="无大盘数据可用",
        )

    sample_60 = eligible.tail(60)
    dr60 = pd.to_numeric(sample_60["median_return_pct"], errors="coerce")
    if dr60.notna().sum() >= 2:
        arr60 = dr60.dropna().to_numpy()
        compounded_60 = (float(np.prod(1.0 + arr60 / 100.0)) - 1.0) * 100.0
        volatility_60 = float(dr60.std(ddof=0) * (252 ** 0.5))
    else:
        compounded_60 = None
        volatility_60 = None

    sample_20 = eligible.tail(20)
    dr20 = pd.to_numeric(sample_20["median_return_pct"], errors="coerce")
    if dr20.notna().sum() >= 2:
        arr20 = dr20.dropna().to_numpy()
        compounded_20 = (float(np.prod(1.0 + arr20 / 100.0)) - 1.0) * 100.0
    else:
        compounded_20 = None

    ret_60 = compounded_60 or 0.0
    vol_60 = volatility_60 or 0.0

    if ret_60 >= bull_return_threshold_pct and vol_60 < panic_volatility_pct:
        regime: MarketRegime = "bull"
        label = "牛市/情绪高涨"
        adjustment = 0.85
        desc = (
            f"上市前60日大盘累计收益{ret_60:+.1f}%，波动率{vol_60:.1f}%。"
            f"市场整体向好，个股破发更可能源于基本面缺陷，系统收紧预警标准。"
        )
    elif ret_60 <= bear_return_threshold_pct or vol_60 >= panic_volatility_pct:
        regime = "bear"
        label = "熊市/恐慌" if ret_60 <= bear_return_threshold_pct else f"高波动(vol={vol_60:.0f}%)"
        adjustment = 1.15
        desc = (
            f"上市前60日大盘累计收益{ret_60:+.1f}%，波动率{vol_60:.1f}%。"
            f"市场偏弱或剧烈波动，个股下跌可能有大盘拖累，系统放宽预警标准。"
        )
    else:
        regime = "neutral"
        label = "震荡/横盘"
        adjustment = 1.0
        desc = (
            f"上市前60日大盘累计收益{ret_60:+.1f}%，波动率{vol_60:.1f}%。"
            f"市场无极端方向，系统按标准阈值判断。"
        )

    return MarketRegimeResult(
        regime=regime,
        label=label,
        adjustment_factor=adjustment,
        market_return_60d_pct=compounded_60,
        market_volatility_60d_pct=volatility_60,
        market_return_20d_pct=compounded_20,
        description=desc,
    )


def adjust_probability(raw: float, regime: MarketRegimeResult) -> float:
    """应用市场环境调节后概率。"""
    adjusted = raw * regime.adjustment_factor
    return min(max(adjusted, 0.0), 1.0)


__all__ = [
    "MarketRegime",
    "MarketRegimeResult",
    "detect_market_regime",
    "adjust_probability",
]