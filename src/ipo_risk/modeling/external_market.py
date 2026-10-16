"""比赛数据覆盖不到新股时的外部上市日期与市场环境兜底。"""

from __future__ import annotations

from datetime import date, datetime, time as datetime_time, timedelta, timezone
from html.parser import HTMLParser
import math
from pathlib import Path
import re
import time
from typing import Any, Mapping
from urllib.parse import quote

import httpx
import pandas as pd

from ipo_risk.config import ExternalMarketConfig
from ipo_risk.domain import FeatureValue
from ipo_risk.modeling.dataset import (
    DEFAULT_MARKET_WINDOWS,
    build_market_feature_snapshot_from_daily_sentiment,
)


class ExternalMarketDataError(RuntimeError):
    """外部市场服务不可用或返回无效数据。"""


class TwelveDataExternalMarketClient:
    """Twelve Data IPO Calendar 与恒生指数日线适配器。"""

    def __init__(
        self,
        config: ExternalMarketConfig,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        if config.provider != "twelve_data":
            raise ValueError(f"unsupported external market provider: {config.provider}")
        self.config = config
        self._owns_client = client is None
        self.client = client or httpx.Client(
            base_url=config.base_url.rstrip("/"),
            timeout=config.timeout_seconds,
        )

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def resolve_listing_date(
        self,
        *,
        company_id: str,
        company_name: str,
        prospectus_date: date,
    ) -> date | None:
        """从招股书日起180天内的港股IPO日历匹配计划上市日期。"""

        payload = self._get(
            "/ipo_calendar",
            {
                "start_date": prospectus_date.isoformat(),
                "end_date": (prospectus_date + timedelta(days=180)).isoformat(),
                "mic_code": self.config.mic_code,
            },
        )
        target_ticker = _normalize_ticker(company_id)
        target_name = _normalize_name(company_name)
        matches: list[tuple[int, date]] = []
        if not isinstance(payload, Mapping):
            raise ExternalMarketDataError("外部IPO日历返回格式无效。")
        for raw_day, rows in payload.items():
            try:
                listing_day = date.fromisoformat(str(raw_day))
            except ValueError:
                continue
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, Mapping) or not _is_hong_kong_listing(row, self.config):
                    continue
                ticker = _normalize_ticker(str(row.get("symbol") or ""))
                name = _normalize_name(str(row.get("name") or ""))
                score = 0
                if ticker and ticker == target_ticker:
                    score += 100
                if name and target_name and name == target_name:
                    score += 80
                elif name and target_name and (name in target_name or target_name in name):
                    score += 40
                if score:
                    matches.append((score, listing_day))
        return max(matches, default=(0, None), key=lambda item: (item[0], -item[1].toordinal()))[1]

    def build_market_features(
        self,
        *,
        listing_date: date,
        windows: tuple[int, ...] = DEFAULT_MARKET_WINDOWS,
    ) -> dict[str, FeatureValue]:
        """用恒生指数日线构建训练字段的外部代理值。"""

        start = listing_date - timedelta(days=max(windows) * 3)
        payload = self._get(
            "/time_series",
            {
                "symbol": self.config.market_index_symbol,
                "exchange": self.config.exchange,
                "interval": "1day",
                "start_date": start.isoformat(),
                "end_date": (listing_date - timedelta(days=1)).isoformat(),
                "outputsize": max(windows) * 3,
                "order": "ASC",
                "timezone": "Exchange",
            },
        )
        values = payload.get("values") if isinstance(payload, Mapping) else None
        if not isinstance(values, list) or len(values) < min(windows):
            raise ExternalMarketDataError("外部恒生指数日线数量不足。")
        rows: list[dict[str, Any]] = []
        previous_close: float | None = None
        for raw in sorted(values, key=lambda item: str(item.get("datetime") or "")):
            if not isinstance(raw, Mapping):
                continue
            try:
                trade_date = date.fromisoformat(str(raw["datetime"])[:10])
                close = float(raw["close"])
                high = float(raw["high"])
                low = float(raw["low"])
                volume = float(raw.get("volume") or 0.0)
            except (KeyError, TypeError, ValueError):
                continue
            daily_return = (
                (close / previous_close - 1.0) * 100.0
                if previous_close is not None and previous_close > 0
                else math.nan
            )
            amplitude = (
                (high - low) / previous_close * 100.0
                if previous_close is not None and previous_close > 0
                else math.nan
            )
            rows.append(
                {
                    "trade_date": trade_date,
                    "median_return_pct": daily_return,
                    # 外部兜底没有全市场横截面，使用恒指上涨日占比作为广度代理。
                    "advance_ratio": (1.0 if daily_return > 0 else 0.0),
                    "median_amplitude_pct": amplitude,
                    "market_turnover_hkd": close * volume,
                    "active_securities": 1.0,
                }
            )
            previous_close = close
        if len(rows) < min(windows):
            raise ExternalMarketDataError("外部恒生指数有效日线数量不足。")
        return build_market_feature_snapshot_from_daily_sentiment(
            pd.DataFrame(rows),
            listing_date=listing_date,
            windows=windows,
            source_ref=(
                "twelve_data:time_series:"
                f"{self.config.market_index_symbol}:{self.config.exchange}:proxy"
            ),
        )

    def _get(self, path: str, params: Mapping[str, Any]) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.client.get(
                    path,
                    params={**params, "apikey": self.config.require_api_key()},
                )
                payload = response.json()
                if isinstance(payload, Mapping) and (
                    payload.get("status") == "error" or payload.get("code")
                ):
                    raise ExternalMarketDataError(
                        str(payload.get("message") or "外部市场API返回错误。")
                    )
                response.raise_for_status()
                return payload
            except ExternalMarketDataError:
                raise
            except (httpx.TransportError, httpx.HTTPStatusError, ValueError) as exc:
                last_error = exc
                retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code == 429
                    or exc.response.status_code >= 500
                )
                if not retryable or attempt >= self.config.max_retries:
                    break
                time.sleep(self.config.retry_initial_seconds * (2**attempt))
        raise ExternalMarketDataError(f"Twelve Data请求失败：{last_error}") from last_error


class _HtmlTableRows(HTMLParser):
    """只提取HTML表格单元格文字，避免引入额外解析依赖。"""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell_parts: list[str] | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],  # noqa: ARG002
    ) -> None:
        if tag.lower() == "tr":
            self._row = []
        elif tag.lower() in {"td", "th"} and self._row is not None:
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"td", "th"} and self._cell_parts is not None:
            if self._row is not None:
                self._row.append(" ".join("".join(self._cell_parts).split()))
            self._cell_parts = None
        elif tag.lower() == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None
            self._cell_parts = None


class HKEXNewListingsClient:
    """查询港交所最近两周新上市证券表中的官方上市日期。"""

    _DATE_PATTERN = re.compile(r"(?P<day>\d{1,2})/(?P<month>\d{1,2})/(?P<year>20\d{2})")
    _CODE_PATTERN = re.compile(r"^\D*(?P<code>\d{4,5})\D*$")

    def __init__(
        self,
        config: ExternalMarketConfig,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self._owns_client = client is None
        self.client = client or httpx.Client(
            timeout=config.timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 IPO-Risk-Research/1.0"},
        )

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def resolve_listing_date(
        self,
        *,
        company_id: str,
        company_name: str,  # noqa: ARG002 - 股票代码是官方表的稳定匹配键
        prospectus_date: date,
    ) -> date | None:
        parser = _HtmlTableRows()
        parser.feed(self._get_html())
        target = _normalize_ticker(company_id)
        matches: list[date] = []
        for cells in parser.rows:
            joined = " ".join(cells)
            if "新上市" not in joined and "new listing" not in joined.lower():
                continue
            date_index: int | None = None
            listing_day: date | None = None
            for index, cell in enumerate(cells):
                match = self._DATE_PATTERN.search(cell)
                if match is None:
                    continue
                try:
                    listing_day = date(
                        int(match.group("year")),
                        int(match.group("month")),
                        int(match.group("day")),
                    )
                except ValueError:
                    listing_day = None
                date_index = index
                break
            if listing_day is None or date_index is None:
                continue
            listed_code: str | None = None
            for cell in cells[date_index + 1 :]:
                match = self._CODE_PATTERN.fullmatch(cell.strip())
                if match is not None:
                    listed_code = _normalize_ticker(match.group("code"))
                    break
            if listed_code != target:
                continue
            if not (
                prospectus_date - timedelta(days=30)
                <= listing_day
                <= prospectus_date + timedelta(days=180)
            ):
                continue
            matches.append(listing_day)
        return min(matches) if matches else None

    def _get_html(self) -> str:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.client.get(self.config.hkex_new_listings_url)
                response.raise_for_status()
                return response.text
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_error = exc
                retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code == 429
                    or exc.response.status_code >= 500
                )
                if not retryable or attempt >= self.config.max_retries:
                    break
                time.sleep(self.config.retry_initial_seconds * (2**attempt))
        raise ExternalMarketDataError(f"港交所新上市资料查询失败：{last_error}") from last_error


class YahooFinanceExternalMarketClient:
    """无需密钥的 Yahoo Finance 港股上市日与恒指日线兜底。"""

    def __init__(
        self,
        config: ExternalMarketConfig,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self._owns_client = client is None
        self.client = client or httpx.Client(
            base_url=config.yahoo_base_url.rstrip("/"),
            timeout=config.timeout_seconds,
            headers={"User-Agent": "Mozilla/5.0 IPO-Risk-Research/1.0"},
        )

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def resolve_listing_date(
        self,
        *,
        company_id: str,
        company_name: str,  # noqa: ARG002 - 与主数据源接口保持一致
        prospectus_date: date,
    ) -> date | None:
        symbol = f"{int(_normalize_ticker(company_id))}.HK"
        result = self._chart(
            symbol,
            start=prospectus_date - timedelta(days=30),
            end=prospectus_date + timedelta(days=180),
        )
        trading_days = sorted(
            datetime.fromtimestamp(int(value), tz=timezone.utc).date()
            for value in (result.get("timestamp") or [])
            if value is not None
        )
        eligible = [day for day in trading_days if day >= prospectus_date]
        return eligible[0] if eligible else None

    def build_market_features(
        self,
        *,
        listing_date: date,
        windows: tuple[int, ...] = DEFAULT_MARKET_WINDOWS,
    ) -> dict[str, FeatureValue]:
        result = self._chart(
            self.config.yahoo_market_symbol,
            start=listing_date - timedelta(days=max(windows) * 3),
            end=listing_date,
        )
        timestamps = result.get("timestamp") or []
        indicators = result.get("indicators") or {}
        quote_blocks = indicators.get("quote") or []
        quote_values = quote_blocks[0] if quote_blocks else {}
        rows: list[dict[str, Any]] = []
        previous_close: float | None = None
        for index, raw_timestamp in enumerate(timestamps):
            try:
                trade_date = datetime.fromtimestamp(
                    int(raw_timestamp), tz=timezone.utc
                ).date()
                close = float(quote_values["close"][index])
                high = float(quote_values["high"][index])
                low = float(quote_values["low"][index])
                volumes = quote_values.get("volume") or []
                volume = float(volumes[index] or 0.0) if index < len(volumes) else 0.0
            except (IndexError, KeyError, TypeError, ValueError):
                continue
            if trade_date >= listing_date:
                continue
            daily_return = (
                (close / previous_close - 1.0) * 100.0
                if previous_close is not None and previous_close > 0
                else math.nan
            )
            amplitude = (
                (high - low) / previous_close * 100.0
                if previous_close is not None and previous_close > 0
                else math.nan
            )
            rows.append(
                {
                    "trade_date": trade_date,
                    "median_return_pct": daily_return,
                    "advance_ratio": 1.0 if daily_return > 0 else 0.0,
                    "median_amplitude_pct": amplitude,
                    "market_turnover_hkd": close * volume,
                    "active_securities": 1.0,
                }
            )
            previous_close = close
        if len(rows) < min(windows):
            raise ExternalMarketDataError("Yahoo恒生指数有效日线数量不足。")
        return build_market_feature_snapshot_from_daily_sentiment(
            pd.DataFrame(rows),
            listing_date=listing_date,
            windows=windows,
            source_ref=f"yahoo_finance:chart:{self.config.yahoo_market_symbol}:proxy",
        )

    def _chart(self, symbol: str, *, start: date, end: date) -> Mapping[str, Any]:
        payload = self._get(
            f"/v8/finance/chart/{quote(symbol, safe='')}",
            {
                "period1": _unix_timestamp(start),
                "period2": _unix_timestamp(end + timedelta(days=1)),
                "interval": "1d",
                "events": "history",
            },
        )
        chart = payload.get("chart") if isinstance(payload, Mapping) else None
        error = chart.get("error") if isinstance(chart, Mapping) else None
        results = chart.get("result") if isinstance(chart, Mapping) else None
        if error or not isinstance(results, list) or not results:
            raise ExternalMarketDataError(f"Yahoo Finance无可用数据：{error or symbol}")
        result = results[0]
        if not isinstance(result, Mapping):
            raise ExternalMarketDataError("Yahoo Finance返回格式无效。")
        return result

    def _get(self, path: str, params: Mapping[str, Any]) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.client.get(path, params=params)
                response.raise_for_status()
                return response.json()
            except (httpx.TransportError, httpx.HTTPStatusError, ValueError) as exc:
                last_error = exc
                retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code == 429
                    or exc.response.status_code >= 500
                )
                if not retryable or attempt >= self.config.max_retries:
                    break
                time.sleep(self.config.retry_initial_seconds * (2**attempt))
        raise ExternalMarketDataError(f"Yahoo Finance请求失败：{last_error}") from last_error


class ExternalMarketClient:
    """外部数据链：上市日期优先港交所，交易后再由Yahoo反查。"""

    def __init__(self, config: ExternalMarketConfig) -> None:
        self.last_listing_date_source: str | None = None
        self.twelve_data = (
            TwelveDataExternalMarketClient(config)
            if config.twelve_data_is_configured
            else None
        )
        self.hkex = HKEXNewListingsClient(config) if config.free_fallback_enabled else None
        self.yahoo = (
            YahooFinanceExternalMarketClient(config)
            if config.free_fallback_enabled
            else None
        )

    def close(self) -> None:
        if self.twelve_data is not None:
            self.twelve_data.close()
        if self.hkex is not None:
            self.hkex.close()
        if self.yahoo is not None:
            self.yahoo.close()

    def resolve_listing_date(self, **kwargs: Any) -> date | None:
        errors: list[str] = []
        for source, client in (
            ("hkex:newly_listed_securities", self.hkex),
            ("yahoo_finance:first_trading_day", self.yahoo),
        ):
            if client is None:
                continue
            try:
                result = client.resolve_listing_date(**kwargs)
                if result is not None:
                    self.last_listing_date_source = source
                    return result
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
        if errors:
            raise ExternalMarketDataError("；".join(errors))
        return None

    def build_market_features(self, **kwargs: Any) -> dict[str, FeatureValue]:
        errors: list[str] = []
        for client in (self.twelve_data, self.yahoo):
            if client is None:
                continue
            try:
                return client.build_market_features(**kwargs)
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
        raise ExternalMarketDataError("；".join(errors) or "没有可用的外部市场数据源。")


def _unix_timestamp(value: date) -> int:
    return int(
        datetime.combine(value, datetime_time.min, tzinfo=timezone.utc).timestamp()
    )


def local_market_data_covers(
    daily_market_path: Path,
    *,
    listing_date: date,
    maximum_gap_days: int,
) -> bool:
    """本地数据最后交易日距离上市日足够近时才视为可覆盖。"""

    if not daily_market_path.is_file():
        return False
    frame = pd.read_csv(daily_market_path, usecols=["trade_date"])
    dates = pd.to_datetime(frame["trade_date"], errors="coerce").dropna()
    if dates.empty:
        return False
    eligible = dates.loc[dates.dt.date < listing_date]
    if eligible.empty:
        return False
    latest = eligible.max().date()
    return (listing_date - latest).days <= maximum_gap_days


def _normalize_ticker(value: str) -> str:
    stem = value.upper().split(".", 1)[0]
    digits = "".join(re.findall(r"\d+", stem))
    return digits.lstrip("0") or "0" if digits else stem.strip()


def _normalize_name(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _is_hong_kong_listing(
    row: Mapping[str, Any], config: ExternalMarketConfig
) -> bool:
    mic = str(row.get("mic_code") or "").upper()
    exchange = str(row.get("exchange") or "").upper()
    return mic == config.mic_code.upper() or any(
        marker in exchange for marker in ("HKEX", "HONG KONG", "SEHK")
    )


__all__ = [
    "ExternalMarketClient",
    "ExternalMarketDataError",
    "HKEXNewListingsClient",
    "TwelveDataExternalMarketClient",
    "YahooFinanceExternalMarketClient",
    "local_market_data_covers",
]
