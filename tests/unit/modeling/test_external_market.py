from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path

import httpx
from pydantic import SecretStr

from ipo_risk.config import ExternalMarketConfig
from ipo_risk.modeling.external_market import (
    ExternalMarketClient,
    ExternalMarketDataError,
    HKEXNewListingsClient,
    TwelveDataExternalMarketClient,
    YahooFinanceExternalMarketClient,
    local_market_data_covers,
)


def _config() -> ExternalMarketConfig:
    return ExternalMarketConfig(
        provider="twelve_data",
        api_key=SecretStr("test-key"),
    )


def test_twelve_data_resolves_hong_kong_listing_date_by_ticker() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["apikey"] == "test-key"
        return httpx.Response(
            200,
            json={
                "2026-09-18": [
                    {
                        "symbol": "01234",
                        "name": "测试科技有限公司",
                        "exchange": "HKEX",
                        "mic_code": "XHKG",
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="https://api.twelvedata.com") as http:
        client = TwelveDataExternalMarketClient(_config(), client=http)
        result = client.resolve_listing_date(
            company_id="01234",
            company_name="测试科技",
            prospectus_date=date(2026, 8, 28),
        )

    assert result == date(2026, 9, 18)


def test_twelve_data_builds_all_market_windows_from_index_daily_data() -> None:
    start = date(2026, 5, 1)
    values = []
    close = 20_000.0
    for offset in range(90):
        day = start + timedelta(days=offset)
        close += 10.0 if offset % 2 == 0 else -4.0
        values.append(
            {
                "datetime": day.isoformat(),
                "open": str(close - 3.0),
                "high": str(close + 12.0),
                "low": str(close - 15.0),
                "close": str(close),
                "volume": "1000000",
            }
        )

    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, content=json.dumps({"values": values}).encode())
    )
    with httpx.Client(transport=transport, base_url="https://api.twelvedata.com") as http:
        client = TwelveDataExternalMarketClient(_config(), client=http)
        features = client.build_market_features(listing_date=date(2026, 8, 15))

    assert features["market_advance_ratio_20d"].value is not None
    assert features["market_volatility_20d_pct"].value is not None
    assert "twelve_data" in features["market_advance_ratio_20d"].source_ref


def test_local_market_coverage_uses_latest_day_before_listing(tmp_path: Path) -> None:
    path = tmp_path / "daily_market_sentiment.csv"
    path.write_text(
        "trade_date\n2025-01-02\n2025-01-09\n2026-01-01\n",
        encoding="utf-8",
    )

    assert local_market_data_covers(
        path,
        listing_date=date(2025, 1, 10),
        maximum_gap_days=10,
    )


def test_yahoo_builds_market_features_without_api_key() -> None:
    start = date(2026, 5, 1)
    timestamps = []
    close_values = []
    for offset in range(90):
        day = start + timedelta(days=offset)
        timestamps.append(
            int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())
        )
        close_values.append(20_000.0 + offset * 10.0)
    quote = {
        "close": close_values,
        "high": [value + 12.0 for value in close_values],
        "low": [value - 15.0 for value in close_values],
        "volume": [1_000_000 for _ in close_values],
    }
    payload = {
        "chart": {
            "result": [
                {
                    "timestamp": timestamps,
                    "indicators": {"quote": [quote]},
                }
            ],
            "error": None,
        }
    }
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(200, json=payload)
    )
    config = _config().model_copy(update={"free_fallback_enabled": True})
    with httpx.Client(transport=transport, base_url=config.yahoo_base_url) as http:
        client = YahooFinanceExternalMarketClient(config, client=http)
        features = client.build_market_features(listing_date=date(2026, 8, 15))

    assert features["market_volatility_20d_pct"].value is not None
    assert "yahoo_finance" in features["market_volatility_20d_pct"].source_ref


def test_external_market_chain_falls_back_to_yahoo() -> None:
    class FailedTwelveData:
        def build_market_features(self, **_kwargs: object) -> dict[str, object]:
            raise ExternalMarketDataError("套餐不支持港交所行情")

    class WorkingYahoo:
        def build_market_features(self, **_kwargs: object) -> dict[str, object]:
            return {"source": "free"}

    client = ExternalMarketClient.__new__(ExternalMarketClient)
    client.twelve_data = FailedTwelveData()
    client.hkex = None
    client.yahoo = WorkingYahoo()
    client.last_listing_date_source = None

    assert client.build_market_features(listing_date=date(2026, 8, 15)) == {
        "source": "free"
    }


def test_listing_date_chain_uses_hkex_then_yahoo_without_twelve_data() -> None:
    class ForbiddenTwelveData:
        def resolve_listing_date(self, **_kwargs: object) -> date | None:
            raise AssertionError("上市日期链不应调用Twelve Data")

    class EmptyHKEX:
        def resolve_listing_date(self, **_kwargs: object) -> date | None:
            return None

    class WorkingYahoo:
        def resolve_listing_date(self, **_kwargs: object) -> date | None:
            return date(2026, 8, 15)

    client = ExternalMarketClient.__new__(ExternalMarketClient)
    client.twelve_data = ForbiddenTwelveData()
    client.hkex = EmptyHKEX()
    client.yahoo = WorkingYahoo()
    client.last_listing_date_source = None

    assert client.resolve_listing_date(company_id="01234") == date(2026, 8, 15)
    assert client.last_listing_date_source == "yahoo_finance:first_trading_day"


def test_hkex_resolves_official_new_listing_date_by_primary_stock_code() -> None:
    html = """
    <table>
      <tr><th>上市/開始交易日期</th><th>股份簡稱</th><th>股份代號</th><th>備註</th></tr>
      <tr><td>30/12/2025*</td><td>測試科技</td><td>01234</td><td>新上市</td></tr>
      <tr><td>31/12/2025</td><td>其他公司</td><td>05678</td><td>新上市</td></tr>
    </table>
    """
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, text=html))
    config = _config()
    with httpx.Client(transport=transport) as http:
        client = HKEXNewListingsClient(config, client=http)
        result = client.resolve_listing_date(
            company_id="01234",
            company_name="测试科技",
            prospectus_date=date(2025, 12, 18),
        )

    assert result == date(2025, 12, 30)
