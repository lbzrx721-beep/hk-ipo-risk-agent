from __future__ import annotations

from datetime import date

from scripts.normalize_fundamental_currency import _convert, _rate_to_hkd, _to_million


def test_rate_to_hkd_uses_latest_common_ecb_business_day() -> None:
    rates = {
        "HKD": {date(2024, 1, 5): 8.55, date(2024, 1, 8): 8.56},
        "CNY": {date(2024, 1, 5): 7.82, date(2024, 1, 8): 7.83},
    }

    rate_date, rate = _rate_to_hkd("RMB", date(2024, 1, 7), rates)

    assert rate_date == date(2024, 1, 5)
    assert rate == 8.55 / 7.82


def test_rate_to_hkd_handles_hkd_mop_and_eur() -> None:
    rates = {"HKD": {date(2024, 1, 5): 8.55}}

    assert _rate_to_hkd("HKD", date(2024, 1, 7), rates) == (
        date(2024, 1, 7),
        1.0,
    )
    assert _rate_to_hkd("MOP", date(2024, 1, 7), rates) == (
        date(2024, 1, 7),
        1.0 / 1.03,
    )
    assert _rate_to_hkd("EUR", date(2024, 1, 7), rates) == (
        date(2024, 1, 5),
        8.55,
    )


def test_convert_distinguishes_audited_base_units_from_manual_millions() -> None:
    assert float(
        _convert(
            "67675515000",
            1.1,
            67675515000,
            field="fundamental_revenue_latest_native",
        )
    ) == 74443.0665
    assert float(
        _convert(
            "1192.75",
            1.1,
            None,
            field="fundamental_revenue_latest_native",
        )
    ) == 1312.025
    assert float(_to_million("373709804", 373709804000)) == 373709.804
    assert float(_to_million("4049.74", 4049738800)) == 4049.74


def test_convert_recognizes_revised_thousand_and_base_unit_values() -> None:
    assert float(_to_million("40677441", 39120348000)) == 40677.441
    assert float(_to_million("11724263", 11546851000)) == 11724.263
    assert float(_to_million("36748483", 43676000)) == 36.748483


def test_unaudited_large_table_values_are_treated_as_thousands() -> None:
    assert float(
        _to_million(
            "35519538",
            None,
            field="fundamental_revenue_latest_native",
            reference_hkd_mn="4000",
        )
    ) == 35519.538
    assert float(
        _to_million(
            "59241.145", None, field="fundamental_revenue_latest_native"
        )
    ) == 59241.145


def test_unaudited_cash_flow_uses_existing_cash_flow_ratio() -> None:
    assert float(
        _to_million(
            "3168.3",
            None,
            field="fundamental_operating_cash_flow_latest_native",
            related_value="35519538",
            related_audited_value=None,
            expected_ratio_pct="0.0089198795322168",
            reference_hkd_mn="4000",
        )
    ) == 3.1683


def test_unaudited_cash_flow_inherits_revenue_table_unit() -> None:
    assert float(
        _to_million(
            "7563842",
            None,
            field="fundamental_operating_cash_flow_latest_native",
            related_value="36748483",
            related_audited_value=43676296,
        )
    ) == 7.563842


def test_revenue_prefers_audited_unit_over_business_reference() -> None:
    assert float(
        _to_million(
            "40677441",
            39120348,
            field="fundamental_revenue_latest_native",
            reference_hkd_mn="4000",
            rate=1.2,
        )
    ) == 40.677441


def test_cash_flow_uses_its_own_reliable_audit_unit() -> None:
    assert float(
        _to_million(
            "2191663000",
            2191663000,
            field="fundamental_operating_cash_flow_latest_native",
            related_value="9901566",
            related_audited_value=None,
            reference_hkd_mn="11802.4",
        )
    ) == 2191.663


def test_unaudited_three_year_cash_flow_selects_plausible_unit() -> None:
    assert float(
        _to_million(
            "1696556",
            None,
            field="fundamental_operating_cash_flow_sum_3y_native",
            related_value="1443.732",
            related_audited_value=1443732000,
            reference_hkd_mn="1085.75",
        )
    ) == 1696.556


def test_zero_revenue_cash_flow_does_not_divide_by_zero() -> None:
    assert float(
        _to_million(
            "-225.212",
            None,
            field="fundamental_operating_cash_flow_latest_native",
            related_value="0",
            related_audited_value=None,
            expected_ratio_pct="",
            reference_hkd_mn="4486",
        )
    ) == -225.212


def test_zero_revenue_three_year_cash_flow_recognizes_thousand_unit() -> None:
    assert float(
        _to_million(
            "-485359",
            None,
            field="fundamental_operating_cash_flow_sum_3y_native",
            related_value="0",
            related_audited_value=None,
            reference_hkd_mn="5920",
        )
    ) == -485.359


def test_different_reporting_period_falls_back_to_business_scale() -> None:
    assert float(
        _to_million(
            "576888",
            10752917,
            field="fundamental_revenue_latest_native",
            reference_hkd_mn="729328",
            rate=1.1,
        )
    ) == 576888.0


def test_verified_divisor_overrides_missing_table_unit() -> None:
    assert float(
        _convert(
            "40677441",
            1.2,
            39120348,
            field="fundamental_revenue_latest_native",
            reference_hkd_mn="4000",
            forced_divisor=1000,
        )
    ) == 48812.9292
