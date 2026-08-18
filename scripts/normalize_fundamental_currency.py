"""将招股书财务金额按招股书日期统一换算为百万港元。"""

from __future__ import annotations

import csv
from datetime import date
import json
import math
from pathlib import Path
import time
from typing import Any
from urllib.request import urlopen

import typer


app = typer.Typer(add_completion=False)

_ECB_CURRENCIES = ("HKD", "CNY", "USD", "SGD", "MYR")
_OFFERING_AMOUNT_COLUMNS = (
    "offering_gross_proceeds_low_hkd",
    "offering_gross_proceeds_high_hkd",
    "offering_gross_proceeds_mid_hkd",
    "offering_net_proceeds_hkd",
    "offering_market_cap_low_hkd",
    "offering_market_cap_high_hkd",
    "offering_market_cap_mid_hkd",
)
_NATIVE_AMOUNT_COLUMNS = (
    "fundamental_revenue_latest_native",
    "fundamental_operating_cash_flow_latest_native",
    "fundamental_operating_cash_flow_sum_3y_native",
)
_MANUAL_CURRENCY_BY_COMPANY = {
    "00013": "USD",
    "00816": "RMB",
    "01376": "SGD",
    "01408": "MOP",
    "01463": "MYR",
    "01477": "RMB",
    "01936": "MYR",
    "01948": "RMB",
    "01953": "MYR",
    "02125": "RMB",
    "02129": "SGD",
    "02131": "RMB",
    "02135": "RMB",
    "02216": "USD",
    "02219": "RMB",
    "02285": "USD",
    "02391": "USD",
    "02409": "USD",
    "02411": "RMB",
    "02425": "USD",
    "02427": "RMB",
    "02443": "RMB",
    "02497": "RMB",
    "02505": "RMB",
    "02507": "USD",
    "02529": "RMB",
    "02599": "RMB",
    "06913": "RMB",
    "09696": "RMB",
    "09995": "RMB",
}

# 少数招股书的主财务表表头漏掉“千元”，但相邻财务表明确使用千元。
# 这里记录经过原表核对的单位，不修改原始数值，只用于统一换算。
_VERIFIED_NATIVE_DIVISORS = {
    ("01024", "fundamental_revenue_latest_native"): 1_000.0,
    ("01024", "fundamental_operating_cash_flow_latest_native"): 1_000.0,
    ("01024", "fundamental_operating_cash_flow_sum_3y_native"): 1_000.0,
}


@app.command()
def main(
    project_root: Path = typer.Option(Path.cwd(), help="项目根目录"),
    through_year: int = typer.Option(2024, help="只换算该年份及以前的公司"),
) -> None:
    root = project_root.resolve()
    matrix_path = (
        root
        / "data"
        / "processed"
        / "modeling"
        / "structured_fundamental_feature_matrix.csv"
    )
    rows = _read_csv(matrix_path)
    audited_native_values = _load_audited_native_values(
        matrix_path.with_name("structured_fundamental_feature_audit.jsonl")
    )
    rates = _load_ecb_rates(root / "tmp" / "fx_rates", 2020, through_year)
    converted = 0
    unresolved: list[str] = []
    for row in rows:
        prospectus_date = date.fromisoformat(row["prospectus_date"])
        if prospectus_date.year > through_year:
            _clear_normalized_fields(row)
            continue
        company_id = str(row["company_id"]).zfill(5)
        currency = (row.get("fundamental_reporting_currency") or "").strip().upper()
        currency = _MANUAL_CURRENCY_BY_COMPANY.get(company_id, currency)
        if not currency:
            unresolved.append(company_id)
            _clear_normalized_fields(row)
            continue
        row["fundamental_reporting_currency"] = currency
        for column in _OFFERING_AMOUNT_COLUMNS:
            row[column] = _to_million(
                row.get(column), audited_native_values.get((company_id, column))
            )
        rate_date, rate = _rate_to_hkd(currency, prospectus_date, rates)
        row["fundamental_fx_to_hkd"] = _number(rate)
        row["fundamental_fx_rate_date"] = rate_date.isoformat()
        row["fundamental_revenue_latest_hkd_mn"] = _convert(
            row.get("fundamental_revenue_latest_native"),
            rate,
            audited_native_values.get((company_id, "fundamental_revenue_latest_native")),
            field="fundamental_revenue_latest_native",
            reference_hkd_mn=row.get("offering_market_cap_mid_hkd"),
            forced_divisor=_VERIFIED_NATIVE_DIVISORS.get(
                (company_id, "fundamental_revenue_latest_native")
            ),
        )
        row["fundamental_operating_cash_flow_latest_hkd_mn"] = _convert(
            row.get("fundamental_operating_cash_flow_latest_native"),
            rate,
            audited_native_values.get(
                (company_id, "fundamental_operating_cash_flow_latest_native")
            ),
            field="fundamental_operating_cash_flow_latest_native",
            related_value=row.get("fundamental_revenue_latest_native"),
            related_audited_value=audited_native_values.get(
                (company_id, "fundamental_revenue_latest_native")
            ),
            expected_ratio_pct=row.get(
                "fundamental_operating_cash_flow_to_revenue_latest_pct"
            ),
            reference_hkd_mn=row.get("offering_market_cap_mid_hkd"),
            forced_divisor=_VERIFIED_NATIVE_DIVISORS.get(
                (company_id, "fundamental_operating_cash_flow_latest_native")
            ),
        )
        row["fundamental_operating_cash_flow_sum_3y_hkd_mn"] = _convert(
            row.get("fundamental_operating_cash_flow_sum_3y_native"),
            rate,
            audited_native_values.get(
                (company_id, "fundamental_operating_cash_flow_sum_3y_native")
            ),
            field="fundamental_operating_cash_flow_sum_3y_native",
            related_value=row.get("fundamental_revenue_latest_native"),
            related_audited_value=audited_native_values.get(
                (company_id, "fundamental_revenue_latest_native")
            ),
            reference_hkd_mn=row.get("offering_market_cap_mid_hkd"),
            forced_divisor=_VERIFIED_NATIVE_DIVISORS.get(
                (company_id, "fundamental_operating_cash_flow_sum_3y_native")
            ),
        )
        converted += 1

    if unresolved:
        raise ValueError(f"仍有未确认币种的公司：{', '.join(unresolved)}")
    _write_csv(matrix_path, rows)
    _sync_modeling_matrices(root, rows, through_year=through_year)
    typer.echo(f"换算完成：{converted} 家（截至 {through_year} 年）。")
    typer.echo(f"统一矩阵：{matrix_path}")


def _load_ecb_rates(
    cache_dir: Path, start_year: int, end_year: int
) -> dict[str, dict[date, float]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, dict[date, float]] = {}
    for currency in _ECB_CURRENCIES:
        path = cache_dir / f"ecb_{currency.lower()}_{start_year}_{end_year}.csv"
        if not path.is_file():
            url = (
                "https://data-api.ecb.europa.eu/service/data/EXR/"
                f"D.{currency}.EUR.SP00.A?startPeriod={start_year}-01-01"
                f"&endPeriod={end_year}-12-31&format=csvdata"
            )
            error: Exception | None = None
            for attempt in range(3):
                try:
                    with urlopen(url, timeout=180) as response:  # noqa: S310
                        path.write_bytes(response.read())
                    error = None
                    break
                except Exception as exc:  # pragma: no cover - 网络抖动分支
                    error = exc
                    if attempt < 2:
                        time.sleep(2 ** attempt)
            if error is not None:
                raise RuntimeError(f"下载 {currency} 历史汇率失败") from error
        observations = _read_csv(path)
        result[currency] = {
            date.fromisoformat(row["TIME_PERIOD"]): float(row["OBS_VALUE"])
            for row in observations
        }
    return result


def _rate_to_hkd(
    currency: str,
    requested_date: date,
    rates: dict[str, dict[date, float]],
) -> tuple[date, float]:
    if currency == "HKD":
        return requested_date, 1.0
    if currency == "MOP":
        return requested_date, 1.0 / 1.03
    if currency == "EUR":
        rate_date = next(
            (item for item in sorted(rates["HKD"], reverse=True) if item <= requested_date),
            None,
        )
        if rate_date is None:
            raise ValueError(f"EUR 在 {requested_date} 前没有可用汇率")
        return rate_date, rates["HKD"][rate_date]
    ecb_currency = "CNY" if currency == "RMB" else currency
    if ecb_currency not in rates:
        raise ValueError(f"不支持的财务币种：{currency}")
    valid_dates = sorted(
        set(rates[ecb_currency]).intersection(rates["HKD"]), reverse=True
    )
    rate_date = next((item for item in valid_dates if item <= requested_date), None)
    if rate_date is None:
        raise ValueError(f"{currency} 在 {requested_date} 前没有可用汇率")
    # ECB 报价为每 1 欧元兑换多少本币；因此 1 本币对应 HKD/CURRENCY 港元。
    return rate_date, rates["HKD"][rate_date] / rates[ecb_currency][rate_date]


def _convert(
    value: Any,
    rate: float,
    audited_value: float | None,
    *,
    field: str,
    related_value: Any = None,
    related_audited_value: float | None = None,
    expected_ratio_pct: Any = None,
    reference_hkd_mn: Any = None,
    forced_divisor: float | None = None,
) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    native_million = float(
        _to_million(
            text,
            audited_value,
            field=field,
            related_value=related_value,
            related_audited_value=related_audited_value,
            expected_ratio_pct=expected_ratio_pct,
            reference_hkd_mn=reference_hkd_mn,
            rate=rate,
            forced_divisor=forced_divisor,
        )
    )
    return _number(native_million * rate)


def _to_million(
    value: Any,
    audited_value: float | None,
    *,
    field: str = "",
    related_value: Any = None,
    related_audited_value: float | None = None,
    expected_ratio_pct: Any = None,
    reference_hkd_mn: Any = None,
    rate: float = 1.0,
    forced_divisor: float | None = None,
) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if forced_divisor is not None:
        return _number(float(text) / forced_divisor)
    numeric = float(text)
    # 有审计记录时，审计值以“原币元”为统一合同，优先据此判断当前 CSV
    # 是元、千元还是百万元。只有人工补充值没有审计记录时，才借助市值
    # 作宽松的业务量级判断。
    if (
        field == "fundamental_revenue_latest_native"
        and audited_value is None
        and str(reference_hkd_mn or "").strip()
    ):
        return _number(
            numeric
            / _native_divisor(
                numeric,
                None,
                field=field,
                reference_hkd_mn=reference_hkd_mn,
                rate=rate,
            )
        )
    if field in {
        "fundamental_operating_cash_flow_latest_native",
        "fundamental_operating_cash_flow_sum_3y_native",
    } and related_value not in (None, ""):
        revenue_million = float(
            _to_million(
                related_value,
                related_audited_value,
                field="fundamental_revenue_latest_native",
                reference_hkd_mn=reference_hkd_mn,
                rate=rate,
            )
        )
        current_audit_unreliable = False
        if audited_value is not None and numeric != 0:
            audited_million = audited_value / 1_000_000.0
            current_audit_unreliable = abs(audited_million) > max(
                abs(revenue_million) * 100.0, 1_000_000.0
            )
        if audited_value is not None and not current_audit_unreliable:
            current_divisor = _native_divisor(
                numeric,
                audited_value,
                field=field,
                reference_hkd_mn=reference_hkd_mn,
                rate=rate,
            )
            return _number(numeric / current_divisor)
        # 未商业化生物科技公司及 SPAC 的收入可能为 0。此时现金流/收入
        # 比例没有定义，改用现金流自身的量级规则，避免除零和错误放大。
        if revenue_million == 0:
            current_divisor = _native_divisor(
                numeric,
                None,
                field=field,
                reference_hkd_mn=reference_hkd_mn,
                rate=rate,
            )
            return _number(numeric / current_divisor)
        if (
            field == "fundamental_operating_cash_flow_latest_native"
            and expected_ratio_pct not in (None, "")
        ):
            target = float(expected_ratio_pct)
            candidates = (numeric, numeric / 1_000.0, numeric / 1_000_000.0)
            selected = min(
                candidates,
                key=lambda candidate: abs(
                    (100.0 * candidate / revenue_million if revenue_million else 0.0)
                    - target
                ),
            )
            return _number(selected)
        # 没有直接比例字段时，使用宽松的典型量级选择单位：最新年度经营
        # 现金流约为收入的20%，三年合计约为最新年收入的60%。这只在
        # 元/千元/百万元三档间选单位，不改变原始数值关系。
        typical_ratio = (
            0.2
            if field == "fundamental_operating_cash_flow_latest_native"
            else 0.6
        )
        candidates = (numeric, numeric / 1_000.0, numeric / 1_000_000.0)
        selected = min(
            candidates,
            key=lambda candidate: abs(
                math.log10(
                    max(abs(candidate / revenue_million), 1e-12) / typical_ratio
                )
            ),
        )
        return _number(selected)
    if audited_value is not None:
        audited_million = audited_value / 1_000_000.0
        if numeric == 0 or audited_million == 0:
            return _number(numeric if numeric == audited_million else audited_million)
        # 历史抽取结果曾混用“元、千元、百万元”。审计记录统一为“元”，
        # 因而从三个候选单位中选择与审计值数量级最接近者。这样既容忍
        # 后续人工修订，也不会把 40,677,441 千元误当成 40 万亿元。
        candidates = (numeric, numeric / 1_000.0, numeric / 1_000_000.0)
        reference = abs(audited_million)
        selected = min(
            candidates,
            key=lambda candidate: abs(
                math.log10(max(abs(candidate), 1e-12) / max(reference, 1e-12))
            ),
        )
        audit_distance = abs(
            math.log10(
                max(abs(selected), 1e-12) / max(reference, 1e-12)
            )
        )
        # 当前值与审计抽取相差超过一个数量级，通常代表后续人工补充了
        # 不同报告期的数据。此时审计值只作旁证，改由当前公司的市值
        # 选择当前值自身的元/千元/百万元单位，避免京东等公司被缩小百万倍。
        if audit_distance > 1.0 and str(reference_hkd_mn or "").strip():
            divisor = _native_divisor(
                numeric,
                None,
                field=field,
                reference_hkd_mn=reference_hkd_mn,
                rate=rate,
            )
            return _number(numeric / divisor)
        # 极端审计值通常来自表格行错位或年份/单位误识别。此时回到当前
        # 补充值及业务参照进行单位判断，不能让错误审计放大训练特征。
        reference_text = str(reference_hkd_mn or "").strip()
        if reference_text and float(reference_text) > 0:
            normalized_hkd = abs(selected) * rate
            reference = float(reference_text)
            if normalized_hkd > max(reference * 100.0, 1_000_000.0):
                divisor = _native_divisor(
                    numeric,
                    None,
                    field=field,
                    reference_hkd_mn=reference_hkd_mn,
                    rate=rate,
                )
                return _number(numeric / divisor)
        return _number(selected)
    # 无审计记录时，人工补充值存在两种历史合同：较新的值是“百万原币”，
    # 较早的表格补值仍保留原表数字（通常是千元）。收入和三年现金流使用
    # 审慎量级规则；最新现金流若带有现金流/收入比，则用该恒等式判定单位。
    divisor = _native_divisor(
        numeric,
        None,
        field=field,
        reference_hkd_mn=reference_hkd_mn,
        rate=rate,
    )
    return _number(numeric / divisor)


def _native_divisor(
    numeric: float,
    audited_value: float | None,
    *,
    field: str,
    reference_hkd_mn: Any = None,
    rate: float = 1.0,
) -> float:
    candidates = (1.0, 1_000.0, 1_000_000.0)
    if field == "fundamental_revenue_latest_native":
        reference_text = str(reference_hkd_mn or "").strip()
        if reference_text and float(reference_text) > 0 and numeric != 0:
            reference = float(reference_text)
            return min(
                candidates,
                key=lambda divisor: abs(
                    math.log10(
                        max(abs(numeric / divisor) * rate / reference, 1e-12)
                    )
                ),
            )
    if audited_value is not None and audited_value != 0 and numeric != 0:
        audited_million = abs(audited_value / 1_000_000.0)
        return min(
            candidates,
            key=lambda divisor: abs(
                math.log10(
                    max(abs(numeric / divisor), 1e-12)
                    / max(audited_million, 1e-12)
                )
            ),
        )
    magnitude = abs(numeric)
    if field == "fundamental_revenue_latest_native":
        return 1_000.0 if magnitude >= 100_000.0 else 1.0
    if field == "fundamental_operating_cash_flow_sum_3y_native":
        # 人工补充的三年合计若达到五位数，通常仍保留招股书“千元”表格
        # 单位；百万原币口径的三年现金流不应达到这一数量级。
        return 1_000.0 if magnitude >= 10_000.0 else 1.0
    return 1_000.0 if magnitude >= 1_000_000.0 else 1.0


def _load_audited_native_values(path: Path) -> dict[tuple[str, str], float]:
    if not path.is_file():
        return {}
    result: dict[tuple[str, str], float] = {}
    names = (*_NATIVE_AMOUNT_COLUMNS, *_OFFERING_AMOUNT_COLUMNS)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            company_id = str(payload.get("company_id") or "").zfill(5)
            features = payload.get("features") or {}
            for name in names:
                value = (features.get(name) or {}).get("value")
                if value is not None:
                    result[(company_id, name)] = float(value)
    return result


def _number(value: float) -> str:
    return format(value, ".12g")


def _clear_normalized_fields(row: dict[str, Any]) -> None:
    for name in (
        "fundamental_fx_to_hkd",
        "fundamental_fx_rate_date",
        "fundamental_revenue_latest_hkd_mn",
        "fundamental_operating_cash_flow_latest_hkd_mn",
        "fundamental_operating_cash_flow_sum_3y_hkd_mn",
    ):
        row[name] = ""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0])
    for name in (
        "fundamental_fx_to_hkd",
        "fundamental_fx_rate_date",
        "fundamental_revenue_latest_hkd_mn",
        "fundamental_operating_cash_flow_latest_hkd_mn",
        "fundamental_operating_cash_flow_sum_3y_hkd_mn",
    ):
        if name not in fieldnames:
            fieldnames.append(name)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sync_modeling_matrices(
    root: Path,
    rows: list[dict[str, Any]],
    *,
    through_year: int,
) -> None:
    modeling = root / "data" / "processed" / "modeling"
    normalized_columns = (
        "fundamental_fx_to_hkd",
        "fundamental_fx_rate_date",
        "fundamental_revenue_latest_hkd_mn",
        "fundamental_operating_cash_flow_latest_hkd_mn",
        "fundamental_operating_cash_flow_sum_3y_hkd_mn",
    )
    by_company = {str(row["company_id"]).zfill(5): row for row in rows}
    for filename in ("prelisting_feature_matrix.csv", "agent_augmented_feature_matrix.csv"):
        path = modeling / filename
        if not path.is_file():
            continue
        target_rows = _read_csv(path)
        for target in target_rows:
            source = by_company.get(str(target["company_id"]).zfill(5))
            if source:
                for column in (*_OFFERING_AMOUNT_COLUMNS, "fundamental_reporting_currency"):
                    target[column] = source.get(column, "")
            for column in normalized_columns:
                target[column] = source.get(column, "") if source else ""
        _write_csv(path, target_rows)

    # 统一金额列加入训练契约，原币种金额不再作为模型输入，但仍保留在 CSV 中审计。
    summary_path = modeling / "feature_matrix_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        feature_columns = list(summary.get("feature_columns") or [])
        native_amount_columns = {
            "fundamental_revenue_latest_native",
            "fundamental_operating_cash_flow_latest_native",
            "fundamental_operating_cash_flow_sum_3y_native",
        }
        feature_columns = [
            column for column in feature_columns if column not in native_amount_columns
        ]
        model_amount_columns = {
            "fundamental_revenue_latest_hkd_mn",
            "fundamental_operating_cash_flow_latest_hkd_mn",
            "fundamental_operating_cash_flow_sum_3y_hkd_mn",
        }
        for column in model_amount_columns:
            if column not in feature_columns:
                feature_columns.append(column)
        feature_columns = [
            column for column in feature_columns if column != "fundamental_fx_to_hkd"
        ]
        feature_columns.sort()
        summary["feature_columns"] = feature_columns
        summary["feature_count"] = len(feature_columns)
        summary["currency_normalization"] = {
            "target_currency": "HKD",
            "amount_unit": "million",
            "rate_date_policy": "latest_ecb_business_day_on_or_before_prospectus_date",
            "mop_policy": "1 HKD = 1.03 MOP",
            "through_year": through_year,
            "native_amount_columns_excluded_from_features": sorted(native_amount_columns),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    app()
