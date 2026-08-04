"""赛事交付数据的流式接入、质量检查和上市后标签生成。

原始交付包含数百份大型 PDF 和一张超过 1GB 的行情 CSV。本模块只对小型主表
做内存读取；行情表采用单次二进制流扫描，只保留赛事招股书对应股票的记录。
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, cast
from uuid import uuid4

import fitz

from ipo_risk.storage.files import sha256_file
from ipo_risk.storage.repositories import CatalogRepository


_DELIVERY_FILES = {
    "hkcompanyinfo.csv",
    "hksharedescription.csv",
    "hkshareeodprices.csv",
}
_PROSPECTUS_PATTERN = re.compile(
    r"^(?P<code>\d{5})_(?P<date>\d{2}-\d{2}-\d{4})_(?P<body>.+)\.pdf$",
    re.IGNORECASE,
)
_YEAR_DIRECTORY_PATTERN = re.compile(r"^(?P<year>20\d{2})_(?P<count>\d+)份$")
_EOD_REQUIRED_COLUMNS = {
    "S_INFO_WINDCODE",
    "TRADE_DT",
    "CRNCY_CODE",
    "S_DQ_OPEN",
    "S_DQ_HIGH",
    "S_DQ_LOW",
    "S_DQ_CLOSE",
    "S_DQ_VOLUME",
    "S_DQ_AMOUNT",
    "S_DQ_PRECLOSE",
    "S_DQ_AVGPRICE",
}
_SPECIAL_SECURITY_TERMS = (
    "ACQ",
    "收購",
    "收购",
    "認股權證",
    "认股权证",
    "Z25",
    "－Z",
    "-Z",
)


@dataclass(frozen=True)
class CompetitionProspectus:
    """由官方 PDF 文件名确定的一家公司和招股书版本。"""

    company_id: str
    company_name: str
    prospectus_date: date
    offering_description: str
    year: int
    source_path: Path


@dataclass(frozen=True, slots=True)
class MarketObservation:
    """一条赛事官方港股日行情。"""

    trade_date: date
    currency: str
    open_hkd: float | None
    high_hkd: float | None
    low_hkd: float | None
    close_hkd: float
    volume_shares: float | None
    turnover_hkd: float | None
    preclose_hkd: float | None
    average_price_hkd: float | None

    @property
    def change_hkd(self) -> float | None:
        if self.preclose_hkd is None:
            return None
        return self.close_hkd - self.preclose_hkd

    @property
    def change_pct(self) -> float | None:
        if self.preclose_hkd is None or self.preclose_hkd <= 0:
            return None
        return (self.close_hkd / self.preclose_hkd - 1) * 100

    @property
    def amplitude_pct(self) -> float | None:
        if (
            self.high_hkd is None
            or self.low_hkd is None
            or self.preclose_hkd is None
            or self.preclose_hkd <= 0
        ):
            return None
        return (self.high_hkd - self.low_hkd) / self.preclose_hkd * 100

    def to_market_row(self) -> dict[str, object]:
        """转换为项目现有日线行情 JSON 契约。"""
        return {
            "date": self.trade_date.isoformat(),
            "open_hkd": self.open_hkd,
            "high_hkd": self.high_hkd,
            "low_hkd": self.low_hkd,
            "close_hkd": self.close_hkd,
            "volume_shares": self.volume_shares,
            "turnover_hkd": self.turnover_hkd,
            "amplitude_pct": _rounded(self.amplitude_pct),
            "change_pct": _rounded(self.change_pct),
            "change_hkd": _rounded(self.change_hkd),
            "turnover_rate_pct": None,
        }


@dataclass(frozen=True)
class CompetitionImportResult:
    """一次完整赛事数据接入的结果。"""

    delivery_root: Path
    company_count: int
    listed_count: int
    full_label_count: int
    partial_label_count: int
    unavailable_count: int
    company_folders: tuple[str, ...]
    manifest_path: Path
    labels_path: Path
    summary_path: Path
    quality_report_path: Path
    hardlinked_files: int
    copied_files: int


@dataclass(frozen=True)
class _ReferenceTables:
    company_rows: tuple[dict[str, str], ...]
    share_rows_by_code: Mapping[str, tuple[dict[str, str], ...]]
    quality: dict[str, object]


@dataclass(frozen=True)
class _EodScan:
    rows_by_code: Mapping[str, tuple[MarketObservation, ...]]
    total_rows: int
    invalid_rows: int
    unique_codes: int
    min_trade_date: date | None
    max_trade_date: date | None
    duplicate_target_keys: int
    amount_scale: float


def normalize_stock_code(value: object) -> str | None:
    """把 Wind/HKEX 代码统一成五位股票代码；衍生品临时代码不强行映射。"""
    candidate = str(value or "").strip().upper()
    if candidate.endswith(".HK"):
        candidate = candidate[:-3]
    if not candidate.isdigit() or len(candidate) > 5:
        return None
    return candidate.zfill(5)


def discover_competition_root(path: Path) -> Path:
    """接受实际数据目录或其交付外层目录，返回含三张 CSV 的根目录。"""
    resolved = path.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    if _DELIVERY_FILES.issubset({item.name.lower() for item in resolved.iterdir()}):
        return resolved
    matches = [
        candidate
        for candidate in resolved.iterdir()
        if candidate.is_dir()
        and _DELIVERY_FILES.issubset(
            {item.name.lower() for item in candidate.iterdir()}
        )
    ]
    if len(matches) != 1:
        raise ValueError(f"无法唯一确定赛事数据根目录：{resolved}")
    return matches[0].resolve()


def parse_prospectus_filename(path: Path, *, year: int | None = None) -> CompetitionProspectus:
    """严格解析赛事 PDF 文件名，避免用模糊匹配把公司或日期关联错。"""
    match = _PROSPECTUS_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"招股书文件名不符合赛事契约：{path.name}")
    document_date = datetime.strptime(match.group("date"), "%d-%m-%Y").date()
    body = match.group("body")
    if "_" in body:
        company_name, offering_description = body.rsplit("_", 1)
    else:
        company_name, offering_description = body, ""
    company_name = company_name.strip()
    if not company_name:
        raise ValueError(f"招股书文件名缺少公司名称：{path.name}")
    return CompetitionProspectus(
        company_id=match.group("code"),
        company_name=company_name,
        prospectus_date=document_date,
        offering_description=offering_description.strip(),
        year=year or document_date.year,
        source_path=path.resolve(),
    )


def _discover_prospectuses(root: Path) -> tuple[CompetitionProspectus, ...]:
    entries: list[CompetitionProspectus] = []
    declared_counts: dict[int, int] = {}
    for directory in sorted(item for item in root.iterdir() if item.is_dir()):
        match = _YEAR_DIRECTORY_PATTERN.fullmatch(directory.name)
        if match is None:
            continue
        year = int(match.group("year"))
        declared_counts[year] = int(match.group("count"))
        entries.extend(
            parse_prospectus_filename(path, year=year)
            for path in sorted(directory.glob("*.pdf"))
        )
    if not entries:
        raise ValueError(f"赛事目录没有年度招股书 PDF：{root}")
    code_counts = Counter(entry.company_id for entry in entries)
    duplicates = sorted(code for code, count in code_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"赛事招股书股票代码重复：{', '.join(duplicates)}")
    actual_by_year = Counter(entry.year for entry in entries)
    mismatches = {
        year: {"declared": declared, "actual": actual_by_year[year]}
        for year, declared in declared_counts.items()
        if declared != actual_by_year[year]
    }
    if mismatches:
        raise ValueError(f"年度招股书数量与目录名不一致：{mismatches}")
    return tuple(sorted(entries, key=lambda item: (item.prospectus_date, item.company_id)))


def _read_dict_rows(path: Path) -> tuple[list[str], tuple[dict[str, str], ...], int]:
    rows: list[dict[str, str]] = []
    incomplete = 0
    with path.open("r", encoding="gb18030", newline="") as source:
        reader = csv.DictReader(source)
        headers = list(reader.fieldnames or [])
        for row in reader:
            if None in row or any(row.get(header) is None for header in headers):
                incomplete += 1
            rows.append({key: str(value or "") for key, value in row.items() if key is not None})
    return headers, tuple(rows), incomplete


def _load_reference_tables(
    root: Path,
    target_codes: set[str],
) -> _ReferenceTables:
    company_path = root / "hkcompanyinfo.csv"
    share_path = root / "hksharedescription.csv"
    company_headers, company_rows, company_incomplete = _read_dict_rows(company_path)
    share_headers, share_rows, share_incomplete = _read_dict_rows(share_path)
    share_by_code: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in share_rows:
        code = normalize_stock_code(row.get("S_INFO_CODE") or row.get("S_INFO_WINDCODE"))
        if code is not None:
            share_by_code[code].append(row)
    matched_codes = target_codes.intersection(share_by_code)
    share_ends_newline = _ends_with_newline(share_path)
    quality: dict[str, object] = {
        "encoding": "GB18030",
        "company_info": {
            "rows": len(company_rows),
            "columns": len(company_headers),
            "incomplete_rows": company_incomplete,
            "ends_with_newline": _ends_with_newline(company_path),
        },
        "share_description": {
            "file_size_bytes": share_path.stat().st_size,
            "rows": len(share_rows),
            "columns": len(share_headers),
            "incomplete_rows": share_incomplete,
            "ends_with_newline": share_ends_newline,
            "matched_prospectus_codes": len(matched_codes),
            "truncated": bool(share_incomplete or not share_ends_newline),
            "usable_as_security_master": bool(
                not share_incomplete
                and share_ends_newline
                and len(matched_codes) == len(target_codes)
            ),
        },
    }
    return _ReferenceTables(
        company_rows=company_rows,
        share_rows_by_code={
            code: tuple(rows) for code, rows in share_by_code.items()
        },
        quality=quality,
    )


def _ends_with_newline(path: Path) -> bool:
    with path.open("rb") as source:
        if path.stat().st_size == 0:
            return False
        source.seek(-1, os.SEEK_END)
        return source.read(1) in {b"\n", b"\r"}


def _fast_quoted_fields(line: bytes) -> list[bytes] | None:
    raw = line.rstrip(b"\r\n")
    if len(raw) < 2:
        return None
    if raw.startswith(b'"') and raw.endswith(b'"'):
        return raw[1:-1].split(b'","')
    try:
        decoded = raw.decode("ascii")
    except UnicodeDecodeError:
        return None
    parsed = next(csv.reader([decoded]), None)
    return [value.encode("ascii") for value in parsed] if parsed is not None else None


def _optional_float(value: bytes) -> float | None:
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _scan_eod(
    path: Path,
    prospectuses: Sequence[CompetitionProspectus],
) -> _EodScan:
    target_dates = {
        entry.company_id: int(entry.prospectus_date.strftime("%Y%m%d"))
        for entry in prospectuses
    }
    rows_by_code: dict[str, list[MarketObservation]] = defaultdict(list)
    all_codes: set[str] = set()
    total_rows = 0
    invalid_rows = 0
    min_trade_date: date | None = None
    max_trade_date: date | None = None
    amount_ratios: list[float] = []
    with path.open("rb", buffering=1024 * 1024) as source:
        header_fields = _fast_quoted_fields(source.readline())
        if not header_fields:
            raise ValueError(f"行情 CSV 缺少表头：{path}")
        headers = [field.decode("ascii") for field in header_fields]
        missing = sorted(_EOD_REQUIRED_COLUMNS.difference(headers))
        if missing:
            raise ValueError(f"行情 CSV 缺少字段：{', '.join(missing)}")
        index = {name: headers.index(name) for name in _EOD_REQUIRED_COLUMNS}
        for line in source:
            total_rows += 1
            fields = _fast_quoted_fields(line)
            if fields is None or len(fields) != len(headers):
                invalid_rows += 1
                continue
            code = normalize_stock_code(fields[index["S_INFO_WINDCODE"]].decode("ascii"))
            raw_trade_date = fields[index["TRADE_DT"]]
            if code is None or len(raw_trade_date) != 8 or not raw_trade_date.isdigit():
                invalid_rows += 1
                continue
            all_codes.add(code)
            trade_date_number = int(raw_trade_date)
            try:
                trade_date = datetime.strptime(
                    raw_trade_date.decode("ascii"), "%Y%m%d"
                ).date()
            except ValueError:
                invalid_rows += 1
                continue
            if min_trade_date is None or trade_date < min_trade_date:
                min_trade_date = trade_date
            if max_trade_date is None or trade_date > max_trade_date:
                max_trade_date = trade_date

            volume = _optional_float(fields[index["S_DQ_VOLUME"]])
            amount = _optional_float(fields[index["S_DQ_AMOUNT"]])
            average = _optional_float(fields[index["S_DQ_AVGPRICE"]])
            if (
                len(amount_ratios) < 2000
                and volume is not None
                and volume > 0
                and amount is not None
                and amount > 0
                and average is not None
                and average > 0
            ):
                amount_ratios.append(average * volume / amount)

            prospectus_date = target_dates.get(code)
            if prospectus_date is None or trade_date_number < prospectus_date:
                continue
            close = _optional_float(fields[index["S_DQ_CLOSE"]])
            if close is None or close <= 0:
                continue
            rows_by_code[code].append(
                MarketObservation(
                    trade_date=trade_date,
                    currency=fields[index["CRNCY_CODE"]].decode("ascii") or "HKD",
                    open_hkd=_optional_float(fields[index["S_DQ_OPEN"]]),
                    high_hkd=_optional_float(fields[index["S_DQ_HIGH"]]),
                    low_hkd=_optional_float(fields[index["S_DQ_LOW"]]),
                    close_hkd=close,
                    volume_shares=volume,
                    turnover_hkd=amount,
                    preclose_hkd=_optional_float(fields[index["S_DQ_PRECLOSE"]]),
                    average_price_hkd=average,
                )
            )
    amount_scale = float(median(amount_ratios)) if amount_ratios else 1000.0
    normalized_scale = 1000.0 if 950 <= amount_scale <= 1050 else amount_scale
    duplicate_target_keys = 0
    normalized_rows: dict[str, tuple[MarketObservation, ...]] = {}
    for code, values in rows_by_code.items():
        ordered = sorted(values, key=lambda item: item.trade_date)
        duplicate_target_keys += len(ordered) - len(
            {item.trade_date for item in ordered}
        )
        normalized_rows[code] = tuple(
            MarketObservation(
                trade_date=item.trade_date,
                currency=item.currency,
                open_hkd=item.open_hkd,
                high_hkd=item.high_hkd,
                low_hkd=item.low_hkd,
                close_hkd=item.close_hkd,
                volume_shares=item.volume_shares,
                turnover_hkd=(
                    item.turnover_hkd * normalized_scale
                    if item.turnover_hkd is not None
                    else None
                ),
                preclose_hkd=item.preclose_hkd,
                average_price_hkd=item.average_price_hkd,
            )
            for item in ordered
        )
    return _EodScan(
        rows_by_code=normalized_rows,
        total_rows=total_rows,
        invalid_rows=invalid_rows,
        unique_codes=len(all_codes),
        min_trade_date=min_trade_date,
        max_trade_date=max_trade_date,
        duplicate_target_keys=duplicate_target_keys,
        amount_scale=normalized_scale,
    )


def _choose_share_row(
    rows: Sequence[dict[str, str]],
    prospectus_date: date,
) -> dict[str, str] | None:
    if not rows:
        return None

    def score(row: dict[str, str]) -> tuple[int, int, int]:
        list_date = _parse_compact_date(row.get("S_INFO_LISTDATE"))
        distance = (
            abs((list_date - prospectus_date).days)
            if list_date is not None
            else 999999
        )
        return (
            int((row.get("SECURITYTYPE") or "").upper() == "ORD"),
            int(list_date is not None and list_date >= prospectus_date),
            -distance,
        )

    return max(rows, key=score)


def _parse_compact_date(value: object) -> date | None:
    candidate = str(value or "").strip()
    if not re.fullmatch(r"\d{8}", candidate):
        return None
    try:
        return datetime.strptime(candidate, "%Y%m%d").date()
    except ValueError:
        return None


def _positive_float(value: object) -> float | None:
    try:
        parsed = float(str(value or "").strip())
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _listing_and_offer(
    entry: CompetitionProspectus,
    rows: Sequence[MarketObservation],
    share_rows: Sequence[dict[str, str]],
) -> tuple[date | None, str | None, float | None, str | None]:
    share = _choose_share_row(share_rows, entry.prospectus_date)
    official_date = _parse_compact_date(share.get("S_INFO_LISTDATE")) if share else None
    official_price = _positive_float(share.get("S_INFO_LISTPRICE")) if share else None
    listing_date = official_date or (rows[0].trade_date if rows else None)
    listing_source = (
        "hksharedescription.S_INFO_LISTDATE"
        if official_date
        else ("hkshareeodprices.first_trade_on_or_after_prospectus" if rows else None)
    )
    proxy_price = (
        rows[0].preclose_hkd
        if rows and rows[0].preclose_hkd is not None and rows[0].preclose_hkd > 0
        else None
    )
    offer_price = official_price or proxy_price
    offer_source = (
        "hksharedescription.S_INFO_LISTPRICE"
        if official_price
        else ("hkshareeodprices.first_day_preclose_proxy" if proxy_price else None)
    )
    return listing_date, listing_source, offer_price, offer_source


def _split_for_year(year: int) -> str:
    if year <= 2023:
        return "train"
    if year == 2024:
        return "validation"
    return "test"


def _max_drawdown_pct(prices: Sequence[float]) -> float | None:
    valid = [value for value in prices if value > 0 and math.isfinite(value)]
    if len(valid) < 2:
        return None
    peak = valid[0]
    drawdown = 0.0
    for value in valid:
        peak = max(peak, value)
        drawdown = min(drawdown, (value / peak - 1) * 100)
    return drawdown


def build_horizon_labels(
    *,
    entry: CompetitionProspectus,
    rows: Sequence[MarketObservation],
    listing_date: date | None,
    listing_date_source: str | None,
    offer_price_hkd: float | None,
    offer_price_source: str | None,
    horizons: Sequence[int] = (1, 5, 20, 60),
    significant_drop_threshold_pct: float = -10.0,
) -> list[dict[str, object]]:
    """生成相对发行价的客观标签；缺少发行价时保留可计算的部分指标。"""
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise ValueError("horizons must contain positive trading-day windows")
    labels: list[dict[str, object]] = []
    split = _split_for_year(entry.year)
    for horizon in horizons:
        base: dict[str, object] = {
            "company_id": entry.company_id,
            "company_name": entry.company_name,
            "prospectus_date": entry.prospectus_date.isoformat(),
            "listing_date": listing_date.isoformat() if listing_date else None,
            "listing_date_source": listing_date_source,
            "horizon": horizon,
            "significant_drop_threshold_pct": significant_drop_threshold_pct,
            "offer_price_hkd": offer_price_hkd,
            "offer_price_source": offer_price_source,
            "available_trading_days": len(rows),
            "split": split,
        }
        if len(rows) < horizon:
            base.update(
                {
                    "observation_date": None,
                    "close_hkd": None,
                    "return_vs_offer_pct": None,
                    "return_vs_first_close_pct": None,
                    "broke_issue_price": None,
                    "significant_drop": None,
                    "max_drawdown_pct": None,
                    "label_status": "unavailable",
                    "exclusion_reason": (
                        "no_market_data"
                        if not rows
                        else f"requires_{horizon}_trading_days"
                    ),
                }
            )
            labels.append(base)
            continue
        window = list(rows[:horizon])
        close = window[-1].close_hkd
        return_vs_first = (
            (close / window[0].close_hkd - 1) * 100
            if window[0].close_hkd > 0
            else None
        )
        return_vs_offer = (
            (close / offer_price_hkd - 1) * 100
            if offer_price_hkd is not None and offer_price_hkd > 0
            else None
        )
        lows = [
            item.low_hkd
            for item in window
            if item.low_hkd is not None and item.low_hkd > 0
        ]
        prices_for_drawdown = [item.close_hkd for item in window]
        if offer_price_hkd is not None and offer_price_hkd > 0:
            prices_for_drawdown.insert(0, offer_price_hkd)
        complete = offer_price_hkd is not None and offer_price_hkd > 0
        broke_issue = (
            min(lows) < offer_price_hkd
            if offer_price_hkd is not None and offer_price_hkd > 0 and lows
            else None
        )
        base.update(
            {
                "observation_date": window[-1].trade_date.isoformat(),
                "close_hkd": close,
                "return_vs_offer_pct": _rounded(return_vs_offer),
                "return_vs_first_close_pct": _rounded(return_vs_first),
                "broke_issue_price": broke_issue,
                "significant_drop": (
                    return_vs_offer <= significant_drop_threshold_pct
                    if return_vs_offer is not None
                    else None
                ),
                "max_drawdown_pct": _rounded(
                    _max_drawdown_pct(prices_for_drawdown)
                ),
                "label_status": "complete" if complete else "partial",
                "exclusion_reason": None if complete else "missing_offer_price",
            }
        )
        labels.append(base)
    return labels


def _rounded(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None and math.isfinite(value) else None


def _special_security(entry: CompetitionProspectus) -> bool:
    name = f"{entry.company_name} {entry.offering_description}".upper()
    return any(term.upper() in name for term in _SPECIAL_SECURITY_TERMS)


def _portable(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _backup_existing(path: Path, backup_directory: Path, prefix: str) -> None:
    if not path.is_file():
        return
    digest = sha256_file(path)
    backup = backup_directory / f"{prefix}_{digest[:12]}{path.suffix}"
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)


def _link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            if os.path.samefile(source, target):
                return "reused"
        except OSError:
            pass
        if source.stat().st_size == target.stat().st_size:
            if sha256_file(source) == sha256_file(target):
                return "reused"
        raise FileExistsError(f"目标文件已存在且内容不同：{target}")
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def materialize_delivery_subset(
    source_root: Path,
    target_root: Path,
) -> dict[str, int]:
    """把运行必需的官方数据安全放入项目；跳过与解压 PDF 重复的年度 ZIP。"""
    source = discover_competition_root(source_root)
    target = target_root.resolve()
    if source == target:
        return {"hardlink": 0, "copy": 0, "reused": 0}
    counts: Counter[str] = Counter()
    for entry in _discover_prospectuses(source):
        relative = entry.source_path.relative_to(source)
        counts[_link_or_copy(entry.source_path, target / relative)] += 1
    for path in sorted(source.iterdir()):
        if not path.is_file() or path.suffix.lower() == ".zip":
            continue
        if path.suffix.lower() not in {".csv", ".pdf"}:
            continue
        counts[_link_or_copy(path, target / path.name)] += 1
    return {
        "hardlink": counts["hardlink"],
        "copy": counts["copy"],
        "reused": counts["reused"],
    }


def _company_folder_for(
    company_root: Path,
    entry: CompetitionProspectus,
) -> str:
    matches = [
        candidate.name
        for candidate in company_root.glob(f"{entry.company_id}_*")
        if candidate.is_dir() and (candidate / "公司信息.json").is_file()
    ]
    if len(matches) > 1:
        raise ValueError(f"股票代码{entry.company_id}存在多个公司目录：{matches}")
    return matches[0] if matches else f"{entry.company_id}_{entry.company_name}"


def _existing_document_by_hash(
    metadata: Mapping[str, Any],
    source_hash: str,
    project_root: Path,
) -> tuple[str, Path] | None:
    documents = metadata.get("documents", {})
    if not isinstance(documents, dict):
        return None
    for key, value in documents.items():
        if not isinstance(value, dict):
            continue
        if str(value.get("sha256", "")).lower() != source_hash:
            continue
        path = project_root / str(value.get("local_file", ""))
        if path.is_file():
            return str(key), path.resolve()
    return None


def _merge_company_metadata(
    *,
    existing: dict[str, Any],
    entry: CompetitionProspectus,
    company_folder: str,
    project_root: Path,
    source_hash: str,
    page_count: int,
    document_path: Path,
    delivery_root: Path,
    listing_date: date | None,
    listing_date_source: str | None,
    offer_price: float | None,
    offer_price_source: str | None,
    share_row: dict[str, str] | None,
    company_row: dict[str, str] | None,
) -> dict[str, Any]:
    metadata = dict(existing)
    metadata.setdefault("schema_version", 2)
    metadata["company_id"] = entry.company_id
    metadata.setdefault("exchange", "HKEX")
    metadata.setdefault("market", "Hong Kong")
    metadata.setdefault("ticker", f"{entry.company_id}.HK")
    metadata.setdefault("company_name_zh_official", entry.company_name)
    metadata.setdefault("as_of_date", entry.prospectus_date.isoformat())
    if listing_date is not None and not metadata.get("listing_date"):
        metadata["listing_date"] = listing_date.isoformat()
        metadata["listing_date_source"] = listing_date_source
    offering = metadata.setdefault("offering", {})
    if not isinstance(offering, dict):
        offering = {}
        metadata["offering"] = offering
    if offer_price is not None and not offering.get("offer_price_hkd"):
        offering["offer_price_hkd"] = offer_price
        offering["offer_price_source"] = offer_price_source

    documents = metadata.setdefault("documents", {})
    if not isinstance(documents, dict):
        raise ValueError(f"{company_folder}/公司信息.json 的 documents 必须是对象")
    existing_document = _existing_document_by_hash(metadata, source_hash, project_root)
    if existing_document is None:
        key = (
            "prospectus"
            if "prospectus" not in documents
            else f"prospectus_competition_{source_hash[:8]}"
        )
        documents[key] = {
            "document_type": "prospectus",
            "title": entry.source_path.stem,
            "local_file": _portable(document_path, project_root),
            "document_date": entry.prospectus_date.isoformat(),
            "available_at_date": entry.prospectus_date.isoformat(),
            "language": "zh-Hant",
            "source_kind": "competition_official",
            "source_file_name": entry.source_path.name,
            "official_delivery_file": _portable(entry.source_path, project_root),
            "file_size_bytes": entry.source_path.stat().st_size,
            "sha256": source_hash,
            "page_count": page_count,
            "text_layer_status": "unknown_until_parse",
        }
    else:
        key, _ = existing_document
        declared = documents[key]
        if isinstance(declared, dict):
            declared.setdefault(
                "official_delivery_file", _portable(entry.source_path, project_root)
            )
            declared.setdefault("official_source_kind", "competition_official")

    metadata["competition_dataset"] = {
        "source": "东吴证券研究生创新大赛官方数据",
        "delivery_root": _portable(delivery_root, project_root),
        "original_file_name": entry.source_path.name,
        "prospectus_year": entry.year,
        "offering_description": entry.offering_description,
        "share_description_matched": share_row is not None,
        "company_info_matched": company_row is not None,
        "risk_gold_standard_available": False,
        "market_label_scope": "post_listing_only",
    }
    if company_row is not None:
        metadata["company_profile_source"] = {
            "company_code": company_row.get("S_INFO_COMPCODE"),
            "company_name": company_row.get("S_INFO_COMPNAME"),
            "company_name_en": company_row.get("COMP_NAME_ENG"),
            "founded_date": company_row.get("FOUNDDATE"),
            "briefing": company_row.get("BRIEFING"),
            "business_scope": company_row.get("BUSINESSSCOPE"),
            "currency": company_row.get("CRNY_CODE"),
        }
        metadata.setdefault("company_name_en", company_row.get("COMP_NAME_ENG") or None)
    metadata["data_layout"] = {
        "company_folder": company_folder,
        "prospectus_dir": f"data/raw/prospectus/{company_folder}",
        "company_info_dir": f"data/raw/company_info/{company_folder}",
        "market_dir": f"data/raw/market/{company_folder}",
    }
    return metadata


def _company_row_for_share(
    share_row: dict[str, str] | None,
    company_rows: Sequence[dict[str, str]],
) -> dict[str, str] | None:
    if share_row is None:
        return None
    company_code = (share_row.get("S_INFO_COMPCODE") or "").strip()
    if not company_code:
        return None
    return next(
        (
            row
            for row in company_rows
            if (row.get("S_INFO_COMPCODE") or "").strip() == company_code
        ),
        None,
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0]) if rows else []
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False, default=str))
                target.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def reconcile_backed_up_market_dataset(
    repository: CatalogRepository,
    *,
    project_root: Path,
    company_folder: str,
) -> int:
    """在登记官方行情前，把同路径旧快照改指向已核验的审计备份。"""
    project = project_root.resolve()
    metadata_path = (
        project / "data" / "raw" / "company_info" / company_folder / "公司信息.json"
    )
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    company_id = str(metadata["company_id"])
    current_path = project / "data" / "raw" / "market" / company_folder / "日线行情.json"
    if not current_path.is_file():
        return 0
    current_portable = _portable(current_path, project)
    current_hash = sha256_file(current_path)
    relocated = 0
    for dataset in repository.list_datasets(company_id):
        if dataset.local_file != current_portable or dataset.source_hash == current_hash:
            continue
        backup = (
            project
            / "data"
            / "interim"
            / "audit"
            / company_folder
            / f"pre_competition_market_{dataset.source_hash[:12]}.json"
        )
        if not backup.is_file() or sha256_file(backup) != dataset.source_hash:
            raise ValueError(
                f"旧行情登记与当前文件冲突，但没有找到哈希一致的备份：{company_folder}"
            )
        repository.relocate_dataset_file(dataset.dataset_id, _portable(backup, project))
        relocated += 1
    return relocated


def audit_competition_delivery(source_root: Path) -> dict[str, object]:
    """执行完整只读审计，返回可序列化报告。"""
    root = discover_competition_root(source_root)
    prospectuses = _discover_prospectuses(root)
    target_codes = {entry.company_id for entry in prospectuses}
    references = _load_reference_tables(root, target_codes)
    eod = _scan_eod(root / "hkshareeodprices.csv", prospectuses)
    coverage = {
        horizon: sum(
            len(eod.rows_by_code.get(entry.company_id, ())) >= horizon
            for entry in prospectuses
        )
        for horizon in (1, 5, 20, 60)
    }
    return {
        "schema_version": 1,
        "delivery_root": str(root),
        "prospectus_count": len(prospectuses),
        "prospectus_by_year": dict(sorted(Counter(item.year for item in prospectuses).items())),
        "reference_tables": references.quality,
        "eod": {
            "rows": eod.total_rows,
            "invalid_rows": eod.invalid_rows,
            "unique_codes": eod.unique_codes,
            "min_trade_date": eod.min_trade_date,
            "max_trade_date": eod.max_trade_date,
            "duplicate_target_keys": eod.duplicate_target_keys,
            "amount_to_hkd_scale": eod.amount_scale,
        },
        "market_coverage": {
            "codes_with_market_data": sum(
                bool(eod.rows_by_code.get(entry.company_id))
                for entry in prospectuses
            ),
            "horizons": coverage,
            "without_market_data": [
                entry.company_id
                for entry in prospectuses
                if not eod.rows_by_code.get(entry.company_id)
            ],
        },
    }


def import_competition_delivery(
    source_root: Path,
    *,
    project_root: Path,
    delivery_target: Path | None = None,
    horizons: Sequence[int] = (1, 5, 20, 60),
    significant_drop_threshold_pct: float = -10.0,
    imported_at: datetime | None = None,
) -> CompetitionImportResult:
    """标准化完整赛事交付，并生成公司资料、官方行情和建模标签。"""
    project = project_root.resolve()
    target = (
        delivery_target.resolve()
        if delivery_target is not None
        else project / "data" / "raw" / "competition" / "dongwu_hk_ipo_2020_2025"
    )
    try:
        target.relative_to(project)
    except ValueError as exc:
        raise ValueError(f"赛事原始数据目标必须位于项目内：{target}") from exc
    materialized = materialize_delivery_subset(source_root, target)
    delivery = discover_competition_root(target)
    prospectuses = _discover_prospectuses(delivery)
    target_codes = {entry.company_id for entry in prospectuses}
    references = _load_reference_tables(delivery, target_codes)
    eod = _scan_eod(delivery / "hkshareeodprices.csv", prospectuses)
    imported = imported_at or datetime.now(timezone.utc)

    company_root = project / "data" / "raw" / "company_info"
    prospectus_root = project / "data" / "raw" / "prospectus"
    market_root = project / "data" / "raw" / "market"
    audit_root = project / "data" / "interim" / "audit"
    processed_root = project / "data" / "processed" / "competition"
    company_root.mkdir(parents=True, exist_ok=True)
    prospectus_root.mkdir(parents=True, exist_ok=True)
    market_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    company_folders: list[str] = []
    full_label_codes: set[str] = set()
    partial_label_codes: set[str] = set()
    unavailable_codes: set[str] = set()

    for position, entry in enumerate(prospectuses, start=1):
        company_folder = _company_folder_for(company_root, entry)
        company_folders.append(company_folder)
        rows = eod.rows_by_code.get(entry.company_id, ())
        share_rows = references.share_rows_by_code.get(entry.company_id, ())
        share_row = _choose_share_row(share_rows, entry.prospectus_date)
        company_row = _company_row_for_share(share_row, references.company_rows)
        listing_date, listing_source, offer_price, offer_source = _listing_and_offer(
            entry, rows, share_rows
        )

        source_hash = sha256_file(entry.source_path)
        document = fitz.open(entry.source_path)
        try:
            if document.needs_pass or document.page_count < 1:
                raise ValueError(f"招股书加密或没有页面：{entry.source_path}")
            page_count = document.page_count
        finally:
            document.close()

        metadata_path = company_root / company_folder / "公司信息.json"
        existing: dict[str, Any] = {}
        if metadata_path.is_file():
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError(f"公司信息顶层必须是对象：{metadata_path}")
            existing = loaded
        existing_document = _existing_document_by_hash(existing, source_hash, project)
        if existing_document is not None:
            document_path = existing_document[1]
        else:
            document_path = (
                prospectus_root
                / company_folder
                / f"招股书_{entry.prospectus_date.isoformat()}_{source_hash[:12]}.pdf"
            )
            _link_or_copy(entry.source_path, document_path)

        metadata = _merge_company_metadata(
            existing=existing,
            entry=entry,
            company_folder=company_folder,
            project_root=project,
            source_hash=source_hash,
            page_count=page_count,
            document_path=document_path,
            delivery_root=delivery,
            listing_date=listing_date,
            listing_date_source=listing_source,
            offer_price=offer_price,
            offer_price_source=offer_source,
            share_row=share_row,
            company_row=company_row,
        )
        if metadata != existing:
            if metadata_path.is_file():
                _backup_existing(
                    metadata_path,
                    audit_root / company_folder,
                    "pre_competition_company_info",
                )
            _atomic_json(metadata_path, metadata)

        market_path = market_root / company_folder / "日线行情.json"
        if rows:
            existing_market: dict[str, Any] | None = None
            existing_imported_at: str | None = None
            if market_path.is_file():
                loaded_market = json.loads(market_path.read_text(encoding="utf-8"))
                if isinstance(loaded_market, dict):
                    existing_market = loaded_market
                    source_metadata = loaded_market.get("source", {})
                    if (
                        isinstance(source_metadata, dict)
                        and source_metadata.get("provider") == "Wind赛事官方数据"
                        and source_metadata.get("source_file")
                        == _portable(delivery / "hkshareeodprices.csv", project)
                    ):
                        existing_imported_at = str(
                            source_metadata.get("imported_at") or ""
                        ) or None
            market_payload = {
                "schema_version": 2,
                "company_id": entry.company_id,
                "ticker": f"{entry.company_id}.HK",
                "currency": rows[0].currency or "HKD",
                "interval": "1d",
                "start_date": rows[0].trade_date.isoformat(),
                "end_date": rows[-1].trade_date.isoformat(),
                "row_count": len(rows),
                "usage_scope": "post_listing_label_and_backtest_only",
                "pre_listing_feature_eligible": False,
                "source": {
                    "provider": "Wind赛事官方数据",
                    "source_file": _portable(
                        delivery / "hkshareeodprices.csv", project
                    ),
                    "imported_at": existing_imported_at or imported.isoformat(),
                    "encoding": "GB18030",
                    "amount_normalization": (
                        f"S_DQ_AMOUNT × {eod.amount_scale:g} = turnover_hkd"
                    ),
                    "adjustment": "none",
                },
                "data_quality": {
                    "decision": "赛事官方行情，仅用于上市后标签和回测。",
                    "anomaly_dates": [],
                    "listing_date_candidate": (
                        listing_date.isoformat() if listing_date else None
                    ),
                    "listing_date_source": listing_source,
                    "offer_price_hkd": offer_price,
                    "offer_price_source": offer_source,
                    "turnover_rate_unavailable": True,
                },
                "rows": [row.to_market_row() for row in rows],
            }
            if market_payload != existing_market:
                if market_path.is_file():
                    _backup_existing(
                        market_path,
                        audit_root / company_folder,
                        "pre_competition_market",
                    )
                _atomic_json(market_path, market_payload)

        company_labels = build_horizon_labels(
            entry=entry,
            rows=rows,
            listing_date=listing_date,
            listing_date_source=listing_source,
            offer_price_hkd=offer_price,
            offer_price_source=offer_source,
            horizons=horizons,
            significant_drop_threshold_pct=significant_drop_threshold_pct,
        )
        labels.extend(company_labels)
        statuses = {str(item["label_status"]) for item in company_labels}
        if statuses == {"complete"}:
            full_label_codes.add(entry.company_id)
        elif "partial" in statuses:
            partial_label_codes.add(entry.company_id)
        else:
            unavailable_codes.add(entry.company_id)
        manifest.append(
            {
                "sequence": position,
                "company_id": entry.company_id,
                "ticker": f"{entry.company_id}.HK",
                "company_name": entry.company_name,
                "prospectus_date": entry.prospectus_date.isoformat(),
                "prospectus_year": entry.year,
                "offering_description": entry.offering_description,
                "company_folder": company_folder,
                "prospectus_file": _portable(document_path, project),
                "source_hash": source_hash,
                "page_count": page_count,
                "listing_date": listing_date.isoformat() if listing_date else None,
                "listing_date_source": listing_source,
                "offer_price_hkd": offer_price,
                "offer_price_source": offer_source,
                "market_row_count": len(rows),
                "market_status": "available" if rows else "unavailable",
                "sample_type": (
                    "special_security"
                    if _special_security(entry)
                    else ("listed_ipo" if rows else "unlisted_or_withdrawn")
                ),
                "label_status": (
                    "complete"
                    if entry.company_id in full_label_codes
                    else (
                        "partial"
                        if entry.company_id in partial_label_codes
                        else "unavailable"
                    )
                ),
                "split": _split_for_year(entry.year),
                "risk_gold_standard_available": False,
            }
        )

    manifest_path = processed_root / "company_manifest.jsonl"
    labels_path = processed_root / "market_labels.csv"
    summary_path = processed_root / "dataset_summary.json"
    quality_report_path = audit_root / "competition" / "data_quality.json"
    _write_jsonl(manifest_path, manifest)
    _write_csv(labels_path, labels)
    _write_jsonl(processed_root / "market_labels.jsonl", labels)
    split_counts = Counter(str(item["split"]) for item in manifest)
    share_quality = cast(
        dict[str, object],
        references.quality["share_description"],
    )
    summary = {
        "schema_version": 1,
        "generated_at": imported.isoformat(),
        "source": _portable(delivery, project),
        "company_count": len(manifest),
        "listed_count": sum(item["market_status"] == "available" for item in manifest),
        "full_label_count": len(full_label_codes),
        "partial_label_count": len(partial_label_codes),
        "unavailable_count": len(unavailable_codes),
        "horizons": list(horizons),
        "primary_horizon": 5,
        "significant_drop_definition": (
            f"第N个交易日收盘价相对发行价收益率 <= "
            f"{significant_drop_threshold_pct:g}%"
        ),
        "split_strategy": {
            "train": "prospectus_year <= 2023",
            "validation": "prospectus_year == 2024",
            "test": "prospectus_year >= 2025",
            "counts": dict(sorted(split_counts.items())),
        },
        "risk_gold_standard_available": False,
        "market_labels_are_post_listing_only": True,
        "official_security_master_usable": share_quality[
            "usable_as_security_master"
        ],
    }
    _atomic_json(summary_path, summary)
    quality_report = {
        "schema_version": 1,
        "generated_at": imported.isoformat(),
        "delivery_root": _portable(delivery, project),
        "prospectus_count": len(prospectuses),
        "reference_tables": references.quality,
        "eod": {
            "rows": eod.total_rows,
            "invalid_rows": eod.invalid_rows,
            "unique_codes": eod.unique_codes,
            "min_trade_date": eod.min_trade_date,
            "max_trade_date": eod.max_trade_date,
            "duplicate_target_keys": eod.duplicate_target_keys,
            "amount_to_hkd_scale": eod.amount_scale,
        },
        "coverage": {
            "listed": len(manifest) - len(unavailable_codes),
            "complete_labels": sorted(full_label_codes),
            "partial_labels": sorted(partial_label_codes),
            "unavailable": sorted(unavailable_codes),
        },
    }
    _atomic_json(quality_report_path, quality_report)
    return CompetitionImportResult(
        delivery_root=delivery,
        company_count=len(manifest),
        listed_count=sum(item["market_status"] == "available" for item in manifest),
        full_label_count=len(full_label_codes),
        partial_label_count=len(partial_label_codes),
        unavailable_count=len(unavailable_codes),
        company_folders=tuple(company_folders),
        manifest_path=manifest_path,
        labels_path=labels_path,
        summary_path=summary_path,
        quality_report_path=quality_report_path,
        hardlinked_files=materialized["hardlink"],
        copied_files=materialized["copy"],
    )


__all__ = [
    "CompetitionImportResult",
    "CompetitionProspectus",
    "MarketObservation",
    "audit_competition_delivery",
    "build_horizon_labels",
    "discover_competition_root",
    "import_competition_delivery",
    "materialize_delivery_subset",
    "normalize_stock_code",
    "parse_prospectus_filename",
    "reconcile_backed_up_market_dataset",
]
