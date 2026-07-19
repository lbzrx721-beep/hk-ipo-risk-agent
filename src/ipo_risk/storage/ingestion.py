"""把标准公司目录中的原始资料幂等登记到SQLite目录。"""

from __future__ import annotations

import mimetypes
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from ipo_risk.config import load_config
from ipo_risk.domain import CompanyRecord, DatasetRecord, DocumentRecord
from ipo_risk.storage.files import read_json, sha256_file
from ipo_risk.storage.repositories import CatalogRepository


_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class CompanyIngestionResult:
    """一次公司资料登记的完整结果。"""

    company: CompanyRecord
    documents: tuple[DocumentRecord, ...]
    datasets: tuple[DatasetRecord, ...]
    warnings: tuple[str, ...]


def _safe_folder_name(value: str) -> str:
    candidate = value.strip()
    if not candidate or candidate in {".", ".."} or Path(candidate).name != candidate:
        raise ValueError("company_folder must be one safe path segment")
    return candidate


def _source_type(value: str) -> str:
    candidate = value.strip().lower().replace("-", "_")
    if not _TYPE_PATTERN.fullmatch(candidate):
        raise ValueError(f"source type must be a lowercase slug: {value}")
    return candidate


def _portable_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"raw source must stay inside project root: {resolved}") from exc


def _resolve_source(project_root: Path, local_file: str) -> Path:
    path = (project_root / local_file).resolve()
    try:
        path.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"source path escapes project root: {local_file}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _parse_date(value: Any, field_name: str) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be an ISO date")
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO date: {value}") from exc


def _optional_date(value: Any, field_name: str) -> date | None:
    return None if value in (None, "") else _parse_date(value, field_name)


def _download_date(metadata: dict[str, Any], fallback: date) -> date:
    source = metadata.get("source", {})
    if isinstance(source, dict) and source.get("downloaded_at"):
        return _parse_date(source["downloaded_at"], "source.downloaded_at")
    if metadata.get("downloaded_at"):
        return _parse_date(metadata["downloaded_at"], "downloaded_at")
    return fallback


def make_document_id(
    company_id: str,
    document_type: str,
    document_date: date,
    source_hash: str,
) -> str:
    """由公司、资料类型、公开日期和内容哈希生成稳定文档版本ID。"""
    kind = _source_type(document_type)
    return f"doc_{company_id}_{kind}_{document_date.isoformat()}_{source_hash[:8]}"


def make_dataset_id(
    company_id: str,
    dataset_type: str,
    available_at: date,
    source_hash: str,
) -> str:
    """由数据集类型、可用日期和内容哈希生成稳定快照ID。"""
    kind = _source_type(dataset_type)
    return f"ds_{company_id}_{kind}_{available_at.isoformat()}_{source_hash[:8]}"


def _verify_declared_file(
    source_path: Path,
    metadata: dict[str, Any],
) -> tuple[str, int]:
    actual_hash = sha256_file(source_path)
    declared_hash = metadata.get("sha256")
    if declared_hash and str(declared_hash).strip().lower() != actual_hash:
        raise ValueError(f"SHA-256 does not match metadata: {source_path}")
    actual_size = source_path.stat().st_size
    declared_size = metadata.get("file_size_bytes")
    if declared_size is not None and int(declared_size) != actual_size:
        raise ValueError(f"file size does not match metadata: {source_path}")
    return actual_hash, actual_size


def _build_document(
    *,
    project_root: Path,
    company_id: str,
    document_type: str,
    metadata: dict[str, Any],
    registered_at: datetime,
) -> DocumentRecord:
    kind = _source_type(str(metadata.get("document_type") or document_type))
    source_path = _resolve_source(project_root, str(metadata["local_file"]))
    source_hash, file_size = _verify_declared_file(source_path, metadata)
    document_date = _parse_date(metadata.get("document_date"), "document_date")
    available_at = _parse_date(
        metadata.get("available_at_date", document_date.isoformat()),
        "available_at_date",
    )
    media_type = mimetypes.guess_type(source_path.name)[0] or "application/octet-stream"
    return DocumentRecord(
        document_id=make_document_id(company_id, kind, document_date, source_hash),
        company_id=company_id,
        document_type=kind,
        title=str(metadata.get("title") or source_path.stem),
        local_file=_portable_path(source_path, project_root),
        source_hash=source_hash,
        document_date=document_date,
        available_at=available_at,
        language=metadata.get("language"),
        source_kind=metadata.get("source_kind"),
        source_url=metadata.get("source_url"),
        media_type=media_type,
        file_size_bytes=file_size,
        page_count=metadata.get("page_count"),
        text_layer_status=metadata.get("text_layer_status"),
        metadata_status="declared",
        metadata=metadata,
        registered_at=registered_at,
    )


def _build_declared_dataset(
    *,
    project_root: Path,
    company_id: str,
    dataset_type: str,
    metadata: dict[str, Any],
    registered_at: datetime,
) -> DatasetRecord:
    kind = _source_type(str(metadata.get("dataset_type") or dataset_type))
    source_path = _resolve_source(project_root, str(metadata["local_file"]))
    source_hash, _ = _verify_declared_file(source_path, metadata)
    available_at = _parse_date(metadata.get("available_at"), "available_at")
    return DatasetRecord(
        dataset_id=make_dataset_id(company_id, kind, available_at, source_hash),
        company_id=company_id,
        dataset_type=kind,
        title=str(metadata.get("title") or source_path.stem),
        local_file=_portable_path(source_path, project_root),
        source_hash=source_hash,
        available_at=available_at,
        file_format=str(metadata.get("file_format") or source_path.suffix.lstrip(".")),
        period_start=_optional_date(metadata.get("period_start"), "period_start"),
        period_end=_optional_date(metadata.get("period_end"), "period_end"),
        row_count=metadata.get("row_count"),
        usage_scope=metadata.get("usage_scope"),
        pre_listing_eligible=bool(metadata.get("pre_listing_eligible", False)),
        metadata=metadata,
        registered_at=registered_at,
    )


def _build_market_dataset(
    *,
    project_root: Path,
    company_id: str,
    market_path: Path,
    registered_at: datetime,
) -> DatasetRecord:
    metadata = read_json(market_path)
    if str(metadata.get("company_id")) != company_id:
        raise ValueError(f"market dataset belongs to another company: {market_path}")
    period_start = _parse_date(metadata.get("start_date"), "start_date")
    period_end = _parse_date(metadata.get("end_date"), "end_date")
    available_at = _download_date(metadata, period_end)
    source_hash = sha256_file(market_path)
    return DatasetRecord(
        dataset_id=make_dataset_id(company_id, "market_daily", available_at, source_hash),
        company_id=company_id,
        dataset_type="market_daily",
        title=market_path.stem,
        local_file=_portable_path(market_path, project_root),
        source_hash=source_hash,
        available_at=available_at,
        file_format="json",
        period_start=period_start,
        period_end=period_end,
        row_count=metadata.get("row_count"),
        usage_scope=metadata.get("usage_scope"),
        pre_listing_eligible=bool(metadata.get("pre_listing_feature_eligible", False)),
        metadata={
            "currency": metadata.get("currency"),
            "interval": metadata.get("interval"),
            "source": metadata.get("source"),
            "data_quality": metadata.get("data_quality"),
        },
        registered_at=registered_at,
    )


def register_company_sources(
    company_folder: str,
    *,
    project_root: Path | None = None,
    repository: CatalogRepository | None = None,
) -> CompanyIngestionResult:
    """登记一家公司声明的全部文档和结构化数据，重复运行不会产生重复记录。"""
    config = load_config()
    root = (project_root or config.project_root).resolve()
    folder = _safe_folder_name(company_folder)
    metadata_path = root / config.paths.company_info_dir / folder / "公司信息.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    company_info = read_json(metadata_path)
    company_id = str(company_info["company_id"])
    registered_at = datetime.now(timezone.utc)
    company = CompanyRecord(
        company_id=company_id,
        company_folder=folder,
        ticker=company_info.get("ticker"),
        exchange=company_info.get("exchange"),
        market=company_info.get("market"),
        company_name_zh=str(company_info["company_name_zh_official"]),
        company_name_en=company_info.get("company_name_en"),
        listing_date=_optional_date(company_info.get("listing_date"), "listing_date"),
        metadata_file=_portable_path(metadata_path, root),
        metadata_hash=sha256_file(metadata_path),
        created_at=registered_at,
        updated_at=registered_at,
    )
    catalog = repository or CatalogRepository(root / config.paths.database)
    catalog.initialize()
    catalog.upsert_company(company)

    documents: list[DocumentRecord] = []
    registered_files = {company.metadata_file}
    raw_documents = company_info.get("documents", {})
    if not isinstance(raw_documents, dict):
        raise ValueError("company_info.documents must be an object")
    for document_type, metadata in raw_documents.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"document metadata must be an object: {document_type}")
        document = _build_document(
            project_root=root,
            company_id=company_id,
            document_type=str(document_type),
            metadata=metadata,
            registered_at=registered_at,
        )
        catalog.upsert_document(document)
        documents.append(document)
        registered_files.add(document.local_file)

    datasets: list[DatasetRecord] = []
    raw_datasets = company_info.get("datasets", {})
    if not isinstance(raw_datasets, dict):
        raise ValueError("company_info.datasets must be an object")
    for dataset_type, metadata in raw_datasets.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"dataset metadata must be an object: {dataset_type}")
        dataset = _build_declared_dataset(
            project_root=root,
            company_id=company_id,
            dataset_type=str(dataset_type),
            metadata=metadata,
            registered_at=registered_at,
        )
        catalog.upsert_dataset(dataset)
        datasets.append(dataset)
        registered_files.add(dataset.local_file)

    market_path = root / config.paths.market_dir / folder / "日线行情.json"
    market_portable = _portable_path(market_path, root) if market_path.is_file() else None
    if market_portable and market_portable not in registered_files:
        market_dataset = _build_market_dataset(
            project_root=root,
            company_id=company_id,
            market_path=market_path,
            registered_at=registered_at,
        )
        catalog.upsert_dataset(market_dataset)
        datasets.append(market_dataset)
        registered_files.add(market_dataset.local_file)

    warnings: list[str] = []
    source_roots = (
        root / config.paths.prospectus_dir / folder,
        root / config.paths.company_info_dir / folder,
        root / config.paths.market_dir / folder,
    )
    for source_root in source_roots:
        if not source_root.is_dir():
            continue
        for source_file in sorted(path for path in source_root.rglob("*") if path.is_file()):
            portable = _portable_path(source_file, root)
            if portable not in registered_files:
                warnings.append(
                    f"未登记文件: {portable}；请加入公司信息.json的documents或datasets。"
                )

    return CompanyIngestionResult(
        company=company,
        documents=tuple(sorted(documents, key=lambda item: item.document_id)),
        datasets=tuple(sorted(datasets, key=lambda item: item.dataset_id)),
        warnings=tuple(warnings),
    )


def register_all_companies(
    *,
    project_root: Path | None = None,
    repository: CatalogRepository | None = None,
) -> tuple[CompanyIngestionResult, ...]:
    """扫描 company_info 下所有标准公司目录并批量登记。"""
    config = load_config()
    root = (project_root or config.project_root).resolve()
    catalog = repository or CatalogRepository(root / config.paths.database)
    company_root = root / config.paths.company_info_dir
    folders = sorted(
        path.name
        for path in company_root.iterdir()
        if path.is_dir() and (path / "公司信息.json").is_file()
    )
    return tuple(
        register_company_sources(folder, project_root=root, repository=catalog)
        for folder in folders
    )


__all__ = [
    "CompanyIngestionResult",
    "make_dataset_id",
    "make_document_id",
    "register_all_companies",
    "register_company_sources",
]
