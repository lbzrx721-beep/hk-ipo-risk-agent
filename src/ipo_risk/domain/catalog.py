"""公司、原始资料、数据集与向量索引的目录契约。"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from .models import (
    NonEmptyStr,
    RiskType,
    Sha256,
    StrictModel,
    ensure_timezone_aware,
    utc_now,
)


class CatalogModel(StrictModel):
    """可持久化目录对象的统一版本。"""

    schema_version: Literal["1.0"] = "1.0"


class CompanyRecord(CatalogModel):
    """一个上市主体及其结构化入口文件。"""

    company_id: NonEmptyStr
    company_folder: NonEmptyStr
    ticker: NonEmptyStr | None = None
    exchange: NonEmptyStr | None = None
    market: NonEmptyStr | None = None
    company_name_zh: NonEmptyStr
    company_name_en: NonEmptyStr | None = None
    listing_date: date | None = None
    metadata_file: NonEmptyStr
    metadata_hash: Sha256
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("metadata_hash", mode="before")
    @classmethod
    def normalize_hash(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("created_at", "updated_at")
    @classmethod
    def timestamps_must_be_aware(cls, value: datetime, info: object) -> datetime:
        return ensure_timezone_aware(value, getattr(info, "field_name", "timestamp"))

    @model_validator(mode="after")
    def updated_at_must_follow_created_at(self) -> CompanyRecord:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not be earlier than created_at")
        return self


class DocumentRecord(CatalogModel):
    """一份不可变原始文档的具体版本。"""

    document_id: NonEmptyStr
    company_id: NonEmptyStr
    document_type: RiskType
    title: NonEmptyStr
    local_file: NonEmptyStr
    source_hash: Sha256
    document_date: date
    available_at: date
    language: NonEmptyStr | None = None
    source_kind: NonEmptyStr | None = None
    source_url: NonEmptyStr | None = None
    media_type: NonEmptyStr
    file_size_bytes: int = Field(ge=1)
    page_count: int | None = Field(default=None, ge=1)
    text_layer_status: NonEmptyStr | None = None
    metadata_status: Literal["declared", "inferred"] = "declared"
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    registered_at: datetime = Field(default_factory=utc_now)

    @field_validator("source_hash", mode="before")
    @classmethod
    def normalize_hash(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("registered_at")
    @classmethod
    def registered_at_must_be_aware(cls, value: datetime) -> datetime:
        return ensure_timezone_aware(value, "registered_at")

    @model_validator(mode="after")
    def availability_must_follow_document_date(self) -> DocumentRecord:
        if self.available_at < self.document_date:
            raise ValueError("available_at must not be earlier than document_date")
        return self


class DatasetRecord(CatalogModel):
    """一份结构化数据快照，例如公司信息或日线行情。"""

    dataset_id: NonEmptyStr
    company_id: NonEmptyStr
    dataset_type: RiskType
    title: NonEmptyStr
    local_file: NonEmptyStr
    source_hash: Sha256
    available_at: date
    file_format: NonEmptyStr
    period_start: date | None = None
    period_end: date | None = None
    row_count: int | None = Field(default=None, ge=0)
    usage_scope: NonEmptyStr | None = None
    pre_listing_eligible: bool
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    registered_at: datetime = Field(default_factory=utc_now)

    @field_validator("source_hash", mode="before")
    @classmethod
    def normalize_hash(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("registered_at")
    @classmethod
    def registered_at_must_be_aware(cls, value: datetime) -> datetime:
        return ensure_timezone_aware(value, "registered_at")

    @model_validator(mode="after")
    def period_must_be_ordered(self) -> DatasetRecord:
        if self.period_start and self.period_end and self.period_end < self.period_start:
            raise ValueError("period_end must not be earlier than period_start")
        return self


class VectorIndexRecord(CatalogModel):
    """一个文档版本在指定向量模型下生成的可重建索引。"""

    index_id: NonEmptyStr
    document_id: NonEmptyStr
    source_hash: Sha256
    embedding_provider: NonEmptyStr
    embedding_model: NonEmptyStr
    dimensions: int = Field(gt=0)
    chunk_count: int = Field(gt=0)
    vectors_path: NonEmptyStr
    metadata_path: NonEmptyStr
    manifest_path: NonEmptyStr
    status: Literal["building", "ready", "stale", "failed"] = "ready"
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("source_hash", mode="before")
    @classmethod
    def normalize_hash(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("created_at")
    @classmethod
    def created_at_must_be_aware(cls, value: datetime) -> datetime:
        return ensure_timezone_aware(value, "created_at")


__all__ = [
    "CompanyRecord",
    "DatasetRecord",
    "DocumentRecord",
    "VectorIndexRecord",
]
