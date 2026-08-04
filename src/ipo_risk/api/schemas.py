"""新版分析工作台的API请求结构。"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class UploadMetadata(BaseModel):
    """一份上传文件经用户确认后的公司信息。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    file_name: str = Field(min_length=1)
    company_id: str = Field(min_length=1, max_length=32)
    company_name: str = Field(min_length=1, max_length=120)
    document_date: date
    force_reanalysis: bool = False


class CatalogBatchRequest(BaseModel):
    """从赛事清单创建分析队列。"""

    model_config = ConfigDict(extra="forbid")

    company_folders: tuple[str, ...] = Field(min_length=1, max_length=50)


__all__ = ["CatalogBatchRequest", "UploadMetadata"]
