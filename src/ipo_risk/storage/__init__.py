"""文件产物、原始资料登记与SQLite仓储接口。"""

from .ingestion import (
    CompanyIngestionResult,
    make_dataset_id,
    make_document_id,
    register_all_companies,
    register_company_sources,
)
from .paths import DocumentPaths, safe_path_segment
from .repositories import AnalysisRunRecord, CatalogCounts, CatalogRepository

__all__ = [
    "AnalysisRunRecord",
    "CatalogCounts",
    "CatalogRepository",
    "CompanyIngestionResult",
    "DocumentPaths",
    "make_dataset_id",
    "make_document_id",
    "register_all_companies",
    "register_company_sources",
    "safe_path_segment",
]
