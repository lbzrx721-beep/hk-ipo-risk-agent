"""标准公司目录自动登记测试。"""

from __future__ import annotations

from pathlib import Path

from ipo_risk.storage.files import sha256_file, write_json
from ipo_risk.storage.ingestion import register_company_sources
from ipo_risk.storage.repositories import CatalogRepository


def _create_company_fixture(root: Path) -> None:
    company_folder = "09999_测试公司"
    prospectus_dir = root / "data" / "raw" / "prospectus" / company_folder
    company_dir = root / "data" / "raw" / "company_info" / company_folder
    market_dir = root / "data" / "raw" / "market" / company_folder
    prospectus_dir.mkdir(parents=True)
    company_dir.mkdir(parents=True)
    market_dir.mkdir(parents=True)

    pdf_path = prospectus_dir / "招股章程_2025-01-02_繁体中文.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n% unit test prospectus\n")
    write_json(
        {
            "schema_version": 2,
            "company_id": "09999",
            "exchange": "HKEX",
            "market": "Main Board",
            "ticker": "09999.HK",
            "company_name_zh_official": "测试公司",
            "company_name_en": "Test Company",
            "listing_date": "2025-01-10",
            "documents": {
                "prospectus_main": {
                    "document_type": "prospectus",
                    "local_file": pdf_path.relative_to(root).as_posix(),
                    "document_date": "2025-01-02",
                    "available_at_date": "2025-01-02",
                    "language": "zh-Hant",
                    "source_kind": "official_hkex_public_filing",
                    "file_size_bytes": pdf_path.stat().st_size,
                    "sha256": sha256_file(pdf_path),
                    "page_count": 1,
                    "text_layer_status": "present",
                }
            },
        },
        company_dir / "公司信息.json",
    )
    write_json(
        {
            "schema_version": 2,
            "company_id": "09999",
            "ticker": "09999.HK",
            "currency": "HKD",
            "interval": "1d",
            "start_date": "2025-01-10",
            "end_date": "2025-01-13",
            "row_count": 2,
            "usage_scope": "post_listing_label_and_backtest_only",
            "pre_listing_feature_eligible": False,
            "source": {
                "provider": "fixture",
                "downloaded_at": "2025-01-14T08:00:00+08:00",
            },
            "rows": [],
        },
        market_dir / "日线行情.json",
    )


def test_company_ingestion_is_repeatable_and_detects_unregistered_files(
    tmp_path: Path,
) -> None:
    _create_company_fixture(tmp_path)
    repository = CatalogRepository(tmp_path / "data" / "app.db")

    first = register_company_sources(
        "09999_测试公司",
        project_root=tmp_path,
        repository=repository,
    )
    second = register_company_sources(
        "09999_测试公司",
        project_root=tmp_path,
        repository=repository,
    )

    assert first.company.company_id == "09999"
    assert len(first.documents) == 1
    assert first.documents[0].document_type == "prospectus"
    assert first.documents[0].document_id.startswith("doc_09999_prospectus_2025-01-02_")
    assert len(first.datasets) == 1
    assert first.datasets[0].dataset_type == "market_daily"
    assert not first.datasets[0].pre_listing_eligible
    assert first.warnings == ()
    assert second.documents[0].document_id == first.documents[0].document_id
    assert second.documents[0].source_hash == first.documents[0].source_hash
    assert second.datasets[0].dataset_id == first.datasets[0].dataset_id
    counts = repository.counts()
    assert counts.companies == 1
    assert counts.documents == 1
    assert counts.datasets == 1

    extra_file = (
        tmp_path
        / "data"
        / "raw"
        / "company_info"
        / "09999_测试公司"
        / "未登记公告_2025-01-05.txt"
    )
    extra_file.write_text("待登记", encoding="utf-8")
    third = register_company_sources(
        "09999_测试公司",
        project_root=tmp_path,
        repository=repository,
    )
    assert len(third.warnings) == 1
    assert "未登记公告" in third.warnings[0]
