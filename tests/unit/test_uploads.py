from __future__ import annotations

from datetime import date
from pathlib import Path

import fitz

from ipo_risk.storage.repositories import CatalogRepository
from ipo_risk.uploads import ProspectusUploadService


def _pdf_bytes() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 100), "IPO prospectus risk disclosure")
    payload = document.tobytes()
    document.close()
    return payload


def _prospectus_cover_bytes() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(
        (50, 50, 550, 300),
        (
            "英矽智能 InSilico Medicine Cayman TopCo\n股份代號：3696\n"
            "2025年12月18日\n預期股份在聯交所開始買賣 2025年12月30日"
        ),
        fontname="china-s",
        fontsize=12,
    )
    payload = document.tobytes()
    document.close()
    return payload


def test_cover_text_suggests_company_code_name_and_prospectus_date(
    tmp_path: Path,
) -> None:
    service = ProspectusUploadService(project_root=tmp_path)

    suggestion = service.inspect(
        file_name="hkex_prospectus.pdf",
        content=_prospectus_cover_bytes(),
    )

    assert suggestion.company_id == "03696"
    assert suggestion.company_name == "英矽智能"
    assert suggestion.document_date == date(2025, 12, 18)
    assert suggestion.listing_date == date(2025, 12, 30)
    assert suggestion.source == "cover_text"
    assert suggestion.page_count == 1


def test_listing_date_parser_supports_traditional_chinese_numerals() -> None:
    result = ProspectusUploadService._listing_date_from_text(
        "預期股份於聯交所開始買賣……二零二零年十二月三十日（星期三）",
        document_date=date(2020, 12, 18),
    )

    assert result == date(2020, 12, 30)


def test_listing_date_parser_infers_year_from_timetable_month_and_day() -> None:
    result = ProspectusUploadService._listing_date_from_text(
        "預期股份開始於聯交所買賣……3月12日（星期四）上午九時正",
        document_date=date(2020, 2, 25),
    )

    assert result == date(2020, 3, 12)


def test_listing_date_parser_supports_class_a_ordinary_shares() -> None:
    result = ProspectusUploadService._listing_date_from_text(
        "預期A類普通股開始於聯交所買賣……2025年11月6日（星期四）",
        document_date=date(2025, 10, 28),
    )

    assert result == date(2025, 11, 6)


def test_official_filename_supplies_company_name_when_cover_has_website_notice() -> None:
    result = ProspectusUploadService._company_name_from_filename(
        "00800_28-10-2025_文遠知行－Ｗ_全球發售.pdf"
    )

    assert result == "文遠知行－Ｗ"
    assert (
        ProspectusUploadService._company_name_from_cover(
            "本招股章程已於聯交所網站 www.hkexnews.hk\n文遠知行"
        )
        is None
    )


def test_official_filename_supplies_day_month_year_document_date() -> None:
    result = ProspectusUploadService._document_date_from_filename(
        "00800_28-10-2025_文遠知行－Ｗ_全球發售.pdf"
    )

    assert result == date(2025, 10, 28)


def test_uploaded_pdf_is_registered_once_as_an_immutable_company_document(
    tmp_path: Path,
) -> None:
    repository = CatalogRepository(tmp_path / "data" / "app.db")
    service = ProspectusUploadService(
        project_root=tmp_path,
        repository=repository,
    )
    content = _pdf_bytes()

    first = service.register(
        file_name="prospectus.pdf",
        content=content,
        company_id="03696",
        company_name="英矽智能",
        document_date=date(2025, 12, 18),
    )
    second = service.register(
        file_name="prospectus.pdf",
        content=content,
        company_id="03696",
        company_name="英矽智能",
        document_date=date(2025, 12, 18),
    )

    assert first.company_folder == "03696_英矽智能"
    assert first.document_id == second.document_id
    assert first.source_path == second.source_path
    assert first.reused is False
    assert second.reused is True
    assert first.page_count == 1
    assert first.source_path.is_file()
    assert len(list(first.source_path.parent.glob("*.pdf"))) == 1
    stored = repository.get_document(first.document_id)
    assert stored is not None
    assert stored.source_hash == first.source_hash
