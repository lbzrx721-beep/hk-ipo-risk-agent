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
        "英矽智能 InSilico Medicine Cayman TopCo\n股份代號：3696\n2025年12月18日",
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
    assert suggestion.source == "cover_text"
    assert suggestion.page_count == 1


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
