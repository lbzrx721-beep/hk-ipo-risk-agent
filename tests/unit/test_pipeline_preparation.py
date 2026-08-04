from __future__ import annotations

from datetime import date
from pathlib import Path

import fitz

from ipo_risk.parser.document_parser import PageResult, TextBlock
from ipo_risk.pipeline import _prepare_prospectus_source
from ipo_risk.storage.files import sha256_file


def test_prepared_prospectus_reuses_versioned_parser_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pdf_path = tmp_path / "prospectus.pdf"
    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()
    calls = 0

    def fake_parse_document(path: Path, *, min_chars: int):
        nonlocal calls
        assert path == pdf_path
        assert min_chars == 80
        calls += 1
        yield PageResult(
            page=1,
            width=595.0,
            height=842.0,
            has_text_layer=True,
            text_blocks=[
                TextBlock(
                    page=1,
                    x0=72.0,
                    y0=100.0,
                    x1=420.0,
                    y1=120.0,
                    text="公司存在客户集中度风险，前五大客户收入占比较高。",
                    font_size=10.0,
                )
            ],
            raw_text="公司存在客户集中度风险，前五大客户收入占比较高。",
            char_count=24,
        )

    monkeypatch.setattr(
        "ipo_risk.pipeline.parse_document",
        fake_parse_document,
    )
    arguments = {
        "pdf_path": pdf_path,
        "company_id": "00001",
        "company_folder": "00001_测试公司",
        "document_id": "doc-test",
        "source_file": "data/raw/prospectus/00001_测试公司/prospectus.pdf",
        "source_hash": sha256_file(pdf_path),
        "available_at": date(2025, 1, 1),
        "interim_root": tmp_path / "interim",
        "artifacts_root": tmp_path / "artifacts",
        "min_text_chars_before_ocr": 80,
        "progress_callback": None,
    }

    first = _prepare_prospectus_source(**arguments)
    second = _prepare_prospectus_source(**arguments)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert calls == 1
    assert second.chunks == first.chunks
