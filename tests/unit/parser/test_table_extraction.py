from __future__ import annotations

from datetime import date
from pathlib import Path

import fitz

from ipo_risk.parser.document_parser import parse_document
from ipo_risk.parser.tables import _clean_borderless_value, extract_page_tables
from ipo_risk.parser.visual import VisualRegion, VisualRegionKind
from ipo_risk.retrieval.chunker import build_page_chunks


def _write_financial_table_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.insert_text((72, 72), "Top five customers", fontsize=12)

    x_positions = (72, 240, 390, 520)
    y_positions = (100, 130, 160, 190)
    for x in x_positions:
        page.draw_line((x, y_positions[0]), (x, y_positions[-1]))
    for y in y_positions:
        page.draw_line((x_positions[0], y), (x_positions[-1], y))

    rows = (
        ("Customer", "Revenue (HKD m)", "Share"),
        ("Customer A", "120", "30%"),
        ("Customer B", "80", "20%"),
    )
    for row_index, row in enumerate(rows):
        baseline = y_positions[row_index] + 20
        for column_index, value in enumerate(row):
            page.insert_text(
                (x_positions[column_index] + 5, baseline),
                value,
                fontsize=9,
            )
    document.save(path)
    document.close()


def _write_borderless_multilevel_table_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.insert_text((72, 72), "Summary income statement", fontsize=12)
    page.insert_text((245, 100), "2023", fontsize=9)
    page.insert_text((405, 100), "2024", fontsize=9)
    page.insert_text((195, 118), "HKD m", fontsize=9)
    page.insert_text((285, 118), "%", fontsize=9)
    page.insert_text((355, 118), "HKD m", fontsize=9)
    page.insert_text((445, 118), "%", fontsize=9)
    rows = (
        ("Revenue", "200", "100", "240", "100"),
        ("Gross profit", "80", "40", "108", "45"),
        ("Net profit", "20", "10", "36", "15"),
        ("Operating cash flow", "(10)", "(5)", "24", "10"),
    )
    x_positions = (72, 195, 285, 355, 445)
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            page.insert_text(
                (x_positions[column_index], 145 + row_index * 20),
                value,
                fontsize=9,
            )
    document.save(path)
    document.close()


def _write_dotted_financial_table_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    for index, year in enumerate(("2021", "2022", "2023", "2024")):
        page.insert_text((230 + index * 75, 90), year, fontsize=9)
    rows = (
        ("R&D cost. . . . . . . . . . . .", "13,703", "14,548", "23,975", "15,267"),
        ("Listing expense. . . . . . . . .", "-", "-", "950", "15,559"),
        ("Staff benefits. . . . . . . . . .", "32,636", "41,652", "46,252", "25,109"),
        ("Total. . . . . . . . . . . . . .", "343,100", "413,887", "548,426", "333,214"),
    )
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            x = 72 if column_index == 0 else 230 + (column_index - 1) * 75
            page.insert_text((x, 125 + row_index * 22), value, fontsize=9)
    document.save(path)
    document.close()


def test_table_becomes_traceable_vector_ready_chunk(tmp_path: Path) -> None:
    pdf_path = tmp_path / "table.pdf"
    _write_financial_table_pdf(pdf_path)

    page = next(parse_document(pdf_path, page_range=(0, 1)))
    chunks = build_page_chunks(
        page,
        company_id="00501",
        document_id="prospectus-2025",
        source_file="table.pdf",
        source_hash="a" * 64,
        available_at=date(2025, 12, 31),
    )

    assert len(page.tables) == 1
    table = page.tables[0]
    assert table.headers == ("Customer", "Revenue (HKD m)", "Share")
    assert table.rows == (
        ("Customer A", "120", "30%"),
        ("Customer B", "80", "20%"),
    )
    assert "|Customer|Revenue (HKD m)|Share|" in table.markdown

    table_chunks = [chunk for chunk in chunks if chunk.chunk_kind == "table"]
    assert len(table_chunks) == 1
    assert table_chunks[0].table_id == table.table_id
    assert "表格证据" in table_chunks[0].text
    assert "Customer A" in table_chunks[0].text
    assert "Revenue (HKD m)=120" in table_chunks[0].text
    assert table_chunks[0].bbox == table.bbox


def test_borderless_table_restores_year_and_unit_hierarchy(tmp_path: Path) -> None:
    pdf_path = tmp_path / "borderless-table.pdf"
    _write_borderless_multilevel_table_pdf(pdf_path)

    page = next(parse_document(pdf_path, min_chars=1, page_range=(0, 1)))

    assert len(page.tables) == 1
    table = page.tables[0]
    assert table.headers == (
        "项目",
        "2023 / HKD m",
        "2023 / %",
        "2024 / HKD m",
        "2024 / %",
    )
    assert table.rows[0] == ("Revenue", "200", "100", "240", "100")
    assert table.rows[-1] == ("Operating cash flow", "(10)", "(5)", "24", "10")


def test_dotted_financial_rows_are_not_mistaken_for_table_of_contents(tmp_path: Path) -> None:
    pdf_path = tmp_path / "dotted-financial-table.pdf"
    _write_dotted_financial_table_pdf(pdf_path)

    page = next(parse_document(pdf_path, min_chars=1, page_range=(0, 1)))

    assert len(page.tables) == 1
    assert page.tables[0].rows[0][0] == "R&D cost"
    assert page.tables[0].rows[-1][-1] == "333,214"


def test_aligned_text_false_positive_never_falls_back_to_find_tables() -> None:
    class PageThatMustNotRunFindTables:
        def find_tables(self, **_kwargs):
            raise AssertionError("aligned text must not call PyMuPDF find_tables")

    candidate = VisualRegion(
        region_id="VR-0007-001",
        page=7,
        kind=VisualRegionKind.TABLE,
        bbox=(70.0, 90.0, 520.0, 570.0),
        confidence=0.72,
        reasons=(
            "aligned_text_columns",
            "horizontal_lines=0",
            "vertical_lines=0",
        ),
    )

    tables = extract_page_tables(
        PageThatMustNotRunFindTables(),  # type: ignore[arg-type]
        page_num=7,
        text_blocks=(),
        visual_regions=(candidate,),
    )

    assert tables == []


def test_long_table_of_contents_dotted_leader_is_cleaned_in_linear_time() -> None:
    dotted_leader = "章節名稱 " + ". " * 2_000

    assert _clean_borderless_value(dotted_leader) == "章節名稱"


def test_timed_out_grid_table_is_marked_for_visual_followup() -> None:
    class TimedOutExtractor:
        def extract(self, **_kwargs):
            return (), "table_extraction_timeout"

    document = fitz.open()
    page = document.new_page(width=595, height=842)
    candidate = VisualRegion(
        region_id="VR-0001-001",
        page=1,
        kind=VisualRegionKind.TABLE,
        bbox=(70.0, 90.0, 520.0, 570.0),
        confidence=0.9,
        reasons=(
            "aligned_grid_lines",
            "horizontal_lines=8",
            "vertical_lines=5",
        ),
    )
    deferred: set[str] = set()

    tables = extract_page_tables(
        page,
        page_num=1,
        text_blocks=(),
        visual_regions=(candidate,),
        isolated_extractor=TimedOutExtractor(),  # type: ignore[arg-type]
        deferred_region_ids=deferred,
    )
    document.close()

    assert tables == []
    assert deferred == {candidate.region_id}
