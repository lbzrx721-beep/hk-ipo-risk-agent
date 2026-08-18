from __future__ import annotations

from pathlib import Path

from ipo_risk.modeling.supplements import read_supplement_records


def test_read_supplement_records_recovers_valid_objects_around_broken_block(
    tmp_path: Path,
) -> None:
    source = tmp_path / "joined.txt"
    source.write_text(
        '[{"pdf_file":"a.pdf","company_id":"1"}]\n'
        '[{"pdf_file":"broken.pdf","company_id":"2"\n'
        '[{"pdf_file":"c.pdf","company_id":"3"}]',
        encoding="utf-8",
    )

    records = read_supplement_records(source)

    assert [record["company_id"] for record in records] == ["1", "3"]


def test_read_supplement_records_reads_standard_json(tmp_path: Path) -> None:
    source = tmp_path / "standard.json"
    source.write_text(
        '[{"pdf_file":"a.pdf","company_id":"00100"}]',
        encoding="utf-8",
    )

    assert read_supplement_records(source)[0]["company_id"] == "00100"
