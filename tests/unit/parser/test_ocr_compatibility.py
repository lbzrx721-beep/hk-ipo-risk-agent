from __future__ import annotations

from dataclasses import dataclass

from ipo_risk.parser.ocr import normalize_ocr_output


@dataclass
class RapidOCR39Output:
    boxes: list[list[list[float]]]
    txts: tuple[str, ...]
    scores: tuple[float, ...]


def test_normalize_rapidocr_39_output_object() -> None:
    output = RapidOCR39Output(
        boxes=[[[10, 20], [110, 20], [110, 40], [10, 40]]],
        txts=("掃描頁文字",),
        scores=(0.96,),
    )

    items = normalize_ocr_output(output)

    assert items == [
        ([[10, 20], [110, 20], [110, 40], [10, 40]], "掃描頁文字", 0.96)
    ]


def test_normalize_legacy_rapidocr_output() -> None:
    output = [
        ([[1, 2], [3, 2], [3, 4], [1, 4]], "legacy", 0.88),
    ]

    assert normalize_ocr_output(output) == output

