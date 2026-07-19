from __future__ import annotations

from ipo_risk.parser.document_parser import TextBlock
from ipo_risk.parser.layout import (
    detect_body_font_size,
    detect_headings,
    extract_printed_page_label,
    merge_blocks_into_paragraphs,
    order_text_blocks,
)


def _block(
    text: str,
    *,
    y0: float,
    y1: float,
    x0: float = 70.0,
    x1: float = 525.0,
    font_size: float = 10.5,
    block_no: int = 0,
) -> TextBlock:
    return TextBlock(
        page=70,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        text=text,
        font_size=font_size,
        block_no=block_no,
    )


def test_reading_order_prevents_footer_from_joining_body() -> None:
    # PyMuPDF can return the footer before body blocks in source order.
    blocks = [
        _block("風險因素", y0=31.98, y1=44.48, x0=267.48, x1=327.79, font_size=12.5),
        _block("– 61 –", y0=749.04, y1=760.75, x0=283.59, x1=311.68, block_no=1),
        _block("候選藥物依賴對外授權合作", y0=70.83, y1=81.33, block_no=2),
        _block("現金流量可能受到重大不利影響。", y0=86.83, y1=97.33, x1=300.0, block_no=3),
        _block("我們預期一部分收入來自合作安排。", y0=117.0, y1=130.47, x0=99.21, block_no=4),
        _block("若無法獲得合作，可能無法支持運營。", y0=133.0, y1=146.47, block_no=5),
    ]

    ordered = order_text_blocks(blocks)
    assert ordered[-1].text == "– 61 –"
    assert extract_printed_page_label(ordered, page_height=842.0) == "61"

    paragraphs = merge_blocks_into_paragraphs(blocks, page_height=842.0)

    assert all("– 61 –" not in paragraph.text for paragraph in paragraphs)
    assert paragraphs[1].text == "候選藥物依賴對外授權合作\n現金流量可能受到重大不利影響。"
    assert paragraphs[1].bbox[3] == 97.33
    assert paragraphs[2].bbox[1] == 117.0


def test_realistic_risk_heading_is_detected() -> None:
    paragraphs = merge_blocks_into_paragraphs(
        [
            _block("風險因素", y0=31.98, y1=44.48, x0=267.48, x1=327.79, font_size=12.5),
            _block("候選藥物開發存在不確定性。", y0=70.0, y1=82.0, font_size=10.5),
            _block("臨床試驗未必能夠取得成功。", y0=87.0, y1=99.0, font_size=10.5),
        ]
    )

    body_font_size = detect_body_font_size(paragraphs)
    detect_headings(paragraphs, body_font_size)

    assert body_font_size == 10.5
    assert paragraphs[0].text == "風險因素"
    assert paragraphs[0].is_heading is True

