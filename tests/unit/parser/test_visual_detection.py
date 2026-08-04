from __future__ import annotations

import fitz

from ipo_risk.parser.visual import VisualRegionKind, detect_visual_regions


def test_local_geometry_detects_table_candidate_without_model_call() -> None:
    document = fitz.open()
    page = document.new_page(width=400, height=400)
    for y in (80, 120, 160, 200):
        page.draw_line((60, y), (340, y))
    for x in (60, 150, 245, 340):
        page.draw_line((x, 80), (x, 200))
    page = document.reload_page(page)

    regions = detect_visual_regions(page, page_num=1, text_blocks=[])

    assert any(region.kind is VisualRegionKind.TABLE for region in regions)
    table = next(region for region in regions if region.kind is VisualRegionKind.TABLE)
    assert table.requires_vision is False
    assert "aligned_grid_lines" in table.reasons
    document.close()
