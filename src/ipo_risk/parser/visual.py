"""PDF 页面中的表格、图片和流程图候选检测。

所有页面先做本地几何检测，不调用视觉模型。候选区域保留页码、坐标、置信度和触发原因；
后续视觉适配器只处理流程图或低置信度表格，避免整本招股书逐页付费扫描。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, Sequence

import fitz


class VisualRegionKind(StrEnum):
    TABLE = "table"
    IMAGE = "image"
    DIAGRAM = "diagram"


class PositionedText(Protocol):
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


@dataclass(frozen=True)
class VisualRegion:
    """一个可能需要表格抽取或视觉理解的页面区域。"""

    region_id: str
    page: int
    kind: VisualRegionKind
    bbox: tuple[float, float, float, float]
    confidence: float
    reasons: tuple[str, ...]
    text_hint: str = ""
    requires_vision: bool = False


class VisualRegionAnalyzer(Protocol):
    """可替换视觉模型的最小接口；本地检测本身不依赖该接口。"""

    model_name: str

    def analyze(
        self,
        *,
        pdf_path: str,
        regions: Sequence[VisualRegion],
    ) -> Sequence[dict[str, Any]]:
        """批量分析已经裁剪定位的候选区域。"""
        ...


def detect_visual_regions(
    page: fitz.Page,
    *,
    page_num: int,
    text_blocks: Sequence[Any],
) -> list[VisualRegion]:
    """使用图片对象、矢量线框和 PyMuPDF 表格检测生成候选区域。"""
    candidates: list[tuple[VisualRegionKind, fitz.Rect, float, tuple[str, ...], bool]] = []
    page_rect = page.rect

    try:
        image_infos = page.get_image_info(xrefs=True)
    except Exception:  # pragma: no cover - 兼容较旧 PyMuPDF
        image_infos = []
    for info in image_infos:
        bbox = _as_rect(info.get("bbox"))
        if bbox is None or bbox.width < 24 or bbox.height < 24:
            continue
        coverage = bbox.get_area() / max(1.0, page_rect.get_area())
        confidence = 0.96 if info.get("xref") else 0.82
        candidates.append(
            (
                VisualRegionKind.IMAGE,
                bbox,
                confidence,
                (
                    "pdf_image_object",
                    f"page_coverage={coverage:.3f}",
                ),
                coverage < 0.9,
            )
        )

    drawings: list[dict[str, Any]]
    try:
        drawings = list(page.get_drawings())
    except Exception:  # pragma: no cover - 损坏页面或旧版本兼容
        drawings = []
    horizontal: list[fitz.Rect] = []
    vertical: list[fitz.Rect] = []
    rectangles: list[fitz.Rect] = []
    connectors: list[fitz.Rect] = []
    for drawing in drawings:
        drawing_rect = _as_rect(drawing.get("rect"))
        for item in drawing.get("items", ()):
            if not item:
                continue
            operation = item[0]
            if operation == "re" and len(item) > 1:
                rect = _as_rect(item[1])
                if rect is not None and rect.width >= 10 and rect.height >= 6:
                    rectangles.append(rect)
            elif operation == "l" and len(item) > 2:
                p1, p2 = item[1], item[2]
                rect = fitz.Rect(
                    min(p1.x, p2.x),
                    min(p1.y, p2.y),
                    max(p1.x, p2.x),
                    max(p1.y, p2.y),
                )
                if abs(p1.y - p2.y) <= 1.5 and abs(p1.x - p2.x) >= 18:
                    horizontal.append(rect)
                elif abs(p1.x - p2.x) <= 1.5 and abs(p1.y - p2.y) >= 12:
                    vertical.append(rect)
                elif drawing_rect is not None:
                    connectors.append(drawing_rect)

    table_rects: list[fitz.Rect] = []
    if len(horizontal) >= 3 and len(vertical) >= 3:
        table_rects.append(_union_rects((*horizontal, *vertical)))
    borderless_table = _detect_aligned_word_table(page)
    if borderless_table is not None:
        table_rects.append(borderless_table)
    for rect in _deduplicate_rects(table_rects):
        candidates.append(
            (
                VisualRegionKind.TABLE,
                rect,
                0.9 if len(horizontal) >= 3 and len(vertical) >= 3 else 0.72,
                (
                    (
                        "aligned_grid_lines"
                        if len(horizontal) >= 3 and len(vertical) >= 3
                        else "aligned_text_columns"
                    ),
                    f"horizontal_lines={len(horizontal)}",
                    f"vertical_lines={len(vertical)}",
                ),
                False,
            )
        )

    # 多个非规则矩形和连接线更像流程图；规则密集网格已在上面标成表格。
    if len(rectangles) >= 3 and connectors and not table_rects:
        diagram_rect = _union_rects((*rectangles, *connectors))
        candidates.append(
            (
                VisualRegionKind.DIAGRAM,
                diagram_rect,
                min(0.92, 0.62 + len(rectangles) * 0.04),
                (
                    "multiple_vector_boxes",
                    f"boxes={len(rectangles)}",
                    f"connectors={len(connectors)}",
                ),
                True,
            )
        )

    regions: list[VisualRegion] = []
    for index, (kind, bbox, confidence, reasons, requires_vision) in enumerate(
        _deduplicate_candidates(candidates),
        start=1,
    ):
        clipped = bbox & page_rect
        if clipped.is_empty or clipped.width <= 0 or clipped.height <= 0:
            continue
        regions.append(
            VisualRegion(
                region_id=f"VR-{page_num:04d}-{index:03d}",
                page=page_num,
                kind=kind,
                bbox=tuple(round(value, 2) for value in clipped),
                confidence=round(confidence, 2),
                reasons=reasons,
                text_hint=_region_text_hint(clipped, text_blocks),
                requires_vision=requires_vision,
            )
        )
    return regions


def _as_rect(value: Any) -> fitz.Rect | None:
    if value is None:
        return None
    try:
        rect = fitz.Rect(value)
    except Exception:
        return None
    if rect.is_empty or rect.width <= 0 or rect.height <= 0:
        return None
    return rect


def _union_rects(rects: Sequence[fitz.Rect]) -> fitz.Rect:
    return fitz.Rect(
        min(rect.x0 for rect in rects),
        min(rect.y0 for rect in rects),
        max(rect.x1 for rect in rects),
        max(rect.y1 for rect in rects),
    )


def _deduplicate_rects(rects: Sequence[fitz.Rect]) -> list[fitz.Rect]:
    kept: list[fitz.Rect] = []
    for rect in rects:
        if any(_intersection_ratio(rect, existing) >= 0.8 for existing in kept):
            continue
        kept.append(rect)
    return kept


def _deduplicate_candidates(
    candidates: Sequence[
        tuple[VisualRegionKind, fitz.Rect, float, tuple[str, ...], bool]
    ],
) -> list[tuple[VisualRegionKind, fitz.Rect, float, tuple[str, ...], bool]]:
    priority = {
        VisualRegionKind.TABLE: 3,
        VisualRegionKind.DIAGRAM: 2,
        VisualRegionKind.IMAGE: 1,
    }
    ordered = sorted(
        candidates,
        key=lambda item: (-priority[item[0]], -item[2], -item[1].get_area()),
    )
    kept: list[tuple[VisualRegionKind, fitz.Rect, float, tuple[str, ...], bool]] = []
    for candidate in ordered:
        if any(
            _intersection_ratio(candidate[1], existing[1]) >= 0.88
            and candidate[0] is existing[0]
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return kept


def _intersection_ratio(first: fitz.Rect, second: fitz.Rect) -> float:
    intersection = first & second
    if intersection.is_empty:
        return 0.0
    return intersection.get_area() / max(1.0, min(first.get_area(), second.get_area()))


def _region_text_hint(
    bbox: fitz.Rect,
    text_blocks: Sequence[PositionedText],
) -> str:
    snippets = [
        block.text.strip()
        for block in text_blocks
        if block.text.strip()
        and _intersection_ratio(
            bbox,
            fitz.Rect(block.x0, block.y0, block.x1, block.y1),
        )
        >= 0.2
    ]
    return " ".join(snippets)[:500]


def _detect_aligned_word_table(page: fitz.Page) -> fitz.Rect | None:
    """用重复列起点识别无边框文本表格，避免昂贵的逐页表格插件。"""
    try:
        raw_words = page.get_text("words")
    except Exception:
        return None
    rows: dict[int, list[tuple[float, float, float, float, str]]] = {}
    for word in raw_words:
        if len(word) < 5:
            continue
        x0, y0, x1, y1, text = word[:5]
        if not str(text).strip():
            continue
        rows.setdefault(round(float(y0) / 4), []).append(
            (float(x0), float(y0), float(x1), float(y1), str(text))
        )
    candidate_rows = [
        sorted(words, key=lambda item: item[0])
        for words in rows.values()
        if len(words) >= 3
    ]
    if len(candidate_rows) < 4:
        return None
    column_counts: dict[int, int] = {}
    for row in candidate_rows:
        for word in row:
            column_counts[round(word[0] / 24)] = (
                column_counts.get(round(word[0] / 24), 0) + 1
            )
    repeated_columns = {
        column for column, count in column_counts.items() if count >= 3
    }
    if len(repeated_columns) < 3:
        return None
    aligned_rows = [
        row
        for row in candidate_rows
        if len({round(word[0] / 24) for word in row} & repeated_columns) >= 3
    ]
    if len(aligned_rows) < 4:
        return None
    words = [word for row in aligned_rows for word in row]
    return fitz.Rect(
        min(word[0] for word in words),
        min(word[1] for word in words),
        max(word[2] for word in words),
        max(word[3] for word in words),
    )


__all__ = [
    "VisualRegion",
    "VisualRegionAnalyzer",
    "VisualRegionKind",
    "detect_visual_regions",
]
