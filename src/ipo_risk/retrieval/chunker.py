"""将页面解析结果转换为可检索、可回链的段落切片。"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from ipo_risk.domain import BoundingBox
from ipo_risk.parser.document_parser import PageResult
from ipo_risk.parser.layout import (
    detect_body_font_size,
    detect_headings,
    extract_printed_page_label,
    merge_blocks_into_paragraphs,
)


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class DocumentChunk(BaseModel):
    """一条供检索使用的段落，保留重建 Evidence 所需的全部定位信息。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    chunk_id: NonEmptyStr
    company_id: NonEmptyStr
    document_id: NonEmptyStr
    source_file: NonEmptyStr
    source_hash: Sha256
    page: int = Field(ge=1)
    printed_page_label: NonEmptyStr | None = None
    bbox: BoundingBox
    text: NonEmptyStr
    is_heading: bool = False
    is_ocr: bool = False
    font_size: float = Field(ge=0.0)
    available_at: date
    chunk_kind: Literal["text", "table"] = "text"
    table_id: NonEmptyStr | None = None
    table_cell_refs: tuple[NonEmptyStr, ...] = ()
    table_cell_values: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_chunk(self) -> DocumentChunk:
        x0, y0, x1, y1 = self.bbox
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError("chunk bbox must satisfy x0 < x1 and y0 < y1")
        if self.chunk_kind == "table" and not self.table_id:
            raise ValueError("table chunks require table_id")
        if self.chunk_kind == "text" and (
            self.table_id or self.table_cell_refs or self.table_cell_values
        ):
            raise ValueError("text chunks cannot reference table metadata")
        if len(set(self.table_cell_refs)) != len(self.table_cell_refs):
            raise ValueError("table_cell_refs must not contain duplicates")
        if len(self.table_cell_refs) != len(self.table_cell_values):
            raise ValueError("table cell references and values must align")
        return self


def build_page_chunks(
    page_result: PageResult,
    *,
    company_id: str,
    document_id: str,
    source_file: str,
    source_hash: str,
    available_at: date,
) -> list[DocumentChunk]:
    """把一页 TextBlock 合并为段落切片并附加文档血缘。"""
    paragraphs = merge_blocks_into_paragraphs(
        page_result.text_blocks,
        page_height=page_result.height,
    )
    body_font_size = detect_body_font_size(paragraphs)
    detect_headings(paragraphs, body_font_size)
    printed_page_label = extract_printed_page_label(
        page_result.text_blocks,
        page_height=page_result.height,
    )

    chunks: list[DocumentChunk] = []
    table_bboxes = tuple(table.bbox for table in page_result.tables if _is_valid_bbox(table.bbox))
    for paragraph_index, paragraph in enumerate(paragraphs, start=1):
        text = paragraph.text.strip()
        if (
            not text
            or not _is_valid_bbox(paragraph.bbox)
            or any(
                _intersection_ratio(paragraph.bbox, table_bbox) >= 0.55
                for table_bbox in table_bboxes
            )
        ):
            continue
        chunks.append(
            DocumentChunk(
                chunk_id=(f"C-{company_id}-{page_result.page:04d}-{paragraph_index:03d}"),
                company_id=company_id,
                document_id=document_id,
                source_file=source_file,
                source_hash=source_hash,
                page=page_result.page,
                printed_page_label=printed_page_label,
                bbox=paragraph.bbox,
                text=text,
                is_heading=paragraph.is_heading,
                is_ocr=any(block.is_ocr for block in paragraph.blocks),
                font_size=paragraph.font_size,
                available_at=available_at,
            )
        )
    for table_index, table in enumerate(page_result.tables, start=1):
        if not _is_valid_bbox(table.bbox):
            continue
        row_groups = _table_row_groups(table.rows)
        for group_index, row_indexes in enumerate(row_groups, start=1):
            grouped_rows = tuple(table.rows[index] for index in row_indexes)
            grouped_cells = tuple(cell for cell in table.cells if cell.row - 1 in row_indexes)
            chunks.append(
                DocumentChunk(
                    chunk_id=(
                        f"C-{company_id}-{page_result.page:04d}-"
                        f"T{table_index:02d}-{group_index:02d}"
                    ),
                    company_id=company_id,
                    document_id=document_id,
                    source_file=source_file,
                    source_hash=source_hash,
                    page=page_result.page,
                    printed_page_label=printed_page_label,
                    bbox=table.bbox,
                    text=_table_chunk_text(
                        table_id=table.table_id,
                        title=table.title,
                        headers=table.headers,
                        rows=grouped_rows,
                        unit=table.unit,
                        currency=table.currency,
                        period=table.period,
                        footnotes=table.footnotes,
                        group_index=group_index,
                        group_count=len(row_groups),
                    ),
                    is_heading=False,
                    is_ocr=False,
                    font_size=0.0,
                    available_at=available_at,
                    chunk_kind="table",
                    table_id=table.table_id,
                    table_cell_refs=tuple(cell.cell_id for cell in grouped_cells),
                    table_cell_values=tuple(cell.value for cell in grouped_cells),
                )
            )
    return chunks


def _is_valid_bbox(bbox: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = bbox
    return x0 >= 0 and y0 >= 0 and x1 > x0 and y1 > y0


def _table_row_groups(
    rows: Sequence[Sequence[str]],
    *,
    max_rows: int = 12,
    max_chars: int = 3600,
) -> tuple[tuple[int, ...], ...]:
    groups: list[tuple[int, ...]] = []
    current: list[int] = []
    current_chars = 0
    for index, row in enumerate(rows):
        row_chars = sum(len(value) for value in row)
        if current and (len(current) >= max_rows or current_chars + row_chars > max_chars):
            groups.append(tuple(current))
            current = []
            current_chars = 0
        current.append(index)
        current_chars += row_chars
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def _table_chunk_text(
    *,
    table_id: str,
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    unit: str | None,
    currency: str | None,
    period: str | None,
    footnotes: Sequence[str],
    group_index: int,
    group_count: int,
) -> str:
    lines = [
        "[表格证据]",
        f"表格ID: {table_id}",
        f"标题: {title or '未识别'}",
        f"分段: {group_index}/{group_count}",
    ]
    if unit:
        lines.append(f"单位: {unit}")
    if currency:
        lines.append(f"币种: {currency}")
    if period:
        lines.append(f"期间: {period}")
    lines.extend(
        (
            "|" + "|".join(headers) + "|",
            "|" + "|".join("---" for _ in headers) + "|",
        )
    )
    lines.extend("|" + "|".join(row) + "|" for row in rows)
    lines.append("逐行语义:")
    for row in rows:
        lines.append(
            "- "
            + "；".join(
                f"{header}={value}" for header, value in zip(headers, row, strict=True) if value
            )
        )
    if footnotes:
        lines.append("附注: " + " ".join(footnotes))
    lines.append("引用要求: 数值结论须同时引用本切片及对应表格单元格来源。")
    return "\n".join(lines)


def _intersection_ratio(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    intersection = (x1 - x0) * (y1 - y0)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / max(1.0, min(first_area, second_area))


__all__ = ["DocumentChunk", "build_page_chunks"]
