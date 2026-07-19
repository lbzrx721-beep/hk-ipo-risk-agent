"""将页面解析结果转换为可检索、可回链的段落切片。"""

from __future__ import annotations

from datetime import date
from typing import Annotated

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

    @model_validator(mode="after")
    def validate_bbox(self) -> DocumentChunk:
        x0, y0, x1, y1 = self.bbox
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError("chunk bbox must satisfy x0 < x1 and y0 < y1")
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
    for paragraph_index, paragraph in enumerate(paragraphs, start=1):
        text = paragraph.text.strip()
        if not text:
            continue
        chunks.append(
            DocumentChunk(
                chunk_id=(
                    f"C-{company_id}-{page_result.page:04d}-{paragraph_index:03d}"
                ),
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
    return chunks


__all__ = ["DocumentChunk", "build_page_chunks"]

