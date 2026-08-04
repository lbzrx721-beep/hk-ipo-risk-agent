"""把 PDF 表格候选转换为可检索、可计算、可回链的结构化工件。"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
import multiprocessing
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any, Sequence

import fitz

from ipo_risk.parser.visual import VisualRegion, VisualRegionKind


_UNIT_PATTERN = re.compile(
    r"(?:单位|Unit)\s*[:：]?\s*([^\n|；;]{1,40})",
    re.IGNORECASE,
)
_PERIOD_PATTERN = re.compile(
    r"(?:截至|止于|for the (?:year|period|six months)|year ended)"
    r"[^|。\n]{0,80}",
    re.IGNORECASE,
)
_CURRENCIES = (
    ("港元", "HKD"),
    ("港币", "HKD"),
    ("港幣", "HKD"),
    ("HKD", "HKD"),
    ("人民币", "RMB"),
    ("人民幣", "RMB"),
    ("RMB", "RMB"),
    ("美元", "USD"),
    ("USD", "USD"),
)

_TABLE_WORKER_DOCUMENT: fitz.Document | None = None
_DEFAULT_TABLE_EXTRACTION_TIMEOUT_SECONDS = 8.0


@dataclass(frozen=True)
class TableCellArtifact:
    """表格中的一个单元格及其原始页面坐标。"""

    cell_id: str
    row: int
    column: int
    value: str
    bbox: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class TableArtifact:
    """表格的规范化表示；Markdown 用于模型，cells 用于核验和计算。"""

    table_id: str
    page: int
    bbox: tuple[float, float, float, float]
    parser: str
    confidence: float
    title: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    cells: tuple[TableCellArtifact, ...]
    markdown: str
    unit: str | None = None
    currency: str | None = None
    period: str | None = None
    footnotes: tuple[str, ...] = ()
    quality_warnings: tuple[str, ...] = ()
    requires_vision: bool = False


def _initialize_table_worker(pdf_path: str) -> None:
    """在隔离进程中只打开一次PDF，供逐页表格解析复用。"""
    global _TABLE_WORKER_DOCUMENT
    _TABLE_WORKER_DOCUMENT = fitz.open(pdf_path)


def _extract_grid_candidate_in_worker(
    page_index: int,
    page_num: int,
    candidate: VisualRegion,
    text_blocks: Sequence[Any],
    first_table_index: int,
) -> tuple[TableArtifact, ...]:
    document = _TABLE_WORKER_DOCUMENT
    if document is None:
        raise RuntimeError("isolated table worker did not open the PDF")
    page = document.load_page(page_index)
    return tuple(
        _extract_grid_candidate(
            page,
            page_num=page_num,
            candidate=candidate,
            text_blocks=text_blocks,
            first_table_index=first_table_index,
        )
    )


class IsolatedTableExtractor:
    """把可能卡死的PyMuPDF表格识别放进可终止的独立进程。"""

    def __init__(
        self,
        pdf_path: Path,
        *,
        timeout_seconds: float = _DEFAULT_TABLE_EXTRACTION_TIMEOUT_SECONDS,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("table extraction timeout must be positive")
        self._pdf_path = pdf_path.resolve()
        self._timeout_seconds = timeout_seconds
        self._context = multiprocessing.get_context("spawn")
        self._pool: Pool | None = None

    def extract(
        self,
        *,
        page_index: int,
        page_num: int,
        candidate: VisualRegion,
        text_blocks: Sequence[Any],
        first_table_index: int,
    ) -> tuple[tuple[TableArtifact, ...], str | None]:
        pool = self._ensure_pool()
        pending = pool.apply_async(
            _extract_grid_candidate_in_worker,
            (
                page_index,
                page_num,
                candidate,
                tuple(text_blocks),
                first_table_index,
            ),
        )
        try:
            return tuple(pending.get(timeout=self._timeout_seconds)), None
        except multiprocessing.TimeoutError:
            self._terminate_pool()
            return (), "table_extraction_timeout"
        except Exception:  # noqa: BLE001 - 单页失败必须降级，不能中断全文解析
            self._terminate_pool()
            return (), "table_extraction_failed"

    def close(self) -> None:
        pool = self._pool
        self._pool = None
        if pool is None:
            return
        pool.close()
        pool.join()

    def _ensure_pool(self) -> Pool:
        if self._pool is None:
            self._pool = self._context.Pool(
                processes=1,
                initializer=_initialize_table_worker,
                initargs=(str(self._pdf_path),),
                maxtasksperchild=200,
            )
        return self._pool

    def _terminate_pool(self) -> None:
        pool = self._pool
        self._pool = None
        if pool is None:
            return
        pool.terminate()
        pool.join()


def extract_page_tables(
    page: fitz.Page,
    *,
    page_num: int,
    text_blocks: Sequence[Any],
    visual_regions: Sequence[VisualRegion],
    isolated_extractor: IsolatedTableExtractor | None = None,
    deferred_region_ids: set[str] | None = None,
) -> list[TableArtifact]:
    """只对本地识别出的表格候选做结构解析，避免逐页调用模型。"""
    candidates = [
        region
        for region in visual_regions
        if region.kind is VisualRegionKind.TABLE
    ]
    artifacts: list[TableArtifact] = []
    for candidate in candidates:
        if "aligned_text_columns" in candidate.reasons:
            if _looks_like_table_of_contents(candidate, text_blocks):
                # 招股书目录经常由“标题 + 点线 + 页码”组成，位置对齐后很像
                # 无边框表格。它不是业务表格，跳过可避免无效解析和伪表格入库。
                continue
            borderless = _extract_borderless_artifacts(
                page_num=page_num,
                first_table_index=len(artifacts) + 1,
                candidate=candidate,
                text_blocks=text_blocks,
            )
            if borderless:
                artifacts.extend(borderless)
            # 对齐文本候选已经由确定性解析器处理。若未形成有效表格，通常是目录、
            # 点线页码等误报；不再回退到可能长时间卡住的 find_tables(text)。
            continue

        warning: str | None = None
        if isolated_extractor is not None:
            extracted, warning = isolated_extractor.extract(
                page_index=page_num - 1,
                page_num=page_num,
                candidate=candidate,
                text_blocks=text_blocks,
                first_table_index=len(artifacts) + 1,
            )
        else:
            try:
                extracted = tuple(
                    _extract_grid_candidate(
                        page,
                        page_num=page_num,
                        candidate=candidate,
                        text_blocks=text_blocks,
                        first_table_index=len(artifacts) + 1,
                    )
                )
            except Exception:  # noqa: BLE001 - 保持直接调用兼容性
                extracted = ()
                warning = "table_extraction_failed"
        if warning is not None and deferred_region_ids is not None:
            deferred_region_ids.add(candidate.region_id)
        for artifact in extracted:
            if any(
                _bbox_overlap(artifact.bbox, item.bbox) >= 0.85
                for item in artifacts
            ):
                continue
            artifacts.append(artifact)
    return artifacts


def _extract_grid_candidate(
    page: fitz.Page,
    *,
    page_num: int,
    candidate: VisualRegion,
    text_blocks: Sequence[Any],
    first_table_index: int,
) -> list[TableArtifact]:
    clip = _expanded_clip(candidate.bbox, page.rect)
    finder = page.find_tables(clip=clip, strategy="lines")
    artifacts: list[TableArtifact] = []
    for table in tuple(getattr(finder, "tables", ()) or ()):
        artifact = _table_to_artifact(
            table,
            page_num=page_num,
            table_index=first_table_index + len(artifacts),
            candidate=candidate,
            text_blocks=text_blocks,
        )
        if artifact is not None:
            artifacts.append(artifact)
    return artifacts


def _extract_borderless_artifacts(
    *,
    page_num: int,
    first_table_index: int,
    candidate: VisualRegion,
    text_blocks: Sequence[Any],
) -> list[TableArtifact]:
    candidate_rect = fitz.Rect(candidate.bbox)
    body_blocks = [
        block
        for block in text_blocks
        if _block_center_in_rect(block, candidate_rect)
    ]
    groups = _group_blocks_by_y(body_blocks)
    if not groups:
        return []
    left_aligned_groups = [
        group
        for group in groups
        if len(group) >= 3
        and min(float(block.x0) for block in group) <= candidate_rect.x0 + 25
    ]
    if len(left_aligned_groups) < 2:
        return []
    column_count = Counter(len(group) for group in left_aligned_groups).most_common(1)[0][0]
    if column_count < 3:
        return []
    core_groups = [
        group
        for group in left_aligned_groups
        if len(group) >= max(3, column_count - 2)
    ]
    segments = _split_table_segments(core_groups)
    artifacts: list[TableArtifact] = []
    for offset, core_segment in enumerate(segments):
        first_y = min(float(block.y0) for block in core_segment[0])
        last_y = max(float(block.y1) for block in core_segment[-1])
        segment_groups = [
            group
            for group in groups
            if min(float(block.y0) for block in group) >= first_y - 1
            and max(float(block.y1) for block in group) <= last_y + 1
            and min(float(block.x0) for block in group) <= candidate_rect.x0 + 25
        ]
        artifact = _build_borderless_artifact(
            page_num=page_num,
            table_index=first_table_index + offset,
            candidate=candidate,
            text_blocks=text_blocks,
            data_groups=segment_groups,
            column_count=column_count,
        )
        if artifact is not None:
            artifacts.append(artifact)
    return artifacts


def _build_borderless_artifact(
    *,
    page_num: int,
    table_index: int,
    candidate: VisualRegion,
    text_blocks: Sequence[Any],
    data_groups: Sequence[Sequence[Any]],
    column_count: int,
) -> TableArtifact | None:
    references = [group for group in data_groups if len(group) == column_count]
    if not references:
        return None
    reference = sorted(references[0], key=lambda block: float(block.x0))
    anchors = tuple(_block_center(block) for block in reference)
    rows_with_blocks = tuple(
        _align_group_to_columns(group, anchors)
        for group in data_groups
    )
    rows = tuple(
        tuple(_clean_borderless_value(block.text) if block is not None else "" for block in row)
        for row in rows_with_blocks
    )
    rows = tuple(row for row in rows if any(row))
    if len(rows) < 2:
        return None

    candidate_rect = fitz.Rect(candidate.bbox)
    first_data_y = min(float(block.y0) for block in data_groups[0])
    header_blocks = [
        block
        for block in text_blocks
        if float(block.y1) <= first_data_y + 1
        and first_data_y - float(block.y1) <= 90
        and float(block.x1) >= candidate_rect.x0 - 5
        and float(block.x0) <= candidate_rect.x1 + 5
    ]
    headers = _borderless_headers(anchors, header_blocks)
    table_id = f"T-P{page_num:04d}-{table_index:03d}"
    cells: list[TableCellArtifact] = []
    for row_index, row in enumerate(rows_with_blocks, start=1):
        for column_index, block in enumerate(row, start=1):
            cells.append(
                TableCellArtifact(
                    cell_id=f"{table_id}:R{row_index}:C{column_index}",
                    row=row_index,
                    column=column_index,
                    value=(
                        _clean_borderless_value(block.text)
                        if block is not None
                        else ""
                    ),
                    bbox=(
                        _round_bbox(block.bbox)
                        if block is not None
                        else None
                    ),
                )
            )
    table_bbox = (
        candidate_rect.x0,
        min(
            (
                float(block.y0)
                for block in header_blocks
                if _is_structural_header(block.text)
            ),
            default=first_data_y,
        ),
        candidate_rect.x1,
        max(float(block.y1) for block in data_groups[-1]),
    )
    title = _nearby_title(table_bbox, text_blocks)
    markdown = _to_markdown(headers, rows)
    searchable_text = " ".join((title, *headers, *(value for row in rows for value in row)))
    unit_match = _UNIT_PATTERN.search(searchable_text)
    period_match = _PERIOD_PATTERN.search(searchable_text)
    warnings = _quality_warnings(headers, rows)
    confidence = max(0.0, min(1.0, candidate.confidence - 0.06 * len(warnings)))
    return TableArtifact(
        table_id=table_id,
        page=page_num,
        bbox=_round_bbox(table_bbox),
        parser="pymupdf-positioned-text",
        confidence=round(confidence, 2),
        title=title,
        headers=headers,
        rows=rows,
        cells=tuple(cells),
        markdown=markdown,
        unit=unit_match.group(1).strip() if unit_match else None,
        currency=_detect_currency(searchable_text),
        period=period_match.group(0).strip() if period_match else None,
        footnotes=_nearby_footnotes(table_bbox, text_blocks),
        quality_warnings=warnings,
        requires_vision=confidence < 0.65,
    )


def _split_table_segments(
    groups: Sequence[Sequence[Any]],
    *,
    max_row_gap: float = 48.0,
) -> tuple[tuple[Sequence[Any], ...], ...]:
    segments: list[list[Sequence[Any]]] = []
    for group in groups:
        if not segments:
            segments.append([group])
            continue
        previous = segments[-1][-1]
        gap = min(float(block.y0) for block in group) - max(
            float(block.y1) for block in previous
        )
        if gap > max_row_gap:
            segments.append([group])
        else:
            segments[-1].append(group)
    return tuple(tuple(segment) for segment in segments if len(segment) >= 2)


def _group_blocks_by_y(
    blocks: Sequence[Any],
    *,
    tolerance: float = 3.5,
) -> list[list[Any]]:
    groups: list[list[Any]] = []
    for block in sorted(blocks, key=lambda item: (float(item.y0), float(item.x0))):
        center_y = (float(block.y0) + float(block.y1)) / 2
        if not groups:
            groups.append([block])
            continue
        previous_y = sum(
            (float(item.y0) + float(item.y1)) / 2
            for item in groups[-1]
        ) / len(groups[-1])
        if abs(center_y - previous_y) <= tolerance:
            groups[-1].append(block)
        else:
            groups.append([block])
    for group in groups:
        group.sort(key=lambda block: float(block.x0))
    return groups


def _align_group_to_columns(
    group: Sequence[Any],
    anchors: Sequence[float],
) -> tuple[Any | None, ...]:
    aligned: list[Any | None] = [None] * len(anchors)
    for block in group:
        target = min(
            range(len(anchors)),
            key=lambda index: abs(_block_center(block) - anchors[index]),
        )
        current = aligned[target]
        if current is None:
            aligned[target] = block
        else:
            current_distance = abs(_block_center(current) - anchors[target])
            if abs(_block_center(block) - anchors[target]) < current_distance:
                aligned[target] = block
    return tuple(aligned)


def _borderless_headers(
    anchors: Sequence[float],
    header_blocks: Sequence[Any],
) -> tuple[str, ...]:
    year_blocks = [
        block
        for block in header_blocks
        if re.fullmatch(r"(?:19|20)\d{2}年?", _normalize_cell(block.text))
    ]
    unit_blocks = [
        block
        for block in header_blocks
        if len(_normalize_cell(block.text)) <= 20
        and re.search(
            r"(?:人民币|人民幣|港元|港幣|HKD|RMB|USD|百萬|百万|千元|%)",
            _normalize_cell(block.text),
            re.IGNORECASE,
        )
    ]
    period_blocks = [
        block
        for block in header_blocks
        if re.search(
            r"(?:截至|止年度|止六個月|止六个月|year ended|months ended)",
            _normalize_cell(block.text),
            re.IGNORECASE,
        )
    ]
    headers = ["项目"]
    for anchor in anchors[1:]:
        year_block = _nearest_header_block(anchor, year_blocks)
        year = _normalize_cell(year_block.text) if year_block is not None else ""
        period = (
            _nearest_header_value(_block_center(year_block), period_blocks)
            if year_block is not None
            else ""
        )
        unit = _nearest_header_value(anchor, unit_blocks)
        parts = tuple(part for part in (period, year, unit) if part)
        headers.append(" / ".join(parts) if parts else f"列{len(headers) + 1}")
    return tuple(headers)


def _nearest_header_value(anchor: float, blocks: Sequence[Any]) -> str:
    block = _nearest_header_block(anchor, blocks)
    if block is None:
        return ""
    return _normalize_cell(block.text)


def _nearest_header_block(anchor: float, blocks: Sequence[Any]) -> Any | None:
    if not blocks:
        return None
    return min(blocks, key=lambda item: abs(_block_center(item) - anchor))


def _is_structural_header(value: object) -> bool:
    text = _normalize_cell(value)
    return bool(
        re.fullmatch(r"(?:19|20)\d{2}年?", text)
        or re.search(
            r"(?:截至|止年度|止六個月|止六个月|人民币|人民幣|港元|港幣|"
            r"HKD|RMB|USD|百萬|百万|千元|%)",
            text,
            re.IGNORECASE,
        )
    )


def _block_center(block: Any) -> float:
    return (float(block.x0) + float(block.x1)) / 2


def _block_center_in_rect(block: Any, rect: fitz.Rect) -> bool:
    center_x = _block_center(block)
    center_y = (float(block.y0) + float(block.y1)) / 2
    return rect.x0 - 1 <= center_x <= rect.x1 + 1 and rect.y0 - 1 <= center_y <= rect.y1 + 1


def _clean_borderless_value(value: object) -> str:
    text = _normalize_cell(value)
    # 不使用 ``(?:\s*\.\s*){2,}$``：面对目录页数百个“点 + 空格”时，
    # 该表达式存在大量等价匹配路径，会产生灾难性回溯。下面的线性扫描
    # 只剥离由空格和点组成、且至少含两个点的尾部。
    trimmed = text.rstrip()
    tail_start = len(trimmed)
    while tail_start > 0 and trimmed[tail_start - 1] in ". ":
        tail_start -= 1
    dotted_tail = trimmed[tail_start:]
    if dotted_tail.count(".") >= 2:
        trimmed = trimmed[:tail_start].rstrip()
    return trimmed


def _looks_like_table_of_contents(
    candidate: VisualRegion,
    text_blocks: Sequence[Any],
) -> bool:
    """识别目录中的点线页码布局，避免把它当成无边框业务表格。"""
    candidate_rect = fitz.Rect(candidate.bbox)
    values = [
        _normalize_cell(block.text)
        for block in text_blocks
        if _block_center_in_rect(block, candidate_rect)
        and _normalize_cell(block.text)
    ]
    if len(values) < 6:
        return False
    dotted_lines = sum(
        value.count(".") >= 8
        and len(value.replace(".", "").replace(" ", "")) <= 20
        for value in values
    )
    page_numbers = sum(
        bool(re.fullmatch(r"(?:[IVXLCDM]+-)?\d{1,4}", value, re.IGNORECASE))
        for value in values
    )
    return dotted_lines >= 2 and page_numbers >= 2


def _table_to_artifact(
    table: Any,
    *,
    page_num: int,
    table_index: int,
    candidate: VisualRegion,
    text_blocks: Sequence[Any],
) -> TableArtifact | None:
    extracted = table.extract()
    if not extracted or int(getattr(table, "col_count", 0)) < 2:
        return None
    normalized = tuple(
        tuple(_normalize_cell(value) for value in row)
        for row in extracted
    )
    if len(normalized) < 2:
        return None

    raw_headers = tuple(
        _normalize_cell(value)
        for value in getattr(getattr(table, "header", None), "names", ())
    )
    column_count = max(len(row) for row in normalized)
    headers = _normalize_headers(raw_headers, normalized[0], column_count)
    rows = normalized
    if _same_row(headers, normalized[0]):
        rows = normalized[1:]
    rows = tuple(_pad_row(row, column_count) for row in rows if any(row))
    if not rows:
        return None

    table_id = f"T-P{page_num:04d}-{table_index:03d}"
    bbox = _round_bbox(getattr(table, "bbox"))
    cells = _extract_cells(
        table,
        table_id=table_id,
        headers=headers,
        rows=rows,
        header_in_first_row=_same_row(headers, normalized[0]),
    )
    title = _nearby_title(bbox, text_blocks)
    markdown = _to_markdown(headers, rows)
    searchable_text = " ".join((title, *headers, *(value for row in rows for value in row)))
    unit_match = _UNIT_PATTERN.search(searchable_text)
    period_match = _PERIOD_PATTERN.search(searchable_text)
    warnings = _quality_warnings(headers, rows)
    confidence = max(
        0.0,
        min(1.0, candidate.confidence - 0.06 * len(warnings)),
    )
    return TableArtifact(
        table_id=table_id,
        page=page_num,
        bbox=bbox,
        parser=f"pymupdf-{('text' if 'aligned_text_columns' in candidate.reasons else 'lines')}",
        confidence=round(confidence, 2),
        title=title,
        headers=headers,
        rows=rows,
        cells=cells,
        markdown=markdown,
        unit=unit_match.group(1).strip() if unit_match else None,
        currency=_detect_currency(searchable_text),
        period=period_match.group(0).strip() if period_match else None,
        footnotes=_nearby_footnotes(bbox, text_blocks),
        quality_warnings=warnings,
        requires_vision=confidence < 0.65,
    )


def _extract_cells(
    table: Any,
    *,
    table_id: str,
    headers: tuple[str, ...],
    rows: tuple[tuple[str, ...], ...],
    header_in_first_row: bool,
) -> tuple[TableCellArtifact, ...]:
    table_rows = tuple(getattr(table, "rows", ()) or ())
    row_offset = 1 if header_in_first_row else 0
    artifacts: list[TableCellArtifact] = []
    for row_index, values in enumerate(rows, start=1):
        source_index = row_index - 1 + row_offset
        source_cells = (
            tuple(getattr(table_rows[source_index], "cells", ()) or ())
            if source_index < len(table_rows)
            else ()
        )
        for column_index, value in enumerate(values, start=1):
            raw_bbox = (
                source_cells[column_index - 1]
                if column_index - 1 < len(source_cells)
                else None
            )
            artifacts.append(
                TableCellArtifact(
                    cell_id=f"{table_id}:R{row_index}:C{column_index}",
                    row=row_index,
                    column=column_index,
                    value=value,
                    bbox=_round_bbox(raw_bbox) if raw_bbox is not None else None,
                )
            )
    return tuple(artifacts)


def _normalize_cell(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\u00a0", " ").split())


def _normalize_headers(
    raw_headers: tuple[str, ...],
    first_row: tuple[str, ...],
    column_count: int,
) -> tuple[str, ...]:
    headers = list(_pad_row(raw_headers, column_count))
    first = _pad_row(first_row, column_count)
    for index in range(column_count):
        if not headers[index]:
            headers[index] = first[index] or f"列{index + 1}"
    return tuple(headers)


def _pad_row(row: Sequence[str], column_count: int) -> tuple[str, ...]:
    return tuple((*row, *("" for _ in range(max(0, column_count - len(row))))))[
        :column_count
    ]


def _same_row(first: Sequence[str], second: Sequence[str]) -> bool:
    return tuple(_normalize_cell(value) for value in first) == tuple(
        _normalize_cell(value) for value in second
    )


def _to_markdown(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    def safe(value: str) -> str:
        return value.replace("|", r"\|").replace("\n", "<br>")

    output = [
        "|" + "|".join(safe(value) for value in headers) + "|",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    output.extend("|" + "|".join(safe(value) for value in row) + "|" for row in rows)
    return "\n".join(output)


def _nearby_title(
    bbox: tuple[float, float, float, float],
    text_blocks: Sequence[Any],
) -> str:
    x0, y0, x1, _ = bbox
    nearby = [
        block
        for block in text_blocks
        if str(getattr(block, "text", "")).strip()
        and float(getattr(block, "y1", 0.0)) <= y0 + 1
        and y0 - float(getattr(block, "y1", 0.0)) <= 180
        and float(getattr(block, "x1", 0.0)) >= x0 - 20
        and float(getattr(block, "x0", 0.0)) <= x1 + 20
    ]
    semantic = [
        block
        for block in nearby
        if not re.fullmatch(
            r"(?:19|20)\d{2}年?|(?:人民币|港元|港幣|HKD|RMB|USD|%)",
            _normalize_cell(block.text),
            re.IGNORECASE,
        )
        and not _normalize_cell(block.text).startswith(("（以", "(以", "（未", "(未"))
    ]
    if not semantic:
        return ""
    semantic.sort(
        key=lambda block: (
            float(getattr(block, "font_size", 0.0)),
            1 if len(_normalize_cell(block.text)) <= 40 else 0,
            -len(_normalize_cell(block.text)),
            float(block.y1),
        ),
        reverse=True,
    )
    return _normalize_cell(semantic[0].text)[:240]


def _nearby_footnotes(
    bbox: tuple[float, float, float, float],
    text_blocks: Sequence[Any],
) -> tuple[str, ...]:
    x0, _, x1, y1 = bbox
    notes = [
        str(block.text).strip()
        for block in text_blocks
        if str(getattr(block, "text", "")).strip()
        and float(getattr(block, "y0", 0.0)) >= y1 - 1
        and float(getattr(block, "y0", 0.0)) - y1 <= 72
        and float(getattr(block, "x1", 0.0)) >= x0 - 20
        and float(getattr(block, "x0", 0.0)) <= x1 + 20
        and re.match(r"^(?:注|附注|Note|\*|\()", str(block.text).strip(), re.IGNORECASE)
    ]
    return tuple(notes[:4])


def _quality_warnings(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> tuple[str, ...]:
    warnings: list[str] = []
    if len(set(headers)) != len(headers):
        warnings.append("duplicate_headers")
    if any(len(row) != len(headers) for row in rows):
        warnings.append("inconsistent_column_count")
    values = [value for row in rows for value in row]
    if values and sum(not value for value in values) / len(values) > 0.35:
        warnings.append("many_empty_cells")
    return tuple(warnings)


def _detect_currency(text: str) -> str | None:
    folded = text.upper()
    for marker, code in _CURRENCIES:
        if marker.upper() in folded:
            return code
    return None


def _expanded_clip(
    bbox: tuple[float, float, float, float],
    page_rect: fitz.Rect,
) -> fitz.Rect:
    rect = fitz.Rect(bbox)
    rect.x0 -= 2
    rect.y0 -= 2
    rect.x1 += 2
    rect.y1 += 2
    return rect & page_rect


def _round_bbox(value: Any) -> tuple[float, float, float, float]:
    rect = fitz.Rect(value)
    return (
        round(float(rect.x0), 2),
        round(float(rect.y0), 2),
        round(float(rect.x1), 2),
        round(float(rect.y1), 2),
    )


def _bbox_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    first_rect = fitz.Rect(first)
    second_rect = fitz.Rect(second)
    intersection = first_rect & second_rect
    if intersection.is_empty:
        return 0.0
    return intersection.get_area() / max(
        1.0,
        min(first_rect.get_area(), second_rect.get_area()),
    )


__all__ = [
    "TableArtifact",
    "TableCellArtifact",
    "extract_page_tables",
]
