"""将招股书解析为带页码和坐标的页面块。

职责：
  这是整个解析流水线的核心入口。用 PyMuPDF 打开 PDF，逐页提取文本块，
  每个文本块都带有物理页码、PDF 点坐标、字号等信息。

数据流：
  PDF 文件
    ↓  parse_document()
  Iterator[PageResult]        ← 每页一个结果
    ↓
  每个 PageResult 包含多个 TextBlock
    ↓  每个 TextBlock 有: page, x0, y0, x1, y1, text, is_ocr, font_size

坐标系统：
  PyMuPDF 使用 PDF 点坐标系（72点=1英寸）。
  原点在页面左上角，x 向右增加，y 向下增加。
  bbox = (x0, y0, x1, y1) 表示一个矩形区域。

OCR 降级：
  如果一页的非空格字符数 < min_chars（默认 80），认为该页是扫描件，
  调用 ocr_page() 尝试 OCR 提取。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF

from ipo_risk.parser.ocr import ocr_page
from ipo_risk.parser.tables import TableArtifact, extract_page_tables
from ipo_risk.parser.visual import VisualRegion, detect_visual_regions


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TextBlock:
    """一个文本块，对应 PDF 页面中的一行或多行文字。

    为什么用 frozen dataclass:
        解析结果一旦生成就不应该被修改，和 Evidence 的不可变原则一致。

    字段说明:
        page:       物理页码，从 1 开始。这是定位和截图的唯一页码主键。
        x0,y0,x1,y1: PDF 点坐标，左上角为原点。
        text:       文本内容，保留原始语言（繁体中文不转换）。
        is_ocr:     是否来自 OCR 兑底，用于区分证据来源。
        font_size:  主要字号，用于后续版面分析（标题 vs 正文）。
        block_no:   PyMuPDF 返回的块编号，用于保持阅读顺序。
        line_no:    块内的行号。
    """

    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    is_ocr: bool = False
    font_size: float = 0.0
    block_no: int = 0
    line_no: int = 0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """返回标准 bbox 元组，和 Evidence.bbox 格式一致。"""
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class PageResult:
    """一页的完整解析结果。

    字段说明:
        page:            物理页码（从 1 开始）。
        width, height:   页面尺寸（PDF 点）。
        has_text_layer:  是否有足够的文字层（False = 扫描页或空白页）。
        text_blocks:     本页所有文本块。
        raw_text:        所有文本块拼接后的原始文本。
        char_count:      非空格字符数，用于判断扫描页。
    """

    page: int
    width: float
    height: float
    has_text_layer: bool
    text_blocks: list[TextBlock] = field(default_factory=list)
    raw_text: str = ""
    char_count: int = 0
    visual_regions: list[VisualRegion] = field(default_factory=list)
    tables: list[TableArtifact] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 单页解析
# ---------------------------------------------------------------------------

def _extract_text_blocks(page: fitz.Page, page_num: int) -> list[TextBlock]:
    """从一页 PDF 提取所有文本块。

    使用 PyMuPDF 的 "dict" 模式，返回嵌套结构:
        page → blocks → lines → spans

    我们在 "line" 级别提取，因为:
        - 太细（字符级）会丢失语义上下文
        - 太粗（整块）可能混入不相关内容

    参数:
        page:     PyMuPDF Page 对象。
        page_num: 物理页码（从 1 开始）。

    返回:
        TextBlock 列表，按阅读顺序排列。
    """
    blocks: list[TextBlock] = []

    # get_text("dict") 返回完整的页面结构，包括字体、颜色等信息
    text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

    for block_idx, block in enumerate(text_dict.get("blocks", [])):
        # type=0 是文字块，type=1 是图片块，我们只处理文字块
        if block.get("type") != 0:
            continue

        for line_idx, line in enumerate(block.get("lines", [])):
            # 一行由多个 span 组成（可能不同字体、字号）
            spans = line.get("spans", [])
            if not spans:
                continue

            # 拼接所有 span 的文本
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue

            # 提取坐标和字号
            bbox = line.get("bbox", (0, 0, 0, 0))
            font_size = max(
                (s.get("size", 0.0) for s in spans),
                default=0.0,
            )

            blocks.append(TextBlock(
                page=page_num,
                x0=round(bbox[0], 2),
                y0=round(bbox[1], 2),
                x1=round(bbox[2], 2),
                y1=round(bbox[3], 2),
                text=text,
                font_size=round(font_size, 1),
                block_no=block_idx,
                line_no=line_idx,
            ))

    return blocks


def parse_page(
    doc: fitz.Document,
    page_index: int,
    min_chars: int = 80,
) -> PageResult:
    """解析 PDF 中的一页。

    流程：
        1. 用 PyMuPDF 提取文字层。
        2. 计算非空格字符数。
        3. 如果字符数 < min_chars，认为是扫描页，调用 OCR。
        4. 返回 PageResult。

    参数:
        doc:        PyMuPDF Document 对象。
        page_index: 页索引（从 0 开始，和 PyMuPDF 一致）。
        min_chars:  触发 OCR 的最小字符数。

    返回:
        PageResult 实例。
    """
    page_num = page_index + 1  # 物理页码从 1 开始
    page = doc.load_page(page_index)
    rect = page.rect

    # --- 第一步：尝试从文字层提取 ---
    text_blocks = _extract_text_blocks(page, page_num)
    raw_text = "\n".join(b.text for b in text_blocks)
    # 计算非空格字符数（空格和换行不算）
    char_count = len(raw_text.replace(" ", "").replace("\n", ""))

    # --- 第二步：判断是否有足够的文字层 ---
    has_text_layer = char_count >= min_chars

    # --- 第三步：扫描页触发 OCR 兑底 ---
    if not has_text_layer:
        ocr_blocks = ocr_page(page, page_num)
        if ocr_blocks:
            # OCR 成功：只有在 OCR 提取的信息更多时才替换原生文字层。
            # 这样可避免把“文字很少但文字层正常”的目录/公司资料页误标为 OCR。
            candidate_blocks = [
                TextBlock(
                    page=ob.page,
                    x0=ob.x0,
                    y0=ob.y0,
                    x1=ob.x1,
                    y1=ob.y1,
                    text=ob.text,
                    is_ocr=True,
                )
                for ob in ocr_blocks
            ]
            candidate_text = "\n".join(b.text for b in candidate_blocks)
            candidate_char_count = len(
                candidate_text.replace(" ", "").replace("\n", "")
            )
            if candidate_char_count > char_count:
                text_blocks = candidate_blocks
                raw_text = candidate_text
                char_count = candidate_char_count

    visual_regions = detect_visual_regions(
        page,
        page_num=page_num,
        text_blocks=text_blocks,
    )
    tables = extract_page_tables(
        page,
        page_num=page_num,
        text_blocks=text_blocks,
        visual_regions=visual_regions,
    )

    return PageResult(
        page=page_num,
        width=round(rect.width, 2),
        height=round(rect.height, 2),
        has_text_layer=has_text_layer,
        text_blocks=text_blocks,
        raw_text=raw_text,
        char_count=char_count,
        visual_regions=visual_regions,
        tables=tables,
    )


# ---------------------------------------------------------------------------
# 全文档解析
# ---------------------------------------------------------------------------

def parse_document(
    pdf_path: Path,
    min_chars: int = 80,
    page_range: tuple[int, int] | None = None,
) -> Iterator[PageResult]:
    """解析整个 PDF 文档，逐页返回结果。

    使用生成器（Iterator），这样 672 页的招股书不需要一次性全部解析到内存。

    参数:
        pdf_path:   PDF 文件路径。
        min_chars:  触发 OCR 的最小字符数。
        page_range: 可选的页范围 (start, end)，索引从 0 开始。
                    例如 (0, 10) 只解析前 10 页，用于测试。
                    None 表示解析全部页面。

    返回:
        PageResult 的生成器，每次 yield 一页。

    用法示例:
        for page_result in parse_document(path, page_range=(0, 5)):
            print(f"第 {page_result.page} 页: {page_result.char_count} 字符")
    """
    doc = fitz.open(str(pdf_path))
    try:
        start = page_range[0] if page_range else 0
        end = page_range[1] if page_range else len(doc)

        for i in range(start, min(end, len(doc))):
            yield parse_page(doc, i, min_chars=min_chars)
    finally:
        doc.close()


def get_page_count(pdf_path: Path) -> int:
    """获取 PDF 的总页数（不解析内容，很快）。"""
    doc = fitz.open(str(pdf_path))
    try:
        return len(doc)
    finally:
        doc.close()
