"""阅读顺序、标题、段落、表格与跨页结构识别。

职责：
  在 document_parser.py 提取的原始 TextBlock 基础上，做进一步版面分析：
  1. 段落合并：将相邻行合并为语义完整的段落。
  2. 章节标题检测：识别招股书中的主要章节（如“风险因素”、“财务资料”）。

为什么需要这个模块：
  原始 TextBlock 粒度是“行”，但证据检索和 LLM 处理需要“段落”级别的上下文。
  例如“风险因素”章节下的一段话可能跨 5 行，合并后才是一个完整的证据候选。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from ipo_risk.parser.document_parser import TextBlock


# ---------------------------------------------------------------------------
# 段落结构
# ---------------------------------------------------------------------------

@dataclass
class Paragraph:
    """一个语义段落，由多个 TextBlock 合并而成。

    字段说明:
        page:       段落起始页码。
        bbox:       段落的外接矩形 (x0, y0, x1, y1)。
        text:       段落全文（各行用空格或换行拼接）。
        blocks:     组成这个段落的原始 TextBlock 列表。
        font_size:  段落的主要字号（用于区分标题 vs 正文）。
        is_heading: 是否被识别为章节标题。
    """

    page: int
    bbox: tuple[float, float, float, float]
    text: str
    blocks: list[TextBlock] = field(default_factory=list)
    font_size: float = 0.0
    is_heading: bool = False


# ---------------------------------------------------------------------------
# 招股书常见章节标题关键词（繁体中文 + 英文）
# ---------------------------------------------------------------------------

# 这些关键词用于检测章节标题，覆盖招股书的主要章节。
# 匹配规则：文本块包含这些关键词 + 字号明显大于正文，就认为是标题。
_SECTION_KEYWORDS = [
    # 繁体中文（招股书原文）
    "概要", "摘要", "風險因素", "行業概覽", "業務",
    "財務資料", "財務概要", "資本結構", "股東", "董事",
    "附錄", "公司資料", "發起人", "包銷", "全球發售",
    "所得款項用途", "基石投資者", "關連交易", "公司架構",
    "股本", "主要股東", "利潤", "股息", "風險",
    # 简体中文（兼容）
    "概要", "摘要", "风险因素", "行业概览", "业务",
    "财务资料", "财务概要", "资本结构", "股东", "董事",
    "附录", "公司资料", "发起人", "包销", "全球发售",
    "所得款项用途", "基石投资者", "关连交易", "公司架构",
    "股本", "主要股东", "利润", "股息",
    # 英文
    "Summary", "Risk Factors", "Industry Overview", "Business",
    "Financial Information", "Capital Structure", "Appendix",
]

# 字号阈值：比正文常见字号（通常 9-11pt）大 20% 以上视为标题
_HEADING_FONT_SIZE_RATIO = 1.15

# 港交所文件常见页脚：– 61 –、- 61 -、I-22 等。
_PRINTED_PAGE_PATTERN = re.compile(
    r"^\s*[–—-]?\s*((?:[A-Z]+-)?(?:\d+|[IVXLCDM]+(?:-\d+)?))\s*[–—-]?\s*$",
    re.IGNORECASE,
)


def order_text_blocks(blocks: list[TextBlock]) -> list[TextBlock]:
    """按页面视觉阅读顺序排列文本块。

    PyMuPDF 的源对象顺序并不总是视觉顺序。港交所 PDF 经常把页脚放在
    正文之前返回；若直接按源顺序合并，页脚会与正文组成一个覆盖整页的段落。
    """
    return sorted(
        blocks,
        key=lambda block: (
            block.page,
            round(block.y0, 1),
            round(block.x0, 1),
            block.block_no,
            block.line_no,
        ),
    )


def extract_printed_page_label(
    blocks: list[TextBlock],
    page_height: float | None = None,
) -> str | None:
    """从页面底部提取展示用印刷页码，不替代物理页码。"""
    for block in sorted(blocks, key=lambda item: item.y0, reverse=True):
        if page_height is not None and block.y0 < page_height * 0.72:
            continue
        match = _PRINTED_PAGE_PATTERN.fullmatch(block.text)
        if match:
            return match.group(1)
    return None


def _is_printed_page_footer(block: TextBlock, page_height: float | None) -> bool:
    """判断文本块是否为页面底部印刷页码。"""
    if page_height is not None and block.y0 < page_height * 0.72:
        return False
    return _PRINTED_PAGE_PATTERN.fullmatch(block.text) is not None


# ---------------------------------------------------------------------------
# 段落合并
# ---------------------------------------------------------------------------

def merge_blocks_into_paragraphs(
    blocks: list[TextBlock],
    vertical_gap_threshold: float = 10.0,
    font_size_tolerance: float = 2.0,
    page_height: float | None = None,
) -> list[Paragraph]:
    """将同一页的文本块合并为段落。

    合并规则：
        1. 垂直间距小于 threshold 的相邻行合并。
        2. 字号差异小于 tolerance 的行合并。
        3. 间距大或字号变化明显时，开始新段落。

    参数:
        blocks:                同一页的 TextBlock 列表（按阅读顺序）。
        vertical_gap_threshold: 垂直间距阈值（PDF 点），超过则分段。
        font_size_tolerance:   字号差异阈值，超过则分段。

    返回:
        Paragraph 列表。
    """
    ordered_blocks = [
        block
        for block in order_text_blocks(blocks)
        if not _is_printed_page_footer(block, page_height)
    ]
    if not ordered_blocks:
        return []

    paragraphs: list[Paragraph] = []
    current_blocks: list[TextBlock] = [ordered_blocks[0]]

    for i in range(1, len(ordered_blocks)):
        prev = ordered_blocks[i - 1]
        curr = ordered_blocks[i]

        # 计算垂直间距：上一块的底部到当前块的顶部
        vertical_gap = curr.y0 - prev.y1
        # 字号差异
        size_diff = abs(curr.font_size - prev.font_size)

        # 判断是否应该开始新段落
        should_break = (
            vertical_gap > vertical_gap_threshold
            or size_diff > font_size_tolerance
            or curr.page != prev.page  # 跨页不合并
        )

        if should_break:
            # 保存当前段落，开始新段落
            paragraphs.append(_build_paragraph(current_blocks))
            current_blocks = [curr]
        else:
            current_blocks.append(curr)

    # 不要忘记最后一个段落
    if current_blocks:
        paragraphs.append(_build_paragraph(current_blocks))

    return paragraphs


def _build_paragraph(blocks: list[TextBlock]) -> Paragraph:
    """从一组文本块构建一个段落。"""
    # 外接矩形：取所有块的最小 x0,y0 和最大 x1,y1
    x0 = min(b.x0 for b in blocks)
    y0 = min(b.y0 for b in blocks)
    x1 = max(b.x1 for b in blocks)
    y1 = max(b.y1 for b in blocks)

    # 拼接文本：用换行连接
    text = "\n".join(b.text for b in blocks)

    # 主要字号：取出现次数最多的字号，而不是简单取第一行。
    sizes = Counter(round(block.font_size, 1) for block in blocks if block.font_size > 0)
    font_size = sizes.most_common(1)[0][0] if sizes else 0.0

    return Paragraph(
        page=blocks[0].page,
        bbox=(round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)),
        text=text,
        blocks=list(blocks),
        font_size=round(font_size, 1),
        is_heading=False,
    )


# ---------------------------------------------------------------------------
# 章节标题检测
# ---------------------------------------------------------------------------

def detect_headings(
    paragraphs: list[Paragraph],
    body_font_size: float = 10.0,
) -> list[Paragraph]:
    """检测并标记章节标题。

    判定规则：
        1. 段落文本包含章节关键词。
        2. 段落字号 > 正文字号 * 1.2（明显比正文大）。
        3. 段落较短（标题通常不会很长）。

    参数:
        paragraphs:    段落列表（会被原地修改 is_heading 字段）。
        body_font_size: 正文的常见字号，默认 10pt。

    返回:
        修改后的段落列表（同一个对象）。
    """
    heading_threshold = body_font_size * _HEADING_FONT_SIZE_RATIO

    for para in paragraphs:
        if para.font_size >= heading_threshold and len(para.text) < 100:
            for keyword in _SECTION_KEYWORDS:
                if keyword in para.text:
                    para.is_heading = True
                    break

    return paragraphs


def detect_body_font_size(paragraphs: list[Paragraph]) -> float:
    """检测正文的主要字号。

    方法：按非空字符数加权统计字号，取承载正文字符最多的字号。
    这比硬编码 10pt 更准确，因为不同招股书的排版可能不同。
    """
    if not paragraphs:
        return 10.0

    sizes: Counter[float] = Counter()
    for paragraph in paragraphs:
        if paragraph.font_size > 0:
            weight = max(len("".join(paragraph.text.split())), 1)
            sizes[paragraph.font_size] += weight
    if not sizes:
        return 10.0
    return sizes.most_common(1)[0][0]
