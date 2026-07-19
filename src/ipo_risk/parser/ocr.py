"""仅在原生文本不足时触发的OCR适配层。

职责：
  对扫描页或文字层缺失的页面，使用 RapidOCR 提取文字并返回带坐标的文本块。

触发条件：
  由 document_parser.py 判断：如果一页的非空格字符数 < min_text_chars_before_ocr
  （默认 80），就调用 ocr_page() 尝试 OCR。

坐标系统：
  OCR 在像素图上检测，返回的坐标是像素值。
  需要根据 (页面尺寸 / 像素尺寸) 的比例换算回 PDF 点坐标。
  这样后续生成的 Evidence.bbox 才能和 PDF 页面对应。

降级策略：
  如果 RapidOCR 未安装，不会崩溃，只记录警告并返回空列表。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    import fitz


logger = logging.getLogger(__name__)


class OCRBlock(NamedTuple):
    """一个 OCR 识别结果，和 TextBlock 结构对齐。

    使用 NamedTuple 而不是 dataclass，因为它是内部传递用的轻量结构，
    会在 document_parser.py 中被转换为 TextBlock。
    """

    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


def normalize_ocr_output(output: Any) -> list[tuple[Any, str, float]]:
    """统一 RapidOCR 2.x/3.x 不同版本的返回结构。

    RapidOCR 3.9 返回 ``RapidOCROutput`` 对象，旧版本通常返回
    ``(results, elapsed)`` 或直接返回结果列表。本函数将它们统一为
    ``[(四点坐标, 文本, 置信度), ...]``。
    """
    if output is None:
        return []

    boxes = getattr(output, "boxes", None)
    texts = getattr(output, "txts", None)
    scores = getattr(output, "scores", None)
    if boxes is not None and texts is not None and scores is not None:
        return [
            (box.tolist() if hasattr(box, "tolist") else box, str(text), float(score))
            for box, text, score in zip(boxes, texts, scores, strict=False)
        ]

    # 旧版常见结构：(results, elapsed)
    if (
        isinstance(output, tuple)
        and len(output) == 2
        and isinstance(output[0], (list, tuple))
        and not isinstance(output[1], (list, tuple))
    ):
        output = output[0]

    normalized: list[tuple[Any, str, float]] = []
    if isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                normalized.append((item[0], str(item[1]), float(item[2])))
    return normalized


@lru_cache(maxsize=1)
def _get_ocr_engine() -> Any:
    """懒加载并复用 OCR 模型，避免扫描文档每页重复初始化。"""
    from rapidocr import RapidOCR

    return RapidOCR()


def ocr_page(
    page: fitz.Page,
    page_num: int,
    dpi: int = 200,
) -> list[OCRBlock]:
    """对一页 PDF 执行 OCR，返回识别到的文本块。

    参数:
        page: PyMuPDF 的 Page 对象。
        page_num: 物理页码（从 1 开始）。
        dpi: OCR 渲染分辨率，200 是速度和质量的平衡点。

    返回:
        OCRBlock 列表，坐标已经是 PDF 点坐标（不是像素）。
        如果 OCR 未安装或失败，返回空列表。
    """
    # ------------------------------------------------------------------
    # 1. 尝试导入 RapidOCR（懒加载，避免每次启动都加载模型）
    # ------------------------------------------------------------------
    try:
        engine = _get_ocr_engine()
    except ImportError:
        logger.warning(
            "RapidOCR 未安装，跳过第 %d 页的 OCR。"
            "如需启用请安装: pip install rapidocr onnxruntime",
            page_num,
        )
        return []

    try:
        # --------------------------------------------------------------
        # 2. 将页面渲染为像素图片
        # --------------------------------------------------------------
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")

        # --------------------------------------------------------------
        # 3. 执行 OCR
        # --------------------------------------------------------------
        result = normalize_ocr_output(engine(img_bytes))

        if not result:
            return []

        # --------------------------------------------------------------
        # 4. 坐标换算：像素坐标 -> PDF 点坐标
        #    公式: pdf_x = pixel_x * (page_width / pixmap_width)
        # --------------------------------------------------------------
        page_rect = page.rect
        scale_x = page_rect.width / pix.width
        scale_y = page_rect.height / pix.height

        blocks: list[OCRBlock] = []
        for item in result:
            # RapidOCR 返回格式: [[x,y], ...], text, confidence
            pts, text, _confidence = item
            if not text or not text.strip():
                continue

            # 从四个角点算出外接矩形
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x0 = min(xs) * scale_x
            y0 = min(ys) * scale_y
            x1 = max(xs) * scale_x
            y1 = max(ys) * scale_y

            blocks.append(OCRBlock(
                page=page_num,
                x0=round(x0, 2),
                y0=round(y0, 2),
                x1=round(x1, 2),
                y1=round(y1, 2),
                text=text.strip(),
            ))

        logger.info("OCR 从第 %d 页提取了 %d 个文本块", page_num, len(blocks))
        return blocks

    except Exception:
        logger.exception("第 %d 页 OCR 失败", page_num)
        return []
