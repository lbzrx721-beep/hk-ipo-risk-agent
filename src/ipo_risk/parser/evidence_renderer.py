"""按照页码与bbox生成证据截图和高亮预览。

职责：
  根据 Evidence 对象的 page + bbox，在 PDF 原页上画高亮框并截图。
  赛题要求“精准映射至招股书原 PDF 页码与段落的证据截图”，
  这个模块就是实现这个要求的核心。

截图流程：
  1. 打开 PDF，加载指定页面。
  2. 在 bbox 位置添加高亮标注（黄色半透明）。
  3. 渲染整个页面为图片。
  4. 裁剪到高亮区域附近（保留上下文）。
  5. 保存为 PNG。

坐标系统：
  输入的 bbox 是 PDF 点坐标，和 PyMuPDF 的 fitz.Rect 直接对应。
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz

from ipo_risk.storage.files import ensure_dir

logger = logging.getLogger(__name__)


def render_evidence_screenshot(
    pdf_path: Path,
    page: int,
    bbox: tuple[float, float, float, float],
    output_path: Path,
    dpi: int = 150,
    highlight_color: tuple[float, float, float] = (1.0, 1.0, 0.0),
    highlight_opacity: float = 0.3,
    padding: float = 30.0,
) -> Path | None:
    """为一条证据生成带高亮框的截图。

    参数:
        pdf_path:          PDF 文件路径。
        page:              物理页码（从 1 开始）。
        bbox:              证据的坐标 (x0, y0, x1, y1)。
        output_path:       截图保存路径（PNG）。
        dpi:               渲染分辨率。
        highlight_color:   高亮颜色，默认黄色 (R,G,B)。
        highlight_opacity: 高亮透明度，0.3 = 半透明。
        padding:           截图四周留白（PDF 点），保留上下文。

    返回:
        截图路径（成功），或 None（失败）。

    为什么返回 None 而不是抛异常：
        截图是辅助产物，不应该因为截图失败就阻断整个解析流程。
        失败了只记日志，Evidence.screenshot_path 留 null。
    """
    try:
        if page < 1:
            raise ValueError("page must be one-based and positive")
        x0, y0, x1, y1 = bbox
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError("invalid evidence bbox")

        # ----------------------------------------------------------
        # 1. 打开 PDF 并加载页面
        # ----------------------------------------------------------
        with fitz.open(str(pdf_path)) as doc:
            page_obj = doc.load_page(page - 1)  # 页索引从 0 开始

        # ----------------------------------------------------------
        # 2. 添加高亮标注
        #    fitz.Rect 接受 (x0, y0, x1, y1)
        # ----------------------------------------------------------
            rect = fitz.Rect(*bbox)
            annot = page_obj.add_highlight_annot(rect)
            annot.set_colors(stroke=highlight_color)
            annot.set_opacity(highlight_opacity)
            annot.update()

        # ----------------------------------------------------------
        # 3. 计算截图区域：高亮框 + 四周 padding
        #    不裁剪到精确 bbox，因为保留上下文对评审更友好
        # ----------------------------------------------------------
            page_rect = page_obj.rect
            clip_rect = fitz.Rect(
                max(0, bbox[0] - padding),
                max(0, bbox[1] - padding),
                min(page_rect.width, bbox[2] + padding),
                min(page_rect.height, bbox[3] + padding),
            )

        # ----------------------------------------------------------
        # 4. 渲染并保存
        # ----------------------------------------------------------
            ensure_dir(output_path.parent)
            pix = page_obj.get_pixmap(dpi=dpi, clip=clip_rect, annots=True)
            pix.save(str(output_path))

        logger.info("证据截图已保存: %s (页 %d)", output_path, page)
        return output_path

    except Exception:
        logger.exception(
            "证据截图失败: pdf=%s, page=%d, bbox=%s",
            pdf_path, page, bbox,
        )
        return None


def render_full_page(
    pdf_path: Path,
    page: int,
    output_path: Path,
    dpi: int = 150,
) -> Path | None:
    """渲染整页 PDF 为图片（用于调试和概览）。

    和 render_evidence_screenshot 的区别：
        - 这个不画高亮框，不裁剪。
        - 用于生成每页的预览图，方便开发时检查解析结果。
    """
    try:
        with fitz.open(str(pdf_path)) as doc:
            page_obj = doc.load_page(page - 1)

            ensure_dir(output_path.parent)
            pix = page_obj.get_pixmap(dpi=dpi)
            pix.save(str(output_path))
        return output_path

    except Exception:
        logger.exception("整页渲染失败: pdf=%s, page=%d", pdf_path, page)
        return None
