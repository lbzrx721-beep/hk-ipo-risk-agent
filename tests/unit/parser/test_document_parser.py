"""PDF 解析器单元测试。

测试策略：
  1. 用真实招股书的前几页做测试（page_range=(0, 5)），避免全量解析。
  2. 验证核心数据结构（TextBlock, PageResult）的字段完整性。
  3. 验证 SHA-256 哈希与公司信息中记录的一致。
  4. 验证段落合并和证据截图功能。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipo_risk.config import load_config
from ipo_risk.parser.document_parser import (
    PageResult,
    get_page_count,
    parse_document,
)
from ipo_risk.parser.layout import (
    Paragraph,
    detect_body_font_size,
    merge_blocks_into_paragraphs,
)
from ipo_risk.storage.files import sha256_file


# ---------------------------------------------------------------------------
# 测试用 PDF 路径
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_PDF_PATH = (
    _PROJECT_ROOT
    / "data"
    / "raw"
    / "prospectus"
    / "03696_英矽智能"
    / "招股章程_2025-12-18_繁体中文.pdf"
)

# 如果 PDF 不存在，跳过所有解析测试
pytestmark = pytest.mark.skipif(
    not _PDF_PATH.exists(),
    reason=f"测试 PDF 不存在: {_PDF_PATH}",
)


# ---------------------------------------------------------------------------
# 基础测试：总页数
# ---------------------------------------------------------------------------

class TestPageCount:
    def test_total_pages_matches_company_info(self) -> None:
        """PDF 总页数应该和 公司信息.json 中记录的 672 页一致。"""
        count = get_page_count(_PDF_PATH)
        assert count == 672, f"预期 672 页，实际 {count} 页"


# ---------------------------------------------------------------------------
# 核心测试：逐页解析
# ---------------------------------------------------------------------------

class TestParseDocument:
    """解析前 5 页，验证输出结构和数据完整性。"""

    @pytest.fixture(autouse=True)
    def _parse_first_pages(self) -> None:
        """预解析前 5 页，供所有测试方法使用。"""
        self.pages: list[PageResult] = list(
            parse_document(_PDF_PATH, page_range=(0, 5))
        )

    def test_returns_correct_number_of_pages(self) -> None:
        """应该返回恰好 5 页的结果。"""
        assert len(self.pages) == 5

    def test_page_numbers_are_one_based(self) -> None:
        """物理页码应该从 1 开始。"""
        assert self.pages[0].page == 1
        assert self.pages[-1].page == 5

    def test_page_dimensions_are_positive(self) -> None:
        """页面宽度和高度应该是正数。"""
        for page in self.pages:
            assert page.width > 0, f"第 {page.page} 页宽度无效"
            assert page.height > 0, f"第 {page.page} 页高度无效"

    def test_text_blocks_have_valid_coordinates(self) -> None:
        """每个文本块的 bbox 应该满足 x0 < x1 且 y0 < y1。"""
        for page in self.pages:
            for block in page.text_blocks:
                assert block.x0 < block.x1, (
                    f"第 {page.page} 页块 {block.block_no}: x0 >= x1"
                )
                assert block.y0 < block.y1, (
                    f"第 {page.page} 页块 {block.block_no}: y0 >= y1"
                )

    def test_text_blocks_have_non_empty_text(self) -> None:
        """每个文本块应该有非空文本。"""
        for page in self.pages:
            for block in page.text_blocks:
                assert block.text.strip(), (
                    f"第 {page.page} 页块 {block.block_no}: 文本为空"
                )

    def test_at_least_one_page_has_text(self) -> None:
        """前 5 页中至少有一页包含文字（不是纯图片页）。"""
        total_chars = sum(p.char_count for p in self.pages)
        assert total_chars > 0, "前 5 页完全没有文字"

    def test_text_blocks_page_matches_parent(self) -> None:
        """文本块的页码应该和所属 PageResult 的页码一致。"""
        for page in self.pages:
            for block in page.text_blocks:
                assert block.page == page.page


# ---------------------------------------------------------------------------
# 哈希测试
# ---------------------------------------------------------------------------

class TestFileHash:
    def test_sha256_matches_company_info(self) -> None:
        """PDF 的 SHA-256 应该和 公司信息.json 中记录的一致。"""
        expected = "c44e3c20c02747bb8768eefedd8703ff841b242d1b8cb96adc50b18b5a51809b"
        actual = sha256_file(_PDF_PATH)
        assert actual == expected, f"哈希不匹配:\n预期: {expected}\n实际: {actual}"


# ---------------------------------------------------------------------------
# 版面分析测试
# ---------------------------------------------------------------------------

class TestLayout:
    """测试段落合并和标题检测。"""

    @pytest.fixture(autouse=True)
    def _parse_and_merge(self) -> None:
        """解析前 3 页并合并段落。"""
        pages = list(parse_document(_PDF_PATH, page_range=(0, 3)))
        self.all_paragraphs: list[Paragraph] = []
        for page in pages:
            if page.text_blocks:
                paras = merge_blocks_into_paragraphs(page.text_blocks)
                self.all_paragraphs.extend(paras)

    def test_paragraphs_are_generated(self) -> None:
        """应该生成至少一个段落。"""
        assert len(self.all_paragraphs) > 0

    def test_paragraphs_have_text(self) -> None:
        """每个段落应该有非空文本。"""
        for para in self.all_paragraphs:
            assert para.text.strip()

    def test_paragraph_bbox_is_valid(self) -> None:
        """段落的 bbox 应该满足 x0 < x1 且 y0 < y1。"""
        for para in self.all_paragraphs:
            x0, y0, x1, y1 = para.bbox
            assert x0 < x1
            assert y0 < y1

    def test_body_font_size_detection(self) -> None:
        """应该能检测到正文字号（通常在 8-14pt 之间）。"""
        size = detect_body_font_size(self.all_paragraphs)
        assert 6.0 <= size <= 16.0, f"检测到的正文字号 {size} 不在合理范围"


# ---------------------------------------------------------------------------
# 配置测试
# ---------------------------------------------------------------------------

class TestConfig:
    def test_load_config_returns_valid_object(self) -> None:
        """load_config 应该成功返回 AppConfig 实例。"""
        config = load_config()
        assert config.project_name == "hk-ipo-risk-agent"
        assert config.parser.min_text_chars_before_ocr == 80
        assert config.parser.render_dpi == 150
        assert config.embedding.provider == "dashscope"
        assert config.embedding.model_name == "qwen3.7-text-embedding"
        assert config.embedding.batch_size == 20
        assert config.llm.provider == "deepseek"
        assert config.llm.model_name == "deepseek-v4-flash"
        assert config.embedding.is_configured
        assert config.llm.is_configured

    def test_resolve_path_returns_absolute(self) -> None:
        """resolve_path 应该返回绝对路径。"""
        config = load_config()
        path = config.resolve_path("data/raw/prospectus")
        assert path.is_absolute()
