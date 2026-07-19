"""公司与文档版本级中间产物路径的唯一生成规则。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_INVALID_WINDOWS_CHARS = re.compile(r"[<>:\"/\\|?*\x00-\x1f]")


def safe_path_segment(value: str, field_name: str) -> str:
    """拒绝目录穿越和Windows非法文件名字符。"""
    candidate = value.strip()
    if (
        not candidate
        or candidate in {".", ".."}
        or _INVALID_WINDOWS_CHARS.search(candidate)
        or Path(candidate).name != candidate
    ):
        raise ValueError(f"{field_name} must be one safe path segment")
    return candidate


@dataclass(frozen=True)
class DocumentPaths:
    """一份具体文档版本的全部可重建产物位置。"""

    pages_path: Path
    chunks_path: Path
    embeddings_root: Path
    evidence_dir: Path
    report_path: Path

    @classmethod
    def build(
        cls,
        *,
        interim_root: Path,
        artifacts_root: Path,
        company_folder: str,
        document_id: str,
    ) -> DocumentPaths:
        company = safe_path_segment(company_folder, "company_folder")
        document = safe_path_segment(document_id, "document_id")
        return cls(
            pages_path=interim_root / "documents" / company / document / "pages.jsonl",
            chunks_path=interim_root / "chunks" / company / document / "chunks.jsonl",
            embeddings_root=interim_root / "embeddings" / company / document,
            evidence_dir=artifacts_root / "evidence" / company / document,
            report_path=(
                artifacts_root / "analysis" / company / document / "risk_analysis.json"
            ),
        )

    def embedding_dir(self, model_name: str) -> Path:
        """返回某个Embedding模型的独立索引目录。"""
        model = safe_path_segment(model_name, "model_name")
        return self.embeddings_root / model


__all__ = ["DocumentPaths", "safe_path_segment"]
