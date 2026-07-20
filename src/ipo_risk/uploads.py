"""网页上传招股书的安全校验、不可变落盘与目录登记。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import fitz

from ipo_risk.config import load_config
from ipo_risk.storage.files import read_json
from ipo_risk.storage.ingestion import register_company_sources
from ipo_risk.storage.paths import safe_path_segment
from ipo_risk.storage.repositories import CatalogRepository


_COMPANY_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
_MAX_PDF_BYTES = 250 * 1024 * 1024
_STOCK_CODE_PATTERN = re.compile(
    r"(?:股份代[號号]|股票代码|Stock\s+Code)\s*[:：]?\s*(\d{4,5})",
    re.IGNORECASE,
)
_CHINESE_DATE_PATTERN = re.compile(
    r"(?P<year>20\d{2})\s*年\s*(?P<month>\d{1,2})\s*月\s*"
    r"(?P<day>\d{1,2})\s*日"
)
_FILE_DATE_PATTERN = re.compile(
    r"(?P<year>20\d{2})[-_](?P<month>\d{1,2})[-_](?P<day>\d{1,2})"
)
_COMPANY_NAME_PATTERN = re.compile(
    r"^\s*(?P<name>[\u3400-\u9fff]{2,20})(?=\s+[A-Za-z])"
)
_GENERIC_COVER_HEADINGS = {
    "招股章程",
    "招股说明书",
    "全球發售",
    "全球发售",
    "重要提示",
    "目錄",
    "目录",
}


@dataclass(frozen=True)
class ProspectusMetadataSuggestion:
    """从已登记资料或PDF封面推断出的可编辑上传字段。"""

    company_id: str | None
    company_name: str | None
    document_date: date | None
    source: str
    page_count: int
    source_hash: str


@dataclass(frozen=True)
class UploadedProspectus:
    """一份已安全登记、可以直接提交分析的招股书版本。"""

    company_folder: str
    document_id: str
    source_path: Path
    source_hash: str
    page_count: int
    reused: bool


class ProspectusUploadService:
    """把一次网页上传转换为现有分析流水线可以读取的标准公司目录。"""

    def __init__(
        self,
        *,
        project_root: Path,
        repository: CatalogRepository | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.config = load_config(
            config_path=self.project_root / "configs" / "base.yaml",
            providers_path=self.project_root / "configs" / "providers.local.yaml",
        )
        self.repository = repository or CatalogRepository(
            self.project_root / self.config.paths.database
        )
        self.repository.initialize()

    @staticmethod
    def _validate_company(company_id: str, company_name: str) -> tuple[str, str]:
        normalized_id = company_id.strip().upper()
        normalized_name = company_name.strip()
        if not _COMPANY_ID_PATTERN.fullmatch(normalized_id):
            raise ValueError("股票代码只能包含字母、数字、下划线或连字符。")
        if not normalized_name:
            raise ValueError("公司名称不能为空。")
        safe_path_segment(f"{normalized_id}_{normalized_name}", "company_folder")
        return normalized_id, normalized_name

    @staticmethod
    def _inspect_pdf(content: bytes) -> int:
        if not content:
            raise ValueError("上传文件为空。")
        if len(content) > _MAX_PDF_BYTES:
            raise ValueError("招股书PDF不能超过250MB。")
        if not content.startswith(b"%PDF-"):
            raise ValueError("上传内容不是有效的PDF文件。")
        try:
            document = fitz.open(stream=content, filetype="pdf")
        except Exception as exc:
            raise ValueError("PDF无法打开或文件已经损坏。") from exc
        try:
            if document.needs_pass:
                raise ValueError("暂不支持带密码的PDF。")
            if document.page_count < 1:
                raise ValueError("PDF没有可分析页面。")
            return document.page_count
        finally:
            document.close()

    def _known_document_suggestion(
        self,
        *,
        source_hash: str,
        page_count: int,
    ) -> ProspectusMetadataSuggestion | None:
        company_root = self.project_root / self.config.paths.company_info_dir
        if not company_root.is_dir():
            return None
        for metadata_path in company_root.glob("*/公司信息.json"):
            try:
                metadata = read_json(metadata_path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(metadata, dict):
                continue
            documents = metadata.get("documents", {})
            if not isinstance(documents, dict):
                continue
            for declared in documents.values():
                if not isinstance(declared, dict):
                    continue
                if str(declared.get("sha256", "")).lower() != source_hash:
                    continue
                return ProspectusMetadataSuggestion(
                    company_id=(
                        str(metadata.get("company_id", "")).strip().upper() or None
                    ),
                    company_name=(
                        str(metadata.get("company_name_zh_official", "")).strip()
                        or None
                    ),
                    document_date=self._parse_iso_date(declared.get("document_date")),
                    source="registered_document",
                    page_count=page_count,
                    source_hash=source_hash,
                )
        return None

    @staticmethod
    def _parse_iso_date(value: object) -> date | None:
        try:
            return date.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _date_from_match(match: re.Match[str] | None) -> date | None:
        if match is None:
            return None
        try:
            return date(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
            )
        except ValueError:
            return None

    @staticmethod
    def _cover_text(content: bytes, *, page_limit: int = 3) -> str:
        document = fitz.open(stream=content, filetype="pdf")
        try:
            return "\n".join(
                document.load_page(page_index).get_text("text")
                for page_index in range(min(page_limit, document.page_count))
            )
        finally:
            document.close()

    @staticmethod
    def _company_name_from_cover(cover_text: str) -> str | None:
        for raw_line in cover_text.splitlines():
            line = raw_line.strip()
            if not line or line in _GENERIC_COVER_HEADINGS:
                continue
            match = _COMPANY_NAME_PATTERN.match(line)
            if match is not None:
                return match.group("name")
        return None

    def inspect(
        self,
        *,
        file_name: str,
        content: bytes,
    ) -> ProspectusMetadataSuggestion:
        """校验PDF，并为上传表单识别代码、公司名和招股书日期。"""
        page_count = self._inspect_pdf(content)
        source_hash = sha256(content).hexdigest()
        known = self._known_document_suggestion(
            source_hash=source_hash,
            page_count=page_count,
        )
        if known is not None:
            return known

        cover_text = self._cover_text(content)
        code_match = _STOCK_CODE_PATTERN.search(cover_text)
        company_id = code_match.group(1).zfill(5) if code_match else None
        company_name = self._company_name_from_cover(cover_text)
        document_date = self._date_from_match(
            _CHINESE_DATE_PATTERN.search(cover_text)
        ) or self._date_from_match(_FILE_DATE_PATTERN.search(file_name))
        source = (
            "cover_text"
            if company_id and company_name and document_date
            else "partial"
        )
        return ProspectusMetadataSuggestion(
            company_id=company_id,
            company_name=company_name,
            document_date=document_date,
            source=source,
            page_count=page_count,
            source_hash=source_hash,
        )

    def _existing_company_folder(self, company_id: str) -> str | None:
        company_root = self.project_root / self.config.paths.company_info_dir
        if not company_root.is_dir():
            return None
        matches: list[str] = []
        for candidate in company_root.glob(f"{company_id}_*"):
            metadata_path = candidate / "公司信息.json"
            if not metadata_path.is_file():
                continue
            metadata = read_json(metadata_path)
            if str(metadata.get("company_id", "")).strip().upper() == company_id:
                matches.append(candidate.name)
        if len(matches) > 1:
            raise ValueError(f"股票代码{company_id}对应多个公司目录，请先整理主数据。")
        return matches[0] if matches else None

    def _metadata_path(self, company_folder: str) -> Path:
        return (
            self.project_root
            / self.config.paths.company_info_dir
            / company_folder
            / "公司信息.json"
        )

    @staticmethod
    def _atomic_write_bytes(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.uploading")
        temporary.write_bytes(content)
        temporary.replace(path)

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.uploading")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)

    def register(
        self,
        *,
        file_name: str,
        content: bytes,
        company_id: str,
        company_name: str,
        document_date: date,
    ) -> UploadedProspectus:
        """校验并幂等登记上传PDF；同一哈希不会复制原始文件。"""
        normalized_id, normalized_name = self._validate_company(
            company_id,
            company_name,
        )
        page_count = self._inspect_pdf(content)
        source_hash = sha256(content).hexdigest()
        company_folder = self._existing_company_folder(normalized_id) or (
            f"{normalized_id}_{normalized_name}"
        )
        metadata_path = self._metadata_path(company_folder)
        if metadata_path.is_file():
            metadata = read_json(metadata_path)
            if not isinstance(metadata, dict):
                raise ValueError("公司信息.json顶层必须是对象。")
        else:
            metadata = {
                "schema_version": 2,
                "company_id": normalized_id,
                "exchange": "HKEX",
                "ticker": f"{normalized_id}.HK",
                "company_name_zh_official": normalized_name,
                "as_of_date": document_date.isoformat(),
                "documents": {},
                "data_layout": {
                    "company_folder": company_folder,
                    "prospectus_dir": (
                        f"{self.config.paths.prospectus_dir}/{company_folder}"
                    ),
                    "company_info_dir": (
                        f"{self.config.paths.company_info_dir}/{company_folder}"
                    ),
                    "market_dir": f"{self.config.paths.market_dir}/{company_folder}",
                },
            }

        documents = metadata.setdefault("documents", {})
        if not isinstance(documents, dict):
            raise ValueError("公司信息.json中的documents必须是对象。")
        reused = False
        document_key: str | None = None
        source_path: Path | None = None
        for key, declared in documents.items():
            if not isinstance(declared, dict):
                continue
            if str(declared.get("sha256", "")).lower() != source_hash:
                continue
            existing = self.project_root / str(declared.get("local_file", ""))
            if existing.is_file():
                document_key = str(key)
                source_path = existing.resolve()
                reused = True
                break

        if source_path is None:
            source_path = (
                self.project_root
                / self.config.paths.prospectus_dir
                / company_folder
                / f"招股书_{document_date.isoformat()}_{source_hash[:12]}.pdf"
            )
            if source_path.exists():
                existing_hash = sha256(source_path.read_bytes()).hexdigest()
                if existing_hash != source_hash:
                    raise FileExistsError(f"上传目标已存在但内容不同：{source_path}")
                reused = True
            else:
                self._atomic_write_bytes(source_path, content)
            document_key = (
                "prospectus"
                if "prospectus" not in documents
                else f"prospectus_upload_{source_hash[:8]}"
            )
            documents[document_key] = {
                "document_type": "prospectus",
                "title": Path(file_name).stem or "上传招股书",
                "local_file": source_path.relative_to(self.project_root).as_posix(),
                "document_date": document_date.isoformat(),
                "available_at_date": document_date.isoformat(),
                "language": "zh-Hant",
                "source_kind": "user_upload",
                "source_file_name": Path(file_name).name,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "file_size_bytes": len(content),
                "sha256": source_hash,
                "page_count": page_count,
                "text_layer_status": "unknown_until_parse",
            }
            self._atomic_write_json(metadata_path, metadata)

        ingestion = register_company_sources(
            company_folder,
            project_root=self.project_root,
            repository=self.repository,
        )
        registered = next(
            (
                document
                for document in ingestion.documents
                if document.document_type == "prospectus"
                and document.source_hash == source_hash
            ),
            None,
        )
        if registered is None:
            raise RuntimeError("上传文件已经保存，但未能登记为招股书版本。")
        return UploadedProspectus(
            company_folder=company_folder,
            document_id=registered.document_id,
            source_path=source_path,
            source_hash=source_hash,
            page_count=page_count,
            reused=reused,
        )


__all__ = [
    "ProspectusMetadataSuggestion",
    "ProspectusUploadService",
    "UploadedProspectus",
]
