"""SQLite目录、分析结果和后续人工复核的统一仓储。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ipo_risk.domain import CompanyRecord, DatasetRecord, DocumentRecord, VectorIndexRecord

if TYPE_CHECKING:
    from ipo_risk.reporting import RiskAnalysisReport


_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class CatalogCounts:
    """数据库中各类主数据的当前数量。"""

    companies: int
    documents: int
    datasets: int
    vector_indexes: int
    analysis_runs: int
    evidence: int
    risk_findings: int


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _json_loads(value: str) -> Any:
    return json.loads(value)


class CatalogRepository:
    """使用标准库 sqlite3 管理所有公司和资料版本。"""

    def __init__(self, database_path: Path, schema_path: Path | None = None) -> None:
        self.database_path = database_path.resolve()
        self.schema_path = (schema_path or Path(__file__).with_name("schema.sql")).resolve()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        """幂等创建数据库表并登记当前结构版本。"""
        schema = self.schema_path.read_text(encoding="utf-8")
        applied_at = datetime.now(timezone.utc).isoformat()
        with self._connection() as connection:
            connection.executescript(schema)
            connection.execute(
                """
                INSERT INTO schema_metadata(schema_name, schema_version, applied_at)
                VALUES ('catalog', ?, ?)
                ON CONFLICT(schema_name) DO UPDATE SET
                    schema_version = excluded.schema_version,
                    applied_at = excluded.applied_at
                """,
                (_SCHEMA_VERSION, applied_at),
            )

    def upsert_company(self, company: CompanyRecord) -> None:
        """新增公司，或刷新同一公司入口文件中的非版本字段。"""
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO companies(
                    company_id, company_folder, ticker, exchange, market,
                    company_name_zh, company_name_en, listing_date,
                    metadata_file, metadata_hash, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(company_id) DO UPDATE SET
                    company_folder = excluded.company_folder,
                    ticker = excluded.ticker,
                    exchange = excluded.exchange,
                    market = excluded.market,
                    company_name_zh = excluded.company_name_zh,
                    company_name_en = excluded.company_name_en,
                    listing_date = excluded.listing_date,
                    metadata_file = excluded.metadata_file,
                    metadata_hash = excluded.metadata_hash,
                    updated_at = excluded.updated_at
                """,
                (
                    company.company_id,
                    company.company_folder,
                    company.ticker,
                    company.exchange,
                    company.market,
                    company.company_name_zh,
                    company.company_name_en,
                    company.listing_date.isoformat() if company.listing_date else None,
                    company.metadata_file,
                    company.metadata_hash,
                    company.created_at.isoformat(),
                    company.updated_at.isoformat(),
                ),
            )

    def get_company(self, company_id: str) -> CompanyRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM companies WHERE company_id = ?",
                (company_id,),
            ).fetchone()
        return self._company_from_row(row) if row else None

    def get_company_by_folder(self, company_folder: str) -> CompanyRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM companies WHERE company_folder = ?",
                (company_folder,),
            ).fetchone()
        return self._company_from_row(row) if row else None

    def list_companies(self) -> tuple[CompanyRecord, ...]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM companies ORDER BY company_id"
            ).fetchall()
        return tuple(self._company_from_row(row) for row in rows)

    def upsert_document(self, document: DocumentRecord) -> None:
        """登记不可变文档版本；同一文件路径若被覆盖会触发唯一约束。"""
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO documents(
                    document_id, company_id, document_type, title, local_file,
                    source_hash, document_date, available_at, language, source_kind,
                    source_url, media_type, file_size_bytes, page_count,
                    text_layer_status, metadata_status, metadata_json, registered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    title = excluded.title,
                    language = excluded.language,
                    source_kind = excluded.source_kind,
                    source_url = excluded.source_url,
                    page_count = excluded.page_count,
                    text_layer_status = excluded.text_layer_status,
                    metadata_status = excluded.metadata_status,
                    metadata_json = excluded.metadata_json
                """,
                (
                    document.document_id,
                    document.company_id,
                    document.document_type,
                    document.title,
                    document.local_file,
                    document.source_hash,
                    document.document_date.isoformat(),
                    document.available_at.isoformat(),
                    document.language,
                    document.source_kind,
                    document.source_url,
                    document.media_type,
                    document.file_size_bytes,
                    document.page_count,
                    document.text_layer_status,
                    document.metadata_status,
                    _json_dumps(document.metadata),
                    document.registered_at.isoformat(),
                ),
            )

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return self._document_from_row(row) if row else None

    def list_documents(
        self,
        company_id: str,
        *,
        available_on_or_before: date | None = None,
        document_types: Sequence[str] | None = None,
    ) -> tuple[DocumentRecord, ...]:
        """按公司、资料类型和可用日期执行 Point-in-Time 选择。"""
        clauses = ["company_id = ?"]
        parameters: list[Any] = [company_id]
        if available_on_or_before is not None:
            clauses.append("available_at <= ?")
            parameters.append(available_on_or_before.isoformat())
        if document_types:
            placeholders = ", ".join("?" for _ in document_types)
            clauses.append(f"document_type IN ({placeholders})")
            parameters.extend(document_types)
        query = (
            "SELECT * FROM documents WHERE "
            + " AND ".join(clauses)
            + " ORDER BY available_at, document_type, document_id"
        )
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(self._document_from_row(row) for row in rows)

    def upsert_dataset(self, dataset: DatasetRecord) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO datasets(
                    dataset_id, company_id, dataset_type, title, local_file,
                    source_hash, available_at, file_format, period_start, period_end,
                    row_count, usage_scope, pre_listing_eligible, metadata_json,
                    registered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_id) DO UPDATE SET
                    title = excluded.title,
                    row_count = excluded.row_count,
                    usage_scope = excluded.usage_scope,
                    pre_listing_eligible = excluded.pre_listing_eligible,
                    metadata_json = excluded.metadata_json
                """,
                (
                    dataset.dataset_id,
                    dataset.company_id,
                    dataset.dataset_type,
                    dataset.title,
                    dataset.local_file,
                    dataset.source_hash,
                    dataset.available_at.isoformat(),
                    dataset.file_format,
                    dataset.period_start.isoformat() if dataset.period_start else None,
                    dataset.period_end.isoformat() if dataset.period_end else None,
                    dataset.row_count,
                    dataset.usage_scope,
                    int(dataset.pre_listing_eligible),
                    _json_dumps(dataset.metadata),
                    dataset.registered_at.isoformat(),
                ),
            )

    def get_dataset(self, dataset_id: str) -> DatasetRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM datasets WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()
        return self._dataset_from_row(row) if row else None

    def list_datasets(
        self,
        company_id: str,
        *,
        available_on_or_before: date | None = None,
        pre_listing_only: bool = False,
    ) -> tuple[DatasetRecord, ...]:
        clauses = ["company_id = ?"]
        parameters: list[Any] = [company_id]
        if available_on_or_before is not None:
            clauses.append("available_at <= ?")
            parameters.append(available_on_or_before.isoformat())
        if pre_listing_only:
            clauses.append("pre_listing_eligible = 1")
        query = (
            "SELECT * FROM datasets WHERE "
            + " AND ".join(clauses)
            + " ORDER BY available_at, dataset_type, dataset_id"
        )
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(self._dataset_from_row(row) for row in rows)

    def upsert_vector_index(self, index: VectorIndexRecord) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO vector_indexes(
                    index_id, document_id, source_hash, embedding_provider,
                    embedding_model, dimensions, chunk_count, vectors_path,
                    metadata_path, manifest_path, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(index_id) DO UPDATE SET
                    chunk_count = excluded.chunk_count,
                    vectors_path = excluded.vectors_path,
                    metadata_path = excluded.metadata_path,
                    manifest_path = excluded.manifest_path,
                    status = excluded.status,
                    created_at = excluded.created_at
                """,
                (
                    index.index_id,
                    index.document_id,
                    index.source_hash,
                    index.embedding_provider,
                    index.embedding_model,
                    index.dimensions,
                    index.chunk_count,
                    index.vectors_path,
                    index.metadata_path,
                    index.manifest_path,
                    index.status,
                    index.created_at.isoformat(),
                ),
            )

    def get_vector_index(self, index_id: str) -> VectorIndexRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM vector_indexes WHERE index_id = ?",
                (index_id,),
            ).fetchone()
        return self._vector_index_from_row(row) if row else None

    def list_vector_indexes(self, document_id: str) -> tuple[VectorIndexRecord, ...]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM vector_indexes
                WHERE document_id = ?
                ORDER BY embedding_model, dimensions
                """,
                (document_id,),
            ).fetchall()
        return tuple(self._vector_index_from_row(row) for row in rows)

    def save_analysis_report(self, report: RiskAnalysisReport, report_path: str) -> None:
        """把导出JSON中的完整证据链同步写入SQLite。"""
        started_at = min(
            (trace.started_at for trace in report.agent_traces),
            default=report.generated_at,
        )
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO analysis_runs(
                    run_id, company_id, as_of_date, status, started_at, ended_at,
                    report_path, error, created_at
                ) VALUES (?, ?, ?, 'succeeded', ?, ?, ?, NULL, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    status = excluded.status,
                    ended_at = excluded.ended_at,
                    report_path = excluded.report_path,
                    error = NULL
                """,
                (
                    report.run_id,
                    report.company_id,
                    report.source_document.available_at.isoformat(),
                    started_at.isoformat(),
                    report.generated_at.isoformat(),
                    report_path,
                    report.generated_at.isoformat(),
                ),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO analysis_run_documents(run_id, document_id)
                VALUES (?, ?)
                """,
                (report.run_id, report.source_document.document_id),
            )
            for evidence in report.evidences:
                connection.execute(
                    """
                    INSERT INTO evidence(
                        run_id, evidence_id, company_id, document_id, page,
                        source_hash, screenshot_path, payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, evidence_id) DO UPDATE SET
                        screenshot_path = excluded.screenshot_path,
                        payload_json = excluded.payload_json
                    """,
                    (
                        report.run_id,
                        evidence.evidence_id,
                        evidence.company_id,
                        evidence.document_id,
                        evidence.page,
                        evidence.source_hash,
                        evidence.screenshot_path,
                        _json_dumps(evidence.model_dump(mode="json")),
                        evidence.created_at.isoformat(),
                    ),
                )
            for finding in report.risk_findings:
                connection.execute(
                    """
                    INSERT INTO risk_findings(
                        run_id, finding_id, company_id, risk_type, status,
                        confidence, payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, finding_id) DO UPDATE SET
                        status = excluded.status,
                        confidence = excluded.confidence,
                        payload_json = excluded.payload_json
                    """,
                    (
                        report.run_id,
                        finding.finding_id,
                        finding.company_id,
                        finding.risk_type,
                        finding.status,
                        finding.confidence,
                        _json_dumps(finding.model_dump(mode="json")),
                        finding.created_at.isoformat(),
                    ),
                )
                for evidence_id in finding.evidence_ids:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO finding_evidence(
                            run_id, finding_id, evidence_id
                        ) VALUES (?, ?, ?)
                        """,
                        (report.run_id, finding.finding_id, evidence_id),
                    )
            for trace in report.agent_traces:
                connection.execute(
                    """
                    INSERT INTO agent_traces(
                        run_id, trace_id, agent, status, payload_json,
                        started_at, ended_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, trace_id) DO UPDATE SET
                        status = excluded.status,
                        payload_json = excluded.payload_json,
                        ended_at = excluded.ended_at
                    """,
                    (
                        report.run_id,
                        trace.trace_id,
                        trace.agent,
                        trace.status,
                        _json_dumps(trace.model_dump(mode="json")),
                        trace.started_at.isoformat(),
                        trace.ended_at.isoformat(),
                    ),
                )

    def counts(self) -> CatalogCounts:
        tables = (
            "companies",
            "documents",
            "datasets",
            "vector_indexes",
            "analysis_runs",
            "evidence",
            "risk_findings",
        )
        with self._connection() as connection:
            values = {
                table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in tables
            }
        return CatalogCounts(**values)

    @staticmethod
    def _company_from_row(row: sqlite3.Row) -> CompanyRecord:
        return CompanyRecord.model_validate(dict(row))

    @staticmethod
    def _document_from_row(row: sqlite3.Row) -> DocumentRecord:
        data = dict(row)
        data["metadata"] = _json_loads(data.pop("metadata_json"))
        return DocumentRecord.model_validate(data)

    @staticmethod
    def _dataset_from_row(row: sqlite3.Row) -> DatasetRecord:
        data = dict(row)
        data["pre_listing_eligible"] = bool(data["pre_listing_eligible"])
        data["metadata"] = _json_loads(data.pop("metadata_json"))
        return DatasetRecord.model_validate(data)

    @staticmethod
    def _vector_index_from_row(row: sqlite3.Row) -> VectorIndexRecord:
        return VectorIndexRecord.model_validate(dict(row))


__all__ = ["CatalogCounts", "CatalogRepository"]
