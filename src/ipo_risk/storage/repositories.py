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

from ipo_risk.domain import (
    CompanyRecord,
    DatasetRecord,
    DocumentRecord,
    Evidence,
    FindingReview,
    ReviewDecision,
    RiskFinding,
    VectorIndexRecord,
    VerificationStatus,
)

if TYPE_CHECKING:
    from ipo_risk.reporting import RiskAnalysisReport


_SCHEMA_VERSION = "1.1"


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


@dataclass(frozen=True)
class AnalysisRunRecord:
    """一条可供复核界面选择的分析运行记录。"""

    run_id: str
    company_id: str
    as_of_date: date
    status: str
    started_at: datetime
    ended_at: datetime | None
    report_path: str | None
    verified_report_path: str | None
    error: str | None
    created_at: datetime


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
            review_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(reviews)").fetchall()
            }
            if "payload_json" not in review_columns:
                connection.execute(
                    "ALTER TABLE reviews "
                    "ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}'"
                )
            run_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(analysis_runs)").fetchall()
            }
            if "verified_report_path" not in run_columns:
                connection.execute(
                    "ALTER TABLE analysis_runs ADD COLUMN verified_report_path TEXT"
                )
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

    def relocate_dataset_file(self, dataset_id: str, local_file: str) -> None:
        """把历史数据集登记指向已核验的不可变备份文件。

        仅用于原先的“当前文件”被新官方快照替换、且旧内容已经按原哈希备份的场景。
        调用方必须先核对备份文件哈希；仓储层只负责原子更新路径并保留原数据集ID。
        """
        normalized = local_file.strip()
        if not normalized:
            raise ValueError("local_file must not be empty")
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE datasets SET local_file = ? WHERE dataset_id = ?",
                (normalized, dataset_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown dataset_id: {dataset_id}")

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
                        status = CASE
                            WHEN risk_findings.status IN ('verified', 'rejected')
                            THEN risk_findings.status ELSE excluded.status END,
                        confidence = CASE
                            WHEN risk_findings.status IN ('verified', 'rejected')
                            THEN risk_findings.confidence ELSE excluded.confidence END,
                        payload_json = CASE
                            WHEN risk_findings.status IN ('verified', 'rejected')
                            THEN risk_findings.payload_json ELSE excluded.payload_json END
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
                stored_status = connection.execute(
                    """
                    SELECT status FROM risk_findings
                    WHERE run_id = ? AND finding_id = ?
                    """,
                    (report.run_id, finding.finding_id),
                ).fetchone()["status"]
                if (
                    stored_status in {"verified", "rejected"}
                    and finding.status
                    in {
                        VerificationStatus.PROPOSED,
                        VerificationStatus.PENDING_VERIFICATION,
                    }
                ):
                    continue
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

    def get_analysis_run(self, run_id: str) -> AnalysisRunRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM analysis_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return self._analysis_run_from_row(row) if row else None

    def delete_analysis_run(self, run_id: str) -> bool:
        """删除单次分析及其派生记录，保留公司、文档与行情数据。"""
        with self._connection() as connection:
            exists = connection.execute(
                "SELECT 1 FROM analysis_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if exists is None:
                return False
            # reviews uses ON DELETE RESTRICT so audit rows must be removed explicitly.
            connection.execute("DELETE FROM reviews WHERE run_id = ?", (run_id,))
            cursor = connection.execute(
                "DELETE FROM analysis_runs WHERE run_id = ?",
                (run_id,),
            )
        return cursor.rowcount == 1

    def list_analysis_runs(
        self,
        *,
        status: str | None = None,
    ) -> tuple[AnalysisRunRecord, ...]:
        query = "SELECT * FROM analysis_runs"
        parameters: list[Any] = []
        if status is not None:
            query += " WHERE status = ?"
            parameters.append(status)
        query += " ORDER BY created_at DESC, run_id DESC"
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(self._analysis_run_from_row(row) for row in rows)

    def get_finding(self, run_id: str, finding_id: str) -> RiskFinding | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM risk_findings
                WHERE run_id = ? AND finding_id = ?
                """,
                (run_id, finding_id),
            ).fetchone()
        return RiskFinding.model_validate(_json_loads(row["payload_json"])) if row else None

    def get_analysis_report_path(self, run_id: str) -> str | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT report_path FROM analysis_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return str(row["report_path"]) if row and row["report_path"] else None

    def set_verified_report_path(self, run_id: str, report_path: str) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE analysis_runs SET verified_report_path = ? WHERE run_id = ?
                """,
                (report_path, run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"analysis run does not exist: {run_id}")

    def get_verified_report_path(self, run_id: str) -> str | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT verified_report_path FROM analysis_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return (
            str(row["verified_report_path"])
            if row and row["verified_report_path"]
            else None
        )

    def list_findings(
        self,
        run_id: str,
        *,
        status: VerificationStatus | None = None,
    ) -> tuple[RiskFinding, ...]:
        query = "SELECT payload_json FROM risk_findings WHERE run_id = ?"
        parameters: list[Any] = [run_id]
        if status is not None:
            query += " AND status = ?"
            parameters.append(status.value)
        query += " ORDER BY finding_id"
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(
            RiskFinding.model_validate(_json_loads(row["payload_json"])) for row in rows
        )

    def list_evidence(
        self,
        run_id: str,
        *,
        evidence_ids: Sequence[str] | None = None,
    ) -> tuple[Evidence, ...]:
        query = "SELECT payload_json FROM evidence WHERE run_id = ?"
        parameters: list[Any] = [run_id]
        if evidence_ids is not None:
            if not evidence_ids:
                return ()
            placeholders = ", ".join("?" for _ in evidence_ids)
            query += f" AND evidence_id IN ({placeholders})"
            parameters.extend(evidence_ids)
        query += " ORDER BY evidence_id"
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(Evidence.model_validate(_json_loads(row["payload_json"])) for row in rows)

    def save_review(self, review: FindingReview) -> None:
        """Atomically persist an immutable review and its resulting finding state."""
        finding = review.resulting_finding
        with self._connection() as connection:
            existing = connection.execute(
                """
                SELECT 1 FROM risk_findings
                WHERE run_id = ? AND finding_id = ?
                """,
                (review.run_id, review.finding_id),
            ).fetchone()
            if existing is None:
                raise KeyError(
                    f"finding does not exist: {review.run_id}/{review.finding_id}"
                )

            known_evidence = {
                row["evidence_id"]
                for row in connection.execute(
                    "SELECT evidence_id FROM evidence WHERE run_id = ?",
                    (review.run_id,),
                ).fetchall()
            }
            missing = set(finding.evidence_ids) - known_evidence
            if missing:
                raise ValueError(f"review references unknown evidence: {sorted(missing)}")

            connection.execute(
                """
                UPDATE risk_findings
                SET status = ?, confidence = ?, payload_json = ?
                WHERE run_id = ? AND finding_id = ?
                """,
                (
                    finding.status.value,
                    finding.confidence,
                    _json_dumps(finding.model_dump(mode="json")),
                    review.run_id,
                    review.finding_id,
                ),
            )
            connection.execute(
                "DELETE FROM finding_evidence WHERE run_id = ? AND finding_id = ?",
                (review.run_id, review.finding_id),
            )
            connection.executemany(
                """
                INSERT INTO finding_evidence(run_id, finding_id, evidence_id)
                VALUES (?, ?, ?)
                """,
                [
                    (review.run_id, review.finding_id, evidence_id)
                    for evidence_id in finding.evidence_ids
                ],
            )
            edited_payload = (
                _json_dumps(finding.model_dump(mode="json"))
                if review.decision is ReviewDecision.EDITED
                else None
            )
            connection.execute(
                """
                INSERT INTO reviews(
                    review_id, run_id, finding_id, decision, reviewer, comment,
                    edited_payload_json, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review.review_id,
                    review.run_id,
                    review.finding_id,
                    review.decision.value,
                    review.reviewer,
                    review.comment,
                    edited_payload,
                    _json_dumps(review.model_dump(mode="json")),
                    review.created_at.isoformat(),
                ),
            )

    def get_review(self, review_id: str) -> FindingReview | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload_json FROM reviews WHERE review_id = ?",
                (review_id,),
            ).fetchone()
        return FindingReview.model_validate(_json_loads(row["payload_json"])) if row else None

    def list_reviews(
        self,
        run_id: str,
        *,
        finding_id: str | None = None,
    ) -> tuple[FindingReview, ...]:
        query = "SELECT payload_json FROM reviews WHERE run_id = ?"
        parameters: list[Any] = [run_id]
        if finding_id is not None:
            query += " AND finding_id = ?"
            parameters.append(finding_id)
        query += " ORDER BY created_at, review_id"
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return tuple(
            FindingReview.model_validate(_json_loads(row["payload_json"])) for row in rows
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

    @staticmethod
    def _analysis_run_from_row(row: sqlite3.Row) -> AnalysisRunRecord:
        return AnalysisRunRecord(
            run_id=str(row["run_id"]),
            company_id=str(row["company_id"]),
            as_of_date=date.fromisoformat(str(row["as_of_date"])),
            status=str(row["status"]),
            started_at=datetime.fromisoformat(str(row["started_at"])),
            ended_at=(
                datetime.fromisoformat(str(row["ended_at"]))
                if row["ended_at"]
                else None
            ),
            report_path=str(row["report_path"]) if row["report_path"] else None,
            verified_report_path=(
                str(row["verified_report_path"])
                if row["verified_report_path"]
                else None
            ),
            error=str(row["error"]) if row["error"] else None,
            created_at=datetime.fromisoformat(str(row["created_at"])),
        )


__all__ = ["AnalysisRunRecord", "CatalogCounts", "CatalogRepository"]
