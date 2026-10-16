"""使用既有 pgvector 与招股书切片补充可审计财务和业务事实。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
import math
from pathlib import Path
import re
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ipo_risk.agents.analysis import JsonChatProvider
from ipo_risk.domain import FeatureValue
from ipo_risk.retrieval.chunker import DocumentChunk
from ipo_risk.retrieval.lexical import BM25Retriever, SearchHit
from ipo_risk.retrieval.pgvector import PgVectorRetriever
from ipo_risk.resilience import RetryExhaustedError, RetryPolicy, retry_model_node_error
from ipo_risk.storage.files import read_jsonl, write_json, write_jsonl


RAG_FUNDAMENTAL_FEATURE_NAMES = (
    "fundamental_revenue_latest_native",
    "fundamental_operating_cash_flow_latest_native",
    "fundamental_ending_cash_latest_native",
    "fundamental_net_profit_latest_native",
    "fundamental_research_development_latest_native",
    "customer_largest_ratio_pct",
    "customer_top5_ratio_pct",
    "supplier_largest_ratio_pct",
    "supplier_top5_ratio_pct",
    "cornerstone_ratio_pct",
    "cornerstone_lockup_months",
    "controlling_shareholder_lockup_months",
    "industry_primary",
    "pipeline_stage_name",
    "pipeline_stage_score",
    "cash_runway_months",
)

_MODEL_FIELD_NAMES = (
    "fundamental_revenue_latest_native",
    "fundamental_operating_cash_flow_latest_native",
    "fundamental_ending_cash_latest_native",
    "fundamental_net_profit_latest_native",
    "fundamental_research_development_latest_native",
    "customer_largest_ratio_pct",
    "customer_top5_ratio_pct",
    "supplier_largest_ratio_pct",
    "supplier_top5_ratio_pct",
    "cornerstone_share_count",
    "global_offering_share_count",
    "cornerstone_ratio_disclosed_pct",
    "cornerstone_lockup_months",
    "controlling_shareholder_lockup_months",
    "industry_primary",
    "pipeline_stage_name",
)

_FINANCIAL_FIELDS = {
    "fundamental_revenue_latest_native",
    "fundamental_operating_cash_flow_latest_native",
    "fundamental_ending_cash_latest_native",
    "fundamental_net_profit_latest_native",
    "fundamental_research_development_latest_native",
}
_PERCENT_FIELDS = {
    "customer_largest_ratio_pct",
    "customer_top5_ratio_pct",
    "supplier_largest_ratio_pct",
    "supplier_top5_ratio_pct",
    "cornerstone_ratio_disclosed_pct",
}
_MONTH_FIELDS = {
    "cornerstone_lockup_months",
    "controlling_shareholder_lockup_months",
}
_SHARE_FIELDS = {"cornerstone_share_count", "global_offering_share_count"}
_FIELD_SCOPED_EVIDENCE_FIELDS = _FINANCIAL_FIELDS | _PERCENT_FIELDS | _MONTH_FIELDS
_PERCENT_VALUE = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)%")
_PRICE_VALUE = re.compile(r"(\d+(?:\.\d+)?)\s*港元")
_NON_EXACT_PERCENT_MARKERS = (
    "少於", "少于", "低於", "低于", "不足", "超過", "超过",
    "高於", "高于", "不低於", "不低于", "不超過", "不超过",
    "less than", "more than", "below", "above", "at least", "not more than",
)
_APPENDIX_MARKERS = ("附錄", "附录", "appendix")
_ISSUER_FINANCIAL_MARKERS = (
    "本集團", "本集团", "節選合併", "节选合并",
    "合併綜合收益表", "合并综合收益表",
    "歸屬於本公司", "归属于本公司",
    "consolidated statement of profit",
)
_SEPARATE_APPENDIX_SUBJECT = re.compile(
    r"(?:附錄|附录|appendix).{0,24}?[a-z0-9][a-z0-9 .,&'()\-/]{3,80}?"
    r"(?:limited|ltd\.?|inc\.?|plc|ag|corporation|company)",
    re.IGNORECASE,
)
_AMBIGUOUS_ZERO_MARKERS = (
    "重大",
    "主要",
    "material",
    "significant",
)
_EXPLICIT_R_AND_D_ZERO_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?:並無|并无|沒有|没有|未有|不存在|不涉及|並未進行|并未进行)"
        r".{0,16}(?:研發活動|研发活动|研發開支|研发开支|研發費用|研发费用)",
        r"no\s+(?:research\s+and\s+development|r\s*&\s*d)\s+"
        r"(?:activities|expenses|expenditure)",
    )
)

_QUERY_SPECS: dict[str, str] = {
    "fundamental_revenue_latest_native": (
        "最近完整财政年度 合并损益表 收入 总收入 收益 "
        "revenue total revenue"
    ),
    "fundamental_operating_cash_flow_latest_native": (
        "最近完整财政年度 合并现金流量表 经营活动现金净额 "
        "经营活动所得现金净额 经营业务所得现金净额 "
        "net cash from operating activities operating cash flow"
    ),
    "fundamental_ending_cash_latest_native": (
        "最近完整财政年度 合并财务状况表 现金及现金等价物 银行结余及现金 "
        "cash and cash equivalents 年末余额 千元"
    ),
    "fundamental_net_profit_latest_native": (
        "最近完整财政年度 发行人 本集团 合并损益表 年度利润 年内利润 净利润 净亏损 "
        "归属于本公司利润 loss for the year"
    ),
    "fundamental_research_development_latest_native": (
        "最近完整财政年度 研发费用 研发开支 研究及开发成本 research and development expenses"
    ),
    "customer_largest_ratio_pct": (
        "最近完整财政年度 最大单一客户 收入占比 largest customer percentage"
    ),
    "customer_top5_ratio_pct": (
        "最近完整财政年度 五大客户 前五大客户 收入占比 five largest customers percentage"
    ),
    "supplier_largest_ratio_pct": (
        "最近完整财政年度 最大单一供应商 采购占比 largest supplier percentage"
    ),
    "supplier_top5_ratio_pct": (
        "最近完整财政年度 五大供应商 前五大供应商 采购占比 five largest suppliers percentage"
    ),
    "cornerstone_share_count": (
        "基石投资者 cornerstone investors 合计认购股份数 number of offer shares subscribed"
    ),
    "global_offering_share_count": (
        "全球发售股份数 global offering total offer shares 发售股份"
    ),
    "cornerstone_ratio_disclosed_pct": (
        "基石投资者占全球发售股份比例 cornerstone percentage of global offering 中位发售价"
    ),
    "cornerstone_lockup_months": (
        "基石投资者 禁售期 锁定期 上市日期后六个月 cornerstone lock-up period"
    ),
    "controlling_shareholder_lockup_months": (
        "控股股东 上市后 禁售期 锁定期 不出售股份 controlling shareholder lock-up period"
    ),
    "industry_primary": "公司主要业务 行业 业务模式 核心产品 服务 业务概览",
    "pipeline_stage_name": (
        "核心产品 研发管线 临床前 临床I期 IIa期 IIb期 III期 NDA BLA 获批 商业化"
    ),
}

_FIELD_ANCHORS: dict[str, tuple[str, ...]] = {
    "fundamental_revenue_latest_native": (
        "收入", "總收入", "总收入", "收益", "Revenue", "Total revenue",
    ),
    "fundamental_operating_cash_flow_latest_native": (
        "經營活動所得", "经营活动所得", "經營業務所得", "经营业务所得",
        "net cash from operating activities", "operating cash flow",
        "經營活動現金流入", "经营活动现金流入",
    ),
    "fundamental_ending_cash_latest_native": (
        "現金及現金等價物",
        "现金及现金等价物",
        "銀行結餘及現金",
        "银行结余及现金",
        "cash and cash equivalents",
    ),
    "fundamental_net_profit_latest_native": (
        "年內溢利",
        "年内利润",
        "年內虧損",
        "年内亏损",
        "淨利潤",
        "净利润",
        "淨虧損",
        "净亏损",
        "profit for the year",
        "loss for the year",
    ),
    "fundamental_research_development_latest_native": (
        "研發開支",
        "研发开支",
        "研發費用",
        "研发费用",
        "研究及開發成本",
        "research and development expenses",
    ),
    "customer_largest_ratio_pct": ("最大客戶", "最大客户", "largest customer"),
    "customer_top5_ratio_pct": (
        "五大客戶",
        "五大客户",
        "five largest customers",
    ),
    "supplier_largest_ratio_pct": (
        "最大供應商",
        "最大供应商",
        "largest supplier",
    ),
    "supplier_top5_ratio_pct": (
        "五大供應商",
        "五大供应商",
        "five largest suppliers",
    ),
    "cornerstone_share_count": ("基石投資者", "基石投资者", "cornerstone investor"),
    "global_offering_share_count": ("全球發售", "全球发售", "global offering"),
    "cornerstone_ratio_disclosed_pct": (
        "基石投資者",
        "基石投资者",
        "cornerstone investor",
    ),
    "cornerstone_lockup_months": (
        "基石投資者",
        "基石投资者",
        "cornerstone investor",
        "禁售期",
        "禁售期內",
        "lock-up period",
    ),
    "controlling_shareholder_lockup_months": (
        "控股股東",
        "控股股东",
        "controlling shareholder",
    ),
    "industry_primary": ("主要業務", "主要业务", "business overview"),
    "pipeline_stage_name": (
        "核心產品",
        "核心产品",
        "研發管線",
        "研发管线",
        "core product",
    ),
}

_STAGE_PATTERNS = (
    (6, ("获批", "批准上市", "商業化", "商业化", "approved", "commercial")),
    (5, ("nda", "bla", "上市申请", "申報上市", "申报上市", "受理")),
    (4, ("iii期", "iiia期", "iiib期", "三期", "phase iii", "phase 3")),
    (3, ("ii期", "iia期", "iib期", "二期", "phase ii", "phase 2")),
    (2, ("i期", "ia期", "ib期", "一期", "phase i", "phase 1")),
    (1, ("临床前", "臨床前", "preclinical")),
)


class RagFieldExtraction(BaseModel):
    """模型返回的一项受证据约束的事实。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str
    status: Literal["found", "not_found", "not_applicable"]
    value: float | str | None = None
    currency: str | None = None
    unit: str | None = None
    period_end: str | None = None
    source_chunk_ids: tuple[str, ...] = Field(default=(), max_length=6)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    note: str = ""

    @model_validator(mode="after")
    def validate_status(self) -> "RagFieldExtraction":
        if self.status == "found":
            if self.value is None or not self.source_chunk_ids:
                raise ValueError("found fields require value and source_chunk_ids")
        elif self.value is not None or self.source_chunk_ids:
            raise ValueError("missing fields must not contain values or evidence")
        return self


class RagExtractionResponse(BaseModel):
    """一家公司的一次模型提取响应。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    financial_period_used: str | None = None
    fields: tuple[RagFieldExtraction, ...]
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_fields(self) -> "RagExtractionResponse":
        names = [field.name for field in self.fields]
        if len(names) != len(set(names)):
            raise ValueError("model returned duplicate field names")
        if set(names) != set(_MODEL_FIELD_NAMES):
            missing = sorted(set(_MODEL_FIELD_NAMES) - set(names))
            extra = sorted(set(names) - set(_MODEL_FIELD_NAMES))
            raise ValueError(f"field inventory mismatch: missing={missing}, extra={extra}")
        return self


@dataclass(frozen=True)
class RagFundamentalDatasetResult:
    feature_matrix_path: Path
    audit_path: Path
    summary_path: Path
    company_count: int
    feature_columns: tuple[str, ...]


ProgressCallback = Callable[[str], None]


class RagFundamentalExtractor:
    """在单份已向量化招股书中检索，再用一次 LLM 提取多项事实。"""

    def __init__(
        self,
        *,
        retriever: PgVectorRetriever | None,
        provider: JsonChatProvider,
        system_prompt: str,
        max_output_tokens: int = 12_000,
        semantic_top_k: int = 8,
        lexical_top_k: int = 8,
        neighbor_radius: int = 2,
        max_chunks_per_query: int = 10,
        max_chars_per_query: int = 7_000,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.retriever = retriever
        self.provider = provider
        self.system_prompt = system_prompt
        self.max_output_tokens = max_output_tokens
        self.semantic_top_k = semantic_top_k
        self.lexical_top_k = lexical_top_k
        self.neighbor_radius = neighbor_radius
        self.max_chunks_per_query = max_chunks_per_query
        self.max_chars_per_query = max_chars_per_query
        self.retry_policy = retry_policy or RetryPolicy(
            max_retries=2,
            retry_if=retry_model_node_error,
        )

    def extract(
        self,
        *,
        manifest_row: Mapping[str, Any],
        document_id: str,
        chunks: Sequence[DocumentChunk],
        existing_operating_cash_flow: float | None,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        if not chunks:
            raise ValueError("RAG extraction requires parsed document chunks")
        semantic_ready = False
        if self.retriever is not None:
            try:
                semantic_ready = self.retriever.activate_document(
                    document_id,
                    source_hash=str(manifest_row["source_hash"]),
                    expected_chunk_count=len(chunks),
                )
            except Exception:  # noqa: BLE001 - database outage must not disable lexical RAG
                semantic_ready = False
        retrieval_mode = (
            "hybrid_semantic_lexical"
            if semantic_ready
            else "lexical_anchor_fallback"
        )
        if not semantic_ready and progress is not None:
            progress("pgvector语义索引不可用，降级为BM25与字段锚点检索")

        lexical = BM25Retriever(list(chunks))
        evidence_groups: dict[str, list[DocumentChunk]] = {}
        for index, (group, query) in enumerate(_QUERY_SPECS.items(), start=1):
            if progress is not None:
                progress(f"RAG检索 {index}/{len(_QUERY_SPECS)}：{group}")
            semantic_hits = []
            if semantic_ready and self.retriever is not None:
                try:
                    semantic_hits = self.retriever.search(query, top_k=self.semantic_top_k)
                except Exception:  # noqa: BLE001
                    semantic_ready = False
                    retrieval_mode = "lexical_anchor_fallback"
                    if progress is not None:
                        progress("语义检索中断，后续字段继续使用BM25与字段锚点检索")
            lexical_hits = lexical.search(query, top_k=self.lexical_top_k)
            anchor_hits = _anchor_search(chunks, field_name=group, top_k=8)
            selected = _expand_hits(
                chunks,
                anchor_hits=anchor_hits,
                semantic_hits=semantic_hits,
                lexical_hits=lexical_hits,
                neighbor_radius=self.neighbor_radius,
                max_chunks=self.max_chunks_per_query,
                max_chars=self.max_chars_per_query,
            )
            evidence_groups[group] = _augment_field_context(
                chunks,
                selected,
                field_name=group,
                max_chunks=self.max_chunks_per_query,
                max_chars=self.max_chars_per_query,
            )

        allowed_chunks = {
            chunk.chunk_id: chunk
            for selected in evidence_groups.values()
            for chunk in selected
        }
        user_prompt = _build_user_prompt(manifest_row, evidence_groups)

        last_response: RagExtractionResponse | None = None
        last_validation_error: Exception | None = None

        def operation(feedback: str | None) -> RagExtractionResponse:
            nonlocal last_response, last_validation_error
            prompt = user_prompt
            if feedback:
                prompt += (
                    "\n\n上次输出校验失败，请仅修正 JSON。"
                    "无法找到本字段直接证据时必须返回not_found，不得借用其他字段片段："
                    f"{feedback}"
                )
            raw = self.provider.complete_json(
                system_prompt=self.system_prompt,
                user_prompt=prompt,
                temperature=0.0,
                max_output_tokens=self.max_output_tokens,
            )
            response = RagExtractionResponse.model_validate(raw)
            last_response = response
            try:
                _validate_response(
                    response,
                    allowed_chunks,
                    allowed_by_field={
                        name: {chunk.chunk_id for chunk in selected}
                        for name, selected in evidence_groups.items()
                    },
                    prospectus_date=date.fromisoformat(
                        str(manifest_row["prospectus_date"])[:10]
                    ),
                )
            except ValueError as error:
                last_validation_error = error
                raise
            return response

        if progress is not None:
            progress("模型正在从RAG证据中提取结构化事实")
        retry_errors: list[str] = []
        try:
            outcome = self.retry_policy.run_with_feedback(
                operation,
                on_retry=lambda error, _attempt, _wait: retry_errors.append(str(error)),
            )
            response = outcome.value
            attempts = outcome.attempts
            errors = outcome.errors
        except RetryExhaustedError as error:
            if last_response is None or error.last_error is not last_validation_error:
                raise
            response = _downgrade_invalid_fields(
                last_response,
                allowed_chunks,
                allowed_by_field={
                    name: {chunk.chunk_id for chunk in selected}
                    for name, selected in evidence_groups.items()
                },
                prospectus_date=date.fromisoformat(
                    str(manifest_row["prospectus_date"])[:10]
                ),
            )
            attempts = error.attempts
            errors = tuple([*retry_errors, str(error.last_error)])
        return _build_company_record(
            manifest_row=manifest_row,
            response=response,
            evidence_groups=evidence_groups,
            allowed_chunks=allowed_chunks,
            existing_operating_cash_flow=existing_operating_cash_flow,
            model_name=self.provider.model_name,
            attempts=attempts,
            retry_errors=errors,
            retrieval_mode=retrieval_mode,
        )


def build_rag_fundamental_dataset(
    project_root: Path | str,
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    output_dir: Path | str | None = None,
) -> RagFundamentalDatasetResult:
    """把已提取记录保存为 CSV、逐字段审计和覆盖率摘要。"""
    root = Path(project_root).resolve()
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / "data" / "processed" / "modeling"
    )
    destination.mkdir(parents=True, exist_ok=True)
    manifest_by_company = {
        str(row.get("company_id") or "").zfill(5): row for row in manifest_rows
    }
    ordered = sorted(
        (dict(record) for record in records),
        key=lambda row: (
            str(row.get("prospectus_date") or ""),
            str(row.get("company_id") or ""),
        ),
    )
    if not ordered:
        raise ValueError("没有可保存的RAG结构化特征")

    rows: list[dict[str, Any]] = []
    for record in ordered:
        company_id = str(record["company_id"]).zfill(5)
        manifest = manifest_by_company.get(company_id, {})
        features = record.get("features") or {}
        rows.append(
            {
                "company_id": company_id,
                "company_name": record.get("company_name"),
                "company_folder": manifest.get("company_folder"),
                "prospectus_date": record.get("prospectus_date"),
                "split": manifest.get("split"),
                **{
                    name: (features.get(name) or {}).get("value")
                    for name in RAG_FUNDAMENTAL_FEATURE_NAMES
                },
            }
        )

    frame = pd.DataFrame(rows)
    matrix_path = destination / "rag_fundamental_feature_matrix.csv"
    audit_path = destination / "rag_fundamental_feature_audit.jsonl"
    summary_path = destination / "rag_fundamental_feature_summary.json"
    frame.to_csv(matrix_path, index=False, encoding="utf-8-sig")
    write_jsonl(ordered, audit_path)
    coverage = {
        name: int(frame[name].notna().sum()) for name in RAG_FUNDAMENTAL_FEATURE_NAMES
    }
    write_json(
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "extraction_policy": "pgvector_rag_plus_llm_with_deterministic_validation",
            "company_count": len(frame),
            "feature_count": len(RAG_FUNDAMENTAL_FEATURE_NAMES),
            "feature_columns": list(RAG_FUNDAMENTAL_FEATURE_NAMES),
            "coverage_non_null": coverage,
            "years": sorted(
                {
                    int(str(value)[:4])
                    for value in frame["prospectus_date"]
                    if str(value)[:4].isdigit()
                }
            ),
        },
        summary_path,
    )
    return RagFundamentalDatasetResult(
        feature_matrix_path=matrix_path,
        audit_path=audit_path,
        summary_path=summary_path,
        company_count=len(frame),
        feature_columns=RAG_FUNDAMENTAL_FEATURE_NAMES,
    )


def load_rag_feature_snapshots(
    audit_path: Path,
) -> dict[str, dict[str, FeatureValue]]:
    """加载 RAG 补充特征，供统一训练宽表合并。"""
    if not audit_path.is_file():
        return {}
    snapshots: dict[str, dict[str, FeatureValue]] = {}
    for row in read_jsonl(audit_path):
        company_id = str(row.get("company_id") or "").zfill(5)
        features = row.get("features")
        if not isinstance(features, dict):
            continue
        snapshots[company_id] = {
            name: FeatureValue.model_validate(payload)
            for name, payload in features.items()
            if name in RAG_FUNDAMENTAL_FEATURE_NAMES and isinstance(payload, dict)
        }
    return snapshots


def _expand_hits(
    chunks: Sequence[DocumentChunk],
    *,
    anchor_hits: Sequence[SearchHit] = (),
    semantic_hits: Sequence[SearchHit],
    lexical_hits: Sequence[SearchHit],
    neighbor_radius: int,
    max_chunks: int,
    max_chars: int,
) -> list[DocumentChunk]:
    index_by_id = {chunk.chunk_id: index for index, chunk in enumerate(chunks)}
    seeds: list[DocumentChunk] = [hit.chunk for hit in anchor_hits]
    for rank in range(max(len(semantic_hits), len(lexical_hits))):
        if rank < len(semantic_hits):
            seeds.append(semantic_hits[rank].chunk)
        if rank < len(lexical_hits):
            seeds.append(lexical_hits[rank].chunk)

    candidates: list[DocumentChunk] = []
    seen: set[str] = set()

    def add(chunk: DocumentChunk) -> None:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            candidates.append(chunk)

    # 先保留每个高分命中本身，再补邻居。否则首个表格的相邻切片可能占满预算，
    # 把后续真正的条款或财务表挤出上下文。
    unique_seeds: list[DocumentChunk] = []
    seed_ids: set[str] = set()
    for seed in seeds:
        if seed.chunk_id not in seed_ids:
            seed_ids.add(seed.chunk_id)
            unique_seeds.append(seed)
    for seed in unique_seeds[:max_chunks]:
        add(seed)

    for seed in unique_seeds:
        position = index_by_id.get(seed.chunk_id)
        if position is not None:
            for offset in range(1, neighbor_radius + 1):
                if position - offset >= 0:
                    add(chunks[position - offset])
                if position + offset < len(chunks):
                    add(chunks[position + offset])
        if seed.table_id:
            for chunk in chunks:
                if chunk.table_id == seed.table_id:
                    add(chunk)
        if len(candidates) >= max_chunks * 2:
            break

    selected: list[DocumentChunk] = []
    char_count = 0
    for chunk in candidates:
        if len(selected) >= max_chunks:
            break
        remaining = max_chars - char_count
        if remaining <= 0:
            break
        if len(chunk.text) > remaining and selected:
            continue
        selected.append(chunk)
        char_count += len(chunk.text)
    return sorted(selected, key=lambda chunk: (chunk.page, chunk.chunk_id))


def _anchor_search(
    chunks: Sequence[DocumentChunk],
    *,
    field_name: str,
    top_k: int,
) -> list[SearchHit]:
    """用字段专属原文词确保精确表格不会被通用语义片段挤掉。"""
    anchors = tuple(item.lower() for item in _FIELD_ANCHORS.get(field_name, ()))
    if not anchors or top_k < 1:
        return []
    scored: list[tuple[float, DocumentChunk]] = []
    for chunk in chunks:
        text = chunk.text.lower()
        matches = sum(text.count(anchor) for anchor in anchors)
        if not matches:
            continue
        if field_name in {
            "cornerstone_lockup_months",
            "controlling_shareholder_lockup_months",
        }:
            # 关系表会重复数十次“控股股东”，不能让重复次数压过真正的禁售条款。
            matches = min(matches, 3)
        digit_count = sum(character.isdigit() for character in text)
        score = matches * 100.0 + min(digit_count, 200)
        if chunk.chunk_kind == "table":
            score += 300.0
        if any(marker in text for marker in ("截至", "於二零", "于二零", "as at")):
            score += 80.0
        if field_name == "fundamental_ending_cash_latest_native":
            if any(
                marker in text
                for marker in (
                    "現金流量",
                    "现金流量",
                    "增加╱",
                    "減少）淨額",
                    "减少）净额",
                )
            ):
                score -= 600.0
            if any(
                marker in text
                for marker in (
                    "綜合財務狀況表",
                    "综合财务状况表",
                    "合併財務狀況表",
                    "合并财务状况表",
                    "流動資產",
                    "流动资产",
                )
            ):
                score += 500.0
            if digit_count < 4 and any(
                marker in text for marker in ("包括", "定義", "定义", "comprise")
            ):
                score -= 250.0
        elif field_name == "fundamental_revenue_latest_native":
            if any(
                marker in text
                for marker in ("收入", "Revenue", "總收入", "总收入", "收益合計", "收益合计")
            ):
                score += 800.0
            if any(marker in text for marker in ("綜合收益", "综合收益", "損益", "损益")):
                score += 500.0
            if any(marker in text for marker in _APPENDIX_MARKERS):
                score -= 450.0
            if _SEPARATE_APPENDIX_SUBJECT.search(text):
                score -= 1400.0
        elif field_name == "fundamental_operating_cash_flow_latest_native":
            if any(
                marker in text
                for marker in (
                    "經營活動", "经营活动", "operating activities",
                    "現金流量表", "现金流量表", "cash flow statement",
                )
            ):
                score += 800.0
            if any(marker in text for marker in ("增加╱", "减少）净额", "net cash")):
                score += 300.0
            if any(marker in text for marker in _APPENDIX_MARKERS):
                score -= 350.0
        elif field_name == "fundamental_net_profit_latest_native":
            if any(marker in text for marker in _ISSUER_FINANCIAL_MARKERS):
                score += 900.0
            if any(marker in text for marker in _APPENDIX_MARKERS):
                score -= 450.0
            if _SEPARATE_APPENDIX_SUBJECT.search(text):
                score -= 1400.0
        elif field_name in {
            "cornerstone_lockup_months",
            "controlling_shareholder_lockup_months",
        }:
            if field_name == "cornerstone_lockup_months":
                if any(
                    marker in text
                    for marker in ("基石投資者", "基石投资者", "cornerstone investor")
                ):
                    score += 800.0
                if any(
                    marker in text
                    for marker in (
                        "上市日期後六個月",
                        "上市日期后六个月",
                        "六個月期間",
                        "六个月期间",
                        "不得處置",
                        "不得处置",
                    )
                ):
                    score += 1000.0
                if any(
                    marker in text
                    for marker in ("控股股東", "控股股东", "controlling shareholder")
                ):
                    score -= 500.0
            else:
                if any(
                    marker in text
                    for marker in ("控股股東的承諾", "控股股东的承诺")
                ):
                    score += 900.0
                if any(
                    marker in text
                    for marker in ("上市日期", "首六個月", "首六个月", "10.07")
                ):
                    score += 500.0
                if any(
                    marker in text
                    for marker in ("「控股股東」指", "「控股股东」指")
                ):
                    score -= 900.0
            if any(marker in text for marker in ("目 錄", "目錄", "目 录", "目录")):
                score -= 1000.0
        scored.append((score, chunk))
    ranked = sorted(scored, key=lambda item: (-item[0], item[1].chunk_id))[:top_k]
    return [
        SearchHit(chunk=chunk, score=score, rank=rank)
        for rank, (score, chunk) in enumerate(ranked, start=1)
    ]


def _augment_field_context(
    chunks: Sequence[DocumentChunk],
    selected: Sequence[DocumentChunk],
    *,
    field_name: str,
    max_chunks: int,
    max_chars: int,
) -> list[DocumentChunk]:
    """补回表格标题/情景表头，避免只召回数值行而丢失列语义。"""
    priority: list[DocumentChunk] = []
    if field_name == "cornerstone_ratio_disclosed_pct":
        table_pages = {chunk.page for chunk in selected if chunk.chunk_kind == "table"}
        priority = [
            chunk
            for chunk in chunks
            if chunk.page in table_pages
            and any(marker in chunk.text for marker in ("基於發售價", "基于发售价"))
            and "港元" in chunk.text
        ]
    ordered = [*priority, *selected]
    kept: list[DocumentChunk] = []
    seen: set[str] = set()
    char_count = 0
    for chunk in ordered:
        if chunk.chunk_id in seen or len(kept) >= max_chunks:
            continue
        if kept and char_count + len(chunk.text) > max_chars:
            continue
        seen.add(chunk.chunk_id)
        kept.append(chunk)
        char_count += len(chunk.text)
    return sorted(kept, key=lambda chunk: (chunk.page, chunk.chunk_id))


def _build_user_prompt(
    manifest_row: Mapping[str, Any],
    evidence_groups: Mapping[str, Sequence[DocumentChunk]],
) -> str:
    inventory = "\n".join(f"- {name}" for name in _MODEL_FIELD_NAMES)
    sections: list[str] = []
    evidence_registry: dict[str, DocumentChunk] = {}
    fields_by_chunk: dict[str, list[str]] = {}
    for group, chunks in evidence_groups.items():
        chunk_ids: list[str] = []
        for chunk in chunks:
            evidence_registry.setdefault(chunk.chunk_id, chunk)
            fields_by_chunk.setdefault(chunk.chunk_id, []).append(group)
            chunk_ids.append(chunk.chunk_id)
        sections.append(f"- {group}: {', '.join(chunk_ids) or '无召回切片'}")

    rendered_evidence: list[str] = []
    for chunk in sorted(
        evidence_registry.values(), key=lambda item: (item.page, item.chunk_id)
    ):
        metadata = (
            f"chunk_id={chunk.chunk_id} page={chunk.page} "
            f"printed_page={chunk.printed_page_label or '-'} kind={chunk.chunk_kind} "
            f"allowed_for={','.join(fields_by_chunk.get(chunk.chunk_id, []))}"
        )
        if chunk.table_id:
            metadata += f" table_id={chunk.table_id}"
        rendered_evidence.append(f"[{metadata}]\n{chunk.text}")
    return (
        f"公司代码：{str(manifest_row.get('company_id') or '').zfill(5)}\n"
        f"公司名称：{manifest_row.get('company_name')}\n"
        f"招股书日期：{manifest_row.get('prospectus_date')}\n\n"
        f"必须返回以下全部字段，每个恰好一次：\n{inventory}\n\n"
        "每个字段应使用其对应召回切片。只有切片原文本身直接出现该字段行名、条款主体或同表表头时，"
        "才可跨组引用；否则必须返回not_found。不要为了填满字段而借用相似数字。\n"
        "字段与召回切片：\n"
        + "\n".join(sections)
        + "\n\n去重后的证据注册表：\n"
        + "\n\n".join(rendered_evidence)
    )


def _validate_response(
    response: RagExtractionResponse,
    allowed_chunks: Mapping[str, DocumentChunk],
    *,
    allowed_by_field: Mapping[str, set[str]] | None = None,
    prospectus_date: date | None = None,
) -> None:
    period = _optional_iso_date(response.financial_period_used)
    if period is not None and prospectus_date is not None and period >= prospectus_date:
        raise ValueError("财务年度必须早于招股书日期")
    found = {field.name: field for field in response.fields if field.status == "found"}
    for field in response.fields:
        if any(chunk_id not in allowed_chunks for chunk_id in field.source_chunk_ids):
            raise ValueError(f"{field.name}引用了本次RAG未返回的chunk_id")
        if (
            field.source_chunk_ids
            and allowed_by_field is not None
            and field.name in _FIELD_SCOPED_EVIDENCE_FIELDS
            and any(
                chunk_id not in allowed_by_field.get(field.name, set())
                for chunk_id in field.source_chunk_ids
            )
            and not _evidence_supports_field(
                field.name,
                field.source_chunk_ids,
                allowed_chunks,
            )
        ):
            permitted = sorted(allowed_by_field.get(field.name, set()))[:12]
            raise ValueError(
                f"{field.name}引用的跨组证据没有直接支持该字段；"
                f"优先使用这些chunk_id：{permitted}；无直接证据请返回not_found"
            )
        if field.status != "found":
            continue
        if field.name in _FINANCIAL_FIELDS:
            if not isinstance(field.value, (int, float)) or isinstance(field.value, bool):
                raise ValueError(f"{field.name}必须为数值")
            if field.unit != "million_native_currency" or not field.currency:
                raise ValueError(f"{field.name}必须使用百万本位币和币种")
            field_period = _optional_iso_date(field.period_end)
            if period is None or field_period != period:
                raise ValueError(f"{field.name}必须与最近完整财政年度一致")
            if (
                field.name == "fundamental_research_development_latest_native"
                and float(field.value) == 0.0
            ):
                _validate_explicit_r_and_d_zero(field, allowed_chunks)
        elif field.name in _PERCENT_FIELDS:
            value = _finite_number(field.value, field.name)
            if field.unit != "percentage_points" or not 0 <= value <= 100:
                raise ValueError(f"{field.name}必须是0到100的百分点")
            _validate_exact_percentage(field, allowed_chunks)
        elif field.name in _MONTH_FIELDS:
            value = _finite_number(field.value, field.name)
            if field.unit != "months" or not 0 < value <= 120:
                raise ValueError(f"{field.name}锁定月数超出合理范围")
            _validate_lockup_subject(field, allowed_chunks)
        elif field.name in _SHARE_FIELDS:
            value = _finite_number(field.value, field.name)
            if field.unit != "shares" or value <= 0:
                raise ValueError(f"{field.name}必须为正的股份数")
        elif field.name == "industry_primary":
            if field.unit != "category" or not isinstance(field.value, str):
                raise ValueError("industry_primary必须是类别文本")
        elif field.name == "pipeline_stage_name":
            if field.unit != "stage_name" or not isinstance(field.value, str):
                raise ValueError("pipeline_stage_name必须是阶段文本")

        if field.name in _FINANCIAL_FIELDS:
            _reject_separate_appendix_subject(field, allowed_chunks)

    for largest_name, top5_name in (
        ("customer_largest_ratio_pct", "customer_top5_ratio_pct"),
        ("supplier_largest_ratio_pct", "supplier_top5_ratio_pct"),
    ):
        largest = found.get(largest_name)
        top5 = found.get(top5_name)
        if largest is not None and top5 is not None:
            if float(largest.value) > float(top5.value):
                raise ValueError(f"{largest_name}不能大于{top5_name}")


def _evidence_context(
    source_chunk_ids: Sequence[str],
    chunks: Mapping[str, DocumentChunk],
) -> str:
    """还原引用切片的同表与同页文本，兼容表头、行名和值被拆开。"""
    sources = [chunks[chunk_id] for chunk_id in source_chunk_ids if chunk_id in chunks]
    if not sources:
        return ""
    pages = {chunk.page for chunk in sources}
    table_ids = {chunk.table_id for chunk in sources if chunk.table_id}
    related = [
        chunk
        for chunk in chunks.values()
        if chunk.chunk_id in source_chunk_ids
        or (chunk.table_id is not None and chunk.table_id in table_ids)
        or chunk.page in pages
    ]
    return " ".join(chunk.text for chunk in related).lower()


def _evidence_supports_field(
    field_name: str,
    source_chunk_ids: Sequence[str],
    chunks: Mapping[str, DocumentChunk],
) -> bool:
    context = _evidence_context(source_chunk_ids, chunks)
    anchors = tuple(anchor.lower() for anchor in _FIELD_ANCHORS.get(field_name, ()))
    return bool(context and anchors and any(anchor in context for anchor in anchors))


def _validate_explicit_r_and_d_zero(
    field: RagFieldExtraction,
    chunks: Mapping[str, DocumentChunk],
) -> None:
    context = _evidence_context(field.source_chunk_ids, chunks)
    for pattern in _EXPLICIT_R_AND_D_ZERO_PATTERNS:
        for match in pattern.finditer(context):
            window = context[max(0, match.start() - 24) : match.end() + 24]
            if not any(marker in window for marker in _AMBIGUOUS_ZERO_MARKERS):
                return
    raise ValueError(
        "fundamental_research_development_latest_native证据只说明无重大或主要研发，"
        "不能据此认定研发费用为零"
    )


def _missing_field(field: RagFieldExtraction, reason: str) -> RagFieldExtraction:
    return field.model_copy(
        update={
            "status": "not_found",
            "value": None,
            "currency": None,
            "unit": None,
            "period_end": None,
            "source_chunk_ids": (),
            "confidence": 0.0,
            "note": f"校验未通过，降级为缺失：{reason}"[:500],
        }
    )


def _downgrade_invalid_fields(
    response: RagExtractionResponse,
    allowed_chunks: Mapping[str, DocumentChunk],
    *,
    allowed_by_field: Mapping[str, set[str]] | None = None,
    prospectus_date: date | None = None,
) -> RagExtractionResponse:
    """重试耗尽后仅移除无可靠证据的字段，保留同公司的其他有效事实。"""
    fields = list(response.fields)
    warnings = list(response.warnings)
    invalid: dict[str, str] = {}
    for candidate in fields:
        if candidate.status != "found":
            continue
        trial_fields = tuple(
            field
            if field.name == candidate.name
            else _missing_field(field, "字段级隔离校验")
            if field.status == "found"
            else field
            for field in fields
        )
        trial_period = (
            response.financial_period_used
            if candidate.name in _FINANCIAL_FIELDS
            else None
        )
        trial = response.model_copy(
            update={"financial_period_used": trial_period, "fields": trial_fields}
        )
        try:
            _validate_response(
                trial,
                allowed_chunks,
                allowed_by_field=allowed_by_field,
                prospectus_date=prospectus_date,
            )
        except ValueError as error:
            invalid[candidate.name] = str(error)

    for largest_name, top5_name in (
        ("customer_largest_ratio_pct", "customer_top5_ratio_pct"),
        ("supplier_largest_ratio_pct", "supplier_top5_ratio_pct"),
    ):
        values = {field.name: field for field in fields if field.status == "found"}
        largest = values.get(largest_name)
        top5 = values.get(top5_name)
        if largest is not None and top5 is not None:
            if float(largest.value) > float(top5.value):
                reason = f"{largest_name}不能大于{top5_name}"
                invalid[largest_name] = reason
                invalid[top5_name] = reason

    downgraded = tuple(
        _missing_field(field, invalid[field.name])
        if field.name in invalid
        else field
        for field in fields
    )
    warnings.extend(
        f"degraded_to_not_found:{name}:{reason}" for name, reason in sorted(invalid.items())
    )
    financial_period = response.financial_period_used
    if not any(
        field.status == "found" and field.name in _FINANCIAL_FIELDS
        for field in downgraded
    ):
        financial_period = None
    result = response.model_copy(
        update={
            "financial_period_used": financial_period,
            "fields": downgraded,
            "warnings": tuple(warnings),
        }
    )
    _validate_response(
        result,
        allowed_chunks,
        allowed_by_field=allowed_by_field,
        prospectus_date=prospectus_date,
    )
    return result


def _build_company_record(
    *,
    manifest_row: Mapping[str, Any],
    response: RagExtractionResponse,
    evidence_groups: Mapping[str, Sequence[DocumentChunk]],
    allowed_chunks: Mapping[str, DocumentChunk],
    existing_operating_cash_flow: float | None,
    model_name: str,
    attempts: int,
    retry_errors: Sequence[str],
    retrieval_mode: str,
) -> dict[str, Any]:
    available_at = date.fromisoformat(str(manifest_row["prospectus_date"])[:10])
    extracted = {field.name: field for field in response.fields}
    features: dict[str, FeatureValue] = {}
    feature_audit: dict[str, Any] = {}

    def add_feature(name: str, value: float | str | None, sources: Sequence[str]) -> None:
        if value is None:
            return
        source_ref = "rag:" + ",".join(sources)
        features[name] = FeatureValue(
            value=value,
            available_at=available_at,
            source_ref=source_ref,
        )

    for name in (
        "fundamental_revenue_latest_native",
        "fundamental_operating_cash_flow_latest_native",
        "fundamental_ending_cash_latest_native",
        "fundamental_net_profit_latest_native",
        "fundamental_research_development_latest_native",
        "customer_largest_ratio_pct",
        "customer_top5_ratio_pct",
        "supplier_largest_ratio_pct",
        "supplier_top5_ratio_pct",
        "cornerstone_lockup_months",
        "controlling_shareholder_lockup_months",
        "industry_primary",
        "pipeline_stage_name",
    ):
        field = extracted[name]
        if field.status == "found":
            add_feature(name, _normalized_value(field), field.source_chunk_ids)
        feature_audit[name] = _field_audit(field, allowed_chunks)

    cornerstone_sources: tuple[str, ...] = ()
    cornerstone_method: str | None = None
    disclosed = extracted["cornerstone_ratio_disclosed_pct"]
    numerator = extracted["cornerstone_share_count"]
    denominator = extracted["global_offering_share_count"]
    cornerstone_ratio: float | None = None
    scenario_ratio, scenario_sources = _cornerstone_midpoint_ratio(
        evidence_groups.get("cornerstone_ratio_disclosed_pct", ())
    )
    if scenario_ratio is not None:
        cornerstone_ratio = scenario_ratio
        cornerstone_sources = scenario_sources
        cornerstone_method = (
            "midpoint_price_scenario_derived"
            if len(scenario_sources) > 1
            else "median_of_disclosed_price_scenarios"
        )
    elif disclosed.status == "found":
        cornerstone_ratio = _finite_number(disclosed.value, disclosed.name)
        cornerstone_sources = disclosed.source_chunk_ids
        cornerstone_method = "direct_disclosure"
    elif numerator.status == denominator.status == "found":
        shares = _finite_number(numerator.value, numerator.name)
        offer_shares = _finite_number(denominator.value, denominator.name)
        if offer_shares > 0:
            cornerstone_ratio = shares / offer_shares * 100.0
            cornerstone_sources = tuple(
                dict.fromkeys((*numerator.source_chunk_ids, *denominator.source_chunk_ids))
            )
            cornerstone_method = "cornerstone_shares/global_offering_shares*100"
    if cornerstone_ratio is not None and 0 <= cornerstone_ratio <= 100:
        add_feature("cornerstone_ratio_pct", cornerstone_ratio, cornerstone_sources)
    feature_audit["cornerstone_ratio_pct"] = {
        "value": cornerstone_ratio,
        "method": cornerstone_method,
        "source_chunk_ids": list(cornerstone_sources),
        "source_pages": _source_pages(cornerstone_sources, allowed_chunks),
        "inputs": {
            "cornerstone_share_count": numerator.value,
            "global_offering_share_count": denominator.value,
            "direct_disclosure_pct": disclosed.value,
        },
    }

    stage = extracted["pipeline_stage_name"]
    stage_score = (
        _pipeline_stage_score(str(stage.value)) if stage.status == "found" else None
    )
    if stage_score is not None:
        add_feature("pipeline_stage_score", float(stage_score), stage.source_chunk_ids)
    feature_audit["pipeline_stage_score"] = {
        "value": stage_score,
        "method": "deterministic_stage_mapping",
        "source_chunk_ids": list(stage.source_chunk_ids),
        "source_pages": _source_pages(stage.source_chunk_ids, allowed_chunks),
    }

    ending_cash = features.get("fundamental_ending_cash_latest_native")
    runway: float | None = None
    if (
        ending_cash is not None
        and existing_operating_cash_flow is not None
        and existing_operating_cash_flow < 0
    ):
        runway = float(ending_cash.value) / abs(existing_operating_cash_flow) * 12.0
        if not math.isfinite(runway) or runway < 0:
            runway = None
    if runway is not None:
        sources = extracted["fundamental_ending_cash_latest_native"].source_chunk_ids
        add_feature("cash_runway_months", runway, sources)
    feature_audit["cash_runway_months"] = {
        "value": runway,
        "method": "ending_cash/abs(latest_operating_cash_flow)*12",
        "inputs": {
            "ending_cash_native_mn": (
                ending_cash.value if ending_cash is not None else None
            ),
            "operating_cash_flow_native_mn": existing_operating_cash_flow,
        },
    }

    for name in RAG_FUNDAMENTAL_FEATURE_NAMES:
        if name not in features:
            features[name] = FeatureValue(
                value=None,
                available_at=available_at,
                source_ref="rag:not_found",
            )

    retrieval_audit = {
        group: [
            {
                "chunk_id": chunk.chunk_id,
                "page": chunk.page,
                "printed_page_label": chunk.printed_page_label,
                "chunk_kind": chunk.chunk_kind,
                "table_id": chunk.table_id,
                "text_preview": " ".join(chunk.text.split())[:800],
            }
            for chunk in chunks
        ]
        for group, chunks in evidence_groups.items()
    }
    return {
        "schema_version": 1,
        "company_id": str(manifest_row.get("company_id") or "").zfill(5),
        "company_name": manifest_row.get("company_name"),
        "prospectus_date": available_at.isoformat(),
        "source_hash": manifest_row.get("source_hash"),
        "financial_period_used": response.financial_period_used,
        "model_name": model_name,
        "model_attempts": attempts,
        "model_retry_errors": list(retry_errors),
        "retrieval_mode": retrieval_mode,
        "features": {
            name: features[name].model_dump(mode="json")
            for name in RAG_FUNDAMENTAL_FEATURE_NAMES
        },
        "feature_audit": feature_audit,
        "model_warnings": list(response.warnings),
        "retrieval": retrieval_audit,
    }


def _normalized_value(field: RagFieldExtraction) -> float | str:
    if isinstance(field.value, str):
        return field.value.strip()
    return _finite_number(field.value, field.name)


def _field_audit(
    field: RagFieldExtraction,
    chunks: Mapping[str, DocumentChunk],
) -> dict[str, Any]:
    return {
        "status": field.status,
        "value": field.value,
        "currency": field.currency,
        "unit": field.unit,
        "period_end": field.period_end,
        "confidence": field.confidence,
        "note": field.note,
        "source_chunk_ids": list(field.source_chunk_ids),
        "source_pages": _source_pages(field.source_chunk_ids, chunks),
    }


def _source_pages(
    chunk_ids: Sequence[str],
    chunks: Mapping[str, DocumentChunk],
) -> list[int]:
    return sorted({chunks[chunk_id].page for chunk_id in chunk_ids if chunk_id in chunks})


def _pipeline_stage_score(value: str) -> int | None:
    normalized = value.lower().replace(" ", "")
    for score, patterns in _STAGE_PATTERNS:
        if any(pattern.replace(" ", "") in normalized for pattern in patterns):
            return score
    return None


def _cornerstone_midpoint_ratio(
    chunks: Sequence[DocumentChunk],
) -> tuple[float | None, tuple[str, ...]]:
    """从上限、中位、下限三组基石表中确定中位发售价对应比例。"""
    candidates: list[tuple[float, str]] = []
    for chunk in chunks:
        if chunk.chunk_kind != "table":
            continue
        for line in chunk.text.splitlines():
            normalized = line.lower()
            if not any(marker in normalized for marker in ("|總計|", "|总计|", "|total|")):
                continue
            values = [float(value) for value in _PERCENT_VALUE.findall(line)]
            if values and 0 <= values[0] <= 100:
                candidates.append((values[0], chunk.chunk_id))
                break
    unique = sorted({value: chunk_id for value, chunk_id in candidates}.items())
    if len(unique) >= 3:
        middle = unique[len(unique) // 2]
        return middle[0], (middle[1],)

    page_context: dict[int, list[DocumentChunk]] = {}
    for chunk in chunks:
        page_context.setdefault(chunk.page, []).append(chunk)
    bounds: list[tuple[float, float, float, str, str]] = []
    for page_chunks in page_context.values():
        price_chunk = next(
            (chunk for chunk in page_chunks if _PRICE_VALUE.search(chunk.text)),
            None,
        )
        if price_chunk is None:
            continue
        prices = [float(value) for value in _PRICE_VALUE.findall(price_chunk.text)]
        for table in (chunk for chunk in page_chunks if chunk.chunk_kind == "table"):
            for line in table.text.splitlines():
                normalized = line.lower()
                if not any(
                    marker in normalized for marker in ("|總計|", "|总计|", "|total|")
                ):
                    continue
                percentages = [float(value) for value in _PERCENT_VALUE.findall(line)]
                numeric_cells = [
                    float(value.replace(",", ""))
                    for value in re.findall(r"(?<![\d.])(\d[\d,]{4,})(?![\d.])", line)
                ]
                if percentages and numeric_cells:
                    bounds.append(
                        (
                            prices[0],
                            numeric_cells[0],
                            percentages[0],
                            table.chunk_id,
                            price_chunk.chunk_id,
                        )
                    )
                    break
    if len(bounds) == 2 and bounds[0][0] != bounds[1][0]:
        low, high = sorted(bounds, key=lambda item: item[0])
        midpoint_price = (low[0] + high[0]) / 2.0
        implied_investment = (low[0] * low[1] + high[0] * high[1]) / 2.0
        midpoint_shares = implied_investment / midpoint_price
        implied_offer_shares = (
            low[1] / (low[2] / 100.0) + high[1] / (high[2] / 100.0)
        ) / 2.0
        if implied_offer_shares > 0:
            sources = tuple(dict.fromkeys((low[3], low[4], high[3], high[4])))
            return midpoint_shares / implied_offer_shares * 100.0, sources
    return None, ()


def _validate_exact_percentage(
    field: RagFieldExtraction,
    chunks: Mapping[str, DocumentChunk],
) -> None:
    value = _finite_number(field.value, field.name)
    exact_match_found = False
    for chunk_id in field.source_chunk_ids:
        text = chunks[chunk_id].text.lower()
        for match in _PERCENT_VALUE.finditer(text):
            if abs(float(match.group(1)) - value) > 1e-9:
                continue
            window = text[max(0, match.start() - 80) : match.end() + 20]
            if any(marker in window for marker in _NON_EXACT_PERCENT_MARKERS):
                raise ValueError(f"{field.name}引用的{value}%不是精確比例")
            exact_match_found = True
    if not exact_match_found:
        raise ValueError(f"{field.name}未在证据中找到精确披露的{value}%")


def _reject_separate_appendix_subject(
    field: RagFieldExtraction,
    chunks: Mapping[str, DocumentChunk],
) -> None:
    source_pages = {
        chunks[chunk_id].page
        for chunk_id in field.source_chunk_ids
        if chunk_id in chunks
    }
    # 标题和主体名经常被拆在相邻切片；校验时把同页及前后两页一并还原。
    context = " ".join(
        chunk.text
        for chunk in chunks.values()
        if any(abs(chunk.page - page) <= 2 for page in source_pages)
    )
    if _SEPARATE_APPENDIX_SUBJECT.search(context):
        raise ValueError(f"{field.name}证据来自其他主体附錄，不能归属发行人")


def _validate_lockup_subject(
    field: RagFieldExtraction,
    chunks: Mapping[str, DocumentChunk],
) -> None:
    source_pages = {
        chunks[chunk_id].page
        for chunk_id in field.source_chunk_ids
        if chunk_id in chunks
    }
    context = " ".join(
        chunk.text.lower()
        for chunk in chunks.values()
        if any(abs(chunk.page - page) <= 1 for page in source_pages)
    )
    if field.name == "cornerstone_lockup_months":
        subject_markers = ("基石投資者", "基石投资者", "cornerstone investor")
    else:
        subject_markers = ("控股股東", "控股股东", "controlling shareholder")
    clause_markers = (
        "上市日期後", "上市日期后", "禁售", "鎖定", "锁定",
        "不得處置", "不得处置", "不會出售", "不会出售", "lock-up",
    )
    if not any(marker in context for marker in subject_markers):
        raise ValueError(f"{field.name}证据主体不匹配")
    if not any(marker in context for marker in clause_markers):
        raise ValueError(f"{field.name}未引用实际禁售条款")


def _finite_number(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name}必须为数值")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name}必须为有限数值")
    return number


def _optional_iso_date(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError as error:
        raise ValueError(f"日期必须为ISO格式：{value}") from error


__all__ = [
    "RAG_FUNDAMENTAL_FEATURE_NAMES",
    "RagExtractionResponse",
    "RagFieldExtraction",
    "RagFundamentalDatasetResult",
    "RagFundamentalExtractor",
    "build_rag_fundamental_dataset",
    "load_rag_feature_snapshots",
]
