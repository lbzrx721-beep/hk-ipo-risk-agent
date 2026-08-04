"""分批扫描整份招股书并从原文动态归纳证据化风险。"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ipo_risk.agents.analysis import (
    JsonChatProvider,
    JsonResponseDecodeError,
    TextDeltaCallback,
)
from ipo_risk.agents.prompting import prompt_audit, with_correction
from ipo_risk.agents.usage import ModelBudgetExceededError
from ipo_risk.domain import AgentRole, RiskSeverity
from ipo_risk.retrieval import DocumentChunk
from ipo_risk.retrieval.search import RiskRule
from ipo_risk.resilience import RetryExhaustedError, RetryPolicy, retry_model_node_error


ProgressCallback = Callable[[str, str], None]
DiscoveryCorpusSelector = Callable[..., Sequence[DocumentChunk]]
_SPECIALIST_ROLES = frozenset(
    {AgentRole.FINANCE, AgentRole.LEGAL, AgentRole.BUSINESS, AgentRole.MARKET}
)


class OpenRiskCandidate(BaseModel):
    """模型在一个全文批次内提出的原始非标风险。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    title: str = Field(min_length=2, max_length=80)
    owner: AgentRole
    severity: RiskSeverity
    claim: str = Field(min_length=8)
    confidence: float = Field(ge=0.0, le=1.0)
    # 单条风险允许引用多段跨页证据；与全局合并后的上限保持一致。
    # 旧版限制为3条，模型返回第4条时会让整个原文包校验失败。
    evidence_chunk_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    search_queries: tuple[str, ...] = Field(min_length=1, max_length=6)
    anchor_terms: tuple[str, ...] = Field(min_length=1, max_length=6)
    uncertainty: str = Field(min_length=2)
    related_known_risk_type: str | None = None

    @field_validator("owner")
    @classmethod
    def owner_must_be_a_specialist(cls, value: AgentRole) -> AgentRole:
        if value not in _SPECIALIST_ROLES:
            raise ValueError("open risk owner must be a specialist Agent")
        return value

    @model_validator(mode="after")
    def references_and_queries_must_be_unique(self) -> OpenRiskCandidate:
        for values, label in (
            (self.evidence_chunk_ids, "evidence_chunk_ids"),
            (self.search_queries, "search_queries"),
            (self.anchor_terms, "anchor_terms"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must not contain duplicates")
        return self


class DiscoveryBatch(BaseModel):
    """单个全文批次的结构化发现输出。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    candidates: tuple[OpenRiskCandidate, ...] = Field(max_length=40)
    has_more: bool = False
    coverage_note: str = ""
    # 仅供本地容错与审计，不属于要求模型生成的JSON契约。
    validation_warnings: tuple[str, ...] = Field(default=(), exclude=True)

    @model_validator(mode="after")
    def continuation_must_make_progress(self) -> DiscoveryBatch:
        if self.has_more and not self.candidates:
            raise ValueError("has_more=true requires at least one new candidate")
        return self


def _validate_discovery_batch(
    payload: dict[str, Any],
    *,
    allowed_evidence_chunk_ids: set[str],
) -> DiscoveryBatch:
    """逐条校验模型候选，避免一个坏对象报废同批全部合法风险。"""
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, (list, tuple)):
        return DiscoveryBatch.model_validate(payload)

    valid_candidates: list[OpenRiskCandidate] = []
    warnings: list[str] = []
    for index, raw_candidate in enumerate(raw_candidates, start=1):
        title = (
            str(raw_candidate.get("title") or f"候选{index}")
            if isinstance(raw_candidate, dict)
            else f"候选{index}"
        )
        try:
            candidate = OpenRiskCandidate.model_validate(raw_candidate)
            unknown = set(candidate.evidence_chunk_ids) - allowed_evidence_chunk_ids
            if unknown:
                raise ValueError(f"cited unknown evidence chunks: {sorted(unknown)}")
        except (TypeError, ValueError) as exc:
            warnings.append(f"{title}：{exc}")
            continue
        valid_candidates.append(candidate)

    if raw_candidates and not valid_candidates:
        raise ValueError("all discovery candidates failed item validation: " + " | ".join(warnings))

    normalized = {
        "candidates": valid_candidates,
        "has_more": bool(payload.get("has_more", False) or warnings) and bool(valid_candidates),
        "coverage_note": str(payload.get("coverage_note") or ""),
    }
    return DiscoveryBatch.model_validate(normalized).model_copy(
        update={"validation_warnings": tuple(warnings)}
    )


def _salvage_incomplete_discovery_response(content: str) -> dict[str, Any]:
    """从被截断的发现响应中恢复已经闭合的候选对象。"""
    match = re.search(r'"candidates"\s*:\s*\[', content)
    if match is None:
        raise ValueError("incomplete discovery response has no candidates array")

    candidates: list[dict[str, Any]] = []
    object_start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index in range(match.end(), len(content)):
        char = content[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                object_start = index
            depth += 1
            continue
        if char != "}" or depth == 0:
            continue
        depth -= 1
        if depth != 0 or object_start is None:
            continue
        try:
            item = json.loads(content[object_start : index + 1])
        except json.JSONDecodeError:
            object_start = None
            continue
        if isinstance(item, dict):
            candidates.append(item)
        object_start = None

    if not candidates:
        raise ValueError("incomplete discovery response contains no recoverable candidate")
    return {
        "candidates": candidates,
        "has_more": True,
        "coverage_note": "响应尾部未闭合，已恢复此前完整候选并安排续扫。",
    }


class AuditedRiskCandidate(OpenRiskCandidate):
    """带批次来源的原始候选，用于全局合并和完整审计。"""

    candidate_id: str = Field(pattern=r"^R-\d{4}-\d{2}$")
    batch_index: int = Field(ge=1)


class ConsolidatedRiskCandidate(BaseModel):
    """全局语义合并模型输出的一组公司特异风险。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    source_candidate_ids: tuple[str, ...] = Field(min_length=1)
    title: str = Field(min_length=2, max_length=120)
    owner: AgentRole
    severity: RiskSeverity
    claim: str = Field(min_length=8)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_chunk_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    search_queries: tuple[str, ...] = Field(min_length=1, max_length=6)
    anchor_terms: tuple[str, ...] = Field(min_length=1, max_length=6)
    uncertainty: str = Field(min_length=2)
    related_catalog_risk_type: str | None = None

    @field_validator("owner")
    @classmethod
    def owner_must_be_a_specialist(cls, value: AgentRole) -> AgentRole:
        if value not in _SPECIALIST_ROLES:
            raise ValueError("consolidated risk owner must be a specialist Agent")
        return value

    @model_validator(mode="after")
    def references_and_queries_must_be_unique(
        self,
    ) -> ConsolidatedRiskCandidate:
        for values, label in (
            (self.source_candidate_ids, "source_candidate_ids"),
            (self.evidence_chunk_ids, "evidence_chunk_ids"),
            (self.search_queries, "search_queries"),
            (self.anchor_terms, "anchor_terms"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} must not contain duplicates")
        return self


class RiskConsolidationBatch(BaseModel):
    """全文候选的全局语义合并输出。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    consolidated_risks: tuple[ConsolidatedRiskCandidate, ...]


class DiscoveryRetrievalQuery(BaseModel):
    """风险发现前用于定位原文的公司特异检索问题。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)
    query: str = Field(min_length=4, max_length=240)
    reason: str = Field(min_length=2, max_length=240)


class DiscoveryRetrievalPlan(BaseModel):
    """由文档目录和少量样本生成的动态检索计划。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    queries: tuple[DiscoveryRetrievalQuery, ...] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def queries_must_be_unique(self) -> DiscoveryRetrievalPlan:
        normalized = ["".join(item.query.lower().split()) for item in self.queries]
        if len(normalized) != len(set(normalized)):
            raise ValueError("retrieval queries must not contain duplicates")
        return self


class RiskConsolidationMapping(BaseModel):
    """原始批次候选到合并后风险的可审计去向。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    risk_type: str = Field(pattern=r"^open_[0-9a-f]{12}$")
    source_candidate_ids: tuple[str, ...] = Field(min_length=1)


class DiscoveredRisk(BaseModel):
    """去重后的动态风险，可转换为专业 Agent 的证据核验任务。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    risk_type: str = Field(pattern=r"^open_[0-9a-f]{12}$")
    title: str
    owner: AgentRole
    severity: RiskSeverity
    claim: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_chunk_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    search_queries: tuple[str, ...] = Field(min_length=1, max_length=6)
    anchor_terms: tuple[str, ...] = Field(min_length=1, max_length=6)
    uncertainty: str
    source_candidate_ids: tuple[str, ...] = ()
    related_catalog_risk_type: str | None = None

    def to_rule(self) -> RiskRule:
        """转成现有检索、专业分析和决策链能够直接消费的风险规则。"""
        return RiskRule(
            id=self.risk_type,
            name=self.title,
            owner=self.owner,
            severity=self.severity,
            query=self.search_queries[0],
            query_variants=self.search_queries[1:],
            required_keywords=self.anchor_terms,
            claim=self.claim,
            max_evidence=min(8, len(self.evidence_chunk_ids)),
        )


class RiskDiscoveryResult(BaseModel):
    """一次全文发现的结果和可审计运行统计。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    risks: tuple[DiscoveredRisk, ...]
    batch_count: int = Field(ge=0)
    model_attempts: int = Field(ge=0)
    batch_errors: tuple[str, ...] = ()
    batch_warnings: tuple[str, ...] = ()
    raw_candidate_count: int = Field(default=0, ge=0)
    consolidated_candidate_count: int = Field(default=0, ge=0)
    consolidation_mode: str = "not_recorded"
    consolidation_attempts: int = Field(default=0, ge=0)
    consolidation_error: str | None = None
    raw_candidates: tuple[AuditedRiskCandidate, ...] = ()
    consolidation_mappings: tuple[RiskConsolidationMapping, ...] = ()
    discovery_mode: str = "full_text_batches"
    source_chunk_count: int = Field(default=0, ge=0)
    selected_chunk_count: int = Field(default=0, ge=0)
    retrieval_query_count: int = Field(default=0, ge=0)


class RiskDiscoverer(Protocol):
    """文档优先风险发现 Agent 的最小公共接口。"""

    provider_name: str
    model_name: str
    prompt_version: str

    def discover(
        self,
        chunks: Sequence[DocumentChunk],
        known_rules: Sequence[RiskRule],
    ) -> RiskDiscoveryResult: ...


class DeepSeekOpenRiskDiscoverer:
    """让长上下文模型阅读整篇或章节包，并严格引用包内真实 chunk_id。

    类名为兼容已有导入而保留；实际供应商由注入的 ``JsonChatProvider`` 决定。
    """

    provider_name = "deepseek"
    prompt_version = "document-first-risk-discovery-v6"

    def __init__(
        self,
        provider: JsonChatProvider,
        *,
        temperature: float,
        max_output_tokens: int,
        system_prompt: str | None = None,
        consolidation_system_prompt: str | None = None,
        prompt_version: str | None = None,
        progress_callback: ProgressCallback | None = None,
        retry_policy: RetryPolicy | None = None,
        batch_char_limit: int = 70_000,
        wave_size: int = 2,
        max_batch_passes: int = 1,
        consolidation_batch_size: int = 24,
        max_discoveries: int | None = None,
        retrieval_planning_system_prompt: str | None = None,
        retrieval_query_count: int = 10,
        retrieval_top_k: int = 12,
        retrieval_lexical_top_k: int = 6,
        retrieval_neighbor_radius: int = 1,
        retrieval_coverage_sections: int = 24,
        retrieval_max_selected_chunks: int = 360,
    ) -> None:
        if batch_char_limit < 500:
            raise ValueError("batch_char_limit must be at least 500")
        if max_discoveries is not None and max_discoveries < 1:
            raise ValueError("max_discoveries must be positive")
        if max_batch_passes < 1:
            raise ValueError("max_batch_passes must be positive")
        if wave_size < 1 or wave_size > 4:
            raise ValueError("wave_size must be between 1 and 4")
        if consolidation_batch_size < 2:
            raise ValueError("consolidation_batch_size must be at least 2")
        if retrieval_query_count < 1 or retrieval_query_count > 20:
            raise ValueError("retrieval_query_count must be between 1 and 20")
        if retrieval_top_k < 1 or retrieval_lexical_top_k < 1:
            raise ValueError("retrieval top-k limits must be positive")
        if retrieval_neighbor_radius < 0 or retrieval_neighbor_radius > 3:
            raise ValueError("retrieval_neighbor_radius must be between 0 and 3")
        if retrieval_coverage_sections < 1:
            raise ValueError("retrieval_coverage_sections must be positive")
        if retrieval_max_selected_chunks < 20:
            raise ValueError("retrieval_max_selected_chunks must be at least 20")
        self.provider = provider
        self.provider_name = str(getattr(provider, "provider_name", "deepseek"))
        self.model_name = provider.model_name
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._progress_callback = progress_callback
        self._retry_policy = retry_policy or RetryPolicy(retry_if=retry_model_node_error)
        self._batch_char_limit = batch_char_limit
        self._wave_size = wave_size
        self._max_batch_passes = max_batch_passes
        self._consolidation_batch_size = consolidation_batch_size
        self._retrieval_query_count = retrieval_query_count
        self._retrieval_top_k = retrieval_top_k
        self._retrieval_lexical_top_k = retrieval_lexical_top_k
        self._retrieval_neighbor_radius = retrieval_neighbor_radius
        self._retrieval_coverage_sections = retrieval_coverage_sections
        self._retrieval_max_selected_chunks = retrieval_max_selected_chunks
        if prompt_version is not None:
            self.prompt_version = prompt_version
        self._system_prompt = system_prompt or (
            "你是港股IPO招股书文档优先风险发现Agent。不得预设行业或风险主题，必须"
            "从用户提供的当前公司原文中归纳风险类型、标题和归属角色。不得使用外部"
            "知识或编造证据ID。必须合并语义重复事项，并检查终止、解除、改善和替代"
            "安排等反证。只输出JSON对象，字段为candidates。"
        )
        self._consolidation_system_prompt = consolidation_system_prompt or (
            "你是港股IPO风险候选全局合并Agent。输入是同一份招股书各批次已经"
            "通过证据白名单校验的原始风险候选。只合并实质上描述同一个底层风险"
            "的候选，不得因为数量多而合并不同风险，也不得删除任何候选；每个"
            "candidate_id必须且只能出现在一个source_candidate_ids中。固定风险"
            "目录只用于标记语义重合，不得反向添加文档没有披露的主题。只能引用"
            "来源候选已经引用的evidence_chunk_ids。只输出JSON对象，字段为"
            "consolidated_risks。"
        )
        self._retrieval_planning_system_prompt = retrieval_planning_system_prompt or (
            "你是港股IPO招股书风险发现的检索规划Agent。根据当前公司的目录和"
            "少量原文样本，生成能够定位重大不确定性、限制条件、依赖关系、"
            "异常变化及其反证的公司特异检索问题。不得套用固定风险目录，不得"
            "臆造文档事实。只输出JSON对象，字段为queries，每项包含query和reason。"
        )

    def discovery_prompt_audit(self) -> dict[str, str]:
        """返回全文发现提示词的版本、内容指纹和运行快照。"""
        return prompt_audit(
            prompt=self._system_prompt,
            version=self.prompt_version,
        )

    def consolidation_prompt_audit(self) -> dict[str, str]:
        """返回全局语义合并提示词的可审计快照。"""
        return prompt_audit(
            prompt=self._consolidation_system_prompt,
            version="risk-candidate-consolidation-v3",
        )

    def retrieval_prompt_audit(self) -> dict[str, str]:
        """返回动态检索规划提示词的可审计快照。"""
        return prompt_audit(
            prompt=self._retrieval_planning_system_prompt,
            version="vector-risk-query-planning-v2",
        )

    def discover_with_retrieval(
        self,
        chunks: Sequence[DocumentChunk],
        known_rules: Sequence[RiskRule],
        *,
        corpus_selector: DiscoveryCorpusSelector,
    ) -> RiskDiscoveryResult:
        """先用向量定位原文，再沿用原结构化风险发现与合并链路。"""
        if not chunks:
            return self.discover(chunks, known_rules)

        if self._progress_callback is not None:
            self._progress_callback(
                "agent:discovery:retrieval_planning",
                "风险发现正在根据招股书目录与样本生成动态原文检索计划。",
            )

        plan_attempts = 0
        try:
            outcome = self._retry_policy.run_with_feedback(
                lambda correction_feedback: self._plan_retrieval_queries(
                    chunks,
                    correction_feedback=correction_feedback,
                )
            )
            plan_attempts = outcome.attempts
            queries = tuple(item.query for item in outcome.value.queries)
        except RetryExhaustedError as error:
            if isinstance(error.last_error, ModelBudgetExceededError):
                raise error.last_error
            plan_attempts = error.attempts
            queries = self._fallback_retrieval_queries(chunks)
            if self._progress_callback is not None:
                self._progress_callback(
                    "agent:discovery:retrieval_plan_fallback",
                    "动态检索计划生成失败，已改用文档目录与样本构造检索问题。",
                )

        query_candidates = list(dict.fromkeys(queries))
        for fallback_query in self._fallback_retrieval_queries(chunks):
            if len(query_candidates) >= self._retrieval_query_count:
                break
            if fallback_query not in query_candidates:
                query_candidates.append(fallback_query)
        queries = tuple(query_candidates[: self._retrieval_query_count])
        selected_chunks = tuple(
            corpus_selector(
                queries,
                semantic_top_k_per_query=self._retrieval_top_k,
                lexical_top_k_per_query=self._retrieval_lexical_top_k,
                neighbor_radius=self._retrieval_neighbor_radius,
                coverage_section_count=self._retrieval_coverage_sections,
                max_chunks=self._retrieval_max_selected_chunks,
            )
        )
        if not selected_chunks:
            selected_chunks = tuple(chunks[: self._retrieval_max_selected_chunks])

        if self._progress_callback is not None:
            self._progress_callback(
                "agent:discovery:retrieval_ready",
                (
                    f"已用{len(queries)}个动态问题从{len(chunks)}个全文切片中筛选"
                    f"{len(selected_chunks)}个原文切片，并补齐相邻上下文与章节覆盖。"
                ),
            )

        result = self.discover(selected_chunks, known_rules)
        return result.model_copy(
            update={
                "model_attempts": result.model_attempts + plan_attempts,
                "discovery_mode": "vector_guided_original_text",
                "source_chunk_count": len(chunks),
                "selected_chunk_count": len(selected_chunks),
                "retrieval_query_count": len(queries),
            }
        )

    def _plan_retrieval_queries(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        correction_feedback: str | None,
    ) -> DiscoveryRetrievalPlan:
        document_map = self._document_map(chunks)
        payload = with_correction(
            {
                "document_identity": {
                    "company_id": chunks[0].company_id,
                    "document_id": chunks[0].document_id,
                    "source_file": chunks[0].source_file,
                },
                "document_outline_and_samples": document_map,
                "constraints": {
                    "target_query_count": self._retrieval_query_count,
                    "derive_queries_from_current_document": True,
                    "do_not_assume_fixed_risk_taxonomy": True,
                    "query_for_risk_evidence_and_possible_counterevidence": True,
                    "no_external_facts": True,
                },
            },
            correction_feedback,
        )
        return DiscoveryRetrievalPlan.model_validate(
            self.provider.complete_json(
                system_prompt=self._retrieval_planning_system_prompt,
                user_prompt=json.dumps(payload, ensure_ascii=False),
                temperature=self._temperature,
                max_output_tokens=min(self._max_output_tokens, 4096),
            )
        )

    def _document_map(
        self,
        chunks: Sequence[DocumentChunk],
    ) -> list[dict[str, object]]:
        headings = [chunk for chunk in chunks if chunk.is_heading]
        selected: list[DocumentChunk] = headings[:120]
        sample_count = min(32, len(chunks))
        if sample_count:
            denominator = max(1, sample_count - 1)
            for index in range(sample_count):
                chunk = chunks[round(index * (len(chunks) - 1) / denominator)]
                if chunk.chunk_id not in {item.chunk_id for item in selected}:
                    selected.append(chunk)
        return [
            {
                "chunk_id": chunk.chunk_id,
                "physical_page": chunk.page,
                "is_heading": chunk.is_heading,
                "source_kind": chunk.chunk_kind,
                "table_id": chunk.table_id,
                "text": chunk.text[:320],
            }
            for chunk in selected
        ]

    def _fallback_retrieval_queries(
        self,
        chunks: Sequence[DocumentChunk],
    ) -> tuple[str, ...]:
        candidates = [
            chunk.text.strip()
            for chunk in chunks
            if chunk.is_heading and len(chunk.text.strip()) >= 4
        ]
        if not candidates:
            candidates = [
                chunk.text.strip()[:160]
                for chunk in self._evenly_sampled_chunks(chunks, self._retrieval_query_count)
            ]
        queries = tuple(dict.fromkeys(candidates))
        if queries:
            return queries[: self._retrieval_query_count]
        return ("本公司招股书披露的重大不确定性、限制条件及相关反证",)

    @staticmethod
    def _evenly_sampled_chunks(
        chunks: Sequence[DocumentChunk],
        count: int,
    ) -> tuple[DocumentChunk, ...]:
        count = min(count, len(chunks))
        if count == 0:
            return ()
        denominator = max(1, count - 1)
        return tuple(
            chunks[round(index * (len(chunks) - 1) / denominator)] for index in range(count)
        )

    def discover(
        self,
        chunks: Sequence[DocumentChunk],
        known_rules: Sequence[RiskRule],
    ) -> RiskDiscoveryResult:
        batches = _build_batches(chunks, self._batch_char_limit)
        catalog_definitions = [
            {
                "id": rule.id,
                "name": rule.name,
                "owner": rule.owner.value,
                "claim": rule.claim,
            }
            for rule in known_rules
        ]
        candidate_records: list[AuditedRiskCandidate] = []
        errors: list[str] = []
        warnings: list[str] = []
        attempts = 0

        def scan_batch(
            index: int,
            batch: Sequence[DocumentChunk],
            prior_records: tuple[AuditedRiskCandidate, ...],
        ) -> tuple[
            int,
            list[AuditedRiskCandidate],
            list[str],
            list[str],
            int,
        ]:
            batch_candidates: list[OpenRiskCandidate] = []
            seen_candidates: set[tuple[str, tuple[str, ...]]] = set()
            batch_errors: list[str] = []
            batch_warnings: list[str] = []
            batch_attempts = 0
            for pass_index in range(1, self._max_batch_passes + 1):
                if self._progress_callback is not None:
                    pass_text = "" if pass_index == 1 else f"（异常恢复第{pass_index}轮）"
                    self._progress_callback(
                        "agent:discovery:prompt",
                        (
                            f"[{index}/{len(batches)}] DeepSeek正在分析"
                            f"风险原文包{pass_text}（每轮并行"
                            f"{min(self._wave_size, len(batches))}批）。"
                        ),
                    )
                try:
                    outcome = self._retry_policy.run_with_feedback(
                        lambda correction_feedback: self._discover_batch(
                            batch,
                            batch_index=index,
                            batch_count=len(batches),
                            prior_document_candidates=prior_records,
                            previous_candidates=batch_candidates,
                            pass_index=pass_index,
                            correction_feedback=correction_feedback,
                        )
                    )
                    batch_attempts += outcome.attempts
                    batch_warnings.extend(outcome.value.validation_warnings)
                except RetryExhaustedError as error:
                    if isinstance(error.last_error, ModelBudgetExceededError):
                        raise error.last_error
                    batch_attempts += error.attempts
                    batch_errors.append(f"batch {index} pass {pass_index}: {error.last_error}")
                    if self._progress_callback is not None:
                        self._progress_callback(
                            "agent:discovery:batch_failed",
                            (
                                f"风险原文包{index}/{len(batches)}分析失败，"
                                f"已重试{max(0, error.attempts - 1)}次："
                                f"{error.last_error}"
                            ),
                        )
                    break

                if self._progress_callback is not None:
                    self._progress_callback(
                        "agent:discovery:structured",
                        (
                            f"[[batch:{index}/{len(batches)}]]"
                            + json.dumps(
                                outcome.value.model_dump(mode="json"),
                                ensure_ascii=False,
                            )
                        ),
                    )

                for candidate in outcome.value.candidates:
                    if candidate.related_known_risk_type is not None:
                        continue
                    identity = (
                        "".join(candidate.title.lower().split()),
                        tuple(candidate.evidence_chunk_ids),
                    )
                    if identity in seen_candidates:
                        continue
                    seen_candidates.add(identity)
                    batch_candidates.append(candidate)
                if not outcome.value.has_more:
                    break
                if pass_index == self._max_batch_passes:
                    batch_warnings.append(
                        f"pack {index}: continuation disabled after "
                        f"{self._max_batch_passes} pass; preserved "
                        f"{len(batch_candidates)} candidates"
                    )

            records = [
                AuditedRiskCandidate(
                    **candidate.model_dump(),
                    candidate_id=f"R-{index:04d}-{item_index:02d}",
                    batch_index=index,
                )
                for item_index, candidate in enumerate(
                    batch_candidates,
                    start=1,
                )
            ]
            return index, records, batch_errors, batch_warnings, batch_attempts

        completed_batches = 0
        for wave_start in range(0, len(batches), self._wave_size):
            wave = list(
                enumerate(
                    batches[wave_start : wave_start + self._wave_size],
                    start=wave_start + 1,
                )
            )
            prior_snapshot = tuple(candidate_records)
            wave_results: list[
                tuple[int, list[AuditedRiskCandidate], list[str], list[str], int]
            ] = []
            with ThreadPoolExecutor(
                max_workers=len(wave),
                thread_name_prefix="risk-discovery",
            ) as executor:
                futures = [
                    executor.submit(scan_batch, index, batch, prior_snapshot)
                    for index, batch in wave
                ]
                for future in as_completed(futures):
                    wave_results.append(future.result())

            for index, new_records, batch_errors, batch_warnings, batch_attempts in sorted(
                wave_results,
                key=lambda item: item[0],
            ):
                attempts += batch_attempts
                errors.extend(batch_errors)
                warnings.extend(batch_warnings)
                candidate_records.extend(new_records)
                completed_batches += 1
                if self._progress_callback is not None:
                    self._progress_callback(
                        "agent:discovery:batch_completed",
                        (
                            f"[{completed_batches}/{len(batches)}] 风险原文包{index}新增"
                            f"{len(new_records)}条，累计{len(candidate_records)}条风险候选。"
                        ),
                    )

        if self._progress_callback is not None:
            self._progress_callback(
                "agent:discovery:raw_completed",
                (
                    f"{len(batches)}个风险原文包共形成"
                    f"{len(candidate_records)}条原始候选，开始全局语义合并。"
                ),
            )

        consolidation_attempts = 0
        consolidation_error: str | None = None
        if not candidate_records:
            risks: list[DiscoveredRisk] = []
            mappings: list[RiskConsolidationMapping] = []
            consolidation_mode = "not_needed"
        else:
            if self._progress_callback is not None:
                self._progress_callback(
                    "agent:discovery:prompt",
                    "正在对全部原始候选进行全局语义合并与目录重合检查。",
                )
            (
                risks,
                mappings,
                consolidation_mode,
                consolidation_attempts,
                consolidation_error,
            ) = self._consolidate_all_candidates(
                candidate_records,
                catalog_definitions,
                known_catalog_types={rule.id for rule in known_rules},
            )

        attempts += consolidation_attempts
        if self._progress_callback is not None:
            mode_text = {
                "semantic_model": "大模型语义合并",
                "semantic_model_partitioned": "分区大模型语义合并",
                "partitioned_with_exact_title_fallback": ("分区语义合并与局部精确标题兜底"),
                "exact_title_fallback": "精确标题兜底（未丢弃候选）",
                "not_needed": "无需合并",
            }.get(consolidation_mode, consolidation_mode)
            self._progress_callback(
                "agent:discovery:consolidated",
                (f"{len(candidate_records)}条原始候选经{mode_text}形成{len(risks)}条待确认风险。"),
            )
        return RiskDiscoveryResult(
            risks=tuple(risks),
            batch_count=len(batches),
            model_attempts=attempts,
            batch_errors=tuple(errors),
            batch_warnings=tuple(warnings),
            raw_candidate_count=len(candidate_records),
            consolidated_candidate_count=len(risks),
            consolidation_mode=consolidation_mode,
            consolidation_attempts=consolidation_attempts,
            consolidation_error=consolidation_error,
            raw_candidates=tuple(candidate_records),
            consolidation_mappings=tuple(mappings),
            source_chunk_count=len(chunks),
            selected_chunk_count=len(chunks),
        )

    def _discover_batch(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        batch_index: int,
        batch_count: int,
        prior_document_candidates: Sequence[AuditedRiskCandidate],
        previous_candidates: Sequence[OpenRiskCandidate],
        pass_index: int,
        correction_feedback: str | None,
    ) -> DiscoveryBatch:
        allowed = [chunk.chunk_id for chunk in chunks]
        first_chunk = chunks[0]
        payload = with_correction(
            {
                "document_identity": {
                    "company_id": first_chunk.company_id,
                    "document_id": first_chunk.document_id,
                    "source_file": first_chunk.source_file,
                    "available_at": first_chunk.available_at.isoformat(),
                },
                "document_scan_state": {
                    "batch_index": batch_index,
                    "batch_count": batch_count,
                    "prior_candidate_count": len(prior_document_candidates),
                },
                "prior_document_candidates": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "title": candidate.title,
                        "owner": candidate.owner.value,
                        "severity": candidate.severity.value,
                        "claim": candidate.claim[:240],
                        "evidence_chunk_ids": list(candidate.evidence_chunk_ids),
                    }
                    for candidate in prior_document_candidates
                ],
                "scan_pass": pass_index,
                "already_returned_candidates": [
                    {
                        "title": candidate.title,
                        "claim": candidate.claim,
                        "evidence_chunk_ids": list(candidate.evidence_chunk_ids),
                    }
                    for candidate in previous_candidates
                ],
                "constraints": {
                    "allowed_evidence_chunk_ids": allowed,
                    "derive_topics_from_document": True,
                    "do_not_assume_fixed_taxonomy": True,
                    "max_candidates_this_pass": 40,
                    "set_has_more_when_material_risks_remain": (self._max_batch_passes > 1),
                    "do_not_repeat_already_returned_candidates": True,
                    "prior_document_candidates_are_memory_not_evidence": True,
                    "return_existing_theme_only_when_current_batch_adds_new_evidence": True,
                    "reuse_prior_title_for_same_underlying_risk": True,
                    "no_external_evidence": True,
                },
                "chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "physical_page": chunk.page,
                        "printed_page_label": chunk.printed_page_label,
                        "source_kind": chunk.chunk_kind,
                        "table_id": chunk.table_id,
                        "table_cell_refs": list(chunk.table_cell_refs),
                        "table_cells": dict(
                            zip(
                                chunk.table_cell_refs,
                                chunk.table_cell_values,
                                strict=True,
                            )
                        ),
                        "text": chunk.text,
                    }
                    for chunk in chunks
                ],
            },
            correction_feedback,
        )
        on_text_delta: TextDeltaCallback | None = None
        if self._progress_callback is not None:

            def report_delta(delta: str) -> None:
                assert self._progress_callback is not None
                self._progress_callback(
                    "agent:discovery:delta",
                    f"[[batch:{batch_index}/{batch_count}]]{delta}",
                )

            on_text_delta = report_delta
        request: dict[str, Any] = {
            "system_prompt": self._system_prompt,
            "user_prompt": json.dumps(payload, ensure_ascii=False),
            "temperature": self._temperature,
            "max_output_tokens": self._max_output_tokens,
        }
        if on_text_delta is not None:
            request["on_text_delta"] = on_text_delta
        allowed_set = set(allowed)
        recovery_warning: str | None = None
        try:
            raw_response = self.provider.complete_json(**request)
        except JsonResponseDecodeError as exc:
            raw_response = _salvage_incomplete_discovery_response(exc.raw_content)
            recovery_warning = "模型响应尾部未闭合；已保留此前完整候选，并通过下一轮补齐覆盖。"
        result = _validate_discovery_batch(
            raw_response,
            allowed_evidence_chunk_ids=allowed_set,
        )
        if recovery_warning is not None:
            result = result.model_copy(
                update={
                    "validation_warnings": (
                        *result.validation_warnings,
                        recovery_warning,
                    )
                }
            )
        return result

    def _consolidate_candidates(
        self,
        candidates: Sequence[AuditedRiskCandidate],
        catalog_definitions: Sequence[dict[str, str]],
        *,
        correction_feedback: str | None,
        partition_index: int,
        partition_count: int,
    ) -> RiskConsolidationBatch:
        payload = with_correction(
            {
                "partition": {
                    "index": partition_index,
                    "count": partition_count,
                    "candidate_count": len(candidates),
                },
                "candidate_records": [
                    candidate.model_dump(mode="json") for candidate in candidates
                ],
                "catalog_risk_definitions": list(catalog_definitions),
                "constraints": {
                    "map_every_candidate_exactly_once": True,
                    "merge_only_same_underlying_risk": True,
                    "preserve_materially_distinct_risks": True,
                    "catalog_is_matching_reference_only": True,
                    "no_silent_truncation": True,
                    "max_evidence_per_risk": 8,
                },
            },
            correction_feedback,
        )
        on_text_delta: TextDeltaCallback | None = None
        if self._progress_callback is not None:

            def report_delta(delta: str) -> None:
                assert self._progress_callback is not None
                self._progress_callback(
                    "agent:discovery:delta",
                    (f"[[consolidation:{partition_index}/{partition_count}]]{delta}"),
                )

            on_text_delta = report_delta
        request: dict[str, Any] = {
            "system_prompt": self._consolidation_system_prompt,
            "user_prompt": json.dumps(payload, ensure_ascii=False),
            "temperature": self._temperature,
            "max_output_tokens": self._max_output_tokens,
        }
        if on_text_delta is not None:
            request["on_text_delta"] = on_text_delta
        result = RiskConsolidationBatch.model_validate(self.provider.complete_json(**request))
        if self._progress_callback is not None:
            self._progress_callback(
                "agent:discovery:structured",
                (
                    f"[[consolidation:{partition_index}/{partition_count}]]"
                    + json.dumps(
                        result.model_dump(mode="json"),
                        ensure_ascii=False,
                    )
                ),
            )
        return result

    def _consolidate_all_candidates(
        self,
        candidates: Sequence[AuditedRiskCandidate],
        catalog_definitions: Sequence[dict[str, str]],
        *,
        known_catalog_types: set[str],
    ) -> tuple[
        list[DiscoveredRisk],
        list[RiskConsolidationMapping],
        str,
        int,
        str | None,
    ]:
        """按角色和容量分区合并，避免一个超长请求吞掉大量候选。"""
        partitions: list[list[AuditedRiskCandidate]] = []
        by_owner: dict[AgentRole, list[AuditedRiskCandidate]] = {}
        for candidate in candidates:
            by_owner.setdefault(candidate.owner, []).append(candidate)
        if len(candidates) <= self._consolidation_batch_size:
            partitions = [list(candidates)]
        else:
            for owner in sorted(by_owner, key=lambda item: item.value):
                owned = by_owner[owner]
                for start in range(0, len(owned), self._consolidation_batch_size):
                    partitions.append(owned[start : start + self._consolidation_batch_size])

        risks: list[DiscoveredRisk] = []
        mappings: list[RiskConsolidationMapping] = []
        errors: list[str] = []
        attempts = 0
        semantic_partitions = 0
        for index, partition in enumerate(partitions, start=1):
            try:
                outcome = self._retry_policy.run_with_feedback(
                    lambda correction_feedback: _risks_from_consolidation(
                        self._consolidate_candidates(
                            partition,
                            catalog_definitions,
                            correction_feedback=correction_feedback,
                            partition_index=index,
                            partition_count=len(partitions),
                        ),
                        partition,
                        known_catalog_types=known_catalog_types,
                    )
                )
                attempts += outcome.attempts
                partition_risks, partition_mappings = outcome.value
                semantic_partitions += 1
            except RetryExhaustedError as error:
                if isinstance(error.last_error, ModelBudgetExceededError):
                    raise error.last_error
                attempts += error.attempts
                errors.append(f"partition {index}: {error.last_error}")
                partition_risks, partition_mappings = _deduplicate_candidates(partition)
            risks.extend(partition_risks)
            mappings.extend(partition_mappings)

        ordered = sorted(
            zip(risks, mappings, strict=True),
            key=lambda pair: (-pair[0].confidence, pair[0].risk_type),
        )
        risks = [pair[0] for pair in ordered]
        mappings = [pair[1] for pair in ordered]
        if not errors and len(partitions) == 1:
            mode = "semantic_model"
        elif not errors:
            mode = "semantic_model_partitioned"
        elif semantic_partitions:
            mode = "partitioned_with_exact_title_fallback"
        else:
            mode = "exact_title_fallback"
        return (
            risks,
            mappings,
            mode,
            attempts,
            " | ".join(errors) if errors else None,
        )


def _build_batches(
    chunks: Sequence[DocumentChunk],
    char_limit: int,
) -> list[tuple[DocumentChunk, ...]]:
    batches: list[tuple[DocumentChunk, ...]] = []
    current: list[DocumentChunk] = []
    current_chars = 0
    for chunk in chunks:
        size = len(chunk.text)
        if current and current_chars + size > char_limit:
            batches.append(tuple(current))
            current = []
            current_chars = 0
        current.append(chunk)
        current_chars += size
    if current:
        batches.append(tuple(current))
    return batches


def _deduplicate_candidates(
    candidates: Sequence[AuditedRiskCandidate],
) -> tuple[list[DiscoveredRisk], list[RiskConsolidationMapping]]:
    """模型不可用时只合并标题完全一致的候选，绝不截断或猜测语义。"""
    by_key: dict[tuple[AgentRole, str], list[AuditedRiskCandidate]] = {}
    for candidate in candidates:
        normalized = "".join(candidate.title.lower().split())
        key = (candidate.owner, normalized)
        by_key.setdefault(key, []).append(candidate)
    risks: list[DiscoveredRisk] = []
    mappings: list[RiskConsolidationMapping] = []
    severity_rank = {
        RiskSeverity.LOW: 0,
        RiskSeverity.MEDIUM: 1,
        RiskSeverity.HIGH: 2,
        RiskSeverity.CRITICAL: 3,
    }
    for (owner, normalized), group in by_key.items():
        candidate = max(group, key=lambda item: item.confidence)
        ordered_group = [
            candidate,
            *sorted(
                (item for item in group if item is not candidate),
                key=lambda item: -item.confidence,
            ),
        ]
        evidence_ids = _unique_strings(
            value for item in ordered_group for value in item.evidence_chunk_ids
        )[:8]
        search_queries = _unique_strings(
            value for item in ordered_group for value in item.search_queries
        )[:6]
        anchor_terms = _unique_strings(
            value for item in ordered_group for value in item.anchor_terms
        )[:6]
        severity = max(group, key=lambda item: severity_rank[item.severity]).severity
        digest = sha256(f"{owner.value}\0{normalized}".encode("utf-8")).hexdigest()[:12]
        risks.append(
            DiscoveredRisk(
                risk_type=f"open_{digest}",
                title=candidate.title,
                owner=candidate.owner,
                severity=severity,
                claim=candidate.claim,
                confidence=candidate.confidence,
                evidence_chunk_ids=evidence_ids,
                search_queries=search_queries,
                anchor_terms=anchor_terms,
                uncertainty=candidate.uncertainty,
                source_candidate_ids=tuple(item.candidate_id for item in ordered_group),
            )
        )
        mappings.append(
            RiskConsolidationMapping(
                risk_type=f"open_{digest}",
                source_candidate_ids=tuple(item.candidate_id for item in ordered_group),
            )
        )
    risk_order = sorted(
        range(len(risks)),
        key=lambda index: (-risks[index].confidence, risks[index].risk_type),
    )
    return (
        [risks[index] for index in risk_order],
        [mappings[index] for index in risk_order],
    )


def _risks_from_consolidation(
    batch: RiskConsolidationBatch,
    candidates: Sequence[AuditedRiskCandidate],
    *,
    known_catalog_types: set[str],
) -> tuple[list[DiscoveredRisk], list[RiskConsolidationMapping]]:
    """验证全局模型没有丢候选或越界引用，再生成稳定的动态风险。"""
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    expected_ids = set(candidate_by_id)
    returned_ids = [
        candidate_id
        for risk in batch.consolidated_risks
        for candidate_id in risk.source_candidate_ids
    ]
    unknown_ids = set(returned_ids) - expected_ids
    missing_ids = expected_ids - set(returned_ids)
    duplicate_ids = sorted(
        {candidate_id for candidate_id in returned_ids if returned_ids.count(candidate_id) > 1}
    )
    if unknown_ids or missing_ids or duplicate_ids:
        raise ValueError(
            "consolidation candidate mapping invalid: "
            f"unknown={sorted(unknown_ids)}, missing={sorted(missing_ids)}, "
            f"duplicates={duplicate_ids}"
        )

    risks: list[DiscoveredRisk] = []
    mappings: list[RiskConsolidationMapping] = []
    seen_digests: set[str] = set()
    for consolidated in batch.consolidated_risks:
        sources = [
            candidate_by_id[candidate_id] for candidate_id in consolidated.source_candidate_ids
        ]
        allowed_evidence = {
            chunk_id for source in sources for chunk_id in source.evidence_chunk_ids
        }
        unknown_evidence = set(consolidated.evidence_chunk_ids) - allowed_evidence
        if unknown_evidence:
            raise ValueError(
                "consolidation cited evidence outside source candidates: "
                f"{sorted(unknown_evidence)}"
            )
        related_catalog = consolidated.related_catalog_risk_type
        if related_catalog is not None and related_catalog not in known_catalog_types:
            raise ValueError(f"consolidation cited unknown catalog risk type: {related_catalog}")

        normalized_title = "".join(consolidated.title.lower().split())
        identity = (
            f"{consolidated.owner.value}\0{normalized_title}\0"
            f"{'|'.join(sorted(consolidated.source_candidate_ids))}"
        )
        digest = sha256(identity.encode("utf-8")).hexdigest()[:12]
        if digest in seen_digests:
            raise ValueError("consolidation produced duplicate risk identity")
        seen_digests.add(digest)
        risk_type = f"open_{digest}"
        risks.append(
            DiscoveredRisk(
                risk_type=risk_type,
                title=consolidated.title,
                owner=consolidated.owner,
                severity=consolidated.severity,
                claim=consolidated.claim,
                confidence=consolidated.confidence,
                evidence_chunk_ids=consolidated.evidence_chunk_ids,
                search_queries=consolidated.search_queries,
                anchor_terms=consolidated.anchor_terms,
                uncertainty=consolidated.uncertainty,
                source_candidate_ids=consolidated.source_candidate_ids,
                related_catalog_risk_type=related_catalog,
            )
        )
        mappings.append(
            RiskConsolidationMapping(
                risk_type=risk_type,
                source_candidate_ids=consolidated.source_candidate_ids,
            )
        )

    risk_order = sorted(
        range(len(risks)),
        key=lambda index: (-risks[index].confidence, risks[index].risk_type),
    )
    return (
        [risks[index] for index in risk_order],
        [mappings[index] for index in risk_order],
    )


def _unique_strings(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


__all__ = [
    "AuditedRiskCandidate",
    "ConsolidatedRiskCandidate",
    "DeepSeekOpenRiskDiscoverer",
    "DiscoveryRetrievalPlan",
    "DiscoveryRetrievalQuery",
    "DiscoveredRisk",
    "OpenRiskCandidate",
    "RiskConsolidationMapping",
    "RiskDiscoverer",
    "RiskDiscoveryResult",
]
