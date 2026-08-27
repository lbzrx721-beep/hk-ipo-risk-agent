"""专业 Agent 的角色内批量证据核验。

四个专业角色由 LangGraph 并行；每个角色内部把全部已分派风险按固定容量打包，一次模型调用
核验一组风险。只有模型明确要求补证的风险才允许再检索一轮，避免“一条风险一次调用”。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ipo_risk.agents.analysis import JsonChatProvider, TextDeltaCallback
from ipo_risk.agents.prompting import compose_prompt, prompt_audit, with_correction
from ipo_risk.agents.usage import ModelBudgetExceededError
from ipo_risk.domain import AgentRole, RiskSeverity
from ipo_risk.retrieval.chunker import DocumentChunk
from ipo_risk.retrieval.search import RiskRule
from ipo_risk.resilience import (
    RetryExhaustedError,
    RetryPolicy,
    retry_model_node_error,
    retry_transient_api_error,
)


ProgressCallback = Callable[[str, str], None]


class SpecialistStatus(StrEnum):
    """机器核验状态；没有“驳回”，最终驳回只属于人工复核。"""

    SUPPORTED = "supported"
    NEEDS_MORE_EVIDENCE = "needs_more_evidence"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    SUBJECT_OR_PERIOD_MISMATCH = "subject_or_period_mismatch"
    DUPLICATE_MERGED = "duplicate_merged"
    CONFLICT = "conflict"
    TECHNICAL_FAILURE = "technical_failure"


@dataclass(frozen=True)
class SpecialistTask:
    """一条风险及本轮允许引用的候选原文。"""

    rule: RiskRule
    candidates: tuple[DocumentChunk, ...]


class SpecialistAssessment(BaseModel):
    """专业 Agent 对一条风险的非删除式核验结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    risk_type: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    status: SpecialistStatus
    claim: str = ""
    severity: RiskSeverity
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_chunk_ids: tuple[str, ...] = ()
    uncertainty: str = Field(min_length=2)
    followup_queries: tuple[str, ...] = Field(default=(), max_length=4)
    duplicate_of: str | None = None

    @field_validator("evidence_chunk_ids", "followup_queries")
    @classmethod
    def values_must_be_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("assessment references and queries must be unique")
        return value

    @model_validator(mode="after")
    def validate_status_contract(self) -> SpecialistAssessment:
        if self.status is SpecialistStatus.SUPPORTED:
            if not self.claim or not self.evidence_chunk_ids:
                raise ValueError("supported assessment requires claim and evidence")
        elif self.claim:
            raise ValueError("non-supported assessment cannot publish a final claim")
        if self.status is SpecialistStatus.NEEDS_MORE_EVIDENCE and not self.followup_queries:
            raise ValueError("needs_more_evidence requires followup_queries")
        if self.status is SpecialistStatus.DUPLICATE_MERGED and not self.duplicate_of:
            raise ValueError("duplicate_merged requires duplicate_of")
        if self.status is not SpecialistStatus.DUPLICATE_MERGED and self.duplicate_of:
            raise ValueError("duplicate_of is only valid for duplicate_merged")
        return self


class SpecialistAssessmentBatch(BaseModel):
    """一次角色批量调用的结构化输出。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    assessments: tuple[SpecialistAssessment, ...]

    @model_validator(mode="after")
    def risk_types_must_be_unique(self) -> SpecialistAssessmentBatch:
        values = [item.risk_type for item in self.assessments]
        if len(values) != len(set(values)):
            raise ValueError("specialist batch risk_types must be unique")
        return self


@dataclass(frozen=True)
class SpecialistBatchResult:
    """所有批次合并后的核验结果与调用审计。"""

    assessments: tuple[SpecialistAssessment, ...]
    attempts_by_risk: Mapping[str, int]
    errors_by_risk: Mapping[str, tuple[str, ...]]

    def by_risk(self) -> dict[str, SpecialistAssessment]:
        return {item.risk_type: item for item in self.assessments}


class SpecialistBatchAnalyzer(Protocol):
    """角色内批量核验模块的接口。"""

    provider_name: str
    model_name: str
    prompt_version: str

    def prompt_audit(self, role: AgentRole) -> dict[str, str]:
        """返回角色实际提示词的版本、指纹和快照。"""
        ...

    def analyze_batch(self, tasks: Sequence[SpecialistTask]) -> SpecialistBatchResult:
        """核验全部任务；任务可以拆请求，但不得因容量上限丢失。"""
        ...


class DeepSeekSpecialistBatchAnalyzer:
    """OpenAI兼容模型的专业角色批量核验适配器。"""

    prompt_version = "specialist-evidence-batch-v3"

    def __init__(
        self,
        provider: JsonChatProvider,
        *,
        temperature: float,
        max_output_tokens: int,
        batch_size: int = 20,
        system_prompt: str | None = None,
        role_prompts: Mapping[AgentRole, str] | None = None,
        providers_by_role: Mapping[AgentRole, JsonChatProvider] | None = None,
        prompt_version: str | None = None,
        progress_callback: ProgressCallback | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        if batch_size < 2:
            raise ValueError("specialist batch_size must be at least 2")
        self.provider = provider
        self.provider_name = str(getattr(provider, "provider_name", "deepseek"))
        self.model_name = provider.model_name
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._batch_size = batch_size
        self._progress_callback = progress_callback
        self._retry_policy = retry_policy or RetryPolicy(retry_if=retry_model_node_error)
        self._role_prompts = dict(role_prompts or {})
        self._providers_by_role = dict(providers_by_role or {})
        if prompt_version is not None:
            self.prompt_version = prompt_version
        self._system_prompt = system_prompt or (
            "你是港股IPO招股书专业证据核验Agent。一次输入包含同一专业角色负责的多条"
            "风险。你必须逐条核验并完整返回，不能因为数量多而遗漏。只能引用每条风险"
            "各自allowed_evidence_chunk_ids。机器只标记支持、需补证、证据不足、主体或"
            "期间错配、重复合并或冲突，不得作出人工驳回。只输出JSON对象assessments。"
        )

    def prompt_audit(self, role: AgentRole) -> dict[str, str]:
        return prompt_audit(
            prompt=compose_prompt(self._system_prompt, self._role_prompts.get(role)),
            version=self.prompt_version,
        )

    def analyze_batch(self, tasks: Sequence[SpecialistTask]) -> SpecialistBatchResult:
        if not tasks:
            return SpecialistBatchResult(assessments=(), attempts_by_risk={}, errors_by_risk={})
        roles = {task.rule.owner for task in tasks}
        if len(roles) != 1:
            raise ValueError("a specialist batch must contain exactly one owning role")
        role = next(iter(roles))
        assessments: list[SpecialistAssessment] = []
        attempts_by_risk: dict[str, int] = {}
        errors_by_risk: dict[str, tuple[str, ...]] = {}
        partitions = [
            tasks[start : start + self._batch_size]
            for start in range(0, len(tasks), self._batch_size)
        ]

        def analyze_partition_with_isolation(
            partition: Sequence[SpecialistTask],
            *,
            partition_index: int,
            inherited_attempts: int = 0,
            inherited_errors: tuple[str, ...] = (),
        ) -> tuple[
            list[SpecialistAssessment],
            dict[str, int],
            dict[str, tuple[str, ...]],
        ]:
            """批次持续失败时自动拆小，避免一条异常拖断整个专业角色。"""
            try:
                outcome = self._retry_policy.run_with_feedback(
                    lambda correction: self._analyze_partition(
                        partition,
                        role=role,
                        partition_index=partition_index,
                        partition_count=len(partitions),
                        correction_feedback=correction,
                    ),
                    on_retry=(
                        lambda error, attempt, wait: self._report_retry(
                            role=role,
                            partition_index=partition_index,
                            partition_count=len(partitions),
                            attempt=attempt,
                            wait_seconds=wait,
                            error=error,
                        )
                    ),
                )
            except RetryExhaustedError as error:
                if isinstance(error.last_error, ModelBudgetExceededError):
                    raise error.last_error
                message = str(error.last_error).strip() or type(error.last_error).__name__
                accumulated_attempts = inherited_attempts + error.attempts
                accumulated_errors = (*inherited_errors, message)
                if len(partition) > 1:
                    midpoint = len(partition) // 2
                    if self._progress_callback is not None:
                        self._progress_callback(
                            f"agent:{role.value}:retrying",
                            (
                                f"[{partition_index}/{len(partitions)}] {_role_name(role)} Agent"
                                f"批次持续异常，已自动拆为{midpoint}条和"
                                f"{len(partition) - midpoint}条继续核验；其他专业角色不受影响。"
                            ),
                        )
                    left = analyze_partition_with_isolation(
                        partition[:midpoint],
                        partition_index=partition_index,
                        inherited_attempts=accumulated_attempts,
                        inherited_errors=accumulated_errors,
                    )
                    right = analyze_partition_with_isolation(
                        partition[midpoint:],
                        partition_index=partition_index,
                        inherited_attempts=accumulated_attempts,
                        inherited_errors=accumulated_errors,
                    )
                    return (
                        [*left[0], *right[0]],
                        {**left[1], **right[1]},
                        {**left[2], **right[2]},
                    )

                task = partition[0]
                status = (
                    SpecialistStatus.TECHNICAL_FAILURE
                    if retry_transient_api_error(error.last_error)
                    else SpecialistStatus.EVIDENCE_INSUFFICIENT
                )
                return (
                    [
                        SpecialistAssessment(
                            risk_type=task.rule.id,
                            status=status,
                            severity=task.rule.severity,
                            confidence=0.0,
                            uncertainty=f"专业Agent核验失败，候选风险仍保留：{message}",
                        )
                    ],
                    {task.rule.id: accumulated_attempts},
                    {task.rule.id: accumulated_errors},
                )

            attempts = inherited_attempts + outcome.attempts
            errors = (*inherited_errors, *outcome.errors)
            risk_ids = (task.rule.id for task in partition)
            return (
                list(outcome.value.assessments),
                {risk_id: attempts for risk_id in risk_ids},
                {
                    task.rule.id: errors
                    for task in partition
                },
            )

        for index, partition in enumerate(partitions, start=1):
            if self._progress_callback is not None:
                self._progress_callback(
                    f"agent:{role.value}:model_requested",
                    (
                        f"[{index}/{len(partitions)}] {_role_name(role)} Agent正在批量核验"
                        f"{len(partition)}条风险。"
                    ),
                )
            partition_assessments, partition_attempts, partition_errors = (
                analyze_partition_with_isolation(
                    partition,
                    partition_index=index,
                )
            )
            assessments.extend(partition_assessments)
            attempts_by_risk.update(partition_attempts)
            errors_by_risk.update(partition_errors)
            if self._progress_callback is not None:
                self._progress_callback(
                    f"agent:{role.value}:structured",
                    (
                        f"[[batch:{index}/{len(partitions)}]]"
                        + json.dumps(
                            SpecialistAssessmentBatch(
                                assessments=tuple(partition_assessments)
                            ).model_dump(mode="json"),
                            ensure_ascii=False,
                        )
                    ),
                )
        return SpecialistBatchResult(
            assessments=tuple(assessments),
            attempts_by_risk=attempts_by_risk,
            errors_by_risk=errors_by_risk,
        )

    def _report_retry(
        self,
        *,
        role: AgentRole,
        partition_index: int,
        partition_count: int,
        attempt: int,
        wait_seconds: float,
        error: Exception,
    ) -> None:
        if self._progress_callback is None:
            return
        message = str(error).strip() or type(error).__name__
        failure_kind = (
            "连接异常" if retry_transient_api_error(error) else "输出校验未通过"
        )
        self._progress_callback(
            f"agent:{role.value}:retrying",
            (
                f"[{partition_index}/{partition_count}] {_role_name(role)} Agent{failure_kind}，"
                f"{wait_seconds:g}秒后进行第{attempt}次尝试：{message[:180]}"
            ),
        )

    def _analyze_partition(
        self,
        tasks: Sequence[SpecialistTask],
        *,
        role: AgentRole,
        partition_index: int,
        partition_count: int,
        correction_feedback: str | None,
    ) -> SpecialistAssessmentBatch:
        expected = {task.rule.id for task in tasks}
        allowed_by_risk = {
            task.rule.id: {chunk.chunk_id for chunk in task.candidates} for task in tasks
        }
        payload = with_correction(
            {
                "role": role.value,
                "constraints": {
                    "required_risk_types": sorted(expected),
                    "return_every_risk_exactly_once": True,
                    "no_external_evidence": True,
                    "human_review_is_final_authority": True,
                    "allowed_statuses": [
                        status.value
                        for status in SpecialistStatus
                        if status is not SpecialistStatus.TECHNICAL_FAILURE
                    ],
                },
                "risks": [
                    {
                        "risk_type": task.rule.id,
                        "risk_name": task.rule.name,
                        "default_severity": task.rule.severity.value,
                        "review_question": task.rule.claim,
                        "search_queries": list(task.rule.search_queries),
                        "allowed_evidence_chunk_ids": [
                            chunk.chunk_id for chunk in task.candidates
                        ],
                        "candidates": [
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
                                "quote": chunk.text,
                            }
                            for chunk in task.candidates
                        ],
                    }
                    for task in tasks
                ],
            },
            correction_feedback,
        )
        on_text_delta: TextDeltaCallback | None = None
        if self._progress_callback is not None:

            def report_delta(delta: str) -> None:
                assert self._progress_callback is not None
                self._progress_callback(
                    f"agent:{role.value}:delta",
                    (
                        f"[[batch:{partition_index}/{partition_count}]]"
                        f"{delta}"
                    ),
                )

            on_text_delta = report_delta
        request: dict[str, Any] = {
            "system_prompt": compose_prompt(
                self._system_prompt,
                self._role_prompts.get(role),
            ),
            "user_prompt": json.dumps(payload, ensure_ascii=False),
            "temperature": self._temperature,
            "max_output_tokens": self._max_output_tokens,
        }
        if on_text_delta is not None:
            request["on_text_delta"] = on_text_delta
        provider = self._providers_by_role.get(role, self.provider)
        batch = SpecialistAssessmentBatch.model_validate(
            provider.complete_json(**request)
        )
        returned = {item.risk_type for item in batch.assessments}
        if returned != expected:
            raise ValueError(
                "specialist batch must cover every submitted risk: "
                f"expected={sorted(expected)}, returned={sorted(returned)}"
            )
        for assessment in batch.assessments:
            unknown = set(assessment.evidence_chunk_ids) - allowed_by_risk[
                assessment.risk_type
            ]
            if unknown:
                raise ValueError(
                    f"{assessment.risk_type} cited unknown evidence chunks: {sorted(unknown)}"
                )
            if assessment.duplicate_of is not None and assessment.duplicate_of not in expected:
                raise ValueError(
                    f"{assessment.risk_type} cited unknown duplicate_of risk"
                )
        return batch


def _role_name(role: AgentRole) -> str:
    return {
        AgentRole.FINANCE: "财务",
        AgentRole.LEGAL: "法律",
        AgentRole.BUSINESS: "业务",
        AgentRole.MARKET: "市场",
    }.get(role, role.value)


__all__ = [
    "DeepSeekSpecialistBatchAnalyzer",
    "SpecialistAssessment",
    "SpecialistAssessmentBatch",
    "SpecialistBatchAnalyzer",
    "SpecialistBatchResult",
    "SpecialistStatus",
    "SpecialistTask",
]
