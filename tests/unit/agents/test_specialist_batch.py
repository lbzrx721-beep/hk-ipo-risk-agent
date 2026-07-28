from __future__ import annotations

import json
from datetime import date

from ipo_risk.agents.grounded import GroundedAnalysisEngine
from ipo_risk.agents.specialist import (
    DeepSeekSpecialistBatchAnalyzer,
    SpecialistTask,
    SpecialistStatus,
)
from ipo_risk.domain import AgentRole, RiskSeverity
from ipo_risk.retrieval import DocumentChunk
from ipo_risk.retrieval.search import RiskRule


def _chunk(index: int) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"C-{index:03d}",
        company_id="00001",
        document_id="doc",
        source_file="prospectus.pdf",
        source_hash="a" * 64,
        page=index,
        bbox=(10.0, 10.0, 300.0, 80.0),
        text=(
            f"风险事项{index}存在直接披露和限制条件，"
            "该事项可能影响公司持续经营表现并需要人工进一步核实。"
        ),
        font_size=10.0,
        available_at=date(2026, 1, 1),
    )


def _rule(index: int) -> RiskRule:
    return RiskRule(
        id=f"open_{index:012x}",
        name=f"风险事项{index}",
        owner=AgentRole.FINANCE,
        severity=RiskSeverity.HIGH,
        query=f"风险事项{index}",
        required_keywords=(f"风险事项{index}",),
        claim=f"核验风险事项{index}是否存在方向性风险。",
        max_evidence=2,
    )


class BatchProvider:
    model_name = "deepseek-test"
    provider_name = "deepseek"

    def __init__(self) -> None:
        self.calls = 0
        self.batch_sizes: list[int] = []
        self.payloads: list[dict[str, object]] = []

    def complete_json(self, **request: object) -> dict[str, object]:
        self.calls += 1
        payload = json.loads(str(request["user_prompt"]))
        self.payloads.append(payload)
        risks = payload["risks"]
        self.batch_sizes.append(len(risks))
        return {
            "assessments": [
                {
                    "risk_type": risk["risk_type"],
                    "status": "supported",
                    "claim": f"{risk['risk_name']}获得原文直接支持。",
                    "severity": "high",
                    "confidence": 0.88,
                    "evidence_chunk_ids": [risk["allowed_evidence_chunk_ids"][0]],
                    "uncertainty": "仍需人工确认风险影响程度。",
                    "followup_queries": [],
                    "duplicate_of": None,
                }
                for risk in risks
            ]
        }


def test_specialist_receives_table_structure_and_cell_references() -> None:
    provider = BatchProvider()
    analyzer = DeepSeekSpecialistBatchAnalyzer(
        provider,
        temperature=0.1,
        max_output_tokens=8_192,
        batch_size=20,
    )
    table_chunk = _chunk(1).model_copy(
        update={
            "text": (
                "[表格证据]\n|客户|收入（港元）|占比|\n|---|---|---|\n"
                "|客户A|120|30%|"
            ),
            "chunk_kind": "table",
            "table_id": "T-P0068-001",
            "table_cell_refs": (
                "T-P0068-001:R1:C1",
                "T-P0068-001:R1:C2",
                "T-P0068-001:R1:C3",
            ),
            "table_cell_values": ("客户A", "120", "30%"),
        }
    )

    analyzer.analyze_batch(
        [SpecialistTask(rule=_rule(1), candidates=(table_chunk,))]
    )

    candidate = provider.payloads[0]["risks"][0]["candidates"][0]
    assert candidate["source_kind"] == "table"
    assert candidate["table_id"] == "T-P0068-001"
    assert candidate["table_cell_refs"] == [
        "T-P0068-001:R1:C1",
        "T-P0068-001:R1:C2",
        "T-P0068-001:R1:C3",
    ]
    assert candidate["table_cells"] == {
        "T-P0068-001:R1:C1": "客户A",
        "T-P0068-001:R1:C2": "120",
        "T-P0068-001:R1:C3": "30%",
    }


def test_role_processes_all_risks_in_bounded_batches_instead_of_per_risk_calls() -> None:
    chunks = [_chunk(index) for index in range(1, 26)]
    rules = [_rule(index) for index in range(1, 26)]
    provider = BatchProvider()
    analyzer = DeepSeekSpecialistBatchAnalyzer(
        provider,
        temperature=0.1,
        max_output_tokens=8_192,
        batch_size=20,
    )
    engine = GroundedAnalysisEngine(
        chunks,
        rules,
        specialist_analyzer=analyzer,
        lexical_top_k=5,
        rerank_top_k=2,
    )

    results = engine.analyze_rules_batched(rules)

    assert provider.calls == 2
    assert provider.batch_sizes == [20, 5]
    assert len(results) == 25
    assert all(item.specialist_status is SpecialistStatus.SUPPORTED for item in results)
    assert all(item.analysis.supported for item in results)


def test_role_uses_one_model_call_when_all_assigned_risks_fit_role_capacity() -> None:
    chunks = [_chunk(index) for index in range(1, 26)]
    rules = [_rule(index) for index in range(1, 26)]
    provider = BatchProvider()
    analyzer = DeepSeekSpecialistBatchAnalyzer(
        provider,
        temperature=0.1,
        max_output_tokens=16_384,
        batch_size=100,
    )
    engine = GroundedAnalysisEngine(
        chunks,
        rules,
        specialist_analyzer=analyzer,
        specialist_followup_rounds=0,
        lexical_top_k=5,
        rerank_top_k=2,
    )

    results = engine.analyze_rules_batched(rules)

    assert provider.calls == 1
    assert provider.batch_sizes == [25]
    assert len(results) == 25


def test_each_role_can_use_an_independent_api_client() -> None:
    finance_provider = BatchProvider()
    legal_provider = BatchProvider()
    analyzer = DeepSeekSpecialistBatchAnalyzer(
        finance_provider,
        temperature=0.1,
        max_output_tokens=4_096,
        batch_size=100,
        providers_by_role={AgentRole.LEGAL: legal_provider},
    )
    legal_rule = _rule(1).model_copy(update={"owner": AgentRole.LEGAL})

    result = analyzer.analyze_batch(
        [
            SpecialistTask(
                rule=legal_rule,
                candidates=(_chunk(1),),
            )
        ]
    )

    assert len(result.assessments) == 1
    assert finance_provider.calls == 0
    assert legal_provider.calls == 1


class FollowupProvider(BatchProvider):
    def complete_json(self, **request: object) -> dict[str, object]:
        self.calls += 1
        payload = json.loads(str(request["user_prompt"]))
        risks = payload["risks"]
        self.batch_sizes.append(len(risks))
        if self.calls == 1:
            return {
                "assessments": [
                    {
                        "risk_type": risk["risk_type"],
                        "status": (
                            "needs_more_evidence"
                            if index == 0
                            else "supported"
                        ),
                        "claim": (
                            ""
                            if index == 0
                            else f"{risk['risk_name']}获得原文直接支持。"
                        ),
                        "severity": "high",
                        "confidence": 0.55 if index == 0 else 0.88,
                        "evidence_chunk_ids": (
                            []
                            if index == 0
                            else [risk["allowed_evidence_chunk_ids"][0]]
                        ),
                        "uncertainty": "需要补充检索反证和期间信息。",
                        "followup_queries": (
                            ["补充检索风险事项1的期间和缓释安排"]
                            if index == 0
                            else []
                        ),
                        "duplicate_of": None,
                    }
                    for index, risk in enumerate(risks)
                ]
            }
        risk = risks[0]
        return {
            "assessments": [
                {
                    "risk_type": risk["risk_type"],
                    "status": "supported",
                    "claim": f"{risk['risk_name']}经补证后获得直接支持。",
                    "severity": "high",
                    "confidence": 0.84,
                    "evidence_chunk_ids": [risk["allowed_evidence_chunk_ids"][0]],
                    "uncertainty": "最终影响程度仍需人工判断。",
                    "followup_queries": [],
                    "duplicate_of": None,
                }
            ]
        }


def test_only_risks_requesting_more_evidence_enter_followup_batch() -> None:
    chunks = [_chunk(1), _chunk(2)]
    rules = [_rule(1), _rule(2)]
    provider = FollowupProvider()
    analyzer = DeepSeekSpecialistBatchAnalyzer(
        provider,
        temperature=0.1,
        max_output_tokens=4_096,
        batch_size=20,
    )
    engine = GroundedAnalysisEngine(
        chunks,
        rules,
        specialist_analyzer=analyzer,
        specialist_followup_rounds=1,
        lexical_top_k=5,
        rerank_top_k=2,
    )

    results = engine.analyze_rules_batched(rules)

    assert provider.batch_sizes == [2, 1]
    assert all(item.analysis.supported for item in results)
    assert results[0].analysis.claim.endswith("经补证后获得直接支持。")
