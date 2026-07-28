from __future__ import annotations

import json

from ipo_risk.ui.agent_output import (
    empty_agent_output_message,
    format_raw_model_output,
    paginate_agent_output,
    parse_agent_output,
)


def test_discovery_output_becomes_business_readable_risk_cards() -> None:
    raw = json.dumps(
        {
            "candidates": [
                {
                    "title": "收入确认高度依赖少数大型项目",
                    "owner": "finance",
                    "severity": "high",
                    "claim": "公司收入增长依赖少数大型项目，验收时间集中。",
                    "confidence": 0.85,
                    "evidence_chunk_ids": ["C-1", "C-2"],
                    "uncertainty": "尚需核对项目延期情况。",
                }
            ]
        },
        ensure_ascii=False,
    )

    presentation = parse_agent_output("discovery", raw)

    assert not presentation.has_incomplete_segment
    assert len(presentation.cards) == 1
    card = presentation.cards[0]
    assert card.title == "收入确认高度依赖少数大型项目"
    assert card.owner == "finance"
    assert card.severity == "high"
    assert card.confidence == 0.85
    assert card.evidence_ids == ("C-1", "C-2")


def test_completed_segments_remain_visible_while_next_json_is_streaming() -> None:
    raw = (
        '{"supported":true,"claim":"现金持续流出","severity":"high",'
        '"confidence":0.91,"evidence_chunk_ids":["C-1"],'
        '"uncertainty":"融资时间仍不确定"}'
        "\n\n--- 新的风险任务 ---\n"
        '{"supported":true,"claim":"下一条尚未生成完成"'
    )

    presentation = parse_agent_output("finance", raw)

    assert len(presentation.cards) == 1
    assert presentation.cards[0].claim == "现金持续流出"
    assert presentation.has_incomplete_segment


def test_decision_output_distinguishes_accepted_and_rejected_verdicts() -> None:
    raw = json.dumps(
        {
            "decisions": [
                {
                    "risk_type": "cash_burn",
                    "accepted": True,
                    "claim": "现金消耗压力较高。",
                    "severity": "critical",
                    "confidence": 0.93,
                    "evidence_chunk_ids": ["C-9"],
                    "uncertainty": "未来融资安排尚未确定。",
                    "rationale": "多项原文证据相互印证。",
                },
                {
                    "risk_type": "customer_concentration",
                    "accepted": False,
                    "claim": "",
                    "severity": "medium",
                    "confidence": 0.42,
                    "evidence_chunk_ids": [],
                    "uncertainty": "",
                    "rationale": "现有证据不足。",
                },
            ]
        },
        ensure_ascii=False,
    )

    presentation = parse_agent_output("decision", raw)

    assert [card.accepted for card in presentation.cards] == [True, False]
    assert presentation.cards[1].rationale == "现有证据不足。"


def test_discovery_presentation_separates_raw_batches_from_global_consolidation() -> None:
    raw_batch = json.dumps(
        {
            "candidates": [
                {
                    "title": "连续净亏损风险",
                    "owner": "finance",
                    "severity": "high",
                    "claim": "公司连续报告净亏损。",
                    "confidence": 0.9,
                    "evidence_chunk_ids": ["C-1"],
                    "uncertainty": "仍需核验融资缓释。",
                },
                {
                    "title": "经营现金流持续为负",
                    "owner": "finance",
                    "severity": "high",
                    "claim": "经营活动现金流持续为负。",
                    "confidence": 0.88,
                    "evidence_chunk_ids": ["C-2"],
                    "uncertainty": "仍需核验现金余额。",
                },
            ]
        },
        ensure_ascii=False,
    )
    consolidated = json.dumps(
        {
            "consolidated_risks": [
                {
                    "title": "持续亏损及经营现金净流出风险",
                    "owner": "finance",
                    "severity": "high",
                    "claim": "公司持续亏损且经营活动现金流为负。",
                    "confidence": 0.92,
                    "evidence_chunk_ids": ["C-1", "C-2"],
                    "uncertainty": "仍需核验融资缓释。",
                }
            ]
        },
        ensure_ascii=False,
    )

    presentation = parse_agent_output(
        "discovery",
        f"{raw_batch}\n\n--- 新的风险任务 ---\n{consolidated}",
    )

    assert presentation.raw_discovery_count == 2
    assert presentation.consolidated_discovery_count == 1
    assert [card.kind for card in presentation.cards] == [
        "discovery",
        "discovery",
        "consolidated",
    ]


def test_specialist_batch_output_becomes_one_card_per_risk_status() -> None:
    raw = json.dumps(
        {
            "assessments": [
                {
                    "risk_type": "open_123456789abc",
                    "status": "supported",
                    "claim": "客户集中度获得原文直接支持。",
                    "severity": "high",
                    "confidence": 0.88,
                    "evidence_chunk_ids": ["C-1"],
                    "uncertainty": "仍需确认后续客户变化。",
                    "followup_queries": [],
                    "duplicate_of": None,
                },
                {
                    "risk_type": "open_abcdef123456",
                    "status": "needs_more_evidence",
                    "claim": "",
                    "severity": "medium",
                    "confidence": 0.55,
                    "evidence_chunk_ids": [],
                    "uncertainty": "需要补充期间信息。",
                    "followup_queries": ["检索最近期间"],
                    "duplicate_of": None,
                },
            ]
        },
        ensure_ascii=False,
    )

    presentation = parse_agent_output(
        "finance",
        raw,
        risk_titles={
            "open_123456789abc": "客户集中度风险",
            "open_abcdef123456": "收入期间证据不足",
        },
    )

    assert len(presentation.cards) == 2
    assert presentation.cards[0].machine_status == "supported"
    assert presentation.cards[1].machine_status == "needs_more_evidence"
    assert presentation.cards[0].title == "客户集中度风险"
    assert presentation.cards[0].risk_type == "open_123456789abc"
    assert presentation.cards[1].title == "收入期间证据不足"


def test_completed_items_in_an_incomplete_batch_are_visible_immediately() -> None:
    raw = (
        '{"assessments":['
        '{"risk_type":"open_123456789abc","status":"supported",'
        '"claim":"客户集中度获得原文支持。","severity":"high",'
        '"confidence":0.88,"evidence_chunk_ids":["C-1"],'
        '"uncertainty":"仍需人工确认。"},'
        '{"risk_type":"open_abcdef123456","status":"supported","claim":"尚未完成'
    )

    presentation = parse_agent_output(
        "finance",
        raw,
        risk_titles={"open_123456789abc": "客户集中度风险"},
    )

    assert presentation.has_incomplete_segment
    assert len(presentation.cards) == 1
    assert presentation.cards[0].title == "客户集中度风险"


def test_completed_discovery_candidate_is_visible_before_array_finishes() -> None:
    raw = (
        '{"candidates":['
        '{"title":"客户依赖风险","owner":"business","severity":"high",'
        '"claim":"收入依赖单一客户。","confidence":0.88,'
        '"evidence_chunk_ids":["C-1"],"uncertainty":"续约仍不确定"},'
        '{"title":"下一条尚未生成完成"'
    )

    presentation = parse_agent_output("discovery", raw)

    assert presentation.has_incomplete_segment
    assert presentation.raw_discovery_count == 1
    assert presentation.cards[0].title == "客户依赖风险"


def test_streaming_cards_skip_one_malformed_item_and_continue_after_it() -> None:
    raw = (
        '{"candidates":['
        '{"title":"客户依赖风险","owner":"business","severity":"high",'
        '"claim":"收入依赖单一客户。","confidence":0.88,'
        '"evidence_chunk_ids":["C-1"],"uncertainty":"续约仍不确定"},'
        '{"title":坏掉的模型对象},'
        '{"title":"供应商依赖风险","owner":"business","severity":"medium",'
        '"claim":"采购依赖少数供应商。","confidence":0.82,'
        '"evidence_chunk_ids":["C-2"],"uncertainty":"替代供应仍待核验"}'
    )

    presentation = parse_agent_output("discovery", raw)

    assert presentation.has_incomplete_segment
    assert [card.title for card in presentation.cards] == [
        "客户依赖风险",
        "供应商依赖风险",
    ]


def test_agent_output_pagination_starts_at_one_and_keeps_global_numbering() -> None:
    raw = json.dumps(
        {
            "candidates": [
                {
                    "title": f"风险{index}",
                    "owner": "business",
                    "severity": "medium",
                    "claim": f"风险结论{index}",
                    "confidence": 0.8,
                    "evidence_chunk_ids": [f"C-{index}"],
                    "uncertainty": "待核验",
                }
                for index in range(1, 24)
            ]
        },
        ensure_ascii=False,
    )
    presentation = parse_agent_output("discovery", raw)

    first = paginate_agent_output(presentation, role="discovery", page=1)
    third = paginate_agent_output(presentation, role="discovery", page=3)

    assert first.first_index == 1
    assert len(first.cards) == 10
    assert first.page_count == 3
    assert third.first_index == 21
    assert [card.title for card in third.cards] == ["风险21", "风险22", "风险23"]


def test_discovery_pagination_can_keep_raw_findings_after_global_merge() -> None:
    raw = json.dumps(
        {
            "candidates": [
                {"title": "原始风险一", "risk_type": "raw_one", "claim": "原始结论一"},
                {"title": "原始风险二", "risk_type": "raw_two", "claim": "原始结论二"},
            ]
        },
        ensure_ascii=False,
    )
    merged = json.dumps(
        {"consolidated_risks": [{"title": "合并风险", "risk_type": "merged", "claim": "合并结论"}]},
        ensure_ascii=False,
    )
    presentation = parse_agent_output(
        "discovery",
        f"{raw}\n\n--- 新的风险任务 ---\n{merged}",
    )

    raw_page = paginate_agent_output(
        presentation,
        role="discovery",
        discovery_view="raw",
    )
    merged_page = paginate_agent_output(
        presentation,
        role="discovery",
        discovery_view="consolidated",
    )

    assert [card.title for card in raw_page.cards] == ["原始风险一", "原始风险二"]
    assert [card.title for card in merged_page.cards] == ["合并风险"]


def test_status_only_payload_does_not_create_a_useless_placeholder_card() -> None:
    presentation = parse_agent_output(
        "business",
        '{"status":"running"}',
    )

    assert presentation.cards == ()


def test_empty_structured_payload_does_not_create_a_placeholder_card() -> None:
    presentation = parse_agent_output("business", "{}")

    assert presentation.parsed_segment_count == 1
    assert presentation.cards == ()


def test_skipped_agent_does_not_claim_to_be_waiting_for_model_output() -> None:
    assert empty_agent_output_message("market", "skipped") == (
        "本次未分配市场类风险，未调用模型。"
    )


def test_complete_json_is_pretty_printed_without_losing_earlier_batches() -> None:
    first = '{"assessments":[{"risk_type":"open_1","status":"supported"}]}'
    second = '{"assessments":[{"risk_type":"open_2","status":"supported"}]}'

    formatted = format_raw_model_output(
        f"{first}\n\n--- 新的风险任务 ---\n{second}"
    )

    assert formatted.count('"assessments": [') == 2
    assert '"risk_type": "open_1"' in formatted
    assert '"risk_type": "open_2"' in formatted
