from __future__ import annotations

from datetime import datetime, timezone

from ipo_risk.runtime.analysis_job import AnalysisProgressEvent
from ipo_risk.runtime.progress import build_analysis_progress_snapshot


def _event(stage: str, detail: str) -> AnalysisProgressEvent:
    return AnalysisProgressEvent(
        stage=stage,
        detail=detail,
        occurred_at=datetime(2026, 7, 22, 8, 17, tzinfo=timezone.utc),
    )


def test_bm25_skip_is_terminal_and_profile_payload_is_not_activity_text() -> None:
    snapshot = build_analysis_progress_snapshot((
        _event("embedding_skipped", "使用BM25检索，无需向量化"),
        _event("agent:discovery:started", "正在提取画像"),
        _event("company_review_plan", '{"profile":{"facts":[]}}'),
        _event("agent:discovery:completed", "画像与审查问题已就绪"),
    ), status="running")
    assert snapshot["background_tasks"]["embedding"]["state"] == "skipped"
    assert snapshot["agents"]["discovery"]["state"] == "complete"
    assert "profile" not in snapshot["current_activity"]
    assert not any('"facts"' in event for event in snapshot["events"])


def test_discovery_batch_scans_refresh_one_dynamic_line_without_flooding_history() -> None:
    events = (
        _event("agents_started", "开始运行专业Agent与决策Agent。"),
        _event("agent:discovery:started", "风险发现Agent开始扫描全文。"),
        _event("agent:discovery:prompt", "[1/32] 正在扫描PDF原文批次。"),
        _event("agent:discovery:prompt", "[2/32] 正在扫描PDF原文批次。"),
        _event("agent:discovery:prompt", "[8/32] 正在扫描PDF原文批次。"),
    )

    snapshot = build_analysis_progress_snapshot(events, status="running")

    discovery = snapshot["agents"]["discovery"]
    assert discovery["progress"] == 22
    assert discovery["progress_text"] == "风险原文包 8/32"
    assert snapshot["current_activity"].endswith("[8/32] 正在扫描PDF原文批次。")
    assert not any("扫描PDF原文批次" in line for line in snapshot["events"])


def test_discovery_batch_completion_updates_live_candidate_count_without_event_spam() -> None:
    events = (
        _event("agent:discovery:started", "风险发现Agent开始扫描全文。"),
        _event(
            "agent:discovery:batch_completed",
            "[1/4] 本批新增6条，累计6条风险候选。",
        ),
        _event(
            "agent:discovery:batch_completed",
            "[2/4] 本批新增4条，累计10条风险候选。",
        ),
    )

    snapshot = build_analysis_progress_snapshot(events, status="running")

    discovery = snapshot["agents"]["discovery"]
    assert discovery["progress"] == 45
    assert discovery["progress_text"] == "风险原文包 2/4 · 已形成候选"
    assert snapshot["funnel"]["document_batches"] == 4
    assert snapshot["funnel"]["raw_candidates"] == 10
    assert not any("本批新增" in line for line in snapshot["events"])


def test_discovery_cards_survive_raw_completion_and_next_batch_stream() -> None:
    first_batch = (
        '{"candidates":[{"title":"客户依赖风险","owner":"business",'
        '"severity":"high","claim":"客户依赖获得原文支持。",'
        '"confidence":0.88,"evidence_chunk_ids":["C-1"],'
        '"search_queries":["客户依赖"],"anchor_terms":["客户"],'
        '"uncertainty":"仍需人工判断。","related_known_risk_type":null}]}'
    )
    events = (
        _event("agent:discovery:delta", f"[[batch:1/2]]{first_batch}"),
        _event(
            "agent:discovery:batch_completed",
            "[1/2] 章节包1新增1条，累计1条风险候选。",
        ),
        _event("agent:discovery:delta", '[[batch:2/2]]{"candidates":['),
        _event(
            "agent:discovery:raw_completed",
            "全文2个批次共形成1条原始候选，开始全局语义合并。",
        ),
    )

    snapshot = build_analysis_progress_snapshot(events, status="running")

    assert "客户依赖风险" in snapshot["outputs"]["discovery"]
    assert "全文2个批次" not in snapshot["outputs"]["discovery"]


def test_structured_batch_snapshot_replaces_incomplete_stream_for_live_cards() -> None:
    complete = (
        '{"candidates":[{"title":"合同终止风险","owner":"legal",'
        '"severity":"high","claim":"关键合同可以提前终止。",'
        '"confidence":0.9,"evidence_chunk_ids":["C-1"],'
        '"search_queries":["合同终止"],"anchor_terms":["终止"],'
        '"uncertainty":"替代安排待核验","related_known_risk_type":null}]}'
    )
    events = (
        _event(
            "agent:discovery:delta",
            '[[batch:1/2]]{"candidates":[{"title":"尚未完成',
        ),
        _event(
            "agent:discovery:structured",
            f"[[batch:1/2]]{complete}",
        ),
    )

    snapshot = build_analysis_progress_snapshot(events, status="running")

    assert "合同终止风险" in snapshot["outputs"]["discovery"]
    assert "尚未完成" not in snapshot["outputs"]["discovery"]
    assert snapshot["agents"]["discovery"]["progress_text"] == "本批风险卡片已生成"


def test_risk_catalog_is_kept_for_business_card_titles() -> None:
    events = (
        _event(
            "agent:business:risk_catalog",
            '{"open_123456789abc":"历史增长率不可持续风险"}',
        ),
    )

    snapshot = build_analysis_progress_snapshot(events, status="running")

    assert snapshot["risk_titles"]["business"] == {"open_123456789abc": "历史增长率不可持续风险"}


def test_failed_discovery_batches_keep_actionable_error_details() -> None:
    events = (
        _event(
            "agent:discovery:batch_failed",
            "章节包4/11扫描失败，已重试2次：JSON输出被截断",
        ),
        _event(
            "agent:discovery:degraded",
            "其余10个章节包扫描成功，继续专业分析。",
        ),
    )

    snapshot = build_analysis_progress_snapshot(events, status="running")

    assert snapshot["agents"]["discovery"]["state"] == "degraded"
    assert snapshot["discovery_failures"] == ["章节包4/11扫描失败，已重试2次：JSON输出被截断"]


def test_progress_snapshot_is_monotonic_when_a_later_event_reports_less_progress() -> None:
    events = (
        _event("embedding_started", "开始建立全文向量索引。"),
        _event("embedding_progress", "[12/12] 正在向量化全文切片。"),
        _event("agent:discovery:started", "风险发现Agent开始扫描全文。"),
        _event("agent:discovery:prompt", "[20/32] 正在扫描PDF原文批次。"),
        _event("agent:discovery:prompt", "[3/32] 重试扫描PDF原文批次。"),
        _event("agent:finance:started", "财务Agent开始分析。"),
        _event("agent:finance:progress", "[2/8] 已完成风险核验。"),
        _event("agent:finance:progress", "[1/8] 重试风险核验。"),
    )

    full = build_analysis_progress_snapshot(events, status="running")
    earlier = build_analysis_progress_snapshot(events[:-2], status="running")

    assert full["max_percent"] >= earlier["max_percent"]
    assert full["agents"]["discovery"]["progress"] == 56
    assert full["agents"]["finance"]["progress"] == 22


def test_specialist_returns_to_complete_after_successful_debate_response() -> None:
    events = (
        _event("agent:finance:started", "财务Agent开始分析。"),
        _event("agent:finance:completed", "财务Agent首轮分析完成。"),
        _event("agent:finance:debate_retrieval", "财务Agent开始定向补证。"),
        _event("agent:finance:delta", '{"assessments":['),
        _event("agent:finance:debate_response", "财务Agent已提交补证回应。"),
    )

    snapshot = build_analysis_progress_snapshot(events, status="running")

    assert snapshot["agents"]["finance"]["state"] == "complete"
    assert snapshot["agents"]["finance"]["progress"] == 100
    assert snapshot["agents"]["finance"]["progress_text"] == "定向补证完成"


def test_specialist_returns_to_complete_when_debate_fails() -> None:
    events = (
        _event("agent:legal:completed", "法律Agent首轮分析完成。"),
        _event("agent:legal:debate_retrieval", "法律Agent开始定向补证。"),
        _event("agent:legal:delta", '{"assessments":['),
        _event("agent:legal:debate_failed", "本轮补证未完成，保留首轮结论。"),
    )

    snapshot = build_analysis_progress_snapshot(events, status="running")

    assert snapshot["agents"]["legal"]["state"] == "complete"
    assert snapshot["agents"]["legal"]["progress"] == 100
    assert snapshot["agents"]["legal"]["progress_text"] == "补证已结束，保留首轮结论"


def test_decision_completion_closes_stale_specialist_state_from_legacy_debate_events() -> None:
    events = (
        _event("agent:finance:completed", "财务Agent首轮分析完成。"),
        _event("agent:finance:delta", '{"assessments":['),
        _event("agent:decision:debate_completed", "决策Agent已完成第二轮裁决。"),
        _event("agent:decision:completed", "决策Agent完成最终整合。"),
    )

    snapshot = build_analysis_progress_snapshot(events, status="completed")

    assert snapshot["agents"]["finance"]["state"] == "complete"
    assert snapshot["agents"]["finance"]["progress_text"] == "定向补证已结束"


def test_pdf_parsing_reports_page_progress_without_flooding_event_history() -> None:
    events = (
        _event("parsing_started", "开始解析PDF页面与坐标。"),
        _event("parsing_progress", "[1/635] 已解析PDF第1页。"),
        _event("parsing_progress", "[120/635] 已解析PDF第120页。"),
    )

    snapshot = build_analysis_progress_snapshot(events, status="running")

    assert 12 < snapshot["max_percent"] < 25
    assert snapshot["current_activity"].endswith("120/635页")
    assert not any("已解析PDF第" in line for line in snapshot["events"])


def test_embedding_is_a_background_task_and_does_not_move_main_progress() -> None:
    events = (
        _event("agents_started", "启动风险分析。"),
        _event("embedding_started", "后台开始建立语义索引。"),
        _event("embedding_progress", "[6/12] 正在向量化全文切片。"),
    )

    snapshot = build_analysis_progress_snapshot(events, status="running")

    assert snapshot["max_percent"] == 30
    assert snapshot["background_tasks"]["embedding"]["state"] == "running"
    assert snapshot["background_tasks"]["embedding"]["progress"] == 48


def test_running_snapshot_reports_elapsed_time_without_unreliable_estimate() -> None:
    started = datetime(2026, 7, 22, 8, 17, tzinfo=timezone.utc)
    events = (
        AnalysisProgressEvent(
            stage="agents_started",
            detail="启动风险分析。",
            occurred_at=started,
        ),
    )

    snapshot = build_analysis_progress_snapshot(
        events,
        status="running",
        now=datetime(2026, 7, 22, 8, 18, tzinfo=timezone.utc),
    )

    assert snapshot["elapsed_seconds"] == 60
    assert "estimated_remaining_seconds" not in snapshot


def test_prediction_steps_finish_before_overall_progress_reaches_100() -> None:
    events = (
        _event("agents_completed", "决策Agent已完成风险整合。"),
        _event("features_started", "正在自动采集上市前候选特征。"),
        _event("features_progress", "[3/6] 文档结构与证据统计已汇总"),
        _event("features_completed", "上市前特征快照已生成。"),
        _event("prediction_started", "正在预测五日显著下跌风险。"),
    )

    running = build_analysis_progress_snapshot(events, status="running")
    predicted = build_analysis_progress_snapshot(
        events + (_event("prediction_completed", "五日风险预测完成。"),),
        status="running",
    )
    completed = build_analysis_progress_snapshot(
        events
        + (
            _event("prediction_completed", "五日风险预测完成。"),
            _event("completed", "分析完成。"),
        ),
        status="completed",
    )

    assert running["max_percent"] == 96
    assert running["current_activity"].startswith("五日下跌预测")
    assert predicted["max_percent"] == 99
    assert completed["max_percent"] == 100


def test_feature_collection_progress_is_projected_before_prediction() -> None:
    snapshot = build_analysis_progress_snapshot(
        (
            _event("features_started", "正在自动采集上市前候选特征。"),
            _event("features_progress", "[4/6] 上市前市场与近期IPO环境已汇总"),
        ),
        status="running",
    )

    assert snapshot["max_percent"] == 94
    assert snapshot["current_activity"] == "上市前特征 · 上市前市场与近期IPO环境已汇总"


def test_elapsed_time_pauses_during_human_scope_review_and_resumes_after_submit() -> None:
    started = datetime(2026, 7, 22, 8, 0, tzinfo=timezone.utc)
    requested = datetime(2026, 7, 22, 8, 1, tzinfo=timezone.utc)
    submitted = datetime(2026, 7, 22, 8, 11, tzinfo=timezone.utc)
    waiting_events = (
        AnalysisProgressEvent("agents_started", "开始分析。", started),
        AnalysisProgressEvent(
            "risk_scope_review_requested",
            "等待人工确认。",
            requested,
        ),
    )

    waiting = build_analysis_progress_snapshot(
        waiting_events,
        status="awaiting_review",
        now=datetime(2026, 7, 22, 8, 6, tzinfo=timezone.utc),
    )
    resumed = build_analysis_progress_snapshot(
        waiting_events
        + (
            AnalysisProgressEvent(
                "risk_scope_review_completed",
                "人工确认完成。",
                submitted,
            ),
        ),
        status="running",
        now=datetime(2026, 7, 22, 8, 12, tzinfo=timezone.utc),
    )

    assert waiting["elapsed_seconds"] == 60
    assert waiting["review_wait_seconds"] == 300
    assert resumed["elapsed_seconds"] == 120
    assert resumed["review_wait_seconds"] == 600


def test_specialist_reasoning_is_visible_before_answer_content() -> None:
    events = (
        _event("agent:finance:started", "财务Agent开始分析9类风险。"),
        _event("agent:finance:model_requested", "财务Agent已提交9条风险给模型。"),
        _event("agent:finance:reasoning", "模型正在推理，尚未输出结构化正文。"),
    )

    snapshot = build_analysis_progress_snapshot(events, status="running")

    finance = snapshot["agents"]["finance"]
    assert finance["state"] == "running"
    assert finance["progress_text"] == "模型正在推理"
    assert "尚未输出结构化正文" in finance["message"]


def test_decision_structured_batches_accumulate_instead_of_overwriting() -> None:
    first = (
        '[[batch:1/2]]{"decisions":['
        '{"risk_type":"open_111111111111","accepted":true,'
        '"claim":"第一条风险。","severity":"high","confidence":0.9,'
        '"evidence_chunk_ids":["C-1"],"uncertainty":"待人工确认",'
        '"rationale":"第一批裁决","source_proposal_ids":["P-1"]}]}'
    )
    second = (
        '[[batch:2/2]]{"decisions":['
        '{"risk_type":"open_222222222222","accepted":true,'
        '"claim":"第二条风险。","severity":"medium","confidence":0.8,'
        '"evidence_chunk_ids":["C-2"],"uncertainty":"待人工确认",'
        '"rationale":"第二批裁决","source_proposal_ids":["P-2"]}]}'
    )
    events = (
        _event("agent:decision:structured", first),
        _event("agent:decision:structured", second),
    )

    snapshot = build_analysis_progress_snapshot(events, status="running")

    output = snapshot["outputs"]["decision"]
    assert "第一条风险" in output
    assert "第二条风险" in output


def test_company_risk_scoping_is_a_named_monotonic_pipeline_stage() -> None:
    events = (
        _event("embedding_completed", "全文向量索引建立完成。"),
        _event("risk_scoping_started", "正在识别公司业务和风险适用范围。"),
        _event(
            "risk_scoping_completed",
            "公司画像：数字孪生软件；启用9类风险，排除3类不适用风险。",
        ),
    )

    snapshot = build_analysis_progress_snapshot(events, status="running")

    assert snapshot["max_percent"] == 58
    assert snapshot["overall_text"].startswith("公司画像：数字孪生软件")
    assert snapshot["current_activity"].startswith("公司画像与风险范围 ·")
    assert any("公司画像与风险范围" in line for line in snapshot["events"])


def test_parallel_agent_overall_progress_uses_all_four_branches() -> None:
    base = (
        _event("risk_scoping_completed", "专业Agent任务已准备完成。"),
        _event("agent:market:started", "市场Agent开始分析。"),
        _event("agent:market:completed", "市场Agent完成分析。"),
    )

    market_only = build_analysis_progress_snapshot(base, status="running")
    all_completed = build_analysis_progress_snapshot(
        base
        + (
            _event("agent:finance:completed", "财务Agent完成分析。"),
            _event("agent:legal:completed", "法律Agent完成分析。"),
            _event("agent:business:completed", "业务Agent完成分析。"),
        ),
        status="running",
    )

    assert market_only["max_percent"] == 64
    assert all_completed["max_percent"] == 83


def test_risk_funnel_counts_remain_distinct_and_traceable() -> None:
    events = (
        _event(
            "agent:discovery:raw_completed",
            "全文32个批次共形成83条原始候选，开始全局语义合并。",
        ),
        _event(
            "agent:discovery:consolidated",
            "83条原始候选经大模型语义合并形成21条待确认风险。",
        ),
        _event(
            "risk_scoping_completed",
            "公司画像：数字孪生；保留18条动态候选，排除3条业务错配候选。",
        ),
        _event(
            "risk_scope_review_completed",
            "复核人已确认风险范围：保留及新增16条，专业Agent继续运行。",
        ),
        _event("agent:finance:started", "财务 Agent开始分析8类风险。"),
        _event("agent:legal:started", "法律 Agent开始分析6类风险。"),
        _event("agent:business:started", "业务 Agent开始分析2类风险。"),
        _event(
            "agent:decision:completed",
            "决策Agent接受7条有证据风险，记录0项角色分歧。",
        ),
    )

    snapshot = build_analysis_progress_snapshot(events, status="running")

    assert snapshot["funnel"] == {
        "document_batches": 32,
        "raw_candidates": 83,
        "consolidated_candidates": 21,
        "scoped_in": 18,
        "scoped_out": 3,
        "human_selected": 16,
        "role_assignments": {
            "finance": 8,
            "legal": 6,
            "business": 2,
            "market": 0,
        },
        "final_accepted": 7,
    }


def test_budget_pause_preserves_current_progress_instead_of_marking_failure() -> None:
    events = (
        _event("agent:finance:completed", "财务Agent已完成批量核验。"),
        _event("model_budget_paused", "模型调用预算已达到40次，任务安全暂停。"),
    )

    snapshot = build_analysis_progress_snapshot(
        events,
        status="paused",
        error="模型调用预算已达到40次，任务安全暂停。",
    )

    assert snapshot["agents"]["finance"]["state"] == "complete"
    assert "安全暂停" in snapshot["current_activity"]
