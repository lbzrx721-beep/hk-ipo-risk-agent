from __future__ import annotations

from threading import Thread
from types import SimpleNamespace

from ipo_risk.agents.intervention import (
    LlmFailureRecoveryController,
    LlmFailureRecoveryDecision,
    LlmFailureRecoveryRequest,
    sanitize_llm_error,
)
from ipo_risk.agents.orchestrator import MultiAgentOrchestrator


def test_llm_failure_recovery_waits_for_matching_decision() -> None:
    controller = LlmFailureRecoveryController()
    request = LlmFailureRecoveryRequest(
        run_id="RUN-RECOVERY",
        reason="模型认证失败。",
        technical_detail="401 Unauthorized",
    )
    returned: list[LlmFailureRecoveryDecision] = []

    thread = Thread(target=lambda: returned.append(controller.request_resolution(request)))
    thread.start()
    assert controller.wait_for_request(timeout=1)
    assert controller.pending_request() == request

    decision = LlmFailureRecoveryDecision(
        request_id=request.request_id,
        run_id=request.run_id,
        action="retry_llm",
    )
    controller.submit(decision)
    thread.join(timeout=1)

    assert returned == [decision]
    assert controller.pending_request() is None


def test_sanitize_llm_error_redacts_key_and_classifies_auth_failure() -> None:
    reason, detail = sanitize_llm_error("401 Unauthorized api_key=ark-1234567890abcdefghijkl")

    assert "认证失败" in reason
    assert "ark-1234567890abcdefghijkl" not in detail
    assert "***" in detail


def test_sanitize_llm_error_classifies_missing_key() -> None:
    reason, detail = sanitize_llm_error("ark API Key 未配置，请检查本地供应商配置")

    assert "认证失败" in reason
    assert "未配置" in detail


def test_discovery_retry_reuses_same_stage_after_user_choice() -> None:
    controller = LlmFailureRecoveryController()
    orchestrator = object.__new__(MultiAgentOrchestrator)
    orchestrator._llm_failure_recovery_controller = controller
    calls = 0
    returned: list[object] = []

    def invocation() -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("503 upstream unavailable")
        return SimpleNamespace(batch_count=1, batch_errors=())

    thread = Thread(
        target=lambda: returned.append(
            orchestrator._discover_with_recovery(
                invocation,
                run_id="RUN-RETRY",
                discoverer=SimpleNamespace(provider_name="ark", model_name="deepseek"),
                progress_callback=None,
            )
        )
    )
    thread.start()
    assert controller.wait_for_request(timeout=1)
    request = controller.pending_request()
    assert request is not None
    controller.submit(
        LlmFailureRecoveryDecision(
            request_id=request.request_id,
            run_id=request.run_id,
            action="retry_llm",
        )
    )
    thread.join(timeout=1)

    assert calls == 2
    assert len(returned) == 1


def test_discovery_can_be_explicitly_degraded_to_bm25() -> None:
    controller = LlmFailureRecoveryController()
    orchestrator = object.__new__(MultiAgentOrchestrator)
    orchestrator._llm_failure_recovery_controller = controller
    errors: list[BaseException] = []

    def run() -> None:
        try:
            orchestrator._discover_with_recovery(
                lambda: (_ for _ in ()).throw(RuntimeError("401 Unauthorized")),
                run_id="RUN-BM25",
                discoverer=SimpleNamespace(provider_name="ark", model_name="deepseek"),
                progress_callback=None,
            )
        except BaseException as exc:  # noqa: BLE001 - 验证显式降级信号
            errors.append(exc)

    thread = Thread(target=run)
    thread.start()
    assert controller.wait_for_request(timeout=1)
    request = controller.pending_request()
    assert request is not None
    controller.submit(
        LlmFailureRecoveryDecision(
            request_id=request.request_id,
            run_id=request.run_id,
            action="fallback_bm25",
        )
    )
    thread.join(timeout=1)

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "401 Unauthorized" in str(errors[0])
