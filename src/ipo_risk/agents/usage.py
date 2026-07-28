"""模型调用用量审计、预算门禁与安全暂停。

这个模块把调用次数、Token 和估算费用集中在一个深模块中。模型适配器只需要在调用前登记
一次、响应后回填供应商 usage；编排、网页和批量实验都读取同一个不可变快照。
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


class ModelBudgetExceededError(RuntimeError):
    """模型预算达到上限，当前分析应安全暂停而不是删减风险。"""


@dataclass(frozen=True)
class ModelBudget:
    """单次公司分析允许消耗的付费模型预算。None 表示不启用该维度门禁。"""

    max_calls: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_cost_rmb: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_calls", "max_input_tokens", "max_output_tokens"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive when configured")
        if self.max_cost_rmb is not None and self.max_cost_rmb <= 0:
            raise ValueError("max_cost_rmb must be positive when configured")


@dataclass(frozen=True)
class ModelPricing:
    """供应商每百万 Token 的人民币单价；未配置价格时费用只显示为未知。"""

    input_per_million_rmb: float | None = None
    output_per_million_rmb: float | None = None
    cached_input_per_million_rmb: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "input_per_million_rmb",
            "output_per_million_rmb",
            "cached_input_per_million_rmb",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative when configured")


@dataclass(frozen=True)
class ModelUsageSnapshot:
    """截至当前时刻的模型调用用量。"""

    calls: int
    successful_calls: int
    failed_calls: int
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    estimated_cost_rmb: float | None
    budget_exhausted: bool
    pause_reason: str | None

    def as_dict(self) -> dict[str, int | float | bool | str | None]:
        return {
            "calls": self.calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "estimated_cost_rmb": self.estimated_cost_rmb,
            "budget_exhausted": self.budget_exhausted,
            "pause_reason": self.pause_reason,
        }


class ModelUsageLedger:
    """线程安全的单公司模型用量账本。

    LangGraph 的专业 Agent 会并行调用同一个账本，因此登记与预算判断必须在同一把锁内完成。
    达到预算时抛出 ``ModelBudgetExceededError``；调用方应把任务标为暂停并保留现有产物。
    """

    def __init__(
        self,
        *,
        budget: ModelBudget | None = None,
        pricing: ModelPricing | None = None,
    ) -> None:
        self._budget = budget or ModelBudget()
        self._pricing = pricing or ModelPricing()
        self._lock = Lock()
        self._calls = 0
        self._successful_calls = 0
        self._failed_calls = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._cached_input_tokens = 0
        self._estimated_cost_rmb = 0.0
        self._all_costs_known = True
        self._pause_reason: str | None = None

    def begin_call(self, *, model_name: str, request_tag: str) -> None:
        """在发出网络请求前占用一次调用额度。"""
        with self._lock:
            self._raise_if_paused()
            maximum = self._budget.max_calls
            if maximum is not None and self._calls >= maximum:
                self._pause(
                    f"模型调用预算已达到 {maximum} 次；"
                    f"在 {request_tag or 'unknown'} / {model_name} 前安全暂停。"
                )
            self._calls += 1

    def complete_call(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        pricing: ModelPricing | None = None,
    ) -> None:
        """记录成功响应，并在响应后检查 Token 与费用上限。"""
        if min(input_tokens, output_tokens, cached_input_tokens) < 0:
            raise ValueError("model usage tokens must be non-negative")
        with self._lock:
            self._successful_calls += 1
            self._input_tokens += input_tokens
            self._output_tokens += output_tokens
            self._cached_input_tokens += cached_input_tokens
            current_pricing = pricing or self._pricing
            incremental_cost = _estimate_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
                pricing=current_pricing,
            )
            if incremental_cost is None:
                self._all_costs_known = False
            else:
                self._estimated_cost_rmb += incremental_cost
            self._check_post_call_budget()

    def fail_call(self) -> None:
        """记录一次已经发出但失败的调用。"""
        with self._lock:
            self._failed_calls += 1

    def snapshot(self) -> ModelUsageSnapshot:
        with self._lock:
            return ModelUsageSnapshot(
                calls=self._calls,
                successful_calls=self._successful_calls,
                failed_calls=self._failed_calls,
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                cached_input_tokens=self._cached_input_tokens,
                estimated_cost_rmb=self._estimated_cost(),
                budget_exhausted=self._pause_reason is not None,
                pause_reason=self._pause_reason,
            )

    def _check_post_call_budget(self) -> None:
        checks = (
            (
                self._budget.max_input_tokens,
                self._input_tokens,
                "输入 Token",
            ),
            (
                self._budget.max_output_tokens,
                self._output_tokens,
                "输出 Token",
            ),
        )
        for maximum, actual, label in checks:
            if maximum is not None and actual >= maximum:
                self._set_pause(f"{label}预算已达到 {actual}/{maximum}；当前分析安全暂停。")
        maximum_cost = self._budget.max_cost_rmb
        estimated_cost = self._estimated_cost()
        if (
            maximum_cost is not None
            and estimated_cost is not None
            and estimated_cost >= maximum_cost
        ):
            self._set_pause(
                f"模型估算费用已达到 ¥{estimated_cost:.4f}/¥{maximum_cost:.4f}；"
                "当前分析安全暂停。"
            )

    def _estimated_cost(self) -> float | None:
        if not self._all_costs_known:
            return None
        return round(self._estimated_cost_rmb, 6)

    def _raise_if_paused(self) -> None:
        if self._pause_reason is not None:
            raise ModelBudgetExceededError(self._pause_reason)

    def _pause(self, reason: str) -> None:
        self._pause_reason = reason
        raise ModelBudgetExceededError(reason)

    def _set_pause(self, reason: str) -> None:
        if self._pause_reason is None:
            self._pause_reason = reason


def _estimate_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int,
    pricing: ModelPricing,
) -> float | None:
    if pricing.input_per_million_rmb is None or pricing.output_per_million_rmb is None:
        return None
    cached_price = (
        pricing.cached_input_per_million_rmb
        if pricing.cached_input_per_million_rmb is not None
        else pricing.input_per_million_rmb
    )
    uncached_input = max(0, input_tokens - cached_input_tokens)
    return (
        uncached_input * pricing.input_per_million_rmb
        + cached_input_tokens * cached_price
        + output_tokens * pricing.output_per_million_rmb
    ) / 1_000_000


__all__ = [
    "ModelBudget",
    "ModelBudgetExceededError",
    "ModelPricing",
    "ModelUsageLedger",
    "ModelUsageSnapshot",
]
