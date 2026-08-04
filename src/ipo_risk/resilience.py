"""可注入等待函数的同步指数退避重试策略。"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import ValidationError


T = TypeVar("T")
RetryCallback = Callable[[Exception, int, float], None]


_TRANSIENT_ERROR_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "ConnectError",
    "ConnectTimeout",
    "ConnectionError",
    "InternalServerError",
    "NetworkError",
    "PoolTimeout",
    "ProtocolError",
    "RateLimitError",
    "ReadError",
    "ReadTimeout",
    "RemoteProtocolError",
    "WriteError",
    "WriteTimeout",
}
_TRANSIENT_MESSAGE_MARKERS = (
    "incomplete chunked read",
    "peer closed connection",
    "connection reset",
    "connection aborted",
    "connection refused",
    "connection timed out",
    "error code: 429",
    "status code: 429",
    "502 bad gateway",
    "503 service unavailable",
    "read timed out",
    "server overloaded",
    "service unavailable",
    "too many requests",
    "temporarily unavailable",
    "upstream connect error",
)


def _exception_chain(error: Exception) -> tuple[Exception, ...]:
    """返回异常及其cause/context，兼容SDK对底层httpx异常的包装。"""
    values: list[Exception] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while isinstance(current, Exception) and id(current) not in seen:
        seen.add(id(current))
        values.append(current)
        current = current.__cause__ or current.__context__
    return tuple(values)


def retry_transient_api_error(error: Exception) -> bool:
    """只重试连接、超时、限流和服务端错误，跳过普通4xx。"""
    for item in _exception_chain(error):
        status_code = getattr(item, "status_code", None)
        if isinstance(status_code, int):
            if status_code in {408, 409, 429} or status_code >= 500:
                return True
            continue
        if isinstance(item, (ConnectionError, TimeoutError, OSError)):
            return True
        if type(item).__name__ in _TRANSIENT_ERROR_NAMES:
            return True
        message = str(item).strip().lower()
        if any(marker in message for marker in _TRANSIENT_MESSAGE_MARKERS):
            return True
    return False


def retry_model_node_error(error: Exception) -> bool:
    """模型节点还会重试可由再次生成修复的结构和证据校验错误。"""
    # 预算门禁属于确定性暂停条件，重试不会恢复额度，必须立即向上传播。
    from ipo_risk.agents.usage import ModelBudgetExceededError

    if isinstance(error, ModelBudgetExceededError):
        return False
    return isinstance(error, ValueError) or retry_transient_api_error(error)


@dataclass(frozen=True)
class RetryResult(Generic[T]):
    value: T
    attempts: int
    errors: tuple[str, ...] = ()


class RetryExhaustedError(RuntimeError):
    """保留最终异常和实际尝试次数，供降级与审计使用。"""

    def __init__(self, attempts: int, last_error: Exception) -> None:
        super().__init__(f"retry exhausted after {attempts} attempts: {last_error}")
        self.attempts = attempts
        self.last_error = last_error


@dataclass(frozen=True)
class RetryPolicy:
    """保留精确尝试次数审计的模型子步骤同步重试策略。"""

    max_retries: int = 0
    initial_seconds: float = 1.0
    backoff_factor: float = 2.0
    max_seconds: float = 8.0
    retry_if: Callable[[Exception], bool] = retry_transient_api_error
    sleep: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.initial_seconds < 0 or self.max_seconds < 0:
            raise ValueError("retry intervals must be non-negative")
        if self.backoff_factor < 1:
            raise ValueError("retry backoff_factor must be at least 1")

    def run(
        self,
        operation: Callable[[], T],
        *,
        on_retry: RetryCallback | None = None,
    ) -> RetryResult[T]:
        attempts = 0
        delay = self.initial_seconds
        errors: list[str] = []
        while True:
            attempts += 1
            try:
                return RetryResult(
                    value=operation(),
                    attempts=attempts,
                    errors=tuple(errors),
                )
            except Exception as error:
                errors.append(_compact_error(error))
                retries_used = attempts - 1
                if retries_used >= self.max_retries or not self.retry_if(error):
                    raise RetryExhaustedError(attempts, error) from error
                wait_seconds = min(delay, self.max_seconds)
                if on_retry is not None:
                    on_retry(error, attempts + 1, wait_seconds)
                self.sleep(wait_seconds)
                delay = min(delay * self.backoff_factor, self.max_seconds)

    def run_with_feedback(
        self,
        operation: Callable[[str | None], T],
        *,
        on_retry: RetryCallback | None = None,
    ) -> RetryResult[T]:
        """重试模型结构错误时，把具体校验问题反馈给下一次生成。"""
        attempts = 0
        delay = self.initial_seconds
        feedback: str | None = None
        errors: list[str] = []
        while True:
            attempts += 1
            try:
                return RetryResult(
                    value=operation(feedback),
                    attempts=attempts,
                    errors=tuple(errors),
                )
            except Exception as error:
                compact = _compact_error(error)
                errors.append(compact)
                retries_used = attempts - 1
                if retries_used >= self.max_retries or not self.retry_if(error):
                    raise RetryExhaustedError(attempts, error) from error
                feedback = (
                    _model_validation_feedback(error)
                    if isinstance(error, (ValueError, ValidationError))
                    else None
                )
                wait_seconds = min(delay, self.max_seconds)
                if on_retry is not None:
                    on_retry(error, attempts + 1, wait_seconds)
                self.sleep(wait_seconds)
                delay = min(delay * self.backoff_factor, self.max_seconds)


def _compact_error(error: Exception, *, limit: int = 4_000) -> str:
    message = str(error).strip() or type(error).__name__
    return message[:limit]


def _model_validation_feedback(error: Exception) -> str:
    """把 Pydantic/JSON 错误压缩成模型可直接修正的结构化反馈。"""
    if isinstance(error, ValidationError):
        details: list[dict[str, object]] = []
        for item in error.errors(include_url=False, include_input=True)[:12]:
            raw_input = item.get("input")
            try:
                rendered_input = json.dumps(raw_input, ensure_ascii=False)
            except TypeError:
                rendered_input = repr(raw_input)
            details.append(
                {
                    "field": ".".join(str(part) for part in item.get("loc", ())),
                    "error": str(item.get("msg", "validation error")),
                    "invalid_value": rendered_input[:500],
                }
            )
        return json.dumps(details, ensure_ascii=False)[:4_000]
    if isinstance(error, json.JSONDecodeError):
        excerpt = error.doc[max(0, error.pos - 300) : error.pos + 300]
        return (f"返回内容不是合法JSON：{error.msg}，位置{error.pos}；错误附近内容：{excerpt}")[
            :4_000
        ]
    return _compact_error(error)


__all__ = [
    "RetryExhaustedError",
    "RetryCallback",
    "RetryPolicy",
    "RetryResult",
    "_model_validation_feedback",
    "retry_model_node_error",
    "retry_transient_api_error",
]
