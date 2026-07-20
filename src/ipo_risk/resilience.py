"""可注入等待函数的同步指数退避重试策略。"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")


def retry_transient_api_error(error: Exception) -> bool:
    """只重试连接、超时、限流和服务端错误，跳过普通4xx。"""
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int):
        return status_code in {408, 409, 429} or status_code >= 500
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return True
    return type(error).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }


def retry_model_node_error(error: Exception) -> bool:
    """模型节点还会重试可由再次生成修复的结构和证据校验错误。"""
    return isinstance(error, ValueError) or retry_transient_api_error(error)


@dataclass(frozen=True)
class RetryResult(Generic[T]):
    value: T
    attempts: int


class RetryExhaustedError(RuntimeError):
    """保留最终异常和实际尝试次数，供降级与审计使用。"""

    def __init__(self, attempts: int, last_error: Exception) -> None:
        super().__init__(f"retry exhausted after {attempts} attempts: {last_error}")
        self.attempts = attempts
        self.last_error = last_error


@dataclass(frozen=True)
class RetryPolicy:
    """与LangGraph节点RetryPolicy等价的同步最小实现。"""

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

    def run(self, operation: Callable[[], T]) -> RetryResult[T]:
        attempts = 0
        delay = self.initial_seconds
        while True:
            attempts += 1
            try:
                return RetryResult(value=operation(), attempts=attempts)
            except Exception as error:
                retries_used = attempts - 1
                if retries_used >= self.max_retries or not self.retry_if(error):
                    raise RetryExhaustedError(attempts, error) from error
                self.sleep(min(delay, self.max_seconds))
                delay = min(delay * self.backoff_factor, self.max_seconds)


__all__ = [
    "RetryExhaustedError",
    "RetryPolicy",
    "RetryResult",
    "retry_model_node_error",
    "retry_transient_api_error",
]
