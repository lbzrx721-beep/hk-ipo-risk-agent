"""结构化应用日志配置。

职责：
  初始化 structlog，让所有模块通过 get_logger() 获取带上下文的日志器。

用法：
    from ipo_risk.observability.logging import setup_logging, get_logger
    setup_logging()                  # 应用启动时调用一次
    log = get_logger("parser")       # 各模块按名字获取日志器
    log.info("parsed_page", page=42, blocks=17)  # 结构化输出

输出示例：
    2026-07-17T10:30:00 [info     ] parsed_page  page=42 blocks=17
"""

from __future__ import annotations

import logging
import sys

import structlog


# ---------------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    """初始化全局结构化日志。

    参数:
        level: 日志级别，例如 "DEBUG", "INFO", "WARNING"。

    设计说明:
        - 开发阶段用 ConsoleRenderer，输出彩色可读格式。
        - 生产/比赛交付时可换成 JSONRenderer，方便机器解析。
        - TimeStamper 统一用 ISO 格式，和数据契约的时间格式一致。
    """
    structlog.configure(
        processors=[
            # 合并上下文变量（支持 contextvars 绑定请求级信息）
            structlog.contextvars.merge_contextvars,
            # 自动添加 level 字段
            structlog.stdlib.add_log_level,
            # 支持 log.info("msg %s", arg) 的格式化
            structlog.stdlib.PositionalArgumentsFormatter(),
            # 添加 ISO 时间戳
            structlog.processors.TimeStamper(fmt="iso"),
            # 渲染为可读的控制台格式
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # 同时配置标准库 logging，确保第三方库的日志也能输出
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level.upper(), logging.INFO),
    )


# ---------------------------------------------------------------------------
# 便捷获取日志器
# ---------------------------------------------------------------------------

def get_logger(name: str = "ipo_risk") -> structlog.stdlib.BoundLogger:
    """获取带模块名的结构化日志器。

    参数:
        name: 模块名称，例如 "parser", "retrieval", "finance"。

    返回:
        structlog 的 BoundLogger 实例。
    """
    return structlog.get_logger(name)
