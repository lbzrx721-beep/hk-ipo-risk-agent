"""提示词组合、指纹与模型纠错输入的公共工具。"""

from __future__ import annotations

from hashlib import sha256
from typing import Any


def compose_prompt(base_prompt: str, role_prompt: str | None = None) -> str:
    """组合公共任务契约与专业角色章程，保持稳定分隔以便生成指纹。"""
    base = base_prompt.strip()
    role = (role_prompt or "").strip()
    if not role:
        return base
    return f"{base}\n\n---\n\n# 当前专业角色章程\n\n{role}"


def prompt_sha256(prompt: str) -> str:
    """返回实际发送给模型的提示词内容指纹。"""
    return sha256(prompt.encode("utf-8")).hexdigest()


def prompt_audit(
    *,
    prompt: str,
    version: str,
) -> dict[str, str]:
    """生成可直接写入 ToolCall 的提示词审计快照。"""
    return {
        "prompt_version": version,
        "prompt_sha256": prompt_sha256(prompt),
        "prompt_snapshot": prompt,
    }


def with_correction(
    payload: dict[str, Any],
    correction_feedback: str | None,
) -> dict[str, Any]:
    """在纠错重试时向模型明确反馈上一轮违反的输出契约。"""
    if not correction_feedback:
        return payload
    return {
        **payload,
        "correction": {
            "previous_attempt_validation_error": correction_feedback,
            "return_a_complete_corrected_json_object": True,
            "do_not_repeat_the_invalid_shape_or_references": True,
        },
    }


__all__ = [
    "compose_prompt",
    "prompt_audit",
    "prompt_sha256",
    "with_correction",
]
