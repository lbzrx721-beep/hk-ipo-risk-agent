"""LangSmith 接入：显式开启、限制配置上传、保留检索与模型调用轨迹。"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

from langsmith import traceable
from langsmith.wrappers import wrap_openai


_ENV_KEYS = (
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_WORKSPACE_ID",
    "LANGSMITH_HIDE_INPUTS",
    "LANGSMITH_HIDE_OUTPUTS",
)


def configure_langsmith(project_env: Mapping[str, str]) -> None:
    """仅注入 LangSmith 配置；部署环境优先，修改后需重启服务。"""
    for key in _ENV_KEYS:
        if key in project_env:
            os.environ.setdefault(key, project_env[key])
    os.environ.setdefault("LANGSMITH_PROJECT", "hk-ipo-risk-agent")
    os.environ.setdefault("LANGSMITH_TRACING", "false")
    if os.environ["LANGSMITH_TRACING"].lower() == "true" and not (
        os.environ.get("LANGSMITH_API_KEY", "").strip()
    ):
        os.environ["LANGSMITH_TRACING"] = "false"
        logging.getLogger(__name__).warning(
            "LangSmith 未配置 API Key，本次启动已关闭云端追踪；分析功能不受影响。"
        )


def trace_step(name: str, *, inputs: Sequence[str], run_type: str = "chain", **kwargs: Any):
    """只记录指定输入，避免将 self、客户端、连接串或回调对象序列化。"""
    allowed = frozenset(inputs)
    return traceable(
        name=name,
        run_type=run_type,
        process_inputs=lambda values: {k: v for k, v in values.items() if k in allowed},
        **kwargs,
    )


def retrieval_outputs(outputs: list[Any]) -> dict[str, Any]:
    """只记录命中原文、页码和排名，不上传整个索引或数据库对象。"""
    return {
        "documents": [
            {
                "page_content": hit.chunk.text,
                "metadata": {
                    "chunk_id": hit.chunk.chunk_id,
                    "document_id": hit.chunk.document_id,
                    "page": hit.chunk.page,
                    "score": hit.score,
                    "rank": hit.rank,
                },
            }
            for hit in outputs
        ]
    }


def wrap_chat_client(client: Any, provider_name: str) -> Any:
    """官方适配器同时记录普通/流式响应和供应商返回的 Token 用量。"""
    return wrap_openai(
        client,
        chat_name=f"{provider_name}.chat",
        tracing_extra={
            "metadata": {
                "ls_invocation_params": {"ls_provider": provider_name},
            }
        },
    )


def wrap_embedding_client(client: Any) -> Any:
    """逐次记录 Embedding 请求，省略体积很大的向量值。"""

    def summarize(response: Any) -> dict[str, Any]:
        return {
            "model": response.model,
            "vector_count": len(response.data),
            "usage": response.usage.model_dump() if response.usage else {},
        }

    client.embeddings.create = traceable(
        name="embeddings.create",
        run_type="embedding",
        process_outputs=summarize,
    )(client.embeddings.create)
    return client
