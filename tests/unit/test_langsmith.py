from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from langgraph.graph import END, START, StateGraph
from langsmith import Client, tracing_context
from langsmith.utils import ContextThreadPoolExecutor
from openai import OpenAI

from ipo_risk.agents.analysis import OpenAICompatibleJsonChatProvider
from ipo_risk.observability.trace import (
    _ENV_KEYS,
    configure_langsmith,
    retrieval_outputs,
    trace_step,
    wrap_chat_client,
    wrap_embedding_client,
)


@pytest.fixture(autouse=True)
def no_cloud_upload(monkeypatch):
    # LangGraph 的 callback 也可能写入，测试中统一拦截 SDK 上传入口。
    create = Mock()
    monkeypatch.setattr(Client, "create_run", create)
    monkeypatch.setattr(Client, "update_run", Mock())
    return create


@pytest.fixture
def clean_env(monkeypatch):
    original = {key: os.environ.get(key) for key in _ENV_KEYS}
    for key in _ENV_KEYS:
        os.environ.pop(key, None)
    yield
    for key, value in original.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_config_is_opt_in_and_only_imports_langsmith(clean_env, monkeypatch):
    monkeypatch.setenv("LANGSMITH_PROJECT", "deployment")
    configure_langsmith(
        {"LANGSMITH_API_KEY": "test-key", "LANGSMITH_PROJECT": "local", "OTHER_SECRET": "secret"}
    )
    assert os.environ["LANGSMITH_PROJECT"] == "deployment"
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert "OTHER_SECRET" not in os.environ


def test_config_missing_key_disables_upload(clean_env, caplog):
    configure_langsmith({"LANGSMITH_TRACING": "true"})
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert "未配置 API Key" in caplog.text


def test_config_can_enable_with_key(clean_env):
    configure_langsmith({"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "test-key"})
    assert os.environ["LANGSMITH_TRACING"] == "true"


def test_disabled_tracing_does_not_upload(no_cloud_upload):
    @trace_step("disabled", inputs=("query",))
    def run(query):
        return query

    with tracing_context(enabled=False):
        assert run("test") == "test"
    no_cloud_upload.assert_not_called()


def test_query_outputs_and_thread_parent_are_captured_without_secrets():
    completed = []
    chunk = SimpleNamespace(text="公开测试段落", chunk_id="c1", document_id="d1", page=2)

    class Retriever:
        api_key = "must-not-be-traced"

        @trace_step(
            "test.search",
            inputs=("query", "top_k"),
            run_type="retriever",
            process_outputs=retrieval_outputs,
        )
        def search(self, query, top_k=2):
            return [SimpleNamespace(chunk=chunk, score=0.8, rank=1)]

    @trace_step("root", inputs=("company_id",))
    def root(company_id, postgres_dsn):
        with ContextThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(Retriever().search, "收入风险").result()

    with tracing_context(enabled="local"):
        result = root("demo", "secret-dsn", langsmith_extra={"on_end": completed.append})
    run = completed[0]
    child = run.child_runs[0]
    assert len(result) == 1
    assert run.inputs == {"company_id": "demo"}
    assert child.parent_run_id == run.id
    assert child.inputs == {"query": "收入风险", "top_k": 2}
    assert child.outputs["documents"][0]["page_content"] == "公开测试段落"
    assert child.outputs["documents"][0]["metadata"]["page"] == 2


@pytest.mark.parametrize("stream", [False, True])
def test_chat_captures_response_usage_and_keeps_streaming(stream):
    requests = []
    completed = []
    deltas = []

    def handle(request):
        requests.append(request)
        if stream:
            events = [
                {
                    "id": "chat-test",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": '{"ok":true}'},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chat-test",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "test-model",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
                },
            ]
            body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
            return httpx.Response(
                200, text=body + "data: [DONE]\n\n", headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(
            200,
            json={
                "id": "chat-test",
                "object": "chat.completion",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"ok":true}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            },
        )

    with OpenAI(
        api_key="local-test-secret",
        base_url="https://offline.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ) as client:
        provider = OpenAICompatibleJsonChatProvider(
            model_name="test-model", client=wrap_chat_client(client, "ark")
        )
        with tracing_context(enabled="local"):
            result = provider.complete_json(
                system_prompt="测试",
                user_prompt="你好",
                temperature=0,
                max_output_tokens=32,
                on_text_delta=deltas.append if stream else None,
                langsmith_extra={"on_end": completed.append},
            )
    run = completed[0]
    model_run = run.child_runs[0]
    assert result == {"ok": True}
    assert len(requests) == 1
    assert model_run.run_type == "llm"
    assert model_run.inputs["messages"][1]["content"] == "你好"
    assert model_run.outputs["choices"][0]["message"]["content"] == '{"ok":true}'
    assert model_run.outputs["usage_metadata"]["total_tokens"] == 14
    assert "local-test-secret" not in str(run.inputs) + str(model_run.inputs)
    if stream:
        assert "".join(deltas) == '{"ok":true}'


def test_embedding_records_summary_without_vectors():
    completed = []

    def handle(request):
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "embed-test",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    with OpenAI(
        api_key="offline",
        base_url="https://offline.test/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    ) as client:
        wrap_embedding_client(client)
        with tracing_context(enabled="local"):
            response = client.embeddings.create(
                model="embed-test",
                input=["公开文字"],
                encoding_format="float",
                langsmith_extra={"on_end": completed.append},
            )
    assert response.data[0].embedding == [0.1, 0.2]
    assert completed[0].outputs["vector_count"] == 1
    assert completed[0].outputs["usage"]["total_tokens"] == 2
    assert "embedding" not in completed[0].outputs


def test_langgraph_nodes_are_nested_under_analysis():
    completed = []
    graph = StateGraph(dict)
    graph.add_node("finance_agent", lambda state: {"ok": True})
    graph.add_edge(START, "finance_agent")
    graph.add_edge("finance_agent", END)
    compiled = graph.compile()

    @trace_step("analysis", inputs=())
    def run():
        return compiled.invoke({})

    with tracing_context(enabled="local"):
        assert run(langsmith_extra={"on_end": completed.append}) == {"ok": True}
    graph_run = completed[0].child_runs[0]
    assert graph_run.parent_run_id == completed[0].id
    assert "finance_agent" in {child.name for child in graph_run.child_runs}


def test_errors_are_recorded_and_still_raised():
    completed = []

    @trace_step("broken_tool", inputs=("query",), run_type="tool")
    def broken(query):
        raise ValueError("offline error")

    with tracing_context(enabled="local"), pytest.raises(ValueError, match="offline error"):
        broken("query", langsmith_extra={"on_end": completed.append})
    assert "offline error" in completed[0].error
