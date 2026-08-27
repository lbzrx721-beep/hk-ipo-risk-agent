"""项目固定供应商配置测试。"""

from pathlib import Path

import pytest

from ipo_risk.config import load_config
from pydantic import ValidationError


def test_vector_store_backend_rejects_typos(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text("vector_store:\n  backend: chromma", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(base_path, tmp_path / "missing-providers.yaml")


def test_numpy_vector_backend_is_supported_for_local_backup(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text("vector_store:\n  backend: numpy", encoding="utf-8")

    config = load_config(base_path, tmp_path / "missing-providers.yaml")

    assert config.vector_store.backend == "numpy"


def test_pgvector_backend_reads_secret_dsn_from_deployment_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        "vector_store:\n  backend: pgvector\n  postgres_schema: ipo_risk",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "IPO_RISK_POSTGRES_DSN",
        "postgresql://appuser:secret@database.local:5432/ragdb",
    )

    config = load_config(base_path, tmp_path / "missing-providers.yaml")

    assert config.vector_store.backend == "pgvector"
    assert config.vector_store.postgres_schema == "ipo_risk"
    assert "secret" not in repr(config)
    assert config.vector_store.require_postgres_dsn().endswith("/ragdb")


def test_provider_file_has_priority_and_environment_is_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """供应商只能由项目 YAML 决定，不能被用户环境变量悄悄覆盖。"""
    base_path = tmp_path / "base.yaml"
    providers_path = tmp_path / "providers.local.yaml"
    base_path.write_text(
        """
embedding:
  provider: zhipu
  model_name: embedding-3-pro
llm:
  provider: deepseek
  model_name: deepseek-v4-flash
""".strip(),
        encoding="utf-8",
    )
    providers_path.write_text(
        """
embedding:
  api_key: project-zhipu-key
llm:
  api_key: project-deepseek-key
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("ZHIPU_API_KEY", "environment-zhipu-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-deepseek-key")

    config = load_config(base_path, providers_path)

    assert config.embedding.require_api_key() == "project-zhipu-key"
    assert config.llm.require_api_key() == "project-deepseek-key"


def test_secret_values_are_masked_in_config_repr(tmp_path: Path) -> None:
    """明文只存在本地 YAML 中，不应被配置对象 repr 意外打印。"""
    providers_path = tmp_path / "providers.local.yaml"
    providers_path.write_text(
        """
embedding:
  api_key: visible-only-in-local-yaml
llm:
  api_key: another-local-secret
""".strip(),
        encoding="utf-8",
    )

    config = load_config(providers_path=providers_path)
    rendered = repr(config)

    assert "visible-only-in-local-yaml" not in rendered
    assert "another-local-secret" not in rendered


def test_ollama_embedding_uses_non_secret_compatibility_key(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        """
embedding:
  provider: ollama
  base_url: http://192.168.3.91:11434/v1
  model_name: qwen3-embedding:4b
  dimensions: 1024
""".strip(),
        encoding="utf-8",
    )

    config = load_config(base_path, tmp_path / "missing-providers.yaml")

    assert config.embedding.is_configured is True
    assert config.embedding.require_api_key() == "ollama"


def test_embedding_api_model_name_can_differ_from_index_name(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        """
embedding:
  provider: siliconflow
  base_url: https://api.siliconflow.cn/v1
  model_name: qwen3-embedding:4b
  api_model_name: Qwen/Qwen3-Embedding-4B
  dimensions: 1024
""".strip(),
        encoding="utf-8",
    )
    providers_path = tmp_path / "providers.local.yaml"
    providers_path.write_text(
        """
embedding:
  api_key: test-key
""".strip(),
        encoding="utf-8",
    )

    config = load_config(base_path, providers_path)

    assert config.embedding.model_name == "qwen3-embedding:4b"
    assert config.embedding.request_model_name == "Qwen/Qwen3-Embedding-4B"


def test_embedding_provider_performance_settings_are_loaded(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        """
embedding:
  request_concurrency: 5
  document_concurrency: 4
  timeout_seconds: 90
  max_retries: 3
  retry_initial_seconds: 0.25
  retry_backoff_factor: 1.5
  retry_max_seconds: 4
""".strip(),
        encoding="utf-8",
    )

    config = load_config(base_path, tmp_path / "missing-providers.yaml")

    assert config.embedding.request_concurrency == 5
    assert config.embedding.document_concurrency == 4
    assert config.embedding.timeout_seconds == 90
    assert config.embedding.max_retries == 3
    assert config.embedding.retry_initial_seconds == 0.25
    assert config.embedding.retry_backoff_factor == 1.5
    assert config.embedding.retry_max_seconds == 4


def test_model_retry_policy_is_loaded_from_project_config(tmp_path: Path) -> None:
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        """
agents:
  max_retries: 4
  retry_initial_seconds: 0.5
  retry_backoff_factor: 1.5
  retry_max_seconds: 6.0
""".strip(),
        encoding="utf-8",
    )

    config = load_config(base_path, tmp_path / "missing-providers.yaml")

    assert config.agents.max_retries == 4
    assert config.agents.retry_initial_seconds == 0.5
    assert config.agents.retry_backoff_factor == 1.5
    assert config.agents.retry_max_seconds == 6.0


def test_default_discovery_pack_is_small_enough_for_early_visible_results(
    tmp_path: Path,
) -> None:
    config = load_config(
        Path("configs/base.yaml"),
        tmp_path / "missing-providers.yaml",
    )

    assert config.agents.discovery_batch_chars == 70_000
    assert config.agents.discovery_wave_size == 4
    assert config.agents.discovery_retrieval_query_count == 16
    assert config.agents.discovery_max_selected_chunks == 520
    assert config.agents.discovery_max_batch_passes == 2
    assert config.agents.specialist_batch_size == 12
    assert config.agents.specialist_followup_rounds == 0
    assert config.agents.model_budget.max_calls is None
    assert config.agents.model_budget.max_input_tokens is None
    assert config.agents.model_budget.max_output_tokens is None
    assert config.llm.provider == "ark"
    assert config.llm.model_name == "deepseek-v4-flash"
    assert config.llm.supports_json_object is False
    assert config.specialist_llm is not None
    assert config.specialist_llm.provider == "ark"
    assert config.specialist_llm.base_url == (
        "https://ark.cn-beijing.volces.com/api/plan/v3"
    )
    assert config.specialist_llm.model_name == "deepseek-v4-flash"
    assert config.specialist_llm.supports_json_object is False
    assert config.specialist_llm.reasoning_effort is None
    assert config.specialist_llm.disable_thinking is True
