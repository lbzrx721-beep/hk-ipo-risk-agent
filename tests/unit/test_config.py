"""项目固定供应商配置测试。"""

from pathlib import Path

import pytest

from ipo_risk.config import load_config


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
  provider: dashscope
  model_name: qwen3.7-text-embedding
llm:
  provider: deepseek
  model_name: deepseek-v4-flash
""".strip(),
        encoding="utf-8",
    )
    providers_path.write_text(
        """
embedding:
  api_key: project-dashscope-key
llm:
  api_key: project-deepseek-key
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "environment-dashscope-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-deepseek-key")

    config = load_config(base_path, providers_path)

    assert config.embedding.require_api_key() == "project-dashscope-key"
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
