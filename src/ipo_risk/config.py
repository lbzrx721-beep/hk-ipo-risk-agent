"""读取项目内 YAML，并生成统一运行配置。

职责：
  1. 读取 configs/base.yaml 中的项目路径、解析器、检索、Agent、预测等配置。
  2. 读取 configs/providers.local.yaml 中的固定模型供应商与 API Key。
  3. 数据库连接串允许从项目.env或部署环境变量注入。
  4. 将配置合并为类型安全的 AppConfig 对象，供其他所有模块使用。

设计原则：
  - 其他模块不直接读 YAML，统一通过 load_config() 获取。
  - 模型密钥只读项目私有配置；数据库DSN允许标准部署环境变量覆盖。
  - API Key 在 Pydantic 对象中使用 SecretStr，避免日志和 repr 意外泄漏。
  - 所有路径都是相对项目根目录的，通过 resolve_path() 转为绝对路径。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, SecretStr

# ---------------------------------------------------------------------------
# 项目根目录：config.py 位于 src/ipo_risk/，往上两级就是项目根目录。
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# 各配置区段 —— 与 base.yaml 的层级一一对应
# ---------------------------------------------------------------------------


class PathsConfig(BaseModel):
    """各类数据目录的相对路径（相对于项目根目录）。"""

    prospectus_dir: str = "data/raw/prospectus"
    company_info_dir: str = "data/raw/company_info"
    market_dir: str = "data/raw/market"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    artifacts_dir: str = "artifacts"
    database: str = "data/app.db"


class ParserConfig(BaseModel):
    """PDF 解析器参数。

    render_dpi: 渲染页面图片的分辨率，越高截图越清晰但越慢。
    min_text_chars_before_ocr: 一页至少要有多少非空格字符才认为有文字层，
                                否则视为扫描页，触发 OCR 兜底。
    preserve_bbox: 是否保留每个文本块的页面坐标。
    preserve_page_image: 是否为每一页渲染一张完整图片（用于调试和截图）。
    """

    render_dpi: int = 150
    min_text_chars_before_ocr: int = 80
    preserve_bbox: bool = True
    preserve_page_image: bool = True


class RetrievalConfig(BaseModel):
    """检索参数：词法 / 语义 / 重排序各取多少候选。"""

    lexical_top_k: int = 80
    semantic_top_k: int = 20
    rerank_top_k: int = 8


class VectorStoreConfig(BaseModel):
    """向量存储后端配置。

    backend:
      - "numpy": 内存向量 + 磁盘缓存（单文档场景）。
      - "pgvector": PostgreSQL + pgvector 正式持久化与检索后端。
    """

    backend: Literal["numpy", "pgvector"] = "pgvector"
    postgres_dsn: SecretStr = Field(default_factory=lambda: SecretStr(""))
    postgres_dsn_env: str = "IPO_RISK_POSTGRES_DSN"
    postgres_schema: str = "ipo_risk"
    postgres_connect_timeout_seconds: float = Field(default=30.0, gt=0.0)

    def require_postgres_dsn(self) -> str:
        """返回数据库连接串；只允许数据库客户端在连接时读取明文。"""
        value = self.postgres_dsn.get_secret_value().strip()
        if not value:
            raise RuntimeError(
                "pgvector连接未配置：请在项目.env中填写IPO_RISK_POSTGRES_DSN"
            )
        return value


class EmbeddingConfig(BaseModel):
    """招股书语义检索使用的向量模型配置。"""

    provider: str = "zhipu"
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    model_name: str = "embedding-3-pro"
    dimensions: int = Field(default=1024, ge=1)
    batch_size: int = Field(default=64, ge=1)
    request_concurrency: int = Field(default=1, ge=1, le=32)
    document_concurrency: int = Field(default=1, ge=1, le=16)
    timeout_seconds: float = Field(default=120.0, gt=0.0)
    max_retries: int = Field(default=2, ge=0, le=10)
    retry_initial_seconds: float = Field(default=1.0, ge=0.0)
    retry_backoff_factor: float = Field(default=2.0, ge=1.0)
    retry_max_seconds: float = Field(default=8.0, ge=0.0)

    @property
    def is_configured(self) -> bool:
        """是否已经在项目本地配置文件中填写有效密钥。"""
        return self.provider.lower() == "ollama" or bool(
            self.api_key.get_secret_value().strip()
        )

    def require_api_key(self) -> str:
        """返回明文密钥；仅允许模型客户端在真正调用接口时使用。"""
        if self.provider.lower() == "ollama":
            return "ollama"
        value = self.api_key.get_secret_value().strip()
        if not value:
            raise RuntimeError("Embedding API Key 未配置：请检查 configs/providers.local.yaml")
        return value


class LLMConfig(BaseModel):
    """多 Agent 分析、核验和裁决使用的大语言模型配置。"""

    provider: str = "deepseek"
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    base_url: str = "https://api.deepseek.com"
    model_name: str = "deepseek-v4-flash"
    temperature: float = 0.1
    max_output_tokens: int = 8192
    supports_json_object: bool = True
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    disable_thinking: bool = True
    input_price_per_million_rmb: float | None = Field(default=None, ge=0.0)
    output_price_per_million_rmb: float | None = Field(default=None, ge=0.0)
    cached_input_price_per_million_rmb: float | None = Field(default=None, ge=0.0)

    @property
    def is_configured(self) -> bool:
        """是否已经在项目本地配置文件中填写有效密钥。"""
        return bool(self.api_key.get_secret_value().strip())

    def require_api_key(self) -> str:
        """返回明文密钥；仅允许模型客户端在真正调用接口时使用。"""
        value = self.api_key.get_secret_value().strip()
        if not value:
            raise RuntimeError(
                f"{self.provider} API Key 未配置：请检查 configs/providers.local.yaml"
            )
        return value


class ModelBudgetConfig(BaseModel):
    """可选的单公司硬预算；默认仅统计用量，不中断分析。"""

    max_calls: int | None = Field(default=None, ge=1)
    max_input_tokens: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    max_cost_rmb: float | None = Field(default=None, gt=0.0)


class AgentsConfig(BaseModel):
    """多 Agent 协作配置。

    模型供应商和密钥由顶层 llm 配置统一管理；这里仅保存编排约束。
    """

    max_retries: int = Field(default=2, ge=0, le=10)
    retry_initial_seconds: float = Field(default=1.0, ge=0.0, le=60.0)
    retry_backoff_factor: float = Field(default=2.0, ge=1.0, le=10.0)
    retry_max_seconds: float = Field(default=8.0, ge=0.0, le=300.0)
    # 批量分析时允许同时推进的公司流水线数量。每家公司内部的专业 Agent
    # 仍由 LangGraph 并行编排，因此这里必须保持受控，避免瞬时放大 API 请求。
    company_concurrency: int = Field(default=2, ge=1, le=8)
    enable_open_discovery: bool = True
    # 风险发现先动态规划检索问题，再从向量索引取回原文和相邻上下文。
    discovery_retrieval_query_count: int = Field(default=16, ge=4, le=20)
    discovery_retrieval_top_k: int = Field(default=18, ge=2, le=50)
    discovery_retrieval_lexical_top_k: int = Field(default=10, ge=1, le=30)
    discovery_retrieval_neighbor_radius: int = Field(default=1, ge=0, le=3)
    discovery_coverage_sections: int = Field(default=48, ge=4, le=100)
    discovery_max_selected_chunks: int = Field(default=520, ge=50, le=2_000)
    # 取回的原文仍按容量分包；同一波并行、下一波共享前序风险主题记忆。
    discovery_batch_chars: int = Field(default=70_000, ge=2_000, le=1_000_000)
    discovery_wave_size: int = Field(default=4, ge=1, le=4)
    # 首轮完整发现，第二轮仅补充首轮遗漏的、证据独立的重大风险。
    discovery_max_batch_passes: int = Field(default=2, ge=1, le=4)
    consolidation_batch_size: int = Field(default=24, ge=2, le=100)
    specialist_batch_size: int = Field(default=100, ge=2, le=200)
    specialist_followup_rounds: int = Field(default=0, ge=0, le=2)
    decision_conflict_batch_size: int = Field(default=12, ge=2, le=30)
    model_budget: ModelBudgetConfig = Field(default_factory=ModelBudgetConfig)
    require_evidence: bool = True
    require_verification: bool = True


class PredictionConfig(BaseModel):
    """预测模块参数。

    horizons: 赛题要求的四个交易日窗口。
    primary_horizon: 主任务窗口（第 5 日权重最高）。
    forbid_post_listing_features: Point-in-Time 约束开关。
    """

    horizons: list[int] = Field(default_factory=lambda: [1, 5, 20, 60])
    primary_horizon: int = 5
    significant_drop_threshold_pct: float = Field(default=-10.0, ge=-100.0, le=0.0)
    split_strategy: str = "time_ordered"
    forbid_post_listing_features: bool = True
    market_feature_windows: list[int] = Field(default_factory=lambda: [5, 20, 60])
    recent_ipo_lookback_days: int = Field(default=180, ge=30, le=730)
    top_k_fraction: float = Field(default=0.2, gt=0.0, le=1.0)
    max_alert_fraction: float = Field(default=0.5, gt=0.0, lt=1.0)
    random_state: int = 42


# ---------------------------------------------------------------------------
# 顶层配置
# ---------------------------------------------------------------------------


class AppConfig(BaseModel):
    """合并基础 YAML 与项目固定供应商 YAML 后的最终运行配置。"""

    project_name: str = "hk-ipo-risk-agent"
    timezone: str = "Asia/Shanghai"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    parser: ParserConfig = Field(default_factory=ParserConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    specialist_llm: LLMConfig | None = None
    # 兼容旧配置名；新代码应使用 specialist_llm。
    discovery_llm: LLMConfig | None = None
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)

    @property
    def project_root(self) -> Path:
        """项目根目录的绝对路径。"""
        return _PROJECT_ROOT

    def resolve_path(self, relative: str) -> Path:
        """将配置中的相对路径转为绝对路径。

        例如: config.resolve_path("data/raw/prospectus")
              => D:\\东吴证券\\data\\raw\\prospectus
        """
        return _PROJECT_ROOT / relative


# ---------------------------------------------------------------------------
# 加载函数
# ---------------------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any]:
    """读取一个 YAML 映射；文件不存在时返回空映射。"""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是 YAML 映射: {path}")
    return data


def _read_project_env(path: Path) -> dict[str, str]:
    """读取项目级.env；仅用于数据库等部署配置，不污染进程环境。"""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith(("'", '"')) and value.endswith(value[:1]):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_config(
    config_path: str | Path | None = None,
    providers_path: str | Path | None = None,
) -> AppConfig:
    """加载基础配置与项目内固定供应商配置。

    优先级：providers.local.yaml > base.yaml > 代码默认值。
    模型供应商仍固定读取项目配置；数据库DSN允许由项目.env或同名环境变量覆盖。

    参数:
        config_path: 可选的 YAML 路径，默认读 configs/base.yaml。
        providers_path: 可选的供应商 YAML 路径，默认读
                        configs/providers.local.yaml。

    返回:
        合并后的 AppConfig 实例。
    """
    # 1. 确定配置文件路径
    if config_path is None:
        config_path = _PROJECT_ROOT / "configs" / "base.yaml"
    config_path = Path(config_path)

    if providers_path is None:
        providers_path = _PROJECT_ROOT / "configs" / "providers.local.yaml"
    providers_path = Path(providers_path)

    # 2. 读取基础配置和固定供应商配置
    raw = _read_yaml(config_path)
    providers = _read_yaml(providers_path)

    # 3. 逐区段提取，缺失的区段用空字典兖底
    project = raw.get("project", {})
    paths = raw.get("paths", {})
    parser_cfg = raw.get("parser", {})
    retrieval_cfg = raw.get("retrieval", {})
    vector_store_cfg = {
        **raw.get("vector_store", {}),
        **providers.get("vector_store", {}),
    }
    postgres_dsn_env = str(
        vector_store_cfg.get("postgres_dsn_env") or "IPO_RISK_POSTGRES_DSN"
    )
    project_env = _read_project_env(_PROJECT_ROOT / ".env")
    postgres_dsn = os.environ.get(postgres_dsn_env) or project_env.get(
        postgres_dsn_env
    )
    if postgres_dsn:
        vector_store_cfg["postgres_dsn"] = postgres_dsn
    embedding_cfg = {**raw.get("embedding", {}), **providers.get("embedding", {})}
    llm_cfg = {**raw.get("llm", {}), **providers.get("llm", {})}
    raw_discovery_llm = raw.get("discovery_llm")
    provider_discovery_llm = providers.get("discovery_llm")
    raw_specialist_llm = raw.get("specialist_llm", raw_discovery_llm)
    provider_specialist_llm = providers.get(
        "specialist_llm",
        provider_discovery_llm,
    )
    discovery_llm_cfg = None
    if isinstance(raw_discovery_llm, dict) or isinstance(provider_discovery_llm, dict):
        discovery_llm_cfg = {
            **(raw_discovery_llm if isinstance(raw_discovery_llm, dict) else {}),
            **(
                provider_discovery_llm
                if isinstance(provider_discovery_llm, dict)
                else {}
            ),
        }
    specialist_llm_cfg = None
    if isinstance(raw_specialist_llm, dict) or isinstance(
        provider_specialist_llm,
        dict,
    ):
        specialist_llm_cfg = {
            **(
                raw_specialist_llm
                if isinstance(raw_specialist_llm, dict)
                else {}
            ),
            **(
                provider_specialist_llm
                if isinstance(provider_specialist_llm, dict)
                else {}
            ),
        }
    agents_cfg = raw.get("agents", {})
    prediction_cfg = raw.get("prediction", {})

    # 4. 构造并返回 AppConfig；不执行任何环境变量覆盖
    return AppConfig(
        project_name=project.get("name", "hk-ipo-risk-agent"),
        timezone=project.get("timezone", "Asia/Shanghai"),
        paths=PathsConfig(**paths),
        parser=ParserConfig(**parser_cfg),
        retrieval=RetrievalConfig(**retrieval_cfg),
        vector_store=VectorStoreConfig(**vector_store_cfg),
        embedding=EmbeddingConfig(**embedding_cfg),
        llm=LLMConfig(**llm_cfg),
        specialist_llm=(
            LLMConfig(**specialist_llm_cfg)
            if specialist_llm_cfg is not None
            else None
        ),
        discovery_llm=(
            LLMConfig(**discovery_llm_cfg)
            if discovery_llm_cfg is not None
            else None
        ),
        agents=AgentsConfig(**agents_cfg),
        prediction=PredictionConfig(**prediction_cfg),
    )
