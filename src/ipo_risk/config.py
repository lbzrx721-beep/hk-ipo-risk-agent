"""读取项目内 YAML，并生成统一运行配置。

职责：
  1. 读取 configs/base.yaml 中的项目路径、解析器、检索、Agent、预测等配置。
  2. 读取 configs/providers.local.yaml 中的固定模型供应商与 API Key。
  3. 将两者合并为一个类型安全的 AppConfig 对象，供其他所有模块使用。

设计原则：
  - 其他模块不直接读 YAML，统一通过 load_config() 获取。
  - 不读取 Windows 用户环境变量，保证实际运行配置只由项目文件决定。
  - API Key 在 Pydantic 对象中使用 SecretStr，避免日志和 repr 意外泄漏。
  - 所有路径都是相对项目根目录的，通过 resolve_path() 转为绝对路径。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    annotations_dir: str = "data/annotations"
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


class EmbeddingConfig(BaseModel):
    """招股书语义检索使用的向量模型配置。"""

    provider: str = "dashscope"
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_name: str = "qwen3.7-text-embedding"
    dimensions: int = 1024
    batch_size: int = 20

    @property
    def is_configured(self) -> bool:
        """是否已经在项目本地配置文件中填写有效密钥。"""
        return bool(self.api_key.get_secret_value().strip())

    def require_api_key(self) -> str:
        """返回明文密钥；仅允许模型客户端在真正调用接口时使用。"""
        value = self.api_key.get_secret_value().strip()
        if not value:
            raise RuntimeError("DashScope API Key 未配置：请检查 configs/providers.local.yaml")
        return value


class LLMConfig(BaseModel):
    """多 Agent 分析、核验和裁决使用的大语言模型配置。"""

    provider: str = "deepseek"
    api_key: SecretStr = Field(default_factory=lambda: SecretStr(""))
    base_url: str = "https://api.deepseek.com"
    model_name: str = "deepseek-v4-flash"
    temperature: float = 0.1
    max_output_tokens: int = 8192

    @property
    def is_configured(self) -> bool:
        """是否已经在项目本地配置文件中填写有效密钥。"""
        return bool(self.api_key.get_secret_value().strip())

    def require_api_key(self) -> str:
        """返回明文密钥；仅允许模型客户端在真正调用接口时使用。"""
        value = self.api_key.get_secret_value().strip()
        if not value:
            raise RuntimeError("DeepSeek API Key 未配置：请检查 configs/providers.local.yaml")
        return value


class AgentsConfig(BaseModel):
    """多 Agent 协作配置。

    模型供应商和密钥由顶层 llm 配置统一管理；这里仅保存编排约束。
    """

    max_retries: int = Field(default=2, ge=0, le=10)
    retry_initial_seconds: float = Field(default=1.0, ge=0.0, le=60.0)
    retry_backoff_factor: float = Field(default=2.0, ge=1.0, le=10.0)
    retry_max_seconds: float = Field(default=8.0, ge=0.0, le=300.0)
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
    split_strategy: str = "time_ordered"
    forbid_post_listing_features: bool = True


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
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
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


def load_config(
    config_path: str | Path | None = None,
    providers_path: str | Path | None = None,
) -> AppConfig:
    """加载基础配置与项目内固定供应商配置。

    优先级：providers.local.yaml > base.yaml > 代码默认值。
    函数有意不读取任何操作系统环境变量。

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

    # 3. 逐区段提取，缺失的区段用空字典兜底
    project = raw.get("project", {})
    paths = raw.get("paths", {})
    parser_cfg = raw.get("parser", {})
    retrieval_cfg = raw.get("retrieval", {})
    embedding_cfg = {**raw.get("embedding", {}), **providers.get("embedding", {})}
    llm_cfg = {**raw.get("llm", {}), **providers.get("llm", {})}
    agents_cfg = raw.get("agents", {})
    prediction_cfg = raw.get("prediction", {})

    # 4. 构造并返回 AppConfig；不执行任何环境变量覆盖
    return AppConfig(
        project_name=project.get("name", "hk-ipo-risk-agent"),
        timezone=project.get("timezone", "Asia/Shanghai"),
        paths=PathsConfig(**paths),
        parser=ParserConfig(**parser_cfg),
        retrieval=RetrievalConfig(**retrieval_cfg),
        embedding=EmbeddingConfig(**embedding_cfg),
        llm=LLMConfig(**llm_cfg),
        agents=AgentsConfig(**agents_cfg),
        prediction=PredictionConfig(**prediction_cfg),
    )
