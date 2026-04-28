"""Pydantic-модели конфигурации CAIRN.

Структура соответствует configs/default.yaml и описанию архитектуры (таблица 3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    state_dim: int = 128
    context_dim: int = 16
    metric_dim: int = 64
    log_dim: int = 32
    trace_dim: int = 16
    gmm_components: int = 5
    latent_confounders: int = 3
    confounder_dim: int = 32
    hypergraph_layers: int = 1
    attention_heads: int = 8
    attention_layers: int = 3
    breakpoint_window: int = 60


class LossWeightsConfig(BaseModel):
    lambda_pe: float = 1.0
    lambda_um: float = 1.0
    lambda_vak: float = 1.0
    lambda_nez: float = 0.5
    lambda_kr: float = 0.5
    lambda_reb: float = 0.1


class TrainingConfig(BaseModel):
    lr: float = 1e-3
    batch_size: int = 32
    seed: int = 42
    pretrain_epochs: int = 50
    main_epochs: int = 100
    finetune_epochs: int = 30
    freeze_epochs: int = 10
    loss_weights: LossWeightsConfig = Field(default_factory=LossWeightsConfig)
    margin: float = 0.1
    tcd_margin: float = 0.3
    beta_kl: float = 1.0
    beta_kl_z: float = 0.1
    cov_reg: float = 1e-4
    anomaly_percentile: float = 0.99


class FunnelConfig(BaseModel):
    l0_top_k: int = 30
    l1_top_k: int = 5
    l2_top_k: int = 1
    alpha_init: float = 0.5
    local_hops: int = 2


class DecompositionConfig(BaseModel):
    additivity_threshold: float = 0.15
    max_joint_size: int = 3


class VerifierConfig(BaseModel):
    temporal_tolerance_sec: float = 15.0
    transitivity_threshold: float = 0.3
    monotonicity_epsilon: float = 0.05
    permutation_tests: int = 10
    edge_significance_threshold: float = 0.05
    confounder_threshold: float = 0.3


class DriftConfig(BaseModel):
    window_size: int = 100
    drift_threshold_factor: float = 1.5
    finetune_steps: int = 10


class ExplanationConfig(BaseModel):
    generator_level: Literal["template", "local_llm", "cloud_llm"] = "template"
    local_llm_model: Optional[str] = None
    cloud_llm_timeout_sec: float = 8.0


class MetricsConnectorConfig(BaseModel):
    type: Literal["csv", "prometheus"] = "csv"
    path: str = "data/sample/metrics.csv"
    interval_sec: float = 1.0


class LogsConnectorConfig(BaseModel):
    type: Literal["file", "elasticsearch"] = "file"
    path: str = "data/sample/logs.txt"


class TracesConnectorConfig(BaseModel):
    type: Literal["file", "jaeger"] = "file"
    path: str = "data/sample/traces.json"


class ConnectorsConfig(BaseModel):
    metrics: MetricsConnectorConfig = Field(default_factory=MetricsConnectorConfig)
    logs: LogsConnectorConfig = Field(default_factory=LogsConnectorConfig)
    traces: TracesConnectorConfig = Field(default_factory=TracesConnectorConfig)


class GuiConfig(BaseModel):
    theme: Literal["light", "dark"] = "light"
    language: Literal["ru", "en"] = "ru"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    rotation: str = "10 MB"
    retention: str = "7 days"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------


class CAIRNConfig(BaseModel):
    seed: int = 42
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    funnel: FunnelConfig = Field(default_factory=FunnelConfig)
    decomposition: DecompositionConfig = Field(default_factory=DecompositionConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    explanation: ExplanationConfig = Field(default_factory=ExplanationConfig)
    connectors: ConnectorsConfig = Field(default_factory=ConnectorsConfig)
    gui: GuiConfig = Field(default_factory=GuiConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> CAIRNConfig:
    """Загружает конфигурацию из YAML-файла."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # Убираем мета-ключ наследования, если он есть
    data.pop("_base_", None)
    return CAIRNConfig(**data)
