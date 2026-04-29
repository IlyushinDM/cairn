"""Фаза восприятия CAIRN — кодировщики данных и построитель гиперграфа.

Публичный API:
    DualBranchMetricEncoder  — двухветвевой кодировщик метрик (SSM + разрыв)
    LogEncoder               — кодировщик журналов (Drain → GRU)
    DrainTokenizer           — парсер шаблонов журналов
    TraceEncoder             — синусоидальное кодирование глубины вызова
    ContextBuilder           — контекстный вектор cᵢ ∈ ℝ¹⁶
    StateBuilder             — объединитель модальностей → hᵢ ∈ ℝ¹²⁸
    HypergraphBuilder        — строитель причинного гиперграфа
    CausalHypergraph         — структура данных гиперграфа
    HyperEdge                — одно гиперребро
    EdgeType                 — тип гиперребра
"""

from cairn.perception.metric_encoder import (
    DualBranchMetricEncoder,
    SSMBranch,
    BreakpointBranch,
    MetricEncoder,  # alias
)
from cairn.perception.log_encoder import LogEncoder, DrainTokenizer
from cairn.perception.trace_encoder import TraceEncoder
from cairn.perception.state_builder import StateBuilder, ContextBuilder
from cairn.perception.hypergraph_builder import (
    HypergraphBuilder,
    CausalHypergraph,
    HyperEdge,
    EdgeType,
)

__all__ = [
    # Кодировщики
    "DualBranchMetricEncoder",
    "MetricEncoder",
    "SSMBranch",
    "BreakpointBranch",
    "LogEncoder",
    "DrainTokenizer",
    "TraceEncoder",
    # Вектор состояния
    "ContextBuilder",
    "StateBuilder",
    # Гиперграф
    "HypergraphBuilder",
    "CausalHypergraph",
    "HyperEdge",
    "EdgeType",
]
