"""Формирование вектора состояния экземпляра (раздел 2.4–2.5).

Объединяет три модальности в итоговый вектор hᵢ ∈ ℝᵈ (формула 2.11).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cairn.perception.metric_encoder import MetricEncoder
from cairn.perception.log_encoder import LogEncoder
from cairn.perception.trace_encoder import TraceEncoder


class StateBuilder(nn.Module):
    """Построитель вектора состояния.

    Параметры
    ----------
    metric_features : int
        F — число входных метрик.
    log_vocab_size : int
        Словарь шаблонов журналов.
    state_dim : int
        d = 128 — итоговая размерность вектора состояния.
    metric_window : int
        W — размер окна для ветви разрыва.
    metric_out : int
        d_met.
    log_out : int
        d_log.
    trace_out : int
        d_tr.
    """

    def __init__(
        self,
        metric_features: int = 10,
        log_vocab_size: int = 1000,
        state_dim: int = 128,
        metric_window: int = 60,
        metric_out: int = 64,
        log_out: int = 32,
        trace_out: int = 16,
    ) -> None:
        super().__init__()
        self.metric_enc = MetricEncoder(metric_features, window=metric_window, out_dim=metric_out)
        self.log_enc = LogEncoder(log_vocab_size, out_dim=log_out)
        self.trace_enc = TraceEncoder(out_dim=trace_out)

        fusion_in = metric_out + log_out + trace_out
        self.proj = nn.Linear(fusion_in, state_dim)  # W_об (формула 2.11)
        self.norm = nn.LayerNorm(state_dim)

    def forward(
        self,
        metrics: torch.Tensor,   # (batch, T, F)
        log_ids: torch.Tensor,   # (batch, T_l)
        trace_depth: torch.Tensor,  # (batch,)
    ) -> torch.Tensor:
        """
        Возвращает
        ----------
        h : Tensor, shape (batch, state_dim)
        """
        h_met = self.metric_enc(metrics)       # (batch, metric_out)
        h_log = self.log_enc(log_ids)          # (batch, log_out)
        h_tr = self.trace_enc(trace_depth)     # (batch, trace_out)

        h_cat = torch.cat([h_met, h_log, h_tr], dim=-1)  # (batch, fusion_in)
        return self.norm(self.proj(h_cat))     # (batch, state_dim)
