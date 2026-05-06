"""Формирование вектора состояния и контекстного вектора экземпляра (разделы 2.4–2.5).

Компоненты:
  ContextBuilder  — строит контекстный вектор cᵢ ∈ ℝ¹⁶ из метаданных среды.
  StateBuilder    — объединяет три модальности в вектор состояния hᵢ ∈ ℝ¹²⁸:

      hᵢ = LayerNorm(W_fuse · [h_met ∥ h_log ∥ h_tr] + b_fuse)   (формула 2.11)

  PerceptionPipeline — точка входа: принимает батч сырых данных,
      возвращает H ∈ ℝ^{N×128} и C ∈ ℝ^{N×16}.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from cairn.perception.metric_encoder import DualBranchMetricEncoder
from cairn.perception.log_encoder import LogEncoder
from cairn.perception.trace_encoder import TraceEncoder


# ---------------------------------------------------------------------------
# Контекстный вектор cᵢ ∈ ℝ¹⁶
# ---------------------------------------------------------------------------


class ContextBuilder(nn.Module):
    """Строит контекстный вектор cᵢ ∈ ℝ¹⁶ (раздел 2.5).

    Входные признаки (в виде числового вектора raw_dim=7):
      - rps_norm     : нормированная интенсивность запросов (1)
      - hour_sin/cos : циклическое кодирование часа суток (2)
      - dow_sin/cos  : циклическое кодирование дня недели (2)
      - cpu_norm     : cpu_limit / cpu_max (1)
      - mem_norm     : memory_limit / mem_max (1)

    Плюс версия деплоя — кодируется как номер в [0, 1] (1).
    Итого raw_dim=8, проецируется в context_dim=16.

    Параметры
    ----------
    context_dim : int
        Размерность выхода (по умолчанию 16).
    raw_dim : int
        Размерность сырых признаков (по умолчанию 8).
    """

    def __init__(self, context_dim: int = 16, raw_dim: int = 8) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, 32),
            nn.ReLU(),
            nn.Linear(32, context_dim),
        )

    def forward(self, context_raw: torch.Tensor) -> torch.Tensor:
        """
        Параметры
        ----------
        context_raw : Tensor, shape (batch, raw_dim)
            Сырые контекстные признаки.

        Возвращает
        ----------
        c : Tensor, shape (batch, context_dim)
        """
        return self.proj(context_raw)

    @staticmethod
    def build_raw(
        rps: torch.Tensor,          # (batch,)
        hour: torch.Tensor,         # (batch,) — час [0,23]
        day_of_week: torch.Tensor,  # (batch,) — день [0,6]
        cpu_norm: torch.Tensor,     # (batch,) — cpu_limit/max_cpu
        mem_norm: torch.Tensor,     # (batch,) — memory_limit/max_mem
        version_norm: torch.Tensor, # (batch,) — версия в [0,1]
        rps_max: float = 1000.0,
    ) -> torch.Tensor:
        """Строит raw-вектор из числовых метаданных среды.

        Возвращает Tensor shape (batch, 8).
        """
        rps_n = (rps / rps_max).clamp(0, 1).unsqueeze(1)       # (batch, 1)
        h_sin = torch.sin(2 * math.pi * hour / 24).unsqueeze(1)
        h_cos = torch.cos(2 * math.pi * hour / 24).unsqueeze(1)
        d_sin = torch.sin(2 * math.pi * day_of_week / 7).unsqueeze(1)
        d_cos = torch.cos(2 * math.pi * day_of_week / 7).unsqueeze(1)
        cpu_n = cpu_norm.clamp(0, 1).unsqueeze(1)
        mem_n = mem_norm.clamp(0, 1).unsqueeze(1)
        ver_n = version_norm.clamp(0, 1).unsqueeze(1)
        return torch.cat([rps_n, h_sin, h_cos, d_sin, d_cos, cpu_n, mem_n, ver_n], dim=1)


# ---------------------------------------------------------------------------
# StateBuilder — объединение модальностей
# ---------------------------------------------------------------------------


class StateBuilder(nn.Module):
    """Объединяет три кодировщика в вектор состояния hᵢ ∈ ℝ^state_dim.

    Параметры
    ----------
    n_metrics : int
        F — число метрик в временном ряду.
    log_vocab_size : int
        Размер словаря шаблонов журналов (включая PAD=0 и UNK=1).
    state_dim : int
        d = 128 — итоговая размерность вектора состояния.
    context_dim : int
        dc = 16 — размерность контекстного вектора.
    d_met : int
        Размерность выхода кодировщика метрик.
    d_log : int
        Размерность выхода кодировщика журналов.
    d_tr : int
        Размерность выхода кодировщика трассировок.
    d_ssm : int
        Размерность SSM-ветви метрик.
    d_brk : int
        Размерность ветви разрыва метрик.
    ssm_state_dim : int
        D — размерность скрытого состояния SSM.
    window : int
        W — размер окна для ветви разрыва.
    context_raw_dim : int
        Размерность сырого контекстного вектора.
    """

    def __init__(
        self,
        n_metrics: int = 4,
        log_vocab_size: int = 500,
        state_dim: int = 128,
        context_dim: int = 16,
        d_met: int = 64,
        d_log: int = 32,
        d_tr: int = 16,
        d_ssm: int = 32,
        d_brk: int = 32,
        ssm_state_dim: int = 64,
        window: int = 60,
        context_raw_dim: int = 16,  # совпадает с context_dim (выход ContextBuilder)
    ) -> None:
        super().__init__()
        self.d_met = d_met
        self.d_log = d_log
        self.d_tr = d_tr

        # Три кодировщика модальностей
        self.metric_enc = DualBranchMetricEncoder(
            n_metrics=n_metrics,
            d_ssm=d_ssm,
            d_brk=d_brk,
            d_out=d_met,
            ssm_state_dim=ssm_state_dim,
            window=window,
        )
        self.log_enc = LogEncoder(
            vocab_size=log_vocab_size,
            embed_dim=min(64, log_vocab_size // 2),
            hidden_dim=64,
            d_out=d_log,
        )
        self.trace_enc = TraceEncoder(d_out=d_tr)

        # Контекстный вектор
        self.context_builder = ContextBuilder(
            context_dim=context_dim,
            raw_dim=context_raw_dim,
        )

        # Фьюжн: [h_met ∥ h_log ∥ h_tr] → state_dim  (формула 2.11)
        fusion_in = d_met + d_log + d_tr
        self.fusion = nn.Linear(fusion_in, state_dim)
        self.norm = nn.LayerNorm(state_dim)

    def forward(
        self,
        metrics: torch.Tensor,                       # (batch, T, F)
        log_ids: torch.Tensor,                       # (batch, T_l) — ID шаблонов
        trace_depth: torch.Tensor,                   # (batch,) или (batch, n_spans)
        context_raw: Optional[torch.Tensor] = None,  # (batch, raw_dim)
        log_lengths: Optional[torch.Tensor] = None,  # (batch,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Возвращает
        ----------
        h : Tensor, shape (batch, state_dim)      — вектор состояния
        c : Tensor, shape (batch, context_dim)    — контекстный вектор

        Если context_raw не передан — c возвращается как нулевой тензор.
        """
        h_met = self.metric_enc(metrics)                    # (batch, d_met)
        h_log = self.log_enc(log_ids, log_lengths)         # (batch, d_log)
        h_tr = self.trace_enc(trace_depth)                  # (batch, d_tr)

        h_cat = torch.cat([h_met, h_log, h_tr], dim=-1)    # (batch, fusion_in)
        h = self.norm(self.fusion(h_cat))                   # (batch, state_dim)

        if context_raw is not None:
            c = self.context_builder(context_raw)           # (batch, context_dim)
        else:
            c = torch.zeros(
                h.shape[0], self.context_builder.context_dim,
                device=h.device, dtype=h.dtype,
            )

        return h, c
