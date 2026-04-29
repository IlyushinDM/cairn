"""Кодировщик трассировок CAIRN (раздел 2.3).

Синусоидальное позиционное кодирование глубины вызова в дереве трассировок.
Подход аналогичен позиционному кодированию трансформеров и CHASE [5].

Дополнительно реализует агрегацию по трассировке:
если у одного экземпляра несколько span'ов — берётся среднее кодирование.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TraceEncoder(nn.Module):
    """Синусоидальное позиционное кодирование глубины вызова (раздел 2.3).

    Параметры
    ----------
    d_out : int
        Размерность выхода h_tr.
    max_depth : int
        Максимальная глубина дерева вызовов.
    """

    def __init__(self, d_out: int = 32, max_depth: int = 20) -> None:
        super().__init__()
        self.d_out = d_out

        # Предвычисляем таблицу позиционного кодирования (max_depth, d_out)
        pe = torch.zeros(max_depth, d_out)
        position = torch.arange(0, max_depth, dtype=torch.float).unsqueeze(1)
        half = d_out // 2
        div_term = torch.exp(
            torch.arange(0, half, dtype=torch.float) * (-math.log(10000.0) / half)
        )
        pe[:, 0::2] = torch.sin(position * div_term[:d_out - half])
        pe[:, 1::2] = torch.cos(position * div_term[:half])
        self.register_buffer("pe", pe)   # не обучается

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Параметры
        ----------
        depth : Tensor, shape (batch,) или (batch, n_spans)
            Глубина вызова (целые числа ≥ 0).
            Если 2D — усредняет по span'ам.

        Возвращает
        ----------
        h_tr : Tensor, shape (batch, d_out)
        """
        if depth.dim() == 2:
            # Несколько span'ов — берём среднее
            depth_clamped = depth.clamp(0, self.pe.shape[0] - 1).long()
            return self.pe[depth_clamped].mean(dim=1)   # (batch, d_out)

        depth_clamped = depth.clamp(0, self.pe.shape[0] - 1).long()
        return self.pe[depth_clamped]   # (batch, d_out)
