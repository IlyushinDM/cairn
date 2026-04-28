"""Кодировщик трассировок — синусоидальное позиционное кодирование глубины вызовов."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TraceEncoder(nn.Module):
    """Кодировщик трассировок (раздел 2.3).

    Из трассировки извлекается глубина экземпляра в дереве вызовов.
    Глубина кодируется синусоидальным позиционным кодированием по аналогии с CHASE [5].

    Параметры
    ----------
    out_dim : int
        d_tr — размерность выхода h_tr.
    max_depth : int
        Максимальная глубина дерева вызовов (для позиционного кодирования).
    """

    def __init__(self, out_dim: int = 16, max_depth: int = 64) -> None:
        super().__init__()
        self.out_dim = out_dim

        # Предвычисляем таблицу позиционного кодирования (max_depth, out_dim)
        pe = torch.zeros(max_depth, out_dim)
        position = torch.arange(0, max_depth, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, out_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / out_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: out_dim // 2])
        self.register_buffer("pe", pe)  # не обучается

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        Параметры
        ----------
        depth : Tensor, shape (batch,) — целые числа, глубина вызова.

        Возвращает
        ----------
        h_tr : Tensor, shape (batch, out_dim)
        """
        depth = depth.clamp(0, self.pe.shape[0] - 1).long()
        return self.pe[depth]  # (batch, out_dim)
