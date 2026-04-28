"""Кодировщики журналов и трассировок (раздел 2.3).

Журналы: Drain → числовые векторы → GRU → h_log
Трассировки: глубина в дереве вызовов → синусоидальное позиционное кодирование → h_tr
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LogEncoder(nn.Module):
    """Кодировщик журналов.

    Параметры
    ----------
    vocab_size : int
        Число шаблонов журналов (выход Drain).
    embed_dim : int
        Размерность эмбеддинга шаблона.
    hidden_dim : int
        Размерность скрытого состояния GRU.
    out_dim : int
        d_log — размерность выхода h_log.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        out_dim: int = 32,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, log_ids: torch.Tensor) -> torch.Tensor:
        """
        Параметры
        ----------
        log_ids : Tensor, shape (batch, T_l)
            Последовательность ID шаблонов (целые числа).

        Возвращает
        ----------
        h_log : Tensor, shape (batch, out_dim)
        """
        emb = self.embed(log_ids)              # (batch, T_l, embed_dim)
        _, hidden = self.gru(emb)              # hidden: (1, batch, hidden_dim)
        return self.proj(hidden.squeeze(0))    # (batch, out_dim)
