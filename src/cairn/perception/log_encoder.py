"""Кодировщик журналов CAIRN (раздел 2.3).

Конвейер: raw logs → DrainTokenizer → шаблонные ID → Embedding → GRU → h_log

DrainTokenizer – упрощённая реализация алгоритма Drain: группирует сообщения
в шаблоны по длине токенов и совпадающим позициям. Не требует внешних зависимостей.
Если установлен drain3 (опционально), используется он.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


# ---------------------------------------------------------------------------
# DrainTokenizer
# ---------------------------------------------------------------------------

_WILDCARD = "<*>"
_SPLIT_RE = re.compile(r"[\s:=,\[\](){}\"']+")


def _tokenize(message: str) -> list[str]:
    return [t for t in _SPLIT_RE.split(message.strip()) if t]


class DrainTokenizer:
    """Упрощённый Drain-подобный парсер шаблонов журналов.

    Группирует сообщения в шаблоны на основе длины токенов и совпадающих позиций.
    Числовые токены заменяются на <*>.

    Параметры
    ----------
    sim_threshold : float
        Минимальная доля совпадающих токенов для отнесения к существующему шаблону.
    max_templates : int
        Максимальный размер словаря (ID ≥ max_templates → UNK = 1).
    """

    _NUM_RE = re.compile(r"^[-+]?\d+(\.\d+)?([eE][-+]?\d+)?$")

    def __init__(self, sim_threshold: float = 0.5, max_templates: int = 498) -> None:
        self.sim_threshold = sim_threshold
        self.max_templates = max_templates
        # 0 = PAD, 1 = UNK, 2+ = шаблоны
        self._templates: list[list[str]] = []          # список шаблонных токенов
        self._by_len: dict[int, list[int]] = defaultdict(list)  # длина → [индексы]

    def _is_numeric(self, token: str) -> bool:
        return bool(self._NUM_RE.match(token))

    def _normalize(self, tokens: list[str]) -> list[str]:
        return [_WILDCARD if self._is_numeric(t) else t for t in tokens]

    def _similarity(self, tmpl: list[str], tokens: list[str]) -> float:
        if len(tmpl) != len(tokens):
            return 0.0
        matches = sum(
            1 for a, b in zip(tmpl, tokens)
            if a == b or a == _WILDCARD or b == _WILDCARD
        )
        return matches / len(tmpl) if tmpl else 0.0

    def _merge(self, tmpl: list[str], tokens: list[str]) -> list[str]:
        return [a if a == b else _WILDCARD for a, b in zip(tmpl, tokens)]

    def fit_transform(self, messages: Iterable[str]) -> list[int]:
        """Обучает словарь и возвращает список ID для каждого сообщения."""
        return [self.transform_one(m) for m in messages]

    def transform_one(self, message: str) -> int:
        """Возвращает ID шаблона для одного сообщения.

        Обновляет словарь, если подходящий шаблон не найден.
        """
        tokens = self._normalize(_tokenize(message))
        if not tokens:
            return 1  # UNK

        length = len(tokens)
        candidates = self._by_len[length]

        best_idx, best_sim = -1, 0.0
        for idx in candidates:
            sim = self._similarity(self._templates[idx], tokens)
            if sim > best_sim:
                best_sim, best_idx = sim, idx

        if best_sim >= self.sim_threshold and best_idx >= 0:
            # Обновляем существующий шаблон
            self._templates[best_idx] = self._merge(self._templates[best_idx], tokens)
            return best_idx + 2  # +2: 0=PAD, 1=UNK
        else:
            # Добавляем новый шаблон (если словарь не переполнен)
            if len(self._templates) < self.max_templates:
                idx = len(self._templates)
                self._templates.append(tokens)
                self._by_len[length].append(idx)
                return idx + 2
            else:
                return 1  # UNK

    @property
    def vocab_size(self) -> int:
        """Реальный размер словаря (включая PAD и UNK)."""
        return len(self._templates) + 2


# ---------------------------------------------------------------------------
# LogEncoder
# ---------------------------------------------------------------------------


class LogEncoder(nn.Module):
    """Кодировщик журналов: ID шаблонов → Embedding → GRU → h_log.

    Параметры
    ----------
    vocab_size : int
        Размер словаря шаблонов (включая PAD=0 и UNK=1).
    embed_dim : int
        Размерность эмбеддинга шаблона.
    hidden_dim : int
        Размерность скрытого состояния GRU.
    d_out : int
        Размерность выхода h_log.
    """

    def __init__(
        self,
        vocab_size: int = 500,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        d_out: int = 32,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(
        self,
        template_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Параметры
        ----------
        template_ids : Tensor, shape (batch, max_len)
            ID шаблонов (целые числа, 0 = PAD).
        lengths : Tensor, shape (batch,) | None
            Реальные длины последовательностей для packed_sequence.
            Если None – используется полная длина для всех.

        Возвращает
        ----------
        h_log : Tensor, shape (batch, d_out)
        """
        emb = self.embed(template_ids)   # (batch, max_len, embed_dim)

        if lengths is not None:
            # Упакованная последовательность для корректной обработки padding
            lengths_cpu = lengths.cpu().clamp(min=1)
            packed = pack_padded_sequence(
                emb, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            _, hidden = self.gru(packed)
        else:
            _, hidden = self.gru(emb)

        # hidden: (1, batch, hidden_dim)
        h = self.proj(hidden.squeeze(0))   # (batch, d_out)
        return self.norm(h)
