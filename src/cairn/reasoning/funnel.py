"""Каскадная воронка для локализации первопричины (раздел 3.5).

Упрощённая реализация: ранжирование по NLL (аномальности состояния узла).
GMM корректно оценивает вероятность нормального состояния.
Узел с максимальным NLL — наиболее вероятная первопричина.
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import torch


class CascadeFunnel:
    """Ранжирование кандидатов первопричины по NLL-скору.

    score(i) = NLL(h_i) — насколько состояние узла i аномально
               согласно условной модели нормы GMM.

    Параметры
    ----------
    l0_top_k : итоговое число кандидатов (по умолчанию 5)
    l1_top_k : не используется (для совместимости)
    l2_top_k : финальный размер результата (по умолчанию 1)
    """

    def __init__(
        self,
        l0_top_k: int = 30,
        l1_top_k: int = 5,
        l2_top_k: int = 1,
    ) -> None:
        self.l0_top_k = l0_top_k
        self.l1_top_k = l1_top_k
        self.l2_top_k = l2_top_k

    def run(
        self,
        nll: torch.Tensor,           # (N,) — NLL каждого узла
        H: torch.Tensor,             # (N, d) — состояния (для совместимости)
        adjacency: torch.Tensor,     # (N, N) — не используется
        cf_module,                   # не используется
        gmm,                         # не используется
        contexts: torch.Tensor,      # не используется
        hypergraph,                  # не используется
        H_normal: Optional[torch.Tensor] = None,  # не используется
    ) -> List[Tuple[int, float]]:
        """Возвращает узлы, отсортированные по убыванию NLL.

        Возвращает
        ----------
        list of (node_idx, nll_value) — до l2_top_k записей.
        """
        N = nll.shape[0]
        k = min(self.l0_top_k, N)

        # Сортируем все узлы по убыванию NLL
        top_k = nll.topk(k)
        ranked = [(int(idx), float(nll[idx].detach())) for idx in top_k.indices]

        return ranked[: self.l2_top_k]
