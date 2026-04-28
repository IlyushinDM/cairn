"""Каскадная воронка фильтрации кандидатов (раздел 3.5, таблица 1).

Три ступени:
  L0 (быстрая): NLL + профиль соседей → top-30   (~2 мс)
  L1 (приближ.): контрфактика на локальном подграфе → top-5  (~50 мс)
  L2 (полная):  полный контрфактический анализ → top-1  (~200 мс)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CascadeFunnel(nn.Module):
    """Каскадная воронка (раздел 3.5).

    Параметры
    ----------
    l0_top_k : int  — число кандидатов после ступени 0 (30)
    l1_top_k : int  — число кандидатов после ступени 1 (5)
    l2_top_k : int  — финальное число кандидатов (1)
    local_hops : int — радиус локального подграфа для ступени 1 (2 шага)
    alpha_init : float — начальный α для ступени 0 (формула 3.32)
    """

    def __init__(
        self,
        l0_top_k: int = 30,
        l1_top_k: int = 5,
        l2_top_k: int = 1,
        local_hops: int = 2,
        alpha_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.l0_top_k = l0_top_k
        self.l1_top_k = l1_top_k
        self.l2_top_k = l2_top_k
        self.local_hops = local_hops
        # α — обучаемый параметр (формула 3.32)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def _stage0_score(
        self,
        nll: torch.Tensor,         # (N,) — аномальность каждого узла
        adjacency: torch.Tensor,   # (N, N) — матрица смежности (нормированная)
    ) -> torch.Tensor:
        """Оценка ступени 0 (формула 3.32):
           Score_i = α * NLL_i + (1-α) * NLL_i * Σ_{j∈N(i)} w_ij * NLL_j
        """
        alpha = torch.sigmoid(self.alpha)              # (0, 1)
        neighbor_score = adjacency @ nll               # (N,) — взвешенная сумма NLL соседей
        score = alpha * nll + (1 - alpha) * nll * neighbor_score
        return score

    def _local_subgraph(
        self,
        node_idx: int,
        adjacency: torch.Tensor,
        hops: int,
    ) -> list[int]:
        """Возвращает индексы узлов в h-шаговой окрестности node_idx."""
        visited = {node_idx}
        frontier = {node_idx}
        for _ in range(hops):
            new_frontier = set()
            for v in frontier:
                neighbors = adjacency[v].nonzero(as_tuple=True)[0].tolist()
                for n in neighbors:
                    if n not in visited:
                        visited.add(n)
                        new_frontier.add(n)
            frontier = new_frontier
        return sorted(visited)

    def run(
        self,
        nll: torch.Tensor,         # (N,)
        states: torch.Tensor,      # (N, d)
        adjacency: torch.Tensor,   # (N, N) нормированная
        cf_module,                 # CounterfactualInterventionModule
        nll_fn,                    # callable(states) -> (N,)
        prototypes: torch.Tensor,  # (N, d)
        incidence: torch.Tensor,   # (N, M)
        edge_weights: torch.Tensor,  # (M,)
    ) -> list[tuple[int, float]]:
        """Запускает трёхступенчатую воронку и возвращает ранжированных кандидатов.

        Возвращает
        ----------
        list of (node_idx, pe_score) — до l2_top_k записей.
        """
        N = nll.shape[0]

        # --- Ступень 0: быстрая фильтрация ---
        score0 = self._stage0_score(nll, adjacency)             # (N,)
        k0 = min(self.l0_top_k, N)
        top0 = score0.topk(k0).indices.tolist()                 # индексы top-30

        # --- Ступень 1: приближённый анализ на локальном подграфе ---
        pe1_scores: list[tuple[int, float]] = []
        for idx in top0:
            local_nodes = self._local_subgraph(idx, adjacency, self.local_hops)
            local_states = states[local_nodes]
            local_proto = prototypes[local_nodes[local_nodes.index(idx)]]
            local_incidence = incidence[local_nodes]

            with torch.no_grad():
                pe = cf_module.causal_effect(
                    local_states, nll_fn,
                    local_nodes.index(idx), local_proto,
                    local_incidence, edge_weights,
                )
            pe1_scores.append((idx, pe.item()))

        pe1_scores.sort(key=lambda x: -x[1])
        top1 = [idx for idx, _ in pe1_scores[: self.l1_top_k]]

        # --- Ступень 2: полный анализ ---
        pe2_scores = cf_module.rank_candidates(
            states, nll_fn, prototypes, top1, incidence, edge_weights
        )

        return pe2_scores[: self.l2_top_k]
