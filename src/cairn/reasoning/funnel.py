"""Каскадная воронка фильтрации кандидатов (раздел 3.5, таблица 1).

Три ступени:
  L0 (~2 мс)   : NLL + распространение по соседям → top-l0_top_k
  L1 (~50 мс)  : приближённый CE на локальном подграфе → top-l1_top_k
  L2 (~200 мс) : полный CE на полном графе → top-l2_top_k

Важно: nll_fn на L1 получает правильные локальные контексты (а не весь contexts),
что устраняет ошибку размерности при передаче замкнутых переменных.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class CascadeFunnel(nn.Module):
    """Каскадная воронка (раздел 3.5).

    Параметры
    ----------
    l0_top_k : int   — кандидаты после L0
    l1_top_k : int   — кандидаты после L1
    l2_top_k : int   — финальные кандидаты
    local_hops : int — радиус локального подграфа для L1
    alpha_init : float — начальный α (обучаемый параметр, формула 3.32)
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
        self.l0_top_k   = l0_top_k
        self.l1_top_k   = l1_top_k
        self.l2_top_k   = l2_top_k
        self.local_hops = local_hops
        self.alpha      = nn.Parameter(torch.tensor(alpha_init))   # обучаемый

    # ------------------------------------------------------------------
    # Ступень 0 — быстрая оценка
    # ------------------------------------------------------------------

    def _l0_score(
        self,
        nll: torch.Tensor,          # (N,)
        adjacency: torch.Tensor,    # (N, N)
    ) -> torch.Tensor:
        """Score_i = α·NLL_i + (1-α)·NLL_i·Σ_j w_ij·NLL_j  (формула 3.32)."""
        alpha = torch.sigmoid(self.alpha)
        neighbor_nll = adjacency @ nll      # (N,)
        return alpha * nll + (1 - alpha) * nll * neighbor_nll

    # ------------------------------------------------------------------
    # Локальный подграф для L1
    # ------------------------------------------------------------------

    def _local_nodes(
        self, node_idx: int, adjacency: torch.Tensor
    ) -> List[int]:
        """BFS до local_hops шагов от node_idx."""
        visited, frontier = {node_idx}, {node_idx}
        for _ in range(self.local_hops):
            new_frontier: set[int] = set()
            for v in frontier:
                nbrs = adjacency[v].nonzero(as_tuple=True)[0].tolist()
                for n in nbrs:
                    if n not in visited:
                        visited.add(n)
                        new_frontier.add(n)
            frontier = new_frontier
        return sorted(visited)

    # ------------------------------------------------------------------
    # Основной метод run()
    # ------------------------------------------------------------------

    def run(
        self,
        nll: torch.Tensor,           # (N,)        — предвычисленный NLL
        H: torch.Tensor,             # (N, d)       — матрица состояний
        adjacency: torch.Tensor,     # (N, N)       — нормированная матрица смежности
        cf_module,                   # CounterfactualModule
        gmm,                         # ConditionalGMM
        contexts: torch.Tensor,      # (N, ctx_dim) — контекстные векторы
        hypergraph,                  # CausalHypergraph
    ) -> List[Tuple[int, float]]:
        """Запускает трёхступенчатую воронку.

        Возвращает
        ----------
        list of (node_idx, ce_value) — до l2_top_k записей, отсортированных по убыванию.
        """
        N = nll.shape[0]

        # ----------------------------------------------------------------
        # L0: быстрая фильтрация
        # ----------------------------------------------------------------
        score0 = self._l0_score(nll, adjacency)
        k0 = min(self.l0_top_k, N)
        top0 = score0.topk(k0).indices.tolist()

        # ----------------------------------------------------------------
        # L1: приближённый CE на локальных подграфах
        # ----------------------------------------------------------------
        incidence    = hypergraph.incidence_matrix().to(H.device)
        edge_weights = hypergraph.edge_weights().to(H.device)

        pe1: List[Tuple[int, float]] = []
        for idx in top0:
            local = self._local_nodes(idx, adjacency)
            local_idx = local.index(idx)

            H_local  = H[local]
            ctx_local = contexts[local]
            proto_local = gmm.prototype(ctx_local[local_idx:local_idx+1]).squeeze(0)  # (d,)

            # Локальная матрица инцидентности (строки = local)
            H_inc_local = incidence[local]     # (N_local, M)

            mask = torch.zeros(len(local), 1, device=H.device, dtype=H.dtype)
            mask[local_idx] = 1.0
            H_cf_local = (1 - mask) * H_local + mask * proto_local.unsqueeze(0)
            H_cf_prop = cf_module._hg_forward(H_cf_local, H_inc_local, edge_weights)

            with torch.no_grad():
                a_before = gmm.nll(H_local, ctx_local).mean().item()
                a_after  = gmm.nll(H_cf_prop, ctx_local).mean().item()

            pe1.append((idx, a_before - a_after))

        pe1.sort(key=lambda x: -x[1])
        top1 = [idx for idx, _ in pe1[: self.l1_top_k]]

        # ----------------------------------------------------------------
        # L2: полный CE на полном графе
        # ----------------------------------------------------------------
        pe2 = cf_module.rank_candidates(H, top1, gmm, contexts, hypergraph)
        return pe2[: self.l2_top_k]
