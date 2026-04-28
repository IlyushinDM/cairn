"""Контрфактический интервенционный модуль — ядро архитектуры CAIRN (раздел 3.3).

Реализует:
  - Виртуальное вмешательство do(i) через дифференцируемую замену (формула 3.29)
  - Гиперграфовую свёртку для распространения эффекта (формулы 3.23–3.24)
  - Вычисление причинного эффекта ПЭ(i) (формулы 3.25–3.28)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HypergraphConv(nn.Module):
    """Нормализованная гиперграфовая свёртка (формула 3.24).

    Xl+1 = D_v^{-1/2} H W_H D_e^{-1} H^T D_v^{-1/2} X^(l) Θ
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.theta = nn.Linear(in_dim, out_dim, bias=False)

    def forward(
        self,
        X: torch.Tensor,        # (N, in_dim)
        H: torch.Tensor,        # (N, M) — матрица инцидентности
        W_H: torch.Tensor,      # (M,)   — веса гиперрёбер
    ) -> torch.Tensor:
        """Возвращает X_new : (N, out_dim)."""
        N = X.shape[0]
        # Степени вершин и рёбер
        D_v = H.sum(dim=1).clamp(min=1e-8)         # (N,)
        D_e = H.sum(dim=0).clamp(min=1e-8)         # (M,)

        Dv_inv_sqrt = (D_v ** -0.5).unsqueeze(1)   # (N,1)
        De_inv = (D_e ** -1).unsqueeze(0)           # (1,M)
        WH = W_H.unsqueeze(0)                        # (1,M)

        # Нормализованная матрица смежности: (N, N)
        theta_H = H * WH * De_inv                   # (N, M)
        A_norm = (Dv_inv_sqrt * H) @ theta_H.T * Dv_inv_sqrt.T  # (N, N)

        return self.theta(A_norm @ X)


class CounterfactualInterventionModule(nn.Module):
    """Контрфактический интервенционный модуль (раздел 3.3).

    Параметры
    ----------
    state_dim : int
        d = 128.
    n_layers : int
        Число слоёв гиперграфовой свёртки (обычно 1).
    """

    def __init__(self, state_dim: int = 128, n_layers: int = 1) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.convs = nn.ModuleList([
            HypergraphConv(state_dim, state_dim) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(state_dim)

    def _hypergraph_forward(
        self,
        H_states: torch.Tensor,    # (N, d) — матрица состояний всех узлов
        incidence: torch.Tensor,   # (N, M) — матрица инцидентности гиперграфа
        edge_weights: torch.Tensor,  # (M,)
    ) -> torch.Tensor:
        """Прогоняет состояния через все слои свёртки."""
        X = H_states
        for conv in self.convs:
            X = torch.relu(conv(X, incidence, edge_weights))
        return self.norm(X)

    def intervene(
        self,
        states: torch.Tensor,         # (N, d)
        candidate_idx: int,
        prototype: torch.Tensor,      # (d,) — μ*(cᵢ) условный прототип
        incidence: torch.Tensor,      # (N, M)
        edge_weights: torch.Tensor,   # (M,)
    ) -> torch.Tensor:
        """Выполняет виртуальное вмешательство do(i) и возвращает HКФ(i).

        Операция дифференцируема через prototype (формула 3.29).
        """
        N = states.shape[0]
        # Маска: единица в позиции candidate_idx, нули везде
        mask = torch.zeros(N, 1, device=states.device)
        mask[candidate_idx] = 1.0

        # Формула 3.29: h_i^КФ = (1-mask)*h + mask*μ*
        states_cf = (1 - mask) * states + mask * prototype.unsqueeze(0)

        # Распространяем эффект через гиперграф (формула 3.23)
        return self._hypergraph_forward(states_cf, incidence, edge_weights)

    def causal_effect(
        self,
        states: torch.Tensor,          # (N, d)
        nll_fn,                        # callable(states) -> (N,) anomaly scores
        candidate_idx: int,
        prototype: torch.Tensor,       # (d,)
        incidence: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Вычисляет ПЭ(i) = A(G) - A^КФ_i(G) (формулы 3.25–3.27).

        Возвращает
        ----------
        pe : Tensor, shape () — скаляр причинного эффекта.
        """
        # Аномальность до вмешательства (формула 3.25)
        nll_before = nll_fn(states)                     # (N,)
        ag_before = nll_before.mean()

        # Состояние после вмешательства
        states_cf = self.intervene(
            states, candidate_idx, prototype, incidence, edge_weights
        )
        nll_after = nll_fn(states_cf)                   # (N,)
        ag_after = nll_after.mean()

        return ag_before - ag_after                      # ПЭ(i) ≥ 0 → снизили аномальность

    def rank_candidates(
        self,
        states: torch.Tensor,
        nll_fn,
        prototypes: torch.Tensor,      # (N, d) — условные прототипы для каждого узла
        candidate_indices: list[int],
        incidence: torch.Tensor,
        edge_weights: torch.Tensor,
    ) -> list[tuple[int, float]]:
        """Ранжирует кандидатов по причинному эффекту (формула 3.28).

        Возвращает
        ----------
        list of (node_idx, pe_value) sorted descending.
        """
        scores = []
        for idx in candidate_indices:
            with torch.no_grad():
                pe = self.causal_effect(
                    states, nll_fn, idx, prototypes[idx], incidence, edge_weights
                )
            scores.append((idx, pe.item()))
        scores.sort(key=lambda x: -x[1])
        return scores
