"""Контрфактический интервенционный модуль – ядро CAIRN (раздел 3.3).

Реализует:
  - HypergraphConv             – нормализованная гиперграфовая свёртка (формула 3.24)
  - CounterfactualModule       – do(i) вмешательство, ПЭ(i), ранжирование
  - CounterfactualInterventionModule  – alias для обратной совместимости

Вмешательство дифференцируемо: градиенты проходят через prototype(c_i).
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Гиперграфовая свёртка
# ---------------------------------------------------------------------------

class HypergraphConv(nn.Module):
    """Нормализованная гиперграфовая свёртка (формула 3.24).

    X^{l+1} = D_v^{-½} H W_H D_e^{-1} H^T D_v^{-½} X^{l} Θ

    Параметры
    ----------
    in_dim : int
    out_dim : int
    use_pyg : bool
        Если True и torch_geometric установлен – используется PyG-реализация.
    """

    def __init__(self, in_dim: int, out_dim: int, use_pyg: bool = False) -> None:
        super().__init__()
        self._pyg_conv = None
        if use_pyg:
            try:
                from torch_geometric.nn import HypergraphConv as PyGHypergraphConv
                self._pyg_conv = PyGHypergraphConv(in_dim, out_dim)
            except ImportError:
                pass

        if self._pyg_conv is None:
            self.theta = nn.Linear(in_dim, out_dim, bias=False)

    def forward(
        self,
        X: torch.Tensor,          # (N, in_dim)
        H: torch.Tensor,          # (N, M)
        W_H: torch.Tensor,        # (M,)
    ) -> torch.Tensor:
        if self._pyg_conv is not None:
            # Преобразуем матрицу инцидентности в sparse формат PyG
            nz = H.nonzero(as_tuple=True)
            hyperedge_index = torch.stack([nz[0], nz[1]])  # (2, nnz)
            return self._pyg_conv(X, hyperedge_index, hyperedge_weight=W_H)

        # Встроенная реализация
        D_v = H.sum(dim=1).clamp(min=1e-8)             # (N,)
        D_e = H.sum(dim=0).clamp(min=1e-8)             # (M,)
        Dv_inv_sqrt = (D_v ** -0.5).unsqueeze(1)       # (N, 1)
        # Θ_H = H · diag(W_H) · diag(D_e^{-1}): (N, M)
        Theta_H = H * W_H.unsqueeze(0) / D_e.unsqueeze(0)
        # A_norm = D_v^{-½} H Θ_H^T D_v^{-½}: (N, N)
        A_norm = (Dv_inv_sqrt * H) @ Theta_H.T * Dv_inv_sqrt.T
        return self.theta(A_norm @ X)


# ---------------------------------------------------------------------------
# CounterfactualModule
# ---------------------------------------------------------------------------

class CounterfactualModule(nn.Module):
    """Дифференцируемый контрфактический интервенционный модуль (раздел 3.3).

    Параметры
    ----------
    state_dim : int
        d = 128.
    n_conv_layers : int
        Число слоёв гиперграфовой свёртки (обычно 1).
    use_pyg : bool
        Использовать PyG HypergraphConv если доступен.
    """

    def __init__(
        self,
        state_dim: int = 128,
        n_conv_layers: int = 1,
        use_pyg: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.convs = nn.ModuleList(
            [HypergraphConv(state_dim, state_dim, use_pyg=use_pyg)
             for _ in range(n_conv_layers)]
        )
        self.norm = nn.LayerNorm(state_dim)

    # ------------------------------------------------------------------
    # Гиперграфовый forward
    # ------------------------------------------------------------------

    def _hg_forward(
        self,
        states: torch.Tensor,      # (N, d)
        incidence: torch.Tensor,   # (N, M)
        edge_weights: torch.Tensor,  # (M,)
    ) -> torch.Tensor:
        X = states
        for conv in self.convs:
            X = torch.relu(conv(X, incidence, edge_weights))
        return self.norm(X)

    # ------------------------------------------------------------------
    # Вмешательство (дифференцируемое, формула 3.29)
    # ------------------------------------------------------------------

    def intervene(
        self,
        H: torch.Tensor,              # (N, d)
        i: int,
        prototype: torch.Tensor,      # (d,) – нормальное состояние узла i
        hypergraph,                   # объект с .incidence_matrix() и .edge_weights()
        apply_conv: bool = False,     # свёртка только для propagation, не для scoring
    ) -> torch.Tensor:
        """do(i): заменяем h_i на prototype.

        apply_conv=False (дефолт): возвращает H_cf без свёртки – для вычисления CE.
        apply_conv=True: прогоняет через гиперграфовую свёртку – для propagation.

        Операция дифференцируема – градиент проходит через ``prototype``.

        Возвращает
        ----------
        H_cf : (N, d)
        """
        N = H.shape[0]
        mask = torch.zeros(N, 1, device=H.device, dtype=H.dtype)
        mask[i] = 1.0
        H_cf = (1 - mask) * H + mask * prototype.unsqueeze(0)

        if apply_conv:
            incidence    = hypergraph.incidence_matrix().to(H.device)
            edge_weights = hypergraph.edge_weights().to(H.device)
            return self._hg_forward(H_cf, incidence, edge_weights)
        return H_cf

    # ------------------------------------------------------------------
    # Причинный эффект ПЭ(i) = A(G) - A_cf(G)
    # ------------------------------------------------------------------

    def causal_effect(
        self,
        H: torch.Tensor,
        H_cf: torch.Tensor,
        gmm,                      # ConditionalGMM с методом nll(h, ctx)
        contexts: torch.Tensor,   # (N, context_dim)
        H_normal: Optional[torch.Tensor] = None,   # (N, d) – нормальные состояния
        hypergraph=None,          # для propagation-based CE
    ) -> float:
        """CE(i) = снижение аномальности системы после вмешательства.

        Если H_normal доступен и гиперграф передан – использует distance-CE
        в post-conv пространстве: CE = Σ||h_anom_conv - h_norm_conv||² - Σ||h_cf_conv - h_norm_conv||²
        Иначе – fallback на NLL-CE без свёртки.
        """
        with torch.no_grad():
            if H_normal is not None and hypergraph is not None:
                # Distance-based CE в post-conv пространстве
                # GMM обучена до свёртки, но CE должен учитывать propagation
                inc = hypergraph.incidence_matrix().to(H.device)
                ew  = hypergraph.edge_weights().to(H.device)
                H_anom_conv  = self._hg_forward(H,        inc, ew)  # аномальный конв
                H_cf_conv    = self._hg_forward(H_cf,     inc, ew)  # после вмешательства
                H_norm_conv  = self._hg_forward(H_normal, inc, ew)  # нормальный конв
                # CE = насколько вмешательство приближает систему к нормальному состоянию
                dist_before = ((H_anom_conv - H_norm_conv) ** 2).sum()
                dist_after  = ((H_cf_conv   - H_norm_conv) ** 2).sum()
                return (dist_before - dist_after).item()
            else:
                # Fallback: NLL-разница без свёртки
                nll_before = gmm.nll(H,    contexts)
                nll_after  = gmm.nll(H_cf, contexts)
                return (nll_before.sum() - nll_after.sum()).item()

    # ------------------------------------------------------------------
    # Ранжирование кандидатов
    # ------------------------------------------------------------------

    def rank_candidates(
        self,
        H: torch.Tensor,
        candidates: List[int],
        gmm,
        contexts: torch.Tensor,   # (N, context_dim)
        hypergraph,
        H_normal: Optional[torch.Tensor] = None,  # (N, d) – нормальные состояния
    ) -> List[Tuple[int, float]]:
        """Ранжирует кандидатов по убыванию CE (формула 3.28).

        Возвращает
        ----------
        list of (node_idx, ce_value) – отсортированный по убыванию CE.
        """
        # Прототип = реальное нормальное состояние узла (если доступно)
        # иначе – среднее нормальное состояние по всем узлам
        if H_normal is not None:
            prototypes = H_normal                          # (N, d)
        else:
            # Fallback: GMM прототип с нулевым контекстом
            C_ref = torch.zeros_like(contexts)
            prototypes = torch.stack([
                gmm.prototype(C_ref[i:i+1]).squeeze(0) for i in range(H.shape[0])
            ])

        scores: List[Tuple[int, float]] = []
        for idx in candidates:
            H_cf = self.intervene(H, idx, prototypes[idx], hypergraph)
            ce   = self.causal_effect(H, H_cf, gmm, contexts,
                                      H_normal=H_normal, hypergraph=hypergraph)
            scores.append((idx, ce))

        scores.sort(key=lambda x: -x[1])
        return scores


# Alias для обратной совместимости со старым кодом
CounterfactualInterventionModule = CounterfactualModule
