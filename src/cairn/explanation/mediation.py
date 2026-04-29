"""Медиационная диагностика модели (раздел 4.2).

Измеряет вклад каждого компонента (слой свёртки, ребро гиперграфа) в итоговый
причинный эффект ПЭ(root). Используется для интерпретации: почему CAIRN пришёл
к конкретному выводу.

Методология:
  CE_layer(l) = CE(full) - CE(bypass_l)   — вклад l-го слоя свёртки
  CE_edge(e)  = CE(full) - CE(mask_e)     — вклад ребра e гиперграфа
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class LayerContribution:
    """Вклад одного слоя гиперграфовой свёртки."""
    layer_idx: int
    contribution: float     # CE_full - CE_bypass
    relative: float         # contribution / CE_full (доля)


@dataclass
class EdgeContribution:
    """Вклад одного гиперребра."""
    edge_idx: int
    edge_type: str
    members: List[int]
    contribution: float
    relative: float


@dataclass
class MediationReport:
    """Итоговый отчёт медиационной диагностики.

    Атрибуты
    ----------
    root_cause_idx : int
    ce_full : float                    — CE при полной модели
    layer_contributions : list
    edge_contributions : list
    top_edges : list[EdgeContribution] — топ-3 важных ребра
    """
    root_cause_idx: int
    ce_full: float
    layer_contributions: List[LayerContribution] = field(default_factory=list)
    edge_contributions: List[EdgeContribution] = field(default_factory=list)

    @property
    def top_edges(self) -> List[EdgeContribution]:
        return sorted(self.edge_contributions, key=lambda e: -e.contribution)[:3]

    def summary(self) -> str:
        lines = [
            f"Медиационная диагностика (первопричина: узел {self.root_cause_idx})",
            f"  CE(полная модель) = {self.ce_full:.4f}",
            "",
            "  Вклад слоёв свёртки:",
        ]
        for lc in self.layer_contributions:
            lines.append(
                f"    Слой {lc.layer_idx}: {lc.contribution:+.4f} ({lc.relative:+.1%})"
            )
        lines.append("")
        lines.append("  Топ-3 значимых гиперребра:")
        for ec in self.top_edges:
            lines.append(
                f"    [{ec.edge_type}] {ec.members}: {ec.contribution:+.4f} ({ec.relative:+.1%})"
            )
        return "\n".join(lines)


class MediationDiagnostic:
    """Медиационная диагностика вклада компонентов в причинный эффект.

    Параметры
    ----------
    cf_module : CounterfactualModule
    gmm : ConditionalGMM
    """

    def __init__(self, cf_module, gmm) -> None:
        self.cf_module = cf_module
        self.gmm = gmm

    def diagnose(
        self,
        H: torch.Tensor,            # (N, d)
        hypergraph,                 # CausalHypergraph
        contexts: torch.Tensor,     # (N, ctx_dim)
        root_cause_idx: int,
    ) -> MediationReport:
        """Запускает полную медиационную диагностику.

        Параметры
        ----------
        H : (N, d) матрица состояний
        hypergraph : CausalHypergraph
        contexts : (N, ctx_dim) контекстные векторы
        root_cause_idx : индекс первопричины

        Возвращает
        ----------
        MediationReport
        """
        # Прототип для вмешательства
        proto = self.gmm.prototype(contexts[root_cause_idx:root_cause_idx+1]).squeeze(0)

        # Базовый CE на полной модели
        H_cf_full = self.cf_module.intervene(H, root_cause_idx, proto, hypergraph)
        ce_full = self.cf_module.causal_effect(H, H_cf_full, self.gmm, contexts)

        layer_contribs = self.diagnose_layers(H, hypergraph, contexts, root_cause_idx, proto, ce_full)
        edge_contribs  = self.diagnose_edges(H, hypergraph, contexts, root_cause_idx, proto, ce_full)

        return MediationReport(
            root_cause_idx=root_cause_idx,
            ce_full=ce_full,
            layer_contributions=layer_contribs,
            edge_contributions=edge_contribs,
        )

    def diagnose_layers(
        self,
        H: torch.Tensor,
        hypergraph,
        contexts: torch.Tensor,
        root_cause_idx: int,
        proto: Optional[torch.Tensor] = None,
        ce_full: Optional[float] = None,
    ) -> List[LayerContribution]:
        """CE_layer(l) = CE(full) - CE(bypass_l) для каждого слоя свёртки.

        «Bypass» слоя l: после вмешательства пропускаем l-й слой (используем
        вход слоя как его выход — это эквивалентно удалению слоя).
        """
        if proto is None:
            proto = self.gmm.prototype(contexts[root_cause_idx:root_cause_idx+1]).squeeze(0)
        if ce_full is None:
            H_cf = self.cf_module.intervene(H, root_cause_idx, proto, hypergraph)
            ce_full = self.cf_module.causal_effect(H, H_cf, self.gmm, contexts)

        incidence    = hypergraph.incidence_matrix().to(H.device)
        edge_weights = hypergraph.edge_weights().to(H.device)
        n_layers     = len(self.cf_module.convs)

        contributions: List[LayerContribution] = []

        for l in range(n_layers):
            # Bypass слоя l: запускаем все слои кроме l
            ce_bypass = self._ce_bypass_layer(
                H, root_cause_idx, proto, incidence, edge_weights, contexts, skip_layer=l
            )
            contrib = ce_full - ce_bypass
            rel     = contrib / (abs(ce_full) + 1e-8)
            contributions.append(LayerContribution(l, round(contrib, 4), round(rel, 4)))

        return contributions

    def diagnose_edges(
        self,
        H: torch.Tensor,
        hypergraph,
        contexts: torch.Tensor,
        root_cause_idx: int,
        proto: Optional[torch.Tensor] = None,
        ce_full: Optional[float] = None,
    ) -> List[EdgeContribution]:
        """CE_edge(e) = CE(full) - CE(mask_e) для каждого гиперребра.

        «Mask» ребра e: нулевой вес для ребра e в матрице весов W_H.
        Это удаляет информационный поток через ребро e.
        """
        if proto is None:
            proto = self.gmm.prototype(contexts[root_cause_idx:root_cause_idx+1]).squeeze(0)
        if ce_full is None:
            H_cf = self.cf_module.intervene(H, root_cause_idx, proto, hypergraph)
            ce_full = self.cf_module.causal_effect(H, H_cf, self.gmm, contexts)

        incidence    = hypergraph.incidence_matrix().to(H.device)
        edge_weights = hypergraph.edge_weights().to(H.device)

        contributions: List[EdgeContribution] = []

        for e_idx, edge in enumerate(hypergraph.edges):
            # Маскируем ребро e
            masked_weights = edge_weights.clone()
            masked_weights[e_idx] = 0.0

            ce_masked = self._ce_with_weights(
                H, root_cause_idx, proto, incidence, masked_weights, contexts
            )
            contrib = ce_full - ce_masked
            rel     = contrib / (abs(ce_full) + 1e-8)
            contributions.append(EdgeContribution(
                edge_idx=e_idx,
                edge_type=str(edge.edge_type),
                members=list(edge.members),
                contribution=round(contrib, 4),
                relative=round(rel, 4),
            ))

        return contributions

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _ce_bypass_layer(
        self,
        H: torch.Tensor,
        root_idx: int,
        proto: torch.Tensor,
        incidence: torch.Tensor,
        edge_weights: torch.Tensor,
        contexts: torch.Tensor,
        skip_layer: int,
    ) -> float:
        """CE при пропуске одного слоя свёртки."""
        N = H.shape[0]
        mask = torch.zeros(N, 1, device=H.device, dtype=H.dtype)
        mask[root_idx] = 1.0
        H_cf = (1 - mask) * H + mask * proto.unsqueeze(0)

        # Прогоняем через слои, пропуская skip_layer
        X = H_cf
        for i, conv in enumerate(self.cf_module.convs):
            if i == skip_layer:
                # Bypass: передаём вход без изменений (skip connection)
                # Проецируем если in_dim != out_dim, иначе identity
                X = X
            else:
                X = torch.relu(conv(X, incidence, edge_weights))
        H_cf_prop = self.cf_module.norm(X)

        with torch.no_grad():
            a_before = self.gmm.nll(H, contexts).mean().item()
            a_after  = self.gmm.nll(H_cf_prop, contexts).mean().item()
        return a_before - a_after

    def _ce_with_weights(
        self,
        H: torch.Tensor,
        root_idx: int,
        proto: torch.Tensor,
        incidence: torch.Tensor,
        edge_weights: torch.Tensor,
        contexts: torch.Tensor,
    ) -> float:
        """CE при заданных весах гиперрёбер."""
        N = H.shape[0]
        mask = torch.zeros(N, 1, device=H.device, dtype=H.dtype)
        mask[root_idx] = 1.0
        H_cf = (1 - mask) * H + mask * proto.unsqueeze(0)
        H_cf_prop = self.cf_module._hg_forward(H_cf, incidence, edge_weights)

        with torch.no_grad():
            a_before = self.gmm.nll(H, contexts).mean().item()
            a_after  = self.gmm.nll(H_cf_prop, contexts).mean().item()
        return a_before - a_after
