"""Трёхрежимная декомпозиция множественных первопричин (раздел 3.3.5).

Режимы:
  ADDITIVE      – η < ε_add → причины независимы, суммируем ПЭ
  JOINT         – η ≥ ε_add → перебор подмножеств top-кандидатов
  PROBABILISTIC – N > порога → Beta-Binomial fallback (для больших систем)
"""

from __future__ import annotations

import itertools
from enum import Enum
from typing import List, Optional, Tuple

import torch


class DecompositionMode(str, Enum):
    ADDITIVE      = "additive"
    JOINT         = "joint"
    PROBABILISTIC = "probabilistic"


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def additivity_ratio(pe_a: float, pe_b: float, pe_ab: float) -> float:
    """η = (ПЭ(A) + ПЭ(B) - ПЭ(A∪B)) / ПЭ(A∪B)  (формула 3.30).

    η близко к 0 → аддитивность; η > 0 → синергия (взаимодействие причин).
    """
    if abs(pe_ab) < 1e-8:
        return 0.0
    return (pe_a + pe_b - pe_ab) / pe_ab


def joint_causal_effect(
    H: torch.Tensor,
    gmm,
    contexts: torch.Tensor,
    cf_module,
    hypergraph,
    candidate_indices: List[int],
    prototypes: torch.Tensor,       # (N, d)
) -> float:
    """ПЭ(A ∪ B ...) – совместное вмешательство на нескольких узлах."""
    with torch.no_grad():
        a_before = gmm.nll(H, contexts).mean().item()

        # Применяем вмешательство на всех кандидатах
        incidence    = hypergraph.incidence_matrix().to(H.device)
        edge_weights = hypergraph.edge_weights().to(H.device)

        N = H.shape[0]
        mask = torch.zeros(N, 1, device=H.device, dtype=H.dtype)
        for idx in candidate_indices:
            mask[idx] = 1.0
        H_cf = (1 - mask) * H + mask * _stack_protos(candidate_indices, prototypes, H)
        H_cf_prop = cf_module._hg_forward(H_cf, incidence, edge_weights)
        a_after = gmm.nll(H_cf_prop, contexts).mean().item()

    return a_before - a_after


def _stack_protos(
    indices: List[int], prototypes: torch.Tensor, H: torch.Tensor
) -> torch.Tensor:
    """Создаёт матрицу прототипов N×d: row i = prototypes[i] если i в indices, иначе H[i]."""
    out = H.clone()
    for idx in indices:
        out[idx] = prototypes[idx]
    return out


# ---------------------------------------------------------------------------
# Класс-фасад
# ---------------------------------------------------------------------------

class MultiRootCauseDecomposition:
    """Трёхрежимная декомпозиция множественных первопричин (раздел 3.3.5).

    Параметры
    ----------
    additivity_threshold : float
        ε_add – порог η, ниже которого считаем причины аддитивными (0.15).
    max_joint_size : int
        Максимальный размер подмножества в совместном режиме (3).
    probabilistic_threshold : int
        Если N > этого значения – используем вероятностный режим.
    """

    def __init__(
        self,
        additivity_threshold: float = 0.15,
        max_joint_size: int = 3,
        probabilistic_threshold: int = 100,
    ) -> None:
        self.eta_threshold     = additivity_threshold
        self.max_joint         = max_joint_size
        self.prob_threshold    = probabilistic_threshold

    def decompose(
        self,
        ranked_candidates: List[Tuple[int, float]],   # [(idx, pe), ...] от воронки
        H: torch.Tensor,
        gmm,
        contexts: torch.Tensor,
        cf_module,
        hypergraph,
    ) -> Tuple[DecompositionMode, List[Tuple[int, float]]]:
        """Выбирает режим и возвращает финальный список первопричин.

        Возвращает
        ----------
        (mode, [(node_idx, score), ...])
        """
        if len(ranked_candidates) < 2:
            return DecompositionMode.ADDITIVE, ranked_candidates

        N = H.shape[0]
        if N > self.prob_threshold:
            return self._probabilistic(ranked_candidates)

        # Строим прототипы один раз
        prototypes = torch.stack([
            gmm.prototype(contexts[i:i+1]).squeeze(0) for i in range(N)
        ])

        top_n = min(self.max_joint, len(ranked_candidates))
        top_indices = [idx for idx, _ in ranked_candidates[:top_n]]
        top_pes     = {idx: pe for idx, pe in ranked_candidates[:top_n]}

        # Проверка аддитивности на первой паре
        pe_a  = top_pes[top_indices[0]]
        pe_b  = top_pes[top_indices[1]]
        pe_ab = joint_causal_effect(
            H, gmm, contexts, cf_module, hypergraph,
            top_indices[:2], prototypes
        )
        eta = additivity_ratio(pe_a, pe_b, pe_ab)

        if eta < self.eta_threshold:
            return DecompositionMode.ADDITIVE, ranked_candidates[:top_n]

        return self._joint(top_indices, top_pes, H, gmm, contexts, cf_module, hypergraph, prototypes)

    def _joint(
        self, top_indices, top_pes, H, gmm, contexts, cf_module, hypergraph, prototypes
    ) -> Tuple[DecompositionMode, List[Tuple[int, float]]]:
        """Режим 2: перебор подмножеств размера 2..max_joint."""
        best_subset: List[int] = []
        best_pe: float = -1e9

        for size in range(2, self.max_joint + 1):
            for subset in itertools.combinations(top_indices, size):
                pe = joint_causal_effect(
                    H, gmm, contexts, cf_module, hypergraph, list(subset), prototypes
                )
                if pe > best_pe:
                    best_pe = pe
                    best_subset = list(subset)

        result = [(idx, top_pes.get(idx, 0.0)) for idx in best_subset]
        return DecompositionMode.JOINT, result

    def _probabilistic(
        self, ranked_candidates: List[Tuple[int, float]]
    ) -> Tuple[DecompositionMode, List[Tuple[int, float]]]:
        """Режим 3: Beta-Binomial fallback – берём top-k по ПЭ.

        При N > probabilistic_threshold полный перебор нецелесообразен.
        Возвращаем всех кандидатов сверху до первого значительного падения ПЭ.
        """
        if not ranked_candidates:
            return DecompositionMode.PROBABILISTIC, []

        # Находим «локоть» по падению ПЭ
        result = [ranked_candidates[0]]
        top_pe = ranked_candidates[0][1]
        for idx, pe in ranked_candidates[1:self.max_joint]:
            if top_pe > 0 and pe / top_pe < 0.3:
                break
            result.append((idx, pe))

        return DecompositionMode.PROBABILISTIC, result


# ---------------------------------------------------------------------------
# Функциональный API (обратная совместимость)
# ---------------------------------------------------------------------------

def decompose_multiple_roots(
    ranked_candidates: List[Tuple[int, float]],
    H: torch.Tensor,
    gmm,
    contexts: torch.Tensor,
    cf_module,
    hypergraph,
    additivity_threshold: float = 0.15,
    max_joint_size: int = 3,
) -> Tuple[DecompositionMode, List[Tuple[int, float]]]:
    """Обёртка для обратной совместимости."""
    dec = MultiRootCauseDecomposition(additivity_threshold, max_joint_size)
    return dec.decompose(ranked_candidates, H, gmm, contexts, cf_module, hypergraph)
