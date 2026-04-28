"""Трёхрежимная декомпозиция множественных первопричин (раздел 3.3.5).

Режимы:
  ADDITIVE    — причины независимы (η < ε_адд)
  JOINT       — перебор подмножеств из top-кандидатов
  PROBABILISTIC — бета-биномиальный апостериорный вывод (для больших систем)
"""

from __future__ import annotations

import itertools
from enum import Enum
from typing import List, Tuple

import torch


class DecompositionMode(str, Enum):
    ADDITIVE = "additive"
    JOINT = "joint"
    PROBABILISTIC = "probabilistic"


def additivity_ratio(
    pe_a: float, pe_b: float, pe_ab: float
) -> float:
    """η = (ПЭ(A) + ПЭ(B) - ПЭ(A∪B)) / ПЭ(A∪B) (формула 3.30)."""
    if abs(pe_ab) < 1e-8:
        return 0.0
    return (pe_a + pe_b - pe_ab) / pe_ab


def joint_causal_effect(
    states: torch.Tensor,
    nll_fn,
    candidate_indices: List[int],
    prototypes: torch.Tensor,
    cf_module,
    incidence: torch.Tensor,
    edge_weights: torch.Tensor,
) -> float:
    """ПЭ(A ∪ B ∪ ...) — совместное вмешательство на нескольких узлах."""
    N = states.shape[0]
    nll_before = nll_fn(states).mean()

    states_cf = states.clone()
    for idx in candidate_indices:
        states_cf[idx] = prototypes[idx]

    # Распространяем через гиперграф
    states_cf_prop = cf_module._hypergraph_forward(states_cf, incidence, edge_weights)
    nll_after = nll_fn(states_cf_prop).mean()
    return (nll_before - nll_after).item()


def decompose_multiple_roots(
    ranked_candidates: List[Tuple[int, float]],   # [(idx, pe), ...] от funnel
    states: torch.Tensor,
    nll_fn,
    prototypes: torch.Tensor,
    cf_module,
    incidence: torch.Tensor,
    edge_weights: torch.Tensor,
    additivity_threshold: float = 0.15,
    max_joint_size: int = 3,
) -> Tuple[DecompositionMode, List[Tuple[int, float]]]:
    """Выбирает режим декомпозиции и возвращает список первопричин с оценками.

    Возвращает
    ----------
    (mode, [(node_idx, score), ...])
    """
    if len(ranked_candidates) < 2:
        return DecompositionMode.ADDITIVE, ranked_candidates

    top_indices = [idx for idx, _ in ranked_candidates[:max_joint_size]]
    top_pes = {idx: pe for idx, pe in ranked_candidates[:max_joint_size]}

    # Проверяем аддитивность для первой пары
    pe_a = top_pes.get(top_indices[0], 0.0)
    pe_b = top_pes.get(top_indices[1], 0.0)
    pe_ab = joint_causal_effect(
        states, nll_fn, top_indices[:2], prototypes, cf_module, incidence, edge_weights
    )
    eta = additivity_ratio(pe_a, pe_b, pe_ab)

    if eta < additivity_threshold:
        # Аддитивный режим: возвращаем кандидатов как независимых
        return DecompositionMode.ADDITIVE, ranked_candidates[:max_joint_size]

    # Совместный режим: перебираем подмножества
    best_subset: List[int] = []
    best_pe: float = 0.0
    for size in range(2, max_joint_size + 1):
        for subset in itertools.combinations(top_indices, size):
            pe = joint_causal_effect(
                states, nll_fn, list(subset), prototypes, cf_module, incidence, edge_weights
            )
            if pe > best_pe:
                best_pe = pe
                best_subset = list(subset)

    result = [(idx, top_pes.get(idx, 0.0)) for idx in best_subset]
    return DecompositionMode.JOINT, result
