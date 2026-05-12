"""Метрики качества локализации первопричин.

Функции:
    compute_precision_at_k  — AC@k (Accuracy@k)
    compute_ndcg            — NDCG@k
    compute_mrr             — Mean Reciprocal Rank
    compute_extended_metrics — все метрики для одного инцидента
"""
from __future__ import annotations
import math


def compute_precision_at_k(
    ranked: list[tuple[int, float]],
    root_cause: int,
    k: int,
) -> float:
    """AC@k: 1 если root_cause в топ-k, иначе 0."""
    top_k = [idx for idx, _ in ranked[:k]]
    return 1.0 if root_cause in top_k else 0.0


def compute_ndcg(
    ranked: list[tuple[int, float]],
    root_cause: int,
    k: int,
) -> float:
    """NDCG@k для задачи RCA (один релевантный документ).

    DCG@k  = 1 / log2(rank+1)  если root_cause найден в топ-k
    IDCG@k = 1 / log2(2) = 1   (идеальный случай: root на #1)
    NDCG@k = DCG@k / IDCG@k
    """
    for rank, (node_idx, _) in enumerate(ranked[:k], start=1):
        if node_idx == root_cause:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def compute_mrr(
    ranked: list[tuple[int, float]],
    root_cause: int,
) -> float:
    """MRR: 1/rank для первого корректного ответа."""
    for rank, (node_idx, _) in enumerate(ranked, start=1):
        if node_idx == root_cause:
            return 1.0 / rank
    return 0.0


def compute_extended_metrics(
    ranked: list[tuple[int, float]],
    root_cause: int,
) -> dict[str, float]:
    """Полный набор метрик для одного инцидента."""
    return {
        "AC@1":   compute_precision_at_k(ranked, root_cause, 1),
        "AC@3":   compute_precision_at_k(ranked, root_cause, 3),
        "AC@5":   compute_precision_at_k(ranked, root_cause, 5),
        "NDCG@1": compute_ndcg(ranked, root_cause, 1),
        "NDCG@3": compute_ndcg(ranked, root_cause, 3),
        "NDCG@5": compute_ndcg(ranked, root_cause, 5),
        "MRR":    compute_mrr(ranked, root_cause),
    }
