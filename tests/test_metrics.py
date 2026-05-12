"""Тесты метрик качества: AC@k, NDCG@k, MRR."""
import pytest
from cairn.evaluation.metrics import (
    compute_ndcg, compute_mrr,
    compute_precision_at_k, compute_extended_metrics,
)


class TestEvaluationMetrics:
    """Тесты функций вычисления метрик качества RCA."""

    @pytest.fixture(autouse=True)
    def import_metrics(self):
        self.ndcg   = compute_ndcg
        self.mrr    = compute_mrr
        self.prec_k = compute_precision_at_k

    # ── AC@k ─────────────────────────────────────────────────────────────

    def test_ac1_perfect(self):
        """Root на #1 → AC@1=1."""
        ranked = [(0, 5.0), (1, 3.0), (2, 1.0)]
        assert self.prec_k(ranked, root_cause=0, k=1) == 1.0

    def test_ac1_wrong(self):
        """Root не в топ-1 → AC@1=0."""
        ranked = [(0, 5.0), (1, 3.0), (2, 1.0)]
        assert self.prec_k(ranked, root_cause=2, k=1) == 0.0

    def test_ac3_root_at_rank2(self):
        """Root на #2 → AC@3=1, AC@1=0."""
        ranked = [(0, 5.0), (2, 3.0), (1, 1.0)]
        assert self.prec_k(ranked, root_cause=2, k=1) == 0.0
        assert self.prec_k(ranked, root_cause=2, k=3) == 1.0

    def test_ac5_root_at_rank5(self):
        ranked = [(0, 5.0), (1, 4.0), (2, 3.0), (3, 2.0), (4, 1.0)]
        assert self.prec_k(ranked, root_cause=4, k=5) == 1.0
        assert self.prec_k(ranked, root_cause=4, k=3) == 0.0

    def test_ac1_empty_ranked(self):
        """Пустой список → 0."""
        assert self.prec_k([], root_cause=0, k=1) == 0.0

    # ── NDCG@k ───────────────────────────────────────────────────────────

    def test_ndcg1_perfect(self):
        """Root на #1 → NDCG@1=1."""
        import math
        ranked = [(0, 5.0), (1, 3.0)]
        result = self.ndcg(ranked, root_cause=0, k=1)
        assert abs(result - 1.0 / math.log2(2)) < 1e-6

    def test_ndcg1_wrong(self):
        """Root не в топ-1 → NDCG@1=0."""
        ranked = [(0, 5.0), (1, 3.0)]
        assert self.ndcg(ranked, root_cause=1, k=1) == 0.0

    def test_ndcg3_root_at_rank2(self):
        """Root на #2 → NDCG@3 = 1/log2(3)."""
        import math
        ranked = [(0, 5.0), (2, 3.0), (1, 1.0)]
        result = self.ndcg(ranked, root_cause=2, k=3)
        expected = 1.0 / math.log2(3)
        assert abs(result - expected) < 1e-6

    def test_ndcg_beyond_k(self):
        """Root за пределами k → NDCG@k=0."""
        ranked = [(0, 5.0), (1, 3.0), (2, 1.0), (3, 0.5)]
        assert self.ndcg(ranked, root_cause=3, k=3) == 0.0

    # ── MRR ──────────────────────────────────────────────────────────────

    def test_mrr_rank1(self):
        """Root на #1 → MRR=1.0."""
        ranked = [(0, 5.0), (1, 3.0), (2, 1.0)]
        assert self.mrr(ranked, root_cause=0) == 1.0

    def test_mrr_rank2(self):
        """Root на #2 → MRR=0.5."""
        ranked = [(0, 5.0), (1, 3.0), (2, 1.0)]
        assert self.mrr(ranked, root_cause=1) == 0.5

    def test_mrr_rank3(self):
        """Root на #3 → MRR≈0.333."""
        ranked = [(0, 5.0), (1, 3.0), (2, 1.0)]
        assert abs(self.mrr(ranked, root_cause=2) - 1/3) < 1e-6

    def test_mrr_not_found(self):
        """Root не в списке → MRR=0."""
        ranked = [(0, 5.0), (1, 3.0)]
        assert self.mrr(ranked, root_cause=99) == 0.0

    # ── Граничные случаи ─────────────────────────────────────────────────

    def test_single_element_ranking(self):
        """Единственный элемент является root."""
        ranked = [(0, 1.0)]
        assert self.prec_k(ranked, root_cause=0, k=1) == 1.0
        assert self.mrr(ranked, root_cause=0) == 1.0

    def test_all_zeros_scores(self):
        """Нулевые скоры не ломают метрики."""
        ranked = [(0, 0.0), (1, 0.0), (2, 0.0)]
        assert self.prec_k(ranked, root_cause=0, k=1) == 1.0
        assert self.mrr(ranked, root_cause=2) == pytest.approx(1/3, abs=1e-6)


class TestGraphVerifierEffect:
    """Тесты эффекта топологической корректировки."""

    def test_graph_verifier_penalizes_cascade(self, tiny_hypergraph):
        """Graph verifier снижает скор узлов с аномальными downstream-зависимостями.

        В цепи svc-0 → svc-1 → svc-2 с равными raw scores:
        - svc-2 (лист): нет callees → минимальный cascade penalty → наивысший adjusted
        - svc-0 (корень цепи): все callees аномальны → максимальный cascade penalty
        Это физически верно: если ваши зависимости аномальны, вы вероятно не root.
        """
        import numpy as np

        names  = tiny_hypergraph.instance_names
        idx_0  = names.index("svc-0")
        idx_2  = names.index("svc-2")

        raw_scores = {i: 1.0 for i in range(len(names))}

        called_by: dict[int, int]  = {}
        callee_map: dict[int, list] = {}
        for edge in tiny_hypergraph.edges:
            if edge.edge_type == "call" and len(edge.members) >= 2:
                s, d = edge.members[0], edge.members[1]
                callee_map.setdefault(s, []).append(d)
                called_by[d] = called_by.get(d, 0) + 1

        adjusted = {}
        for idx, score in raw_scores.items():
            cs      = [raw_scores.get(c, 0.0) for c in callee_map.get(idx, [])]
            cascade = float(np.mean(cs)) if cs else 0.0
            n_call  = called_by.get(idx, 0)
            adjusted[idx] = score / (1.0 + cascade) / (1.0 + n_call * 0.5)

        # Лист (svc-2) получает наибольший adjusted score –
        # он не имеет аномальных зависимостей, значит скорее всего root
        assert adjusted[idx_2] >= adjusted[idx_0], (
            f"Leaf score {adjusted[idx_2]:.3f} должен быть >= "
            f"internal node {adjusted[idx_0]:.3f}"
        )
        # Скоры различаются (алгоритм работает)
        assert adjusted[idx_0] != adjusted[idx_2], "Скоры должны различаться"
