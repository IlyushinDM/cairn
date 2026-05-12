"""Тесты модуля восприятия (perception): StateBuilder и энкодеры."""
import pytest
import torch
import numpy as np


class TestStateBuilder:
    """StateBuilder: кодирует метрики, логи, трассировки → state вектор."""

    def test_output_shape(self, cairn_model, small_arch):
        """Выход StateBuilder должен быть (batch, state_dim)."""
        B = 2
        W = small_arch["window"]
        F = small_arch["n_metrics"]
        D = small_arch["state_dim"]
        C = small_arch["context_dim"]

        m_t     = torch.randn(B, W, F)
        log_ids = torch.zeros(B, 3, dtype=torch.long)
        log_len = torch.ones(B, dtype=torch.long) * 3
        dummy_d = torch.zeros(B, small_arch["d_met"])

        with torch.no_grad():
            H, ctx = cairn_model.state_builder(m_t, log_ids, log_len, dummy_d)

        assert H.shape   == (B, D), f"H shape: {H.shape}"
        assert ctx.shape == (B, C), f"ctx shape: {ctx.shape}"

    def test_no_nan_output(self, cairn_model, small_arch):
        """Нет NaN в выходе при нормальном вводе."""
        W = small_arch["window"]
        F = small_arch["n_metrics"]

        m_t     = torch.randn(1, W, F)
        log_ids = torch.zeros(1, 1, dtype=torch.long)
        log_len = torch.ones(1, dtype=torch.long)
        dummy_d = torch.zeros(1, small_arch["d_met"])

        with torch.no_grad():
            H, ctx = cairn_model.state_builder(m_t, log_ids, log_len, dummy_d)

        assert not torch.isnan(H).any(),   "NaN в H"
        assert not torch.isnan(ctx).any(), "NaN в ctx"

    def test_batch_consistency(self, cairn_model, small_arch):
        """Результат не зависит от batch size."""
        W = small_arch["window"]
        F = small_arch["n_metrics"]

        m_t     = torch.randn(1, W, F)
        log_ids = torch.zeros(1, 1, dtype=torch.long)
        log_len = torch.ones(1, dtype=torch.long)
        dummy_d = torch.zeros(1, small_arch["d_met"])

        with torch.no_grad():
            H1, _ = cairn_model.state_builder(m_t, log_ids, log_len, dummy_d)

        m_t_batch     = m_t.repeat(3, 1, 1)
        log_ids_batch = log_ids.repeat(3, 1)
        log_len_batch = log_len.repeat(3)
        dummy_d_batch = dummy_d.repeat(3, 1)

        with torch.no_grad():
            H3, _ = cairn_model.state_builder(
                m_t_batch, log_ids_batch, log_len_batch, dummy_d_batch
            )

        torch.testing.assert_close(H1[0], H3[0], atol=1e-5, rtol=1e-5)

    def test_different_inputs_different_outputs(self, cairn_model, small_arch):
        """Разные входы → разные выходы."""
        W = small_arch["window"]
        F = small_arch["n_metrics"]

        dummy_d = torch.zeros(1, small_arch["d_met"])
        log_ids = torch.zeros(1, 1, dtype=torch.long)
        log_len = torch.ones(1, dtype=torch.long)

        with torch.no_grad():
            H1, _ = cairn_model.state_builder(
                torch.randn(1, W, F), log_ids, log_len, dummy_d)
            H2, _ = cairn_model.state_builder(
                torch.randn(1, W, F) * 5, log_ids, log_len, dummy_d)

        assert not torch.allclose(H1, H2), "Разные входы дали одинаковый выход"

    def test_zero_metrics(self, cairn_model, small_arch):
        """Нулевые метрики → стабильный выход без NaN."""
        W = small_arch["window"]
        F = small_arch["n_metrics"]

        m_t     = torch.zeros(1, W, F)
        log_ids = torch.zeros(1, 1, dtype=torch.long)
        log_len = torch.ones(1, dtype=torch.long)
        dummy_d = torch.zeros(1, small_arch["d_met"])

        with torch.no_grad():
            H, ctx = cairn_model.state_builder(m_t, log_ids, log_len, dummy_d)

        assert not torch.isnan(H).any()
        assert not torch.isinf(H).any()


class TestHypergraphBuilder:
    """HypergraphBuilder: строит гиперграф из топологии."""

    def test_build_from_topology(self, tiny_hypergraph):
        """Гиперграф строится без ошибок."""
        assert tiny_hypergraph is not None
        assert len(tiny_hypergraph.instance_names) == 3

    def test_instance_names(self, tiny_hypergraph):
        assert "svc-0" in tiny_hypergraph.instance_names
        assert "svc-1" in tiny_hypergraph.instance_names
        assert "svc-2" in tiny_hypergraph.instance_names

    def test_edges_present(self, tiny_hypergraph):
        """Рёбра корректно создаются."""
        assert len(tiny_hypergraph.edges) >= 2

    def test_adjacency_matrix_shape(self, tiny_hypergraph):
        """Матрица смежности правильной формы."""
        import torch
        adj = tiny_hypergraph.adjacency_matrix()
        N   = len(tiny_hypergraph.instance_names)
        assert adj.shape == (N, N)

    def test_adjacency_non_negative(self, tiny_hypergraph):
        adj = tiny_hypergraph.adjacency_matrix()
        assert (adj >= 0).all()
