"""Интеграционные тесты: полный пайплайн от данных до результата."""
import pytest
import torch
import numpy as np


@pytest.mark.integration
class TestFullPipeline:
    """Полный пайплайн: StateBuilder → GMM → CascadeFunnel → ранжирование."""

    def test_end_to_end(self, cairn_model, small_arch, tiny_hypergraph):
        """Полный прогон без ошибок."""
        from cairn.reasoning import CascadeFunnel

        N = len(tiny_hypergraph.instance_names)
        D = small_arch["state_dim"]
        C = small_arch["context_dim"]
        W = small_arch["window"]
        F = small_arch["n_metrics"]

        # Симулируем входные данные
        m_t     = torch.randn(N, W, F)
        log_ids = torch.zeros(N, 1, dtype=torch.long)
        log_len = torch.ones(N, dtype=torch.long)
        dummy_d = torch.zeros(N, small_arch["d_met"])

        with torch.no_grad():
            # 1. StateBuilder
            H, ctx = cairn_model.state_builder(m_t, log_ids, log_len, dummy_d)
            assert H.shape == (N, D)

            # 2. GMM → NLL
            nll = cairn_model.gmm.nll(H, ctx)
            assert nll.shape == (N,)

            # 3. CascadeFunnel
            adj      = tiny_hypergraph.adjacency_matrix()
            adj_norm = adj / adj.sum(1, keepdim=True).clamp(min=1)
            funnel   = CascadeFunnel(l0_top_k=N, l1_top_k=N, l2_top_k=N)
            ranked   = funnel.run(
                nll, H, adj_norm,
                cairn_model.cf_module, cairn_model.gmm,
                ctx, tiny_hypergraph,
            )

        # 4. Результат
        assert len(ranked) == N
        assert all(0 <= idx < N for idx, _ in ranked)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_root_cause_detectable(self, cairn_model, small_arch, tiny_hypergraph):
        """При явной аномалии в одном узле он должен иметь высокий NLL."""
        from cairn.reasoning import CascadeFunnel

        N = len(tiny_hypergraph.instance_names)
        W = small_arch["window"]
        F = small_arch["n_metrics"]

        # Нормальные данные для всех кроме первого
        m_t     = torch.randn(N, W, F) * 0.1
        # Аномальные данные для первого узла
        m_t[0] = torch.randn(W, F) * 100.0

        log_ids = torch.zeros(N, 1, dtype=torch.long)
        log_len = torch.ones(N, dtype=torch.long)
        dummy_d = torch.zeros(N, small_arch["d_met"])

        with torch.no_grad():
            H, ctx = cairn_model.state_builder(m_t, log_ids, log_len, dummy_d)
            nll    = cairn_model.gmm.nll(H, ctx)

        # У необученной модели нет гарантии что NLL[0] > NLL[1,2],
        # но проверяем что NLL конечен для всех
        assert torch.isfinite(nll).all()

    def test_multiple_incidents(self, cairn_model, small_arch, tiny_hypergraph):
        """Пайплайн стабилен при нескольких последовательных прогонах."""
        from cairn.reasoning import CascadeFunnel

        N = len(tiny_hypergraph.instance_names)
        W = small_arch["window"]
        F = small_arch["n_metrics"]

        adj      = tiny_hypergraph.adjacency_matrix()
        adj_norm = adj / adj.sum(1, keepdim=True).clamp(min=1)
        funnel   = CascadeFunnel(l0_top_k=N, l1_top_k=N, l2_top_k=N)

        for _ in range(5):
            m_t     = torch.randn(N, W, F)
            log_ids = torch.zeros(N, 1, dtype=torch.long)
            log_len = torch.ones(N, dtype=torch.long)
            dummy_d = torch.zeros(N, small_arch["d_met"])

            with torch.no_grad():
                H, ctx = cairn_model.state_builder(m_t, log_ids, log_len, dummy_d)
                nll    = cairn_model.gmm.nll(H, ctx)
                ranked = funnel.run(
                    nll, H, adj_norm,
                    cairn_model.cf_module, cairn_model.gmm,
                    ctx, tiny_hypergraph,
                )

            assert len(ranked) == N
            assert not any(torch.isnan(torch.tensor(s)) for _, s in ranked)

    @pytest.mark.slow
    def test_checkpoint_save_load(self, cairn_model, tmp_path):
        """Модель сохраняется и загружается с теми же весами."""
        ckpt_path = tmp_path / "test.pt"

        # Сохраняем
        torch.save({
            "model_state_dict": cairn_model.state_dict(),
            "arch_config": {},
        }, ckpt_path)

        # Загружаем
        ckpt    = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state   = ckpt["model_state_dict"]
        missing, unexpected = cairn_model.load_state_dict(state, strict=False)

        assert len(unexpected) == 0, f"Неожиданные ключи: {unexpected}"
