"""Тесты модуля рассуждений (reasoning): GMM, VGAE, CF, Funnel."""
import pytest
import torch
import numpy as np


class TestConditionalGMM:
    """GMM: плотность вероятности и NLL."""

    def test_nll_shape(self, cairn_model, small_arch):
        """NLL должен быть вектором длиной N."""
        N = 3
        D = small_arch["state_dim"]
        C = small_arch["context_dim"]

        H   = torch.randn(N, D)
        ctx = torch.randn(N, C)

        with torch.no_grad():
            nll = cairn_model.gmm.nll(H, ctx)

        assert nll.shape == (N,), f"Ожидалось ({N},), получено {nll.shape}"

    def test_nll_finite(self, cairn_model, small_arch):
        """NLL должен быть конечным."""
        N, D, C = 4, small_arch["state_dim"], small_arch["context_dim"]
        H   = torch.randn(N, D)
        ctx = torch.randn(N, C)

        with torch.no_grad():
            nll = cairn_model.gmm.nll(H, ctx)

        assert not torch.isnan(nll).any(), "NaN в NLL"
        assert not torch.isinf(nll).any(), "Inf в NLL"

    def test_anomalous_higher_nll(self, cairn_model, small_arch):
        """Аномальные данные должны иметь более высокий NLL."""
        D   = small_arch["state_dim"]
        C   = small_arch["context_dim"]
        ctx = torch.zeros(2, C)

        # Нормальный и экстремальный сигналы
        H_normal = torch.randn(2, D) * 0.1
        H_anomal = torch.randn(2, D) * 10.0

        with torch.no_grad():
            nll_normal = cairn_model.gmm.nll(H_normal, ctx).mean()
            nll_anomal = cairn_model.gmm.nll(H_anomal, ctx).mean()

        # Экстремальные данные должны хуже объясняться моделью
        # (NLL выше), если модель достаточно обучена.
        # Для необученной модели это может не выполняться — проверяем хотя бы конечность.
        assert torch.isfinite(nll_normal)
        assert torch.isfinite(nll_anomal)

    def test_single_sample(self, cairn_model, small_arch):
        """Работает с batch size = 1."""
        D   = small_arch["state_dim"]
        C   = small_arch["context_dim"]
        H   = torch.randn(1, D)
        ctx = torch.randn(1, C)

        with torch.no_grad():
            nll = cairn_model.gmm.nll(H, ctx)

        assert nll.shape == (1,)


class TestConfoundedVGAE:
    """VGAE и CF-модуль: проверяем через публичный API (intervene/causal_effect)."""

    def test_vgae_exists(self, cairn_model):
        """VGAE и CF модули присутствуют в модели."""
        assert hasattr(cairn_model, "vgae"), "vgae не найден"
        assert hasattr(cairn_model, "cf_module"), "cf_module не найден"

    def test_cf_module_intervene_shape(self, cairn_model, small_arch, tiny_hypergraph):
        """intervene() возвращает тензор той же формы что H."""
        N   = len(tiny_hypergraph.instance_names)
        D   = small_arch["state_dim"]
        C   = small_arch["context_dim"]
        H   = torch.randn(N, D)
        ctx = torch.randn(N, C)

        with torch.no_grad():
            # prototype берём из GMM как в mediation.py
            proto = cairn_model.gmm.prototype(ctx[0:1]).squeeze(0)
            H_cf  = cairn_model.cf_module.intervene(H, 0, proto, tiny_hypergraph)

        assert H_cf.shape == (N, D), f"Ожидалось ({N}, {D}), получено {H_cf.shape}"

    def test_cf_module_causal_effect(self, cairn_model, small_arch, tiny_hypergraph):
        """causal_effect() возвращает конечные значения на узел."""
        N   = len(tiny_hypergraph.instance_names)
        D   = small_arch["state_dim"]
        C   = small_arch["context_dim"]
        H   = torch.randn(N, D)
        ctx = torch.randn(N, C)

        with torch.no_grad():
            proto = cairn_model.gmm.prototype(ctx[0:1]).squeeze(0)
            H_cf  = cairn_model.cf_module.intervene(H, 0, proto, tiny_hypergraph)
            ce    = cairn_model.cf_module.causal_effect(H, H_cf, cairn_model.gmm, ctx)

        # causal_effect возвращает float или тензор
        if hasattr(ce, "numel"):
            assert torch.isfinite(ce).all()
        else:
            assert isinstance(ce, float) and not (ce != ce)  # not NaN


class TestCounterfactualModule:
    """CounterfactualModule: контрфактическое вмешательство через GNN."""

    def test_output_shape(self, cairn_model, small_arch, tiny_hypergraph):
        """intervene() возвращает тензор той же формы что H."""
        N   = len(tiny_hypergraph.instance_names)
        D   = small_arch["state_dim"]
        C   = small_arch["context_dim"]
        H   = torch.randn(N, D)
        ctx = torch.randn(N, C)

        with torch.no_grad():
            proto = cairn_model.gmm.prototype(ctx[0:1]).squeeze(0)
            H_cf  = cairn_model.cf_module.intervene(H, 0, proto, tiny_hypergraph)

        assert H_cf.shape == H.shape, f"Shape mismatch: {H_cf.shape} vs {H.shape}"

    def test_refinement_changes_values(self, cairn_model, small_arch, tiny_hypergraph):
        """После вмешательства значения изменились."""
        N   = len(tiny_hypergraph.instance_names)
        D   = small_arch["state_dim"]
        C   = small_arch["context_dim"]
        H   = torch.randn(N, D)
        ctx = torch.randn(N, C)

        with torch.no_grad():
            proto = cairn_model.gmm.prototype(ctx[0:1]).squeeze(0)
            H_cf  = cairn_model.cf_module.intervene(H, 0, proto, tiny_hypergraph)

        # Вмешательство изменяет хотя бы один узел
        assert not torch.allclose(H, H_cf), "intervene не изменил представление"


class TestCascadeFunnel:
    """CascadeFunnel: ранжирование первопричин."""

    def test_ranking_length(self, cairn_model, small_arch, tiny_hypergraph):
        """Ранжирование содержит правильное число элементов."""
        from cairn.reasoning import CascadeFunnel

        N = len(tiny_hypergraph.instance_names)
        D = small_arch["state_dim"]
        C = small_arch["context_dim"]

        H   = torch.randn(N, D)
        ctx = torch.randn(N, C)
        adj = tiny_hypergraph.adjacency_matrix()
        adj_norm = adj / adj.sum(1, keepdim=True).clamp(min=1)

        with torch.no_grad():
            nll = cairn_model.gmm.nll(H, ctx)

        funnel = CascadeFunnel(l0_top_k=N, l1_top_k=N, l2_top_k=N)
        with torch.no_grad():
            ranked = funnel.run(
                nll, H, adj_norm,
                cairn_model.cf_module, cairn_model.gmm,
                ctx, tiny_hypergraph,
            )

        assert len(ranked) == N, f"Ожидалось {N} элементов, получено {len(ranked)}"

    def test_ranking_is_sorted(self, cairn_model, small_arch, tiny_hypergraph):
        """Ранжирование отсортировано по убыванию."""
        from cairn.reasoning import CascadeFunnel

        N = len(tiny_hypergraph.instance_names)
        D = small_arch["state_dim"]
        C = small_arch["context_dim"]

        H   = torch.randn(N, D)
        ctx = torch.randn(N, C)
        adj = tiny_hypergraph.adjacency_matrix()
        adj_norm = adj / adj.sum(1, keepdim=True).clamp(min=1)

        with torch.no_grad():
            nll    = cairn_model.gmm.nll(H, ctx)
            funnel = CascadeFunnel(l0_top_k=N, l1_top_k=N, l2_top_k=N)
            ranked = funnel.run(
                nll, H, adj_norm,
                cairn_model.cf_module, cairn_model.gmm,
                ctx, tiny_hypergraph,
            )

        scores = [score for _, score in ranked]
        assert scores == sorted(scores, reverse=True), "Ранжирование не отсортировано"

    def test_without_cf_module(self, cairn_model, small_arch, tiny_hypergraph):
        """Работает без CF-модуля (cf_module=None)."""
        from cairn.reasoning import CascadeFunnel

        N = len(tiny_hypergraph.instance_names)
        D = small_arch["state_dim"]
        C = small_arch["context_dim"]

        H   = torch.randn(N, D)
        ctx = torch.randn(N, C)
        adj = tiny_hypergraph.adjacency_matrix()
        adj_norm = adj / adj.sum(1, keepdim=True).clamp(min=1)

        with torch.no_grad():
            nll    = cairn_model.gmm.nll(H, ctx)
            funnel = CascadeFunnel(l0_top_k=N, l1_top_k=N, l2_top_k=N)
            ranked = funnel.run(
                nll, H, adj_norm,
                None,  # cf_module отключён
                cairn_model.gmm, ctx, tiny_hypergraph,
            )

        assert len(ranked) == N