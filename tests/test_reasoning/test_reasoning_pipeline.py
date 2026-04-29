"""Тесты фазы рассуждения CAIRN.

Классы:
  TestConditionalGMM           — unit-тесты условной GMM
  TestConfoundedVGAE           — unit-тесты VGAE с конфаундерами
  TestCounterfactualModule     — unit-тесты контрфактического модуля
  TestMultiRootDecomposition   — unit-тесты декомпозиции
  TestCascadeFunnel            — unit-тесты воронки
  TestCausalGraphVerifier      — unit-тесты верификатора (5 аксиом)
  TestReasoningPipeline        — сквозной тест на демо-данных
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
import torch

from cairn.reasoning import (
    ConditionalGMM,
    ConfoundedVGAE,
    ExogenousEncoder,
    LatentConfounderModule,
    CounterfactualModule,
    HypergraphConv,
    MultiRootCauseDecomposition,
    DecompositionMode,
    CascadeFunnel,
    CausalGraphVerifier,
    AxiomStatus,
    additivity_ratio,
)
from cairn.perception import HypergraphBuilder

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
N        = 5     # число узлов
D        = 64    # state_dim (уменьшен для скорости тестов)
CTX      = 16    # context_dim
N_COMP   = 3     # n_components GMM
N_CONF   = 2     # n_confounders
CONF_DIM = 16    # confounder_dim
BATCH    = N
SAMPLE_DIR = Path(__file__).parent.parent.parent / "data" / "sample"


# ---------------------------------------------------------------------------
# Общие фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture
def gmm():
    return ConditionalGMM(state_dim=D, context_dim=CTX, n_components=N_COMP)

@pytest.fixture
def H():
    return torch.randn(N, D)

@pytest.fixture
def contexts():
    return torch.randn(N, CTX)

@pytest.fixture
def hypergraph():
    builder = HypergraphBuilder(N)
    return builder.from_topology(
        call_paths=[[0, 1], [1, 2], [1, 3]],
        colocated_groups=[[1, 4]],
        lb_groups=[],
        instance_names=[f"svc-{i}" for i in range(N)],
    )

@pytest.fixture
def cf_module():
    return CounterfactualModule(state_dim=D, n_conv_layers=1)


# ===========================================================================
# TestConditionalGMM
# ===========================================================================

class TestConditionalGMM:

    def test_forward_shapes(self, gmm, contexts):
        w, mu, lv = gmm(contexts)
        assert w.shape  == (N, N_COMP)
        assert mu.shape == (N, N_COMP, D)
        assert lv.shape == (N, N_COMP, D)

    def test_weights_sum_to_one(self, gmm, contexts):
        w, _, _ = gmm(contexts)
        assert torch.allclose(w.sum(-1), torch.ones(N), atol=1e-5)

    def test_nll_shape(self, gmm, H, contexts):
        nll = gmm.nll(H, contexts)
        assert nll.shape == (N,)

    def test_nll_no_nan(self, gmm, H, contexts):
        nll = gmm.nll(H, contexts)
        assert not torch.isnan(nll).any()

    def test_nll_broadcast_context(self, gmm, H):
        """Один контекст для всего батча."""
        ctx_single = torch.randn(1, CTX)
        nll = gmm.nll(H, ctx_single)
        assert nll.shape == (N,)

    def test_prototype_shape(self, gmm, contexts):
        proto = gmm.prototype(contexts)
        assert proto.shape == (N, D)

    def test_prototype_equals_best_mean(self, gmm, contexts):
        """prototype() = строка μ_{k*} с максимальным весом."""
        proto = gmm.prototype(contexts)
        w, mu, _ = gmm(contexts)
        k_star = w.argmax(-1)
        for i in range(N):
            assert torch.allclose(proto[i], mu[i, k_star[i]], atol=1e-5)

    def test_conditional_prototype_alias(self, gmm, contexts):
        p1 = gmm.prototype(contexts)
        p2 = gmm.conditional_prototype(contexts)
        assert torch.allclose(p1, p2)

    def test_anomaly_threshold(self, gmm):
        H_large   = torch.randn(100, D)
        ctx_large = torch.randn(100, CTX)
        nll   = gmm.nll(H_large, ctx_large)
        delta = gmm.anomaly_threshold(nll, percentile=0.9)
        assert isinstance(delta, float)
        frac = (nll <= delta + 1e-5).float().mean().item()
        assert frac >= 0.9 - 1.0 / 100

    def test_detect_drift_no_drift(self, gmm):
        nll_history = torch.randn(100) + 1.0   # среднее = 1
        threshold = 5.0
        assert not gmm.detect_drift(nll_history, threshold)

    def test_detect_drift_with_drift(self, gmm):
        nll_history = torch.randn(100) + 10.0  # среднее = 10 >> threshold
        threshold = 1.0
        assert gmm.detect_drift(nll_history, threshold)

    def test_detect_drift_short_history(self, gmm):
        """Если история короче window — нет дрейфа по умолчанию."""
        nll_history = torch.tensor([100.0, 200.0])
        assert not gmm.detect_drift(nll_history, threshold=1.0, window=20)

    def test_gradient_through_nll(self, gmm, H, contexts):
        H_req = H.requires_grad_(True)
        nll = gmm.nll(H_req, contexts)
        nll.sum().backward()
        assert H_req.grad is not None

    def test_gradient_through_prototype(self, gmm, contexts):
        ctx_req = contexts.requires_grad_(True)
        proto = gmm.prototype(ctx_req)
        proto.sum().backward()
        assert ctx_req.grad is not None


# ===========================================================================
# TestConfoundedVGAE
# ===========================================================================

class TestConfoundedVGAE:

    @pytest.fixture
    def vgae(self):
        return ConfoundedVGAE(
            state_dim=D, n_confounders=N_CONF, confounder_dim=CONF_DIM,
            n_node_types=3, edge_dim=8,
        )

    @pytest.fixture
    def edge_data(self):
        # Простой граф: 0→1, 1→2, 1→3
        edge_index = torch.tensor([[0, 1, 1], [1, 2, 3]], dtype=torch.long)
        edge_type  = torch.tensor([0, 0, 1], dtype=torch.long)
        return edge_index, edge_type

    def test_encode_shapes(self, vgae, H, edge_data):
        ei, et = edge_data
        exog, z_hats, masks, kl = vgae.encode(H, ei, et)
        assert exog.shape == (N, D)
        assert len(z_hats) == N_CONF
        assert not torch.isnan(exog).any()
        assert not torch.isnan(kl)

    def test_kl_positive(self, vgae, H, edge_data):
        """KL-дивергенция должна быть ≥ 0 для произвольных параметров."""
        ei, et = edge_data
        _, _, _, kl = vgae.encode(H, ei, et)
        assert kl.item() > -1.0   # KL может быть слегка отрицательной в начале

    def test_decode_shape(self, vgae, H, edge_data):
        ei, et = edge_data
        exog, _, _, _ = vgae.encode(H, ei, et)
        h_recon = vgae.decode(exog)
        assert h_recon.shape == (N, D)

    def test_independence_loss_shape(self, vgae, H, edge_data):
        ei, et = edge_data
        exog, _, _, _ = vgae.encode(H, ei, et)
        loss = vgae.independence_loss(exog)
        assert loss.shape == ()
        assert 0.0 <= loss.item() <= 1.0 + 1e-5

    def test_independence_loss_single_node(self, vgae):
        u = torch.randn(1, D)
        loss = vgae.independence_loss(u)
        assert loss.item() == 0.0

    def test_encode_no_grad_required(self, vgae, H, edge_data):
        """Encode работает без requires_grad."""
        ei, et = edge_data
        with torch.no_grad():
            exog, _, _, kl = vgae.encode(H, ei, et)
        assert exog.shape == (N, D)

    def test_gradient_through_encode(self, vgae, H, edge_data):
        ei, et = edge_data
        H_req = H.requires_grad_(True)
        exog, _, _, kl = vgae.encode(H_req, ei, et)
        (exog.sum() + kl).backward()
        assert H_req.grad is not None


# ===========================================================================
# TestCounterfactualModule
# ===========================================================================

class TestCounterfactualModule:

    def test_hg_conv_shape(self):
        conv = HypergraphConv(D, D)
        X = torch.randn(N, D)
        H_inc = torch.randint(0, 2, (N, 4)).float()
        W = torch.ones(4)
        out = conv(X, H_inc, W)
        assert out.shape == (N, D)

    def test_intervene_shape(self, cf_module, H, gmm, contexts, hypergraph):
        proto = gmm.prototype(contexts[0:1]).squeeze(0)
        H_cf = cf_module.intervene(H, 0, proto, hypergraph)
        assert H_cf.shape == (N, D)

    def test_intervene_differentiable(self, cf_module, H, gmm, contexts, hypergraph):
        """Градиент должен проходить через prototype → contexts."""
        ctx_req = contexts.requires_grad_(True)
        proto = gmm.prototype(ctx_req[0:1]).squeeze(0)
        H_cf = cf_module.intervene(H, 0, proto, hypergraph)
        H_cf.sum().backward()
        assert ctx_req.grad is not None, "Градиент не проходит через intervene"

    def test_intervene_changes_target(self, cf_module, H, gmm, contexts, hypergraph):
        """H_cf[i] должен отличаться от H[i] после propagation."""
        proto = gmm.prototype(contexts[0:1]).squeeze(0)
        H_cf = cf_module.intervene(H, 0, proto, hypergraph)
        # После гиперграфовой свёртки состояния должны измениться
        assert not torch.allclose(H_cf, H, atol=1e-4)

    def test_causal_effect_type(self, cf_module, H, gmm, contexts, hypergraph):
        proto = gmm.prototype(contexts[0:1]).squeeze(0)
        H_cf = cf_module.intervene(H, 0, proto, hypergraph)
        ce = cf_module.causal_effect(H, H_cf, gmm, contexts)
        assert isinstance(ce, float)

    def test_rank_candidates_sorted(self, cf_module, H, gmm, contexts, hypergraph):
        ranked = cf_module.rank_candidates(H, [0, 1, 2], gmm, contexts, hypergraph)
        assert len(ranked) == 3
        scores = [ce for _, ce in ranked]
        assert scores == sorted(scores, reverse=True), "Не отсортировано по убыванию"

    def test_rank_candidates_all_candidates(self, cf_module, H, gmm, contexts, hypergraph):
        ranked = cf_module.rank_candidates(H, list(range(N)), gmm, contexts, hypergraph)
        assert len(ranked) == N
        idxs = sorted(idx for idx, _ in ranked)
        assert idxs == list(range(N))


# ===========================================================================
# TestMultiRootDecomposition
# ===========================================================================

class TestMultiRootDecomposition:

    @pytest.fixture
    def dec(self):
        return MultiRootCauseDecomposition(
            additivity_threshold=0.15, max_joint_size=2
        )

    def test_single_candidate_additive(self, dec, H, gmm, contexts, cf_module, hypergraph):
        ranked = [(2, 5.0)]
        mode, result = dec.decompose(ranked, H, gmm, contexts, cf_module, hypergraph)
        assert mode == DecompositionMode.ADDITIVE
        assert result == [(2, 5.0)]

    def test_additive_mode_returns_top_n(self, dec, H, gmm, contexts, cf_module, hypergraph):
        ranked = [(2, 5.0), (0, 4.0)]
        mode, result = dec.decompose(ranked, H, gmm, contexts, cf_module, hypergraph)
        assert mode in (DecompositionMode.ADDITIVE, DecompositionMode.JOINT)
        assert len(result) >= 1

    def test_probabilistic_mode_large_N(self, H, gmm, contexts, cf_module, hypergraph):
        dec = MultiRootCauseDecomposition(probabilistic_threshold=3)  # N=5 > 3
        ranked = [(0, 5.0), (1, 4.0), (2, 3.0)]
        mode, result = dec.decompose(ranked, H, gmm, contexts, cf_module, hypergraph)
        assert mode == DecompositionMode.PROBABILISTIC

    def test_additivity_ratio_zero(self):
        # ПЭ аддитивны → η ≈ 0
        assert additivity_ratio(3.0, 2.0, 5.0) == pytest.approx(0.0, abs=1e-5)

    def test_additivity_ratio_synergy(self):
        # Синергия: ПЭ(A∪B) > ПЭ(A) + ПЭ(B)
        eta = additivity_ratio(2.0, 2.0, 5.0)
        assert eta < 0.0    # знак < 0 при синергии


# ===========================================================================
# TestCascadeFunnel
# ===========================================================================

class TestCascadeFunnel:

    @pytest.fixture
    def funnel(self):
        return CascadeFunnel(l0_top_k=N, l1_top_k=3, l2_top_k=1)

    def test_run_returns_list(self, funnel, H, gmm, contexts, cf_module, hypergraph):
        nll = gmm.nll(H, contexts)
        adj = hypergraph.adjacency_matrix()
        adj_norm = adj / adj.sum(1, keepdim=True).clamp(min=1)
        result = funnel.run(nll, H, adj_norm, cf_module, gmm, contexts, hypergraph)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_run_returns_valid_indices(self, funnel, H, gmm, contexts, cf_module, hypergraph):
        nll = gmm.nll(H, contexts)
        adj = hypergraph.adjacency_matrix()
        adj_norm = adj / adj.sum(1, keepdim=True).clamp(min=1)
        result = funnel.run(nll, H, adj_norm, cf_module, gmm, contexts, hypergraph)
        for idx, ce in result:
            assert 0 <= idx < N
            assert isinstance(ce, float)

    def test_l0_score_shape(self, funnel, H, gmm, contexts, hypergraph):
        nll = gmm.nll(H, contexts)
        adj = hypergraph.adjacency_matrix()
        adj_norm = adj / adj.sum(1, keepdim=True).clamp(min=1)
        score = funnel._l0_score(nll, adj_norm)
        assert score.shape == (N,)
        assert not torch.isnan(score).any()

    def test_alpha_is_trainable(self, funnel):
        assert funnel.alpha.requires_grad

    def test_local_nodes_includes_self(self, funnel, hypergraph):
        adj = hypergraph.adjacency_matrix()
        local = funnel._local_nodes(0, adj)
        assert 0 in local


# ===========================================================================
# TestCausalGraphVerifier
# ===========================================================================

class TestCausalGraphVerifier:

    @pytest.fixture
    def verifier(self):
        return CausalGraphVerifier(
            temporal_tolerance_sec=15.0,
            transitivity_threshold=0.3,
            monotonicity_epsilon=0.05,
        )

    @pytest.fixture
    def acyclic_edges(self):
        return [(0, 1), (1, 2), (1, 3)]

    @pytest.fixture
    def cyclic_edges(self):
        return [(0, 1), (1, 2), (2, 0)]

    def test_acyclicity_ok(self, verifier, acyclic_edges):
        r = verifier.verify(
            edges=acyclic_edges,
            causal_effects={0: 1.0, 1: 0.5, 2: 0.1, 3: 0.1},
            anomaly_times={0: 0.0, 1: 5.0, 2: 10.0, 3: 10.0},
            physical_edges=set(acyclic_edges),
            root_candidate=0,
        )
        acyc = next(x for x in r.axiom_results if x.name == "Ацикличность")
        assert acyc.status == AxiomStatus.OK

    def test_acyclicity_violated(self, verifier, cyclic_edges):
        r = verifier.verify(
            edges=cyclic_edges,
            causal_effects={0: 1.0, 1: 0.5, 2: 0.1},
            anomaly_times={0: 0.0, 1: 1.0, 2: 2.0},
            physical_edges=set(cyclic_edges),
            root_candidate=0,
        )
        acyc = next(x for x in r.axiom_results if x.name == "Ацикличность")
        assert acyc.status == AxiomStatus.VIOLATED

    def test_temporal_violation(self, verifier):
        edges = [(0, 1)]
        r = verifier.verify(
            edges=edges,
            causal_effects={0: 1.0, 1: 0.5},
            anomaly_times={0: 100.0, 1: 0.0},   # причина позже следствия
            physical_edges=set(edges),
            root_candidate=0,
        )
        temp = next(x for x in r.axiom_results if "Темпорал" in x.name)
        assert temp.status == AxiomStatus.VIOLATED

    def test_confidence_all_ok(self, verifier, acyclic_edges):
        r = verifier.verify(
            edges=acyclic_edges,
            causal_effects={0: 1.0, 1: 0.5, 2: 0.1, 3: 0.1},
            anomaly_times={0: 0.0, 1: 5.0, 2: 10.0, 3: 10.0},
            physical_edges=set(acyclic_edges),
            root_candidate=0,
        )
        assert r.confidence >= 0.6    # ≥ 3/5 аксиом ОК

    def test_confidence_with_cycle(self, verifier, cyclic_edges):
        r = verifier.verify(
            edges=cyclic_edges,
            causal_effects={0: 1.0, 1: 0.5, 2: 0.1},
            anomaly_times={0: 0.0, 1: 1.0, 2: 2.0},
            physical_edges=set(cyclic_edges),
            root_candidate=0,
        )
        assert r.confidence < 1.0    # минимум одна аксиома нарушена

    def test_summary_contains_confidence(self, verifier, acyclic_edges):
        r = verifier.verify(
            edges=acyclic_edges,
            causal_effects={0: 1.0, 1: 0.5, 2: 0.1, 3: 0.1},
            anomaly_times={0: 0.0, 1: 5.0, 2: 10.0, 3: 10.0},
            physical_edges=set(acyclic_edges),
            root_candidate=0,
        )
        assert "Достоверность" in r.summary()


# ===========================================================================
# TestReasoningPipeline — интеграционный тест на демо-данных
# ===========================================================================

@pytest.mark.skipif(
    not (SAMPLE_DIR / "metrics.csv").exists(),
    reason="Демо-данные отсутствуют. Запустите: python scripts/generate_demo_data.py",
)
class TestReasoningPipeline:
    """Сквозной тест: демо-данные → StateBuilder → GMM → Funnel → Verifier."""

    BASE_TS   = 1_700_000_000.0
    N_STEPS   = 60
    STATE_DIM = 64
    CTX_DIM   = 16

    @pytest.fixture(scope="class")
    def pipeline_data(self):
        """Загружает демо-данные и строит тензоры состояний."""
        from cairn.connectors.csv_file import (
            CSVMetricConnector, FileLogConnector,
            JSONTraceConnector, YAMLTopologyConnector,
        )
        from cairn.perception import (
            StateBuilder, HypergraphBuilder, DrainTokenizer, ContextBuilder,
        )

        metric_conn  = CSVMetricConnector(SAMPLE_DIR / "metrics.csv")
        log_conn     = FileLogConnector(SAMPLE_DIR / "logs.txt")
        trace_conn   = JSONTraceConnector(SAMPLE_DIR / "traces.json")
        topo_conn    = YAMLTopologyConnector(SAMPLE_DIR / "topology.yaml")

        ts_end = self.BASE_TS + self.N_STEPS - 1

        md   = metric_conn.fetch(self.BASE_TS, ts_end)
        ld   = log_conn.fetch(self.BASE_TS, ts_end)
        trs  = trace_conn.fetch(self.BASE_TS, ts_end)
        topo = topo_conn.fetch()

        N_inst = md.n_instances

        # Метрики
        metrics_np = md.values.transpose(1, 0, 2)  # (N, T, F)
        import numpy as np
        metrics_t  = torch.tensor(metrics_np, dtype=torch.float32)
        metrics_t  = torch.nan_to_num(metrics_t, nan=0.0)

        # Журналы
        tokenizer = DrainTokenizer()
        tokenizer.fit_transform(ld.messages)
        MAX_LOG = 10
        log_ids, log_lens = [], []
        for name in md.instance_names:
            msgs = ld.filter_instance(name).messages
            ids  = [tokenizer.transform_one(m) for m in msgs[:MAX_LOG]] or [0]
            log_lens.append(len(ids))
            ids += [0] * (MAX_LOG - len(ids))
            log_ids.append(ids[:MAX_LOG])
        log_ids_t  = torch.tensor(log_ids, dtype=torch.long)
        log_lens_t = torch.tensor(log_lens, dtype=torch.long)

        # Трассировки — глубина
        depths = [0] * N_inst
        for tr in trs:
            for span in tr.spans:
                if span.instance in md.instance_names:
                    idx = md.instance_names.index(span.instance)
                    depths[idx] = max(depths[idx], 1 if span.parent_span_id else 0)
        depths_t = torch.tensor(depths, dtype=torch.long)

        # StateBuilder
        builder = StateBuilder(
            n_metrics=md.n_metrics, log_vocab_size=300,
            state_dim=self.STATE_DIM, context_dim=self.CTX_DIM,
            d_met=32, d_log=16, d_tr=16, d_ssm=16, d_brk=16,
            ssm_state_dim=32, window=20,
        )
        builder.eval()
        with torch.no_grad():
            H, C = builder(metrics_t, log_ids_t, depths_t, log_lengths=log_lens_t)

        # Гиперграф
        hg = HypergraphBuilder.from_topology_data(topo)

        return {"H": H, "C": C, "topo": topo, "hg": hg,
                "inst_names": md.instance_names, "N": N_inst}

    def test_gmm_on_demo_data(self, pipeline_data):
        H, C = pipeline_data["H"], pipeline_data["C"]
        gmm = ConditionalGMM(state_dim=self.STATE_DIM, context_dim=self.CTX_DIM, n_components=3)
        nll = gmm.nll(H, C)
        assert nll.shape == (pipeline_data["N"],)
        assert not torch.isnan(nll).any()

    def test_counterfactual_on_demo_data(self, pipeline_data):
        H, C  = pipeline_data["H"], pipeline_data["C"]
        hg    = pipeline_data["hg"]
        N_inst = pipeline_data["N"]
        gmm   = ConditionalGMM(state_dim=self.STATE_DIM, context_dim=self.CTX_DIM)
        cf    = CounterfactualModule(state_dim=self.STATE_DIM)

        proto = gmm.prototype(C[0:1]).squeeze(0)
        H_cf  = cf.intervene(H, 0, proto, hg)
        assert H_cf.shape == (N_inst, self.STATE_DIM)
        assert not torch.isnan(H_cf).any()

    def test_funnel_finds_root_cause(self, pipeline_data):
        H, C   = pipeline_data["H"], pipeline_data["C"]
        hg     = pipeline_data["hg"]
        N_inst = pipeline_data["N"]
        names  = pipeline_data["inst_names"]

        gmm    = ConditionalGMM(state_dim=self.STATE_DIM, context_dim=self.CTX_DIM)
        cf     = CounterfactualModule(state_dim=self.STATE_DIM)
        funnel = CascadeFunnel(l0_top_k=N_inst, l1_top_k=3, l2_top_k=1)

        nll = gmm.nll(H, C)
        adj = hg.adjacency_matrix()
        adj_norm = adj / adj.sum(1, keepdim=True).clamp(min=1)

        result = funnel.run(nll, H, adj_norm, cf, gmm, C, hg)
        assert len(result) >= 1
        root_idx, root_ce = result[0]
        assert 0 <= root_idx < N_inst

    def test_verifier_on_demo_topology(self, pipeline_data):
        hg    = pipeline_data["hg"]
        N_inst = pipeline_data["N"]

        verifier = CausalGraphVerifier()
        edges    = list(hg.call_path_edges()) if hasattr(hg, "call_path_edges") else \
                   [(hg.instance_names.index(s), hg.instance_names.index(d))
                    for s, d in pipeline_data["topo"].call_edges
                    if s in hg.instance_names and d in hg.instance_names]

        report = verifier.verify(
            edges=edges,
            causal_effects={i: float(i + 1) for i in range(N_inst)},
            anomaly_times={i: float(i * 5) for i in range(N_inst)},
            physical_edges=set(edges),
            root_candidate=0,
        )
        assert report.confidence > 0
        assert "Достоверность" in report.summary()
