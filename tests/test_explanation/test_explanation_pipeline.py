"""Тесты фазы объяснения CAIRN.

Классы:
  TestEvidenceChain        — unit-тесты EvidenceChain и EvidenceChainBuilder
  TestTextGenerator        — unit-тесты TemplateTextGenerator
  TestALPVerifier          — unit-тесты верификатора (IC1–IC5)
  TestMediationDiagnostic  — unit-тесты медиационной диагностики
  TestExplanationPipeline  — сквозной тест на демо-данных
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
import torch

from cairn.explanation import (
    EvidenceChain, NodeAnnotation, EdgeAnnotation,
    EvidenceChainBuilder,
    TemplateTextGenerator, TextExplanationGenerator,
    ALPVerifier, ALPVerificationResult,
    MediationDiagnostic, MediationReport,
)
from cairn.perception import HypergraphBuilder
from cairn.reasoning import ConditionalGMM, CounterfactualModule

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
N   = 5
D   = 64
CTX = 16
SAMPLE_DIR = Path(__file__).parent.parent.parent / "data" / "sample"


# ---------------------------------------------------------------------------
# Общие фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture
def hypergraph():
    builder = HypergraphBuilder(N)
    return builder.from_topology(
        call_paths=[[0, 1], [1, 2], [1, 3]],
        colocated_groups=[[1, 4]],
        lb_groups=[],
        instance_names=["frontend-1", "order-service-1", "payment-service-1",
                        "database-1", "cache-service-1"],
    )

@pytest.fixture
def simple_chain():
    """Простая цепочка с 3 узлами для unit-тестов."""
    nodes = [
        NodeAnnotation(1, "order-service-1", nll=8.0, causal_effect=4.2, failure_type="cpu_exhaustion"),
        NodeAnnotation(0, "frontend-1",      nll=2.5, causal_effect=0.3, failure_type="latency_spike"),
        NodeAnnotation(2, "payment-service-1", nll=3.1, causal_effect=0.8),
    ]
    edges = [
        EdgeAnnotation(src=1, dst=0, edge_type="call",      strength=0.85),
        EdgeAnnotation(src=1, dst=2, edge_type="call",      strength=0.72),
    ]
    return EvidenceChain(
        root_cause_idx=1,
        path_nodes=nodes,
        path_edges=edges,
        causal_effect=4.2,
        confidence=0.8,
    )

@pytest.fixture
def gmm():
    return ConditionalGMM(state_dim=D, context_dim=CTX, n_components=3)

@pytest.fixture
def cf_module():
    return CounterfactualModule(state_dim=D, n_conv_layers=1)

@pytest.fixture
def H():
    return torch.randn(N, D)

@pytest.fixture
def contexts():
    return torch.randn(N, CTX)


# ===========================================================================
# TestEvidenceChain
# ===========================================================================

class TestEvidenceChain:

    def test_to_dict_structure(self, simple_chain):
        d = simple_chain.to_dict()
        assert "root_cause" in d
        assert "path" in d
        assert "edges" in d
        assert "confidence" in d
        assert "warnings" in d

    def test_to_dict_path_length(self, simple_chain):
        d = simple_chain.to_dict()
        assert len(d["path"]) == 3

    def test_to_dict_edge_fields(self, simple_chain):
        d = simple_chain.to_dict()
        e = d["edges"][0]
        assert "src" in e and "dst" in e
        assert "type" in e and "strength" in e
        assert "has_confounder" in e

    def test_summary_contains_root(self, simple_chain):
        s = simple_chain.summary()
        assert "order-service-1" in s

    def test_summary_contains_path(self, simple_chain):
        s = simple_chain.summary()
        assert "→" in s or "->" in s

    def test_confounder_warning_in_summary(self):
        chain = EvidenceChain(
            root_cause_idx=0,
            path_nodes=[NodeAnnotation(0, "svc", nll=5.0)],
            confounder_warnings=["Обнаружен скрытый фактор у 'svc'."],
        )
        assert "скрытый" in chain.summary()

    def test_drift_warning_in_summary(self):
        chain = EvidenceChain(
            root_cause_idx=0,
            path_nodes=[NodeAnnotation(0, "svc", nll=5.0)],
            drift_warning=True,
        )
        assert "дрейф" in chain.summary().lower()

    def test_to_dict_node_has_causal_effect(self, simple_chain):
        d = simple_chain.to_dict()
        assert "causal_effect" in d["path"][0]

    def test_root_name_in_to_dict(self, simple_chain):
        d = simple_chain.to_dict()
        assert d["root_name"] == "order-service-1"   # path_nodes[0]


# ===========================================================================
# TestEvidenceChainBuilder
# ===========================================================================

class TestEvidenceChainBuilder:

    @pytest.fixture
    def builder(self):
        return EvidenceChainBuilder(confounder_threshold=0.3, max_path_depth=4)

    @pytest.fixture
    def nll_scores(self):
        # order-service-1 (idx=1) аномальный
        return {0: 3.0, 1: 9.5, 2: 4.2, 3: 3.8, 4: 2.1}

    @pytest.fixture
    def ce_scores(self):
        return {0: 0.5, 1: 6.3, 2: 1.2, 3: 0.9, 4: 0.3}

    def test_build_returns_evidence_chain(self, builder, hypergraph, nll_scores, ce_scores):
        chain = builder.build(
            root_cause=1,
            causal_graph=hypergraph,
            ce_scores=ce_scores,
            nll_scores=nll_scores,
            anomaly_threshold=2.0,
        )
        assert isinstance(chain, EvidenceChain)

    def test_root_cause_is_first_node(self, builder, hypergraph, nll_scores, ce_scores):
        chain = builder.build(
            root_cause=1,
            causal_graph=hypergraph,
            ce_scores=ce_scores,
            nll_scores=nll_scores,
            anomaly_threshold=2.0,
        )
        assert chain.path_nodes[0].node_idx == 1

    def test_root_name_from_hypergraph(self, builder, hypergraph, nll_scores, ce_scores):
        chain = builder.build(
            root_cause=1,
            causal_graph=hypergraph,
            ce_scores=ce_scores,
            nll_scores=nll_scores,
            anomaly_threshold=2.0,
        )
        assert chain.path_nodes[0].node_name == "order-service-1"

    def test_causal_effect_assigned(self, builder, hypergraph, nll_scores, ce_scores):
        chain = builder.build(
            root_cause=1,
            causal_graph=hypergraph,
            ce_scores=ce_scores,
            nll_scores=nll_scores,
        )
        assert chain.causal_effect == ce_scores[1]

    def test_confounder_warning_added(self, builder, hypergraph, nll_scores, ce_scores):
        chain = builder.build(
            root_cause=1,
            causal_graph=hypergraph,
            ce_scores=ce_scores,
            nll_scores=nll_scores,
            confounder_flags={1: True},
        )
        assert len(chain.confounder_warnings) > 0

    def test_drift_warning_propagated(self, builder, hypergraph, nll_scores, ce_scores):
        chain = builder.build(
            root_cause=1,
            causal_graph=hypergraph,
            ce_scores=ce_scores,
            nll_scores=nll_scores,
            drift_warning=True,
        )
        assert chain.drift_warning

    def test_metadata_failure_type(self, builder, hypergraph, nll_scores, ce_scores):
        chain = builder.build(
            root_cause=1,
            causal_graph=hypergraph,
            ce_scores=ce_scores,
            nll_scores=nll_scores,
            metadata={1: {"failure_type": "cpu_exhaustion", "dominant_metric": "cpu"}},
        )
        root_node = chain.path_nodes[0]
        assert root_node.failure_type == "cpu_exhaustion"
        assert root_node.dominant_metric == "cpu"

    def test_to_text_delegates_to_template(self, builder, hypergraph, nll_scores, ce_scores):
        chain = builder.build(
            root_cause=1,
            causal_graph=hypergraph,
            ce_scores=ce_scores,
            nll_scores=nll_scores,
        )
        text = builder.to_text(chain)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_to_dict_delegates(self, builder, hypergraph, nll_scores, ce_scores):
        chain = builder.build(1, hypergraph, ce_scores, nll_scores)
        d = builder.to_dict(chain)
        assert isinstance(d, dict)
        assert "root_cause" in d


# ===========================================================================
# TestTextGenerator
# ===========================================================================

class TestTextGenerator:

    @pytest.fixture
    def gen(self):
        return TemplateTextGenerator()

    def test_contains_root_name(self, gen, simple_chain):
        text = gen.generate(simple_chain)
        assert "order-service-1" in text

    def test_contains_fault_type(self, gen, simple_chain):
        text = gen.generate(simple_chain)
        assert "cpu_exhaustion" in text

    def test_contains_recommendation(self, gen, simple_chain):
        text = gen.generate(simple_chain)
        assert "Рекомендация:" in text

    def test_contains_confidence(self, gen, simple_chain):
        text = gen.generate(simple_chain)
        # confidence = 0.8 → "80%"
        assert "80%" in text

    def test_contains_ce_value(self, gen, simple_chain):
        text = gen.generate(simple_chain)
        # CE = 4.2 → "4.20"
        assert "4.20" in text

    def test_confounder_warning_in_text(self, gen):
        chain = EvidenceChain(
            root_cause_idx=0,
            path_nodes=[NodeAnnotation(0, "svc", nll=5.0, failure_type="cpu_exhaustion")],
            causal_effect=2.0,
            confidence=0.9,
            confounder_warnings=["Конфаундер обнаружен."],
        )
        text = gen.generate(chain)
        assert "⚠" in text

    def test_empty_chain(self, gen):
        chain = EvidenceChain(root_cause_idx=42)
        text = gen.generate(chain)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_facade_template_level(self, simple_chain):
        gen = TextExplanationGenerator(level="template")
        text = gen.generate(simple_chain)
        assert "order-service-1" in text

    def test_path_format_with_edge_type(self, gen, simple_chain):
        text = gen.generate(simple_chain)
        # Путь должен содержать имена узлов
        assert "frontend-1" in text or "order-service-1" in text


# ===========================================================================
# TestALPVerifier
# ===========================================================================

class TestALPVerifier:

    @pytest.fixture
    def verifier(self):
        return ALPVerifier(anomaly_threshold=2.0, ce_threshold=0.5)

    def test_passes_all_rules(self, verifier, simple_chain):
        gen  = TemplateTextGenerator()
        text = gen.generate(simple_chain)
        result = verifier.verify(simple_chain, text)
        assert result.passed
        assert len(result.violated_rules) == 0

    def test_ic1_low_nll(self, verifier):
        chain = EvidenceChain(
            root_cause_idx=0,
            path_nodes=[NodeAnnotation(0, "svc", nll=0.1, failure_type="unknown")],
            causal_effect=2.0,
            confidence=0.9,
        )
        gen  = TemplateTextGenerator()
        text = gen.generate(chain)
        result = verifier.verify(chain, text)
        assert not result.passed
        assert any("IC1" in r for r in result.violated_rules)

    def test_ic2_low_ce(self, verifier):
        chain = EvidenceChain(
            root_cause_idx=0,
            path_nodes=[NodeAnnotation(0, "svc", nll=5.0, failure_type="cpu_exhaustion")],
            causal_effect=0.01,   # ниже порога 0.5
            confidence=0.9,
        )
        gen  = TemplateTextGenerator()
        text = gen.generate(chain)
        result = verifier.verify(chain, text)
        assert not result.passed
        assert any("IC2" in r for r in result.violated_rules)

    def test_ic3_empty_chain(self, verifier):
        chain = EvidenceChain(root_cause_idx=0, causal_effect=2.0)
        result = verifier.verify(chain, "some text")
        assert not result.passed
        assert any("IC3" in r or "IC1" in r for r in result.violated_rules)

    def test_ic4_name_absent(self, verifier):
        chain = EvidenceChain(
            root_cause_idx=0,
            path_nodes=[NodeAnnotation(0, "order-service-1", nll=5.0, failure_type="cpu_exhaustion")],
            causal_effect=2.0,
            confidence=0.9,
        )
        text = "Проблема обнаружена в payment-service."  # не упоминает order-service-1
        result = verifier.verify(chain, text)
        assert not result.passed
        assert any("IC4" in r for r in result.violated_rules)

    def test_f1_confidence_reduced_on_violation(self, verifier):
        chain = EvidenceChain(
            root_cause_idx=0,
            path_nodes=[NodeAnnotation(0, "svc", nll=0.1)],  # IC1 сработает
            causal_effect=2.0,
            confidence=1.0,
        )
        result = verifier.verify(chain, "svc")
        assert result.f1_confidence < 1.0

    def test_counter_hypothesis_low_confidence(self, verifier):
        chain = EvidenceChain(
            root_cause_idx=0,
            path_nodes=[
                NodeAnnotation(0, "root-svc", nll=3.0, failure_type="cpu_exhaustion"),
                NodeAnnotation(1, "alt-svc",  nll=9.0),  # аномальнее root
            ],
            causal_effect=2.0,
            confidence=0.5,
        )
        gen  = TemplateTextGenerator()
        text = gen.generate(chain)
        result = verifier.verify(chain, text)
        assert result.counter_hypothesis is not None
        assert "alt-svc" in result.counter_hypothesis

    def test_warnings_separate_from_violations(self, verifier, simple_chain):
        gen  = TemplateTextGenerator()
        text = gen.generate(simple_chain)
        result = verifier.verify(simple_chain, text)
        # Предупреждения IC5 не должны делать тест failed
        assert isinstance(result.warnings, list)


# ===========================================================================
# TestMediationDiagnostic
# ===========================================================================

class TestMediationDiagnostic:

    @pytest.fixture
    def diag(self, gmm, cf_module):
        return MediationDiagnostic(cf_module, gmm)

    def test_diagnose_returns_report(self, diag, H, contexts, hypergraph):
        report = diag.diagnose(H, hypergraph, contexts, root_cause_idx=1)
        assert isinstance(report, MediationReport)

    def test_report_has_ce_full(self, diag, H, contexts, hypergraph):
        report = diag.diagnose(H, hypergraph, contexts, root_cause_idx=1)
        assert isinstance(report.ce_full, float)

    def test_layer_contributions_count(self, diag, H, contexts, hypergraph):
        report = diag.diagnose(H, hypergraph, contexts, root_cause_idx=1)
        # cf_module имеет 1 слой → 1 вклад
        assert len(report.layer_contributions) == 1

    def test_layer_contribution_fields(self, diag, H, contexts, hypergraph):
        report = diag.diagnose(H, hypergraph, contexts, root_cause_idx=1)
        lc = report.layer_contributions[0]
        assert hasattr(lc, "layer_idx")
        assert hasattr(lc, "contribution")
        assert hasattr(lc, "relative")

    def test_edge_contributions_count(self, diag, H, contexts, hypergraph):
        report = diag.diagnose(H, hypergraph, contexts, root_cause_idx=1)
        n_edges = len(hypergraph.edges)
        assert len(report.edge_contributions) == n_edges

    def test_edge_contribution_fields(self, diag, H, contexts, hypergraph):
        report = diag.diagnose(H, hypergraph, contexts, root_cause_idx=1)
        ec = report.edge_contributions[0]
        assert hasattr(ec, "edge_type")
        assert hasattr(ec, "members")
        assert hasattr(ec, "contribution")

    def test_top_edges_at_most_3(self, diag, H, contexts, hypergraph):
        report = diag.diagnose(H, hypergraph, contexts, root_cause_idx=1)
        assert len(report.top_edges) <= 3

    def test_summary_contains_ce(self, diag, H, contexts, hypergraph):
        report = diag.diagnose(H, hypergraph, contexts, root_cause_idx=1)
        s = report.summary()
        assert "CE" in s
        assert str(report.root_cause_idx) in s

    def test_diagnose_layers_standalone(self, diag, H, contexts, hypergraph):
        proto = diag.gmm.prototype(contexts[0:1]).squeeze(0)
        contribs = diag.diagnose_layers(H, hypergraph, contexts, root_cause_idx=0, proto=proto)
        assert len(contribs) >= 1

    def test_diagnose_edges_standalone(self, diag, H, contexts, hypergraph):
        proto = diag.gmm.prototype(contexts[0:1]).squeeze(0)
        contribs = diag.diagnose_edges(H, hypergraph, contexts, root_cause_idx=0, proto=proto)
        assert len(contribs) == len(hypergraph.edges)


# ===========================================================================
# TestExplanationPipeline — сквозной тест на демо-данных
# ===========================================================================

@pytest.mark.skipif(
    not (SAMPLE_DIR / "metrics.csv").exists(),
    reason="Демо-данные отсутствуют. Запустите: python scripts/generate_demo_data.py",
)
class TestExplanationPipeline:
    """Сквозной тест: StateBuilder → GMM → Funnel → EvidenceChainBuilder → ALPVerifier."""

    BASE_TS   = 1_700_000_000.0
    N_STEPS   = 60
    STATE_DIM = 64
    CTX_DIM   = 16

    @pytest.fixture(scope="class")
    def pipeline_result(self):
        """Запускает фазы восприятия и рассуждения, возвращает результаты."""
        from cairn.connectors.csv_file import (
            CSVMetricConnector, FileLogConnector,
            JSONTraceConnector, YAMLTopologyConnector,
        )
        from cairn.perception import (
            StateBuilder, HypergraphBuilder, DrainTokenizer,
        )
        from cairn.reasoning import (
            ConditionalGMM, CounterfactualModule, CascadeFunnel,
        )

        # -- Загрузка данных --
        ts_end = self.BASE_TS + self.N_STEPS - 1
        md   = CSVMetricConnector(SAMPLE_DIR / "metrics.csv").fetch(self.BASE_TS, ts_end)
        ld   = FileLogConnector(SAMPLE_DIR / "logs.txt").fetch(self.BASE_TS, ts_end)
        trs  = JSONTraceConnector(SAMPLE_DIR / "traces.json").fetch(self.BASE_TS, ts_end)
        topo = YAMLTopologyConnector(SAMPLE_DIR / "topology.yaml").fetch()

        N_inst = md.n_instances

        import numpy as np
        metrics_t = torch.nan_to_num(
            torch.tensor(md.values.transpose(1, 0, 2), dtype=torch.float32)
        )

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
        depths_t   = torch.zeros(N_inst, dtype=torch.long)

        builder = StateBuilder(
            n_metrics=md.n_metrics, log_vocab_size=300,
            state_dim=self.STATE_DIM, context_dim=self.CTX_DIM,
            d_met=32, d_log=16, d_tr=16, d_ssm=16, d_brk=16,
            ssm_state_dim=32, window=20,
        )
        builder.eval()
        with torch.no_grad():
            H, C = builder(metrics_t, log_ids_t, depths_t, log_lengths=log_lens_t)

        hg  = HypergraphBuilder.from_topology_data(topo)
        gmm = ConditionalGMM(state_dim=self.STATE_DIM, context_dim=self.CTX_DIM)
        cf  = CounterfactualModule(state_dim=self.STATE_DIM)
        fn  = CascadeFunnel(l0_top_k=N_inst, l1_top_k=3, l2_top_k=1)

        nll = gmm.nll(H, C)
        adj = hg.adjacency_matrix()
        adj_norm = adj / adj.sum(1, keepdim=True).clamp(min=1)
        ranked = fn.run(nll, H, adj_norm, cf, gmm, C, hg)

        root_idx, root_ce = ranked[0]
        nll_scores = {i: nll[i].item() for i in range(N_inst)}
        ce_scores  = dict(ranked)

        return {
            "H": H, "C": C, "hg": hg, "gmm": gmm, "cf": cf,
            "nll": nll, "nll_scores": nll_scores,
            "ce_scores": ce_scores, "root_idx": root_idx,
            "inst_names": md.instance_names, "N": N_inst,
        }

    def test_evidence_chain_built(self, pipeline_result):
        pr = pipeline_result
        chain_builder = EvidenceChainBuilder()
        chain = chain_builder.build(
            root_cause=pr["root_idx"],
            causal_graph=pr["hg"],
            ce_scores=pr["ce_scores"],
            nll_scores=pr["nll_scores"],
            anomaly_threshold=0.0,
        )
        assert isinstance(chain, EvidenceChain)
        assert chain.root_cause_idx == pr["root_idx"]
        assert len(chain.path_nodes) >= 1

    def test_text_generated(self, pipeline_result):
        pr = pipeline_result
        chain_builder = EvidenceChainBuilder()
        chain = chain_builder.build(
            pr["root_idx"], pr["hg"], pr["ce_scores"], pr["nll_scores"],
            anomaly_threshold=0.0,
            metadata={pr["root_idx"]: {"failure_type": "cpu_exhaustion", "dominant_metric": "cpu"}},
        )
        gen  = TemplateTextGenerator()
        text = gen.generate(chain)
        assert isinstance(text, str) and len(text) > 20
        assert "cpu_exhaustion" in text

    def test_alp_verification_runs(self, pipeline_result):
        pr = pipeline_result
        chain_builder = EvidenceChainBuilder()
        chain = chain_builder.build(
            pr["root_idx"], pr["hg"], pr["ce_scores"], pr["nll_scores"],
            anomaly_threshold=0.0,
            metadata={pr["root_idx"]: {"failure_type": "cpu_exhaustion"}},
        )
        gen  = TemplateTextGenerator()
        text = gen.generate(chain)
        verifier = ALPVerifier(anomaly_threshold=0.0, ce_threshold=0.0)
        result = verifier.verify(chain, text)
        assert isinstance(result, ALPVerificationResult)
        assert isinstance(result.f1_confidence, float)

    def test_mediation_runs(self, pipeline_result):
        pr  = pipeline_result
        diag = MediationDiagnostic(pr["cf"], pr["gmm"])
        report = diag.diagnose(pr["H"], pr["hg"], pr["C"], pr["root_idx"])
        assert isinstance(report, MediationReport)
        assert len(report.layer_contributions) >= 1
        assert len(report.edge_contributions) >= 1

    def test_full_report_to_dict(self, pipeline_result):
        pr = pipeline_result
        chain_builder = EvidenceChainBuilder()
        chain = chain_builder.build(
            pr["root_idx"], pr["hg"], pr["ce_scores"], pr["nll_scores"],
            anomaly_threshold=0.0,
        )
        d = chain.to_dict()
        assert d["root_cause"] == pr["root_idx"]
        assert isinstance(d["path"], list)
        assert isinstance(d["edges"], list)
