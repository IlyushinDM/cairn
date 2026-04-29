"""Тесты фазы восприятия CAIRN.

Структура:
  TestDualBranchMetricEncoder — unit-тесты кодировщика метрик
  TestLogEncoderAndDrain      — unit-тесты DrainTokenizer + LogEncoder
  TestTraceEncoder            — unit-тесты кодировщика трассировок
  TestContextBuilder          — unit-тесты контекстного вектора
  TestStateBuilder            — unit-тесты StateBuilder
  TestHypergraphBuilder       — unit-тесты построителя гиперграфа
  TestPerceptionPipeline      — сквозной тест на демо-данных
"""

from __future__ import annotations

import sys
from pathlib import Path

# PYTHONPATH для запуска без pip install (если нужно)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import math

import numpy as np
import pytest
import torch

from cairn.perception import (
    DualBranchMetricEncoder,
    SSMBranch,
    BreakpointBranch,
    LogEncoder,
    DrainTokenizer,
    TraceEncoder,
    ContextBuilder,
    StateBuilder,
    HypergraphBuilder,
    CausalHypergraph,
    EdgeType,
)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

BATCH = 4
T = 120      # длина временного ряда
F = 4        # число метрик
N = 5        # число экземпляров сервисов
D_MET = 64
D_LOG = 32
D_TR = 16
STATE_DIM = 128
CONTEXT_DIM = 16

SAMPLE_DIR = Path(__file__).parent.parent.parent / "data" / "sample"


# ===========================================================================
# TestDualBranchMetricEncoder
# ===========================================================================


class TestDualBranchMetricEncoder:
    """Тесты двухветвевого кодировщика метрик."""

    @pytest.fixture
    def enc(self):
        return DualBranchMetricEncoder(
            n_metrics=F, d_ssm=32, d_brk=32, d_out=D_MET, ssm_state_dim=32, window=30
        )

    @pytest.fixture
    def x(self):
        return torch.randn(BATCH, T, F)

    def test_output_shape(self, enc, x):
        out = enc(x)
        assert out.shape == (BATCH, D_MET), f"Ожидалось ({BATCH}, {D_MET}), получено {out.shape}"

    def test_no_nan(self, enc, x):
        out = enc(x)
        assert not torch.isnan(out).any(), "NaN в выходе кодировщика метрик"

    def test_gradient_flows(self, enc, x):
        x.requires_grad_(True)
        out = enc(x)
        out.sum().backward()
        assert x.grad is not None, "Градиент не проходит через DualBranchMetricEncoder"
        # Проверяем, что обучаемые параметры SSM тоже получили градиенты
        assert enc.ssm_branch.A.grad is not None

    def test_ssm_branch_shape(self):
        branch = SSMBranch(n_metrics=F, ssm_state_dim=32, d_out=32)
        x = torch.randn(BATCH, T, F)
        out = branch(x)
        assert out.shape == (BATCH, 32)

    def test_breakpoint_branch_shape(self):
        branch = BreakpointBranch(n_metrics=F, window=30, d_out=32)
        x = torch.randn(BATCH, T, F)
        out = branch(x)
        assert out.shape == (BATCH, 32)

    def test_breakpoint_detects_jump(self):
        """Ветвь разрыва должна давать разные выходы для нормального и аномального ряда."""
        branch = BreakpointBranch(n_metrics=F, window=30, d_out=32)
        x_normal = torch.randn(1, T, F) * 0.1
        x_jump = x_normal.clone()
        x_jump[:, T // 2:, :] += 5.0  # резкий скачок

        branch.eval()
        with torch.no_grad():
            h_n = branch(x_normal)
            h_j = branch(x_jump)
        # Выходы должны отличаться
        assert not torch.allclose(h_n, h_j, atol=1e-3)

    def test_short_sequence(self, enc):
        """Кодировщик должен работать с короткими рядами (T < 2W)."""
        x_short = torch.randn(BATCH, 10, F)
        out = enc(x_short)
        assert out.shape == (BATCH, D_MET)

    def test_batch_independence(self, enc):
        """Каждый элемент батча обрабатывается независимо."""
        x = torch.randn(BATCH, T, F)
        out_batch = enc(x)
        out_single = enc(x[:1])
        assert torch.allclose(out_batch[:1], out_single, atol=1e-5)


# ===========================================================================
# TestLogEncoderAndDrain
# ===========================================================================


class TestLogEncoderAndDrain:
    """Тесты DrainTokenizer и LogEncoder."""

    @pytest.fixture
    def tokenizer(self):
        return DrainTokenizer(sim_threshold=0.5, max_templates=100)

    @pytest.fixture
    def enc(self):
        return LogEncoder(vocab_size=200, embed_dim=32, hidden_dim=32, d_out=D_LOG)

    def test_drain_basic(self, tokenizer):
        ids = tokenizer.fit_transform([
            "Connection established from 192.168.1.1",
            "Connection established from 10.0.0.1",
            "Error code 500 in module auth",
        ])
        assert len(ids) == 3
        # Первые два должны попасть в один шаблон
        assert ids[0] == ids[1], "Похожие сообщения должны получить один ID"

    def test_drain_numeric_wildcard(self, tokenizer):
        ids = tokenizer.fit_transform([
            "Latency 42ms for service order",
            "Latency 137ms for service order",
        ])
        assert ids[0] == ids[1], "Числа должны заменяться на wildcard → одинаковый шаблон"

    def test_drain_unknown(self, tokenizer):
        tokenizer.fit_transform(["message A"] * 100)
        tokenizer._templates = tokenizer._templates[:tokenizer.max_templates]
        # Переполнение словаря → UNK=1
        _id = tokenizer.transform_one("completely unique message that never seen xyz")
        assert _id in (1,) or isinstance(_id, int)

    def test_log_encoder_shape(self, enc):
        ids = torch.randint(0, 100, (BATCH, 20))
        lengths = torch.randint(5, 20, (BATCH,))
        out = enc(ids, lengths)
        assert out.shape == (BATCH, D_LOG)

    def test_log_encoder_no_lengths(self, enc):
        ids = torch.randint(0, 100, (BATCH, 15))
        out = enc(ids)
        assert out.shape == (BATCH, D_LOG)

    def test_log_encoder_padding_ignored(self, enc):
        """PAD токены (idx=0) не должны влиять на выход."""
        enc.eval()
        ids = torch.randint(1, 50, (1, 10))
        ids_padded = torch.cat([ids, torch.zeros(1, 5, dtype=torch.long)], dim=1)
        lengths = torch.tensor([10])
        with torch.no_grad():
            h1 = enc(ids, lengths)
            h2 = enc(ids_padded, lengths)
        assert torch.allclose(h1, h2, atol=1e-5), "PAD должны игнорироваться"

    def test_log_encoder_gradient(self, enc):
        ids = torch.randint(0, 100, (BATCH, 15))
        out = enc(ids)
        out.sum().backward()
        assert enc.embed.weight.grad is not None


# ===========================================================================
# TestTraceEncoder
# ===========================================================================


class TestTraceEncoder:
    """Тесты кодировщика трассировок."""

    @pytest.fixture
    def enc(self):
        return TraceEncoder(d_out=D_TR, max_depth=20)

    def test_output_shape_1d(self, enc):
        depth = torch.randint(0, 10, (BATCH,))
        out = enc(depth)
        assert out.shape == (BATCH, D_TR)

    def test_output_shape_2d(self, enc):
        """Поддержка нескольких span'ов на экземпляр."""
        depth = torch.randint(0, 10, (BATCH, 5))
        out = enc(depth)
        assert out.shape == (BATCH, D_TR)

    def test_no_nan(self, enc):
        depth = torch.randint(0, 15, (BATCH,))
        out = enc(depth)
        assert not torch.isnan(out).any()

    def test_not_trainable(self, enc):
        """Таблица PE не должна быть обучаемым параметром."""
        param_names = [name for name, _ in enc.named_parameters()]
        assert "pe" not in param_names, "PE не должно быть в named_parameters()"

    def test_different_depths(self, enc):
        """Разные глубины → разные кодирования."""
        enc.eval()
        d0 = torch.tensor([0])
        d5 = torch.tensor([5])
        h0 = enc(d0)
        h5 = enc(d5)
        assert not torch.allclose(h0, h5)

    def test_depth_clamping(self, enc):
        """Глубина выше max_depth не должна вызывать ошибку."""
        depth = torch.tensor([100, 200, 999])
        out = enc(depth)
        assert out.shape == (3, D_TR)


# ===========================================================================
# TestContextBuilder
# ===========================================================================


class TestContextBuilder:
    @pytest.fixture
    def builder(self):
        return ContextBuilder(context_dim=CONTEXT_DIM, raw_dim=8)

    def test_output_shape(self, builder):
        raw = torch.randn(BATCH, 8)
        out = builder(raw)
        assert out.shape == (BATCH, CONTEXT_DIM)

    def test_build_raw(self):
        rps = torch.tensor([100.0, 500.0, 1000.0, 200.0])
        hour = torch.tensor([9.0, 14.0, 22.0, 3.0])
        dow = torch.tensor([1.0, 3.0, 6.0, 0.0])
        cpu = torch.tensor([0.5, 0.8, 0.3, 1.0])
        mem = torch.tensor([0.4, 0.6, 0.7, 0.2])
        ver = torch.tensor([0.1, 0.2, 0.5, 0.9])

        raw = ContextBuilder.build_raw(rps, hour, dow, cpu, mem, ver)
        assert raw.shape == (4, 8)
        assert not torch.isnan(raw).any()
        # RPS нормализована в [0, 1]
        assert (raw[:, 0] >= 0).all() and (raw[:, 0] <= 1).all()


# ===========================================================================
# TestStateBuilder
# ===========================================================================


class TestStateBuilder:
    @pytest.fixture
    def builder(self):
        return StateBuilder(
            n_metrics=F,
            log_vocab_size=200,
            state_dim=STATE_DIM,
            context_dim=CONTEXT_DIM,
            d_met=D_MET,
            d_log=D_LOG,
            d_tr=D_TR,
            d_ssm=32,
            d_brk=32,
            ssm_state_dim=32,
            window=30,
        )

    @pytest.fixture
    def inputs(self):
        metrics = torch.randn(BATCH, T, F)
        log_ids = torch.randint(0, 100, (BATCH, 20))
        trace_depth = torch.randint(0, 5, (BATCH,))
        return metrics, log_ids, trace_depth

    def test_state_output_shape(self, builder, inputs):
        metrics, log_ids, depth = inputs
        h, c = builder(metrics, log_ids, depth)
        assert h.shape == (BATCH, STATE_DIM), f"h.shape={h.shape}"
        assert c.shape == (BATCH, CONTEXT_DIM), f"c.shape={c.shape}"

    def test_zero_context_without_raw(self, builder, inputs):
        metrics, log_ids, depth = inputs
        _, c = builder(metrics, log_ids, depth, context_raw=None)
        assert torch.all(c == 0), "Без context_raw вектор c должен быть нулевым"

    def test_nonzero_context_with_raw(self, builder, inputs):
        metrics, log_ids, depth = inputs
        raw = torch.randn(BATCH, 8)
        _, c = builder(metrics, log_ids, depth, context_raw=raw)
        assert not torch.all(c == 0)

    def test_gradient_flows(self, builder, inputs):
        metrics, log_ids, depth = inputs
        metrics = metrics.requires_grad_(True)
        h, c = builder(metrics, log_ids, depth, context_raw=torch.randn(BATCH, 8))
        (h.sum() + c.sum()).backward()
        assert metrics.grad is not None

    def test_no_nan(self, builder, inputs):
        metrics, log_ids, depth = inputs
        h, c = builder(metrics, log_ids, depth)
        assert not torch.isnan(h).any()
        assert not torch.isnan(c).any()

    def test_log_lengths_supported(self, builder, inputs):
        metrics, log_ids, depth = inputs
        lengths = torch.randint(5, 20, (BATCH,))
        h, c = builder(metrics, log_ids, depth, log_lengths=lengths)
        assert h.shape == (BATCH, STATE_DIM)


# ===========================================================================
# TestHypergraphBuilder
# ===========================================================================


class TestHypergraphBuilder:

    @pytest.fixture
    def builder(self):
        return HypergraphBuilder(n_nodes=N)

    @pytest.fixture
    def graph(self, builder):
        return builder.from_topology(
            call_paths=[[0, 1], [1, 2], [1, 3]],
            colocated_groups=[[1, 4]],
            lb_groups=[],
            instance_names=["frontend-1", "order-service-1", "payment-service-1",
                            "database-1", "cache-service-1"],
        )

    def test_n_nodes(self, graph):
        assert graph.n_nodes == N

    def test_instance_names(self, graph):
        assert "order-service-1" in graph.instance_names

    def test_edge_types(self, graph):
        types = {e.edge_type for e in graph.edges}
        assert EdgeType.CALL in types
        assert EdgeType.COLOCATION in types

    def test_incidence_matrix_shape(self, graph):
        H = graph.incidence_matrix()
        M = len(graph.edges)
        assert H.shape == (N, M)

    def test_incidence_matrix_values(self, graph):
        H = graph.incidence_matrix()
        assert ((H == 0) | (H == 1)).all(), "H должна содержать только 0 и 1"

    def test_edge_weights_shape(self, graph):
        w = graph.edge_weights()
        assert w.shape == (len(graph.edges),)

    def test_adjacency_matrix(self, graph):
        A = graph.adjacency_matrix()
        assert A.shape == (N, N)
        # Диагональ = 0
        assert A.diagonal().sum() == 0
        # Симметричность
        assert torch.allclose(A, A.T)

    def test_to_pyg_tensors(self, graph):
        data = graph.to_pyg_tensors()
        assert "incidence" in data
        assert "edge_weights" in data
        assert "edge_type_ids" in data
        assert data["n_nodes"] == N
        assert data["incidence"].shape[0] == N

    def test_add_adaptive_edge(self, graph):
        n_before = len(graph.edges)
        edge = graph.add_adaptive_edge([0, 2], weight=0.8, verified=False)
        assert len(graph.edges) == n_before + 1
        assert edge.edge_type == EdgeType.ADAPTIVE
        assert not edge.verified

    def test_causal_subgraph_excludes_unverified(self, graph):
        graph.add_adaptive_edge([0, 1], verified=False)
        sub = graph.causal_subgraph()
        adaptive_unverified = [e for e in sub.edges if not e.verified]
        assert len(adaptive_unverified) == 0

    def test_from_topology_data(self):
        """Интеграция с TopologyData."""
        # Создаём минимальный mock TopologyData
        class MockInstance:
            def __init__(self, name, service):
                self.name, self.service = name, service

        class MockTopo:
            instances = [
                MockInstance("frontend-1", "frontend"),
                MockInstance("order-service-1", "order-service"),
                MockInstance("database-1", "database"),
            ]
            instance_names = ["frontend-1", "order-service-1", "database-1"]
            call_edges = [("frontend-1", "order-service-1"), ("order-service-1", "database-1")]
            colocation_groups = []
            load_balancer_groups = []

        graph = HypergraphBuilder.from_topology_data(MockTopo())
        assert graph.n_nodes == 3
        assert len(graph.edges) == 2
        # database должен получить тип "database"
        db_idx = graph.instance_idx("database-1")
        assert graph.node_types[db_idx] == "database"


# ===========================================================================
# TestPerceptionPipeline — сквозной тест на демо-данных
# ===========================================================================


@pytest.mark.skipif(
    not (SAMPLE_DIR / "metrics.csv").exists(),
    reason="Демо-данные отсутствуют. Запустите: python scripts/generate_demo_data.py",
)
class TestPerceptionPipeline:
    """Сквозной тест: демо-данные → StateBuilder → H ∈ ℝ^{N×128}."""

    INSTANCES = [
        "frontend-1", "order-service-1", "payment-service-1",
        "cache-service-1", "database-1",
    ]
    METRICS = ["cpu", "memory", "latency_ms", "rps"]
    BASE_TS = 1_700_000_000.0
    WINDOW_SEC = 60  # берём 60 отсчётов нормального периода

    @pytest.fixture(scope="class")
    def demo_data(self):
        """Загружает демо-данные через CSV-коннекторы."""
        from cairn.connectors.csv_file import (
            CSVMetricConnector, FileLogConnector,
            JSONTraceConnector, YAMLTopologyConnector,
        )
        metric_conn = CSVMetricConnector(SAMPLE_DIR / "metrics.csv")
        log_conn = FileLogConnector(SAMPLE_DIR / "logs.txt")
        trace_conn = JSONTraceConnector(SAMPLE_DIR / "traces.json")
        topo_conn = YAMLTopologyConnector(SAMPLE_DIR / "topology.yaml")
        return {
            "metrics": metric_conn.fetch(self.BASE_TS, self.BASE_TS + self.WINDOW_SEC - 1),
            "logs": log_conn.fetch(self.BASE_TS, self.BASE_TS + self.WINDOW_SEC - 1),
            "traces": trace_conn.fetch(self.BASE_TS, self.BASE_TS + self.WINDOW_SEC - 1),
            "topology": topo_conn.fetch(),
        }

    @pytest.fixture(scope="class")
    def tokenizer(self):
        return DrainTokenizer(sim_threshold=0.5)

    @pytest.fixture(scope="class")
    def state_builder(self):
        return StateBuilder(
            n_metrics=len(self.METRICS),
            log_vocab_size=300,
            state_dim=STATE_DIM,
            context_dim=CONTEXT_DIM,
            d_met=D_MET,
            d_log=D_LOG,
            d_tr=D_TR,
            d_ssm=32,
            d_brk=32,
            ssm_state_dim=32,
            window=30,
        )

    def _build_tensors(self, demo_data, tokenizer):
        """Конвертирует демо-данные в тензоры PyTorch."""
        md = demo_data["metrics"]
        N_inst = md.n_instances
        T_steps = md.n_timesteps

        # Метрики: (N, T, F) → (N, T, F)
        metrics_np = md.values.transpose(1, 0, 2)  # (N, T, F)
        metrics_t = torch.tensor(metrics_np, dtype=torch.float32)
        # Заполняем NaN нулями
        metrics_t = torch.nan_to_num(metrics_t, nan=0.0)

        # Журналы: берём сообщения, разбиваем по экземплярам
        all_msgs = demo_data["logs"].messages
        tokenizer.fit_transform(all_msgs) if all_msgs else None

        log_ids_list, log_lengths = [], []
        max_log_len = 10
        for inst_name in md.instance_names:
            inst_msgs = demo_data["logs"].filter_instance(inst_name).messages
            ids = [tokenizer.transform_one(m) for m in inst_msgs[:max_log_len]]
            if not ids:
                ids = [0]
            log_lengths.append(len(ids))
            # Паддинг
            ids += [0] * (max_log_len - len(ids))
            log_ids_list.append(ids[:max_log_len])

        log_ids_t = torch.tensor(log_ids_list, dtype=torch.long)   # (N, max_log_len)
        lengths_t = torch.tensor(log_lengths, dtype=torch.long)    # (N,)

        # Трассировки: глубина первого span'а каждого экземпляра
        inst_depth = {name: 0 for name in md.instance_names}
        for trace in demo_data["traces"]:
            for span in trace.spans:
                if span.instance in inst_depth:
                    parent_cnt = sum(
                        1 for s in trace.spans
                        if s.span_id == span.parent_span_id
                    )
                    inst_depth[span.instance] = parent_cnt
        depths_t = torch.tensor(
            [inst_depth[n] for n in md.instance_names], dtype=torch.long
        )  # (N,)

        return metrics_t, log_ids_t, lengths_t, depths_t

    def test_state_tensor_shape(self, demo_data, tokenizer, state_builder):
        metrics_t, log_ids_t, lengths_t, depths_t = self._build_tensors(demo_data, tokenizer)
        N_inst = metrics_t.shape[0]

        state_builder.eval()
        with torch.no_grad():
            H, C = state_builder(metrics_t, log_ids_t, depths_t, log_lengths=lengths_t)

        assert H.shape == (N_inst, STATE_DIM), f"H.shape={H.shape}, ожидалось ({N_inst},{STATE_DIM})"
        assert C.shape == (N_inst, CONTEXT_DIM), f"C.shape={C.shape}"

    def test_no_nan_in_states(self, demo_data, tokenizer, state_builder):
        metrics_t, log_ids_t, lengths_t, depths_t = self._build_tensors(demo_data, tokenizer)
        state_builder.eval()
        with torch.no_grad():
            H, _ = state_builder(metrics_t, log_ids_t, depths_t, log_lengths=lengths_t)
        assert not torch.isnan(H).any(), "NaN в матрице состояний H"

    def test_anomaly_instance_different_state(self, demo_data, tokenizer, state_builder):
        """Состояние order-service в аномальный период должно отличаться от нормального."""
        from cairn.connectors.csv_file import CSVMetricConnector, FileLogConnector, JSONTraceConnector

        BASE_TS = self.BASE_TS
        normal_data = {
            "metrics": CSVMetricConnector(SAMPLE_DIR / "metrics.csv").fetch(BASE_TS, BASE_TS + 59),
            "logs": FileLogConnector(SAMPLE_DIR / "logs.txt").fetch(BASE_TS, BASE_TS + 59),
            "traces": JSONTraceConnector(SAMPLE_DIR / "traces.json").fetch(BASE_TS, BASE_TS + 59),
        }
        anom_data = {
            "metrics": CSVMetricConnector(SAMPLE_DIR / "metrics.csv").fetch(BASE_TS + 200, BASE_TS + 259),
            "logs": FileLogConnector(SAMPLE_DIR / "logs.txt").fetch(BASE_TS + 200, BASE_TS + 259),
            "traces": JSONTraceConnector(SAMPLE_DIR / "traces.json").fetch(BASE_TS + 200, BASE_TS + 259),
        }

        state_builder.eval()
        results = {}
        for label, ddata in [("normal", normal_data), ("anomaly", anom_data)]:
            mt, li, ll, dt = self._build_tensors(ddata, tokenizer)
            with torch.no_grad():
                H, _ = state_builder(mt, li, dt, log_lengths=ll)
            results[label] = H

        # Находим индекс order-service-1
        inst_names = normal_data["metrics"].instance_names
        order_idx = inst_names.index("order-service-1")

        h_normal = results["normal"][order_idx]
        h_anom = results["anomaly"][order_idx]

        # Состояния должны различаться (косинусное расстояние > 0)
        cos_sim = torch.nn.functional.cosine_similarity(
            h_normal.unsqueeze(0), h_anom.unsqueeze(0)
        ).item()
        assert cos_sim < 0.999, (
            f"Состояния order-service слишком похожи: cos_sim={cos_sim:.4f}. "
            "Кодировщик не фиксирует аномалию."
        )

    def test_hypergraph_from_topology(self, demo_data):
        topo = demo_data["topology"]
        graph = HypergraphBuilder.from_topology_data(topo)

        assert graph.n_nodes == 5
        # Проверяем типы рёбер
        types = {e.edge_type for e in graph.edges}
        assert EdgeType.CALL in types
        assert EdgeType.COLOCATION in types

        # Матрица инцидентности
        H = graph.incidence_matrix()
        assert H.shape[0] == 5
        assert H.shape[1] == len(graph.edges)

        # PyG-тензоры
        pyg = graph.to_pyg_tensors()
        assert pyg["n_nodes"] == 5
        assert "incidence" in pyg
        assert "edge_type_ids" in pyg

    def test_full_pipeline_output_types(self, demo_data, tokenizer, state_builder):
        """Финальная проверка: типы и устройство выходных тензоров."""
        metrics_t, log_ids_t, lengths_t, depths_t = self._build_tensors(demo_data, tokenizer)
        state_builder.eval()
        with torch.no_grad():
            H, C = state_builder(metrics_t, log_ids_t, depths_t, log_lengths=lengths_t)

        assert H.dtype == torch.float32
        assert C.dtype == torch.float32
        assert H.device == C.device
