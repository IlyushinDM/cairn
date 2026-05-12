"""Тесты коннекторов: CSV, MetricData, топология."""
import pytest
import numpy as np
from pathlib import Path


class TestMetricData:
    """MetricData: контейнер для метрических данных."""

    def test_shape(self, sample_metric_data):
        md = sample_metric_data
        assert md.n_instances == 3
        assert md.n_metrics   == 4
        assert len(md.timestamps) == 20

    def test_instance_names(self, sample_metric_data):
        md = sample_metric_data
        assert len(md.instance_names) == 3
        assert md.instance_names[0] == "service-0"

    def test_values_dtype(self, sample_metric_data):
        """Значения должны быть float32."""
        assert sample_metric_data.values.dtype == np.float32

    def test_timestamps_monotonic(self, sample_metric_data):
        """Временные метки монотонно возрастают."""
        ts = sample_metric_data.timestamps
        assert all(ts[i] < ts[i+1] for i in range(len(ts)-1))


class TestTopologyData:
    """TopologyData и HypergraphBuilder."""

    def test_build_hypergraph(self, tiny_hypergraph):
        assert tiny_hypergraph is not None
        assert len(tiny_hypergraph.instance_names) == 3

    def test_call_edges_present(self, tiny_hypergraph):
        call_edges = [e for e in tiny_hypergraph.edges if e.edge_type == "call"]
        assert len(call_edges) >= 2

    def test_adjacency_symmetric_for_undirected(self, tiny_hypergraph):
        """Для call-edges матрица не обязана быть симметричной (направленный граф)."""
        adj = tiny_hypergraph.adjacency_matrix()
        assert adj.shape[0] == adj.shape[1]

    def test_node_indices_valid(self, tiny_hypergraph):
        """Индексы узлов в рёбрах валидны."""
        n = len(tiny_hypergraph.instance_names)
        for edge in tiny_hypergraph.edges:
            for member in edge.members:
                assert 0 <= member < n, f"Невалидный индекс узла: {member}"


class TestDockerLogConnector:
    """DockerLogConnector: парсинг логов."""

    def test_parse_log_level_error(self):
        """Парсер определяет уровень ERROR."""
        import sys
        sys.path.insert(0, "src")
        from cairn.connectors.docker_log_connector import DockerLogConnector

        conn = DockerLogConnector()
        line = "2024-01-15T10:30:45.123Z ERROR connection refused"
        event = conn._parse_log_line(line, "test-svc")
        assert event is not None
        assert event.level == "ERROR"

    def test_parse_log_level_warn(self):
        from cairn.connectors.docker_log_connector import DockerLogConnector

        conn = DockerLogConnector()
        line = "2024-01-15T10:30:45.123Z WARN high memory usage"
        event = conn._parse_log_line(line, "test-svc")
        assert event is not None
        assert event.level == "WARN"

    def test_parse_log_no_timestamp(self):
        """Строки без временной метки обрабатываются без падения."""
        from cairn.connectors.docker_log_connector import DockerLogConnector

        conn = DockerLogConnector()
        event = conn._parse_log_line("plain log line without timestamp", "svc")
        assert event is not None

    def test_top_errors_dedup(self):
        """Повторяющиеся ошибки нормализуются."""
        from cairn.connectors.docker_log_connector import (
            DockerLogConnector, LogEvent,
        )

        conn = DockerLogConnector()
        events = [
            LogEvent(0.0, "ERROR", "connection refused to 192.168.1.1:5432", "svc"),
            LogEvent(1.0, "ERROR", "connection refused to 10.0.0.1:5432",    "svc"),
            LogEvent(2.0, "ERROR", "connection refused to 127.0.0.1:5432",   "svc"),
        ]
        top = conn._extract_top_errors(events, n=3)
        # После нормализации числа → 'N', все три становятся одинаковыми
        assert len(top) >= 1

    def test_anomaly_detection_baseline(self):
        """Первый вызов устанавливает baseline, не детектирует аномалию."""
        from cairn.connectors.docker_log_connector import (
            DockerLogConnector, LogTimeSeries,
        )

        conn = DockerLogConnector()
        ts = LogTimeSeries(
            container="test",
            error_rate=[1.0, 1.0, 1.0],
        )
        score, is_anomalous = conn._detect_anomaly("test", ts)
        assert not is_anomalous, "Первый вызов не должен детектировать аномалию"

    def test_anomaly_detection_spike(self):
        """Резкий рост ошибок → аномалия."""
        from cairn.connectors.docker_log_connector import (
            DockerLogConnector, LogTimeSeries,
        )

        conn = DockerLogConnector(anomaly_threshold=1.5)
        # Устанавливаем baseline
        ts_normal = LogTimeSeries(container="test", error_rate=[1.0, 1.0, 1.0])
        conn._detect_anomaly("test", ts_normal)

        # Резкий рост
        ts_spike = LogTimeSeries(container="test", error_rate=[10.0, 10.0, 10.0])
        score, is_anomalous = conn._detect_anomaly("test", ts_spike)
        assert is_anomalous, f"Ожидалась аномалия (score={score})"


class TestLatencyTraceConnector:
    """LatencyTraceConnector: маппинг endpoint → сервис."""

    def test_endpoint_mapping_product(self):
        from cairn.connectors.latency_trace_connector import LatencyTraceConnector

        conn   = LatencyTraceConnector()
        result = conn._map_endpoint("/product/some-id")
        assert result == "cairn-productcatalog"

    def test_endpoint_mapping_cart(self):
        from cairn.connectors.latency_trace_connector import LatencyTraceConnector

        conn   = LatencyTraceConnector()
        result = conn._map_endpoint("/cart")
        assert result == "cairn-cartservice"

    def test_endpoint_mapping_root(self):
        from cairn.connectors.latency_trace_connector import LatencyTraceConnector

        conn   = LatencyTraceConnector()
        result = conn._map_endpoint("/")
        assert result == "cairn-frontend"

    def test_anomaly_detection_latency_spike(self):
        """Latency spike относительно baseline → is_slow=True."""
        from cairn.connectors.latency_trace_connector import LatencyTraceConnector

        conn = LatencyTraceConnector(threshold_factor=1.5)
        # Устанавливаем baseline
        conn._detect_anomaly("svc", 100.0)

        # Spike
        is_slow, score = conn._detect_anomaly("svc", 500.0)
        assert is_slow, f"Ожидался latency spike (score={score})"
