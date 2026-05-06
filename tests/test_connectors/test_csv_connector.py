"""Тесты CSV/file коннекторов CAIRN.

Проверяют загрузку демо-данных и соответствие унифицированным форматам.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Обеспечиваем видимость пакета без pip install
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pytest

from cairn.connectors.csv_file import (
    CSVMetricConnector,
    FileLogConnector,
    JSONTraceConnector,
    YAMLTopologyConnector,
)
from cairn.connectors.base import (
    ConnectorConfigError,
    LogData,
    MetricData,
    TopologyData,
    TraceData,
)

SAMPLE_DIR = Path(__file__).parent.parent.parent / "data" / "sample"
BASE_TS = 1_700_000_000.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def metric_conn():
    return CSVMetricConnector(SAMPLE_DIR / "metrics.csv")

@pytest.fixture
def log_conn():
    return FileLogConnector(SAMPLE_DIR / "logs.txt")

@pytest.fixture
def trace_conn():
    return JSONTraceConnector(SAMPLE_DIR / "traces.json")

@pytest.fixture
def topology_conn():
    return YAMLTopologyConnector(SAMPLE_DIR / "topology.yaml")


# ---------------------------------------------------------------------------
# MetricData
# ---------------------------------------------------------------------------

class TestCSVMetricConnector:
    def test_available_instances(self, metric_conn):
        insts = metric_conn.available_instances()
        assert len(insts) == 5
        assert "order-service-1" in insts

    def test_available_metrics(self, metric_conn):
        metrics = metric_conn.available_metrics()
        assert set(metrics) >= {"cpu", "memory", "latency_ms", "rps"}

    def test_fetch_returns_metric_data(self, metric_conn):
        end = BASE_TS + 299
        data = metric_conn.fetch(BASE_TS, end)
        assert isinstance(data, MetricData)
        assert data.n_timesteps == 300
        assert data.n_instances == 5
        assert data.n_metrics == 4

    def test_fetch_shape_consistency(self, metric_conn):
        data = metric_conn.fetch(BASE_TS, BASE_TS + 299)
        T, N, M = data.values.shape
        assert T == len(data.timestamps)
        assert N == len(data.instance_names)
        assert M == len(data.metric_names)

    def test_fetch_filter_instances(self, metric_conn):
        data = metric_conn.fetch(
            BASE_TS, BASE_TS + 10,
            instances=["order-service-1", "frontend-1"]
        )
        assert data.n_instances == 2
        assert data.instance_names == ["frontend-1", "order-service-1"]

    def test_fetch_filter_metrics(self, metric_conn):
        data = metric_conn.fetch(BASE_TS, BASE_TS + 10, metrics=["cpu"])
        assert data.n_metrics == 1
        assert data.metric_names == ["cpu"]

    def test_cpu_anomaly_visible(self, metric_conn):
        """CPU order-service-1 должен быть значимо выше в аномальном периоде."""
        normal = metric_conn.fetch(
            BASE_TS, BASE_TS + 299,
            instances=["order-service-1"], metrics=["cpu"]
        )
        anom = metric_conn.fetch(
            BASE_TS + 300, BASE_TS + 399,
            instances=["order-service-1"], metrics=["cpu"]
        )
        cpu_normal = np.nanmean(normal.values[:, 0, 0])
        cpu_anom = np.nanmean(anom.values[:, 0, 0])
        assert cpu_anom > cpu_normal + 0.3, (
            f"Аномалия CPU не обнаружена: normal={cpu_normal:.3f}, anom={cpu_anom:.3f}"
        )

    def test_fetch_unknown_instance_raises(self, metric_conn):
        with pytest.raises(ConnectorConfigError, match="Неизвестные экземпляры"):
            metric_conn.fetch(BASE_TS, BASE_TS + 10, instances=["nonexistent"])

    def test_fetch_unknown_metric_raises(self, metric_conn):
        with pytest.raises(ConnectorConfigError, match="Неизвестные метрики"):
            metric_conn.fetch(BASE_TS, BASE_TS + 10, metrics=["nonexistent"])

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ConnectorConfigError):
            CSVMetricConnector(tmp_path / "nonexistent.csv")

    def test_slice_instances(self, metric_conn):
        data = metric_conn.fetch(BASE_TS, BASE_TS + 50)
        sliced = data.slice_instances(["frontend-1"])
        assert sliced.n_instances == 1
        assert sliced.values.shape == (51, 1, 4)


# ---------------------------------------------------------------------------
# LogData
# ---------------------------------------------------------------------------

class TestFileLogConnector:
    def test_fetch_returns_log_data(self, log_conn):
        data = log_conn.fetch(BASE_TS, BASE_TS + 299)
        assert isinstance(data, LogData)
        assert len(data.entries) > 0

    def test_entries_in_time_range(self, log_conn):
        end = BASE_TS + 199
        data = log_conn.fetch(BASE_TS, end)
        for entry in data.entries:
            assert BASE_TS <= entry.timestamp <= end

    def test_error_logs_in_anomaly_period(self, log_conn):
        data = log_conn.fetch(BASE_TS + 300, BASE_TS + 399)
        levels = data.levels
        assert "ERROR" in levels or "WARN" in levels, "В аномальном периоде нет ERROR/WARN"

    def test_filter_instance(self, log_conn):
        data = log_conn.fetch(BASE_TS, BASE_TS + 299)
        filtered = data.filter_instance("order-service-1")
        for entry in filtered.entries:
            assert entry.instance_name == "order-service-1"

    def test_min_level_filter(self, log_conn):
        warn_only = log_conn.fetch(BASE_TS, BASE_TS + 299, min_level="WARN")
        for entry in warn_only.entries:
            assert entry.level in ("WARN", "WARNING", "ERROR")


# ---------------------------------------------------------------------------
# TraceData
# ---------------------------------------------------------------------------

class TestJSONTraceConnector:
    def test_fetch_returns_traces(self, trace_conn):
        traces = trace_conn.fetch(BASE_TS, BASE_TS + 299)
        assert isinstance(traces, list)
        assert len(traces) > 0
        assert isinstance(traces[0], TraceData)

    def test_trace_has_spans(self, trace_conn):
        traces = trace_conn.fetch(BASE_TS, BASE_TS + 10)
        for t in traces:
            assert len(t.spans) >= 2
            assert t.root_span is not None

    def test_call_path_non_empty(self, trace_conn):
        traces = trace_conn.fetch(BASE_TS, BASE_TS + 10)
        for t in traces:
            path = t.call_path()
            assert len(path) >= 1, "Цепочка вызовов не должна быть пустой"

    def test_time_filter(self, trace_conn):
        all_traces = trace_conn.fetch(BASE_TS, BASE_TS + 299)
        half_traces = trace_conn.fetch(BASE_TS, BASE_TS + 149)
        assert len(half_traces) < len(all_traces)


# ---------------------------------------------------------------------------
# TopologyData
# ---------------------------------------------------------------------------

class TestYAMLTopologyConnector:
    def test_fetch_returns_topology(self, topology_conn):
        topo = topology_conn.fetch()
        assert isinstance(topo, TopologyData)

    def test_instances_count(self, topology_conn):
        topo = topology_conn.fetch()
        assert len(topo.instances) == 5

    def test_instance_names(self, topology_conn):
        topo = topology_conn.fetch()
        assert "order-service-1" in topo.instance_names
        assert "frontend-1" in topo.instance_names

    def test_call_edges(self, topology_conn):
        topo = topology_conn.fetch()
        assert len(topo.call_edges) >= 2
        # frontend → order-service должен быть
        assert ("frontend-1", "order-service-1") in topo.call_edges

    def test_colocation_groups(self, topology_conn):
        topo = topology_conn.fetch()
        assert len(topo.colocation_groups) >= 1
        # order-service и cache-service на одном хосте
        colocated = [set(g) for g in topo.colocation_groups]
        assert {"order-service-1", "cache-service-1"} in colocated

    def test_callers_of(self, topology_conn):
        topo = topology_conn.fetch()
        callers = topo.callers_of("order-service-1")
        assert "frontend-1" in callers

    def test_callees_of(self, topology_conn):
        topo = topology_conn.fetch()
        callees = topo.callees_of("order-service-1")
        assert "payment-service-1" in callees or "database-1" in callees

    def test_get_instance(self, topology_conn):
        topo = topology_conn.fetch()
        inst = topo.get_instance("database-1")
        assert inst is not None
        assert inst.service == "database"
        assert inst.host == "node-4"
