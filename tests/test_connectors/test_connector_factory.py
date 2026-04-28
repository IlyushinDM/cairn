"""Тесты фабрики коннекторов CAIRN."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest

from cairn.connectors.base import (
    BaseLogConnector,
    BaseMetricConnector,
    BaseTopologyConnector,
    BaseTraceConnector,
    ConnectorConfigError,
)
from cairn.connectors.factory import create_connectors

SAMPLE_DIR = str(Path(__file__).parent.parent.parent / "data" / "sample")


def _demo_config() -> dict:
    return {
        "metrics": {"type": "csv", "path": f"{SAMPLE_DIR}/metrics.csv"},
        "logs":    {"type": "file", "path": f"{SAMPLE_DIR}/logs.txt"},
        "traces":  {"type": "file", "path": f"{SAMPLE_DIR}/traces.json"},
        "topology": {"type": "yaml", "path": f"{SAMPLE_DIR}/topology.yaml"},
    }


class TestConnectorFactory:
    def test_create_all_connectors(self):
        cfg = _demo_config()
        mc, lc, tc, topc = create_connectors(cfg)
        assert isinstance(mc, BaseMetricConnector)
        assert isinstance(lc, BaseLogConnector)
        assert isinstance(tc, BaseTraceConnector)
        assert isinstance(topc, BaseTopologyConnector)

    def test_csv_metric_connector_created(self):
        from cairn.connectors.csv_file import CSVMetricConnector
        mc, *_ = create_connectors(_demo_config())
        assert isinstance(mc, CSVMetricConnector)

    def test_file_log_connector_created(self):
        from cairn.connectors.csv_file import FileLogConnector
        _, lc, *_ = create_connectors(_demo_config())
        assert isinstance(lc, FileLogConnector)

    def test_unknown_metric_type_raises(self):
        cfg = _demo_config()
        cfg["metrics"]["type"] = "unknown_db"
        with pytest.raises(ConnectorConfigError, match="Неизвестный тип"):
            create_connectors(cfg)

    def test_unknown_log_type_raises(self):
        cfg = _demo_config()
        cfg["logs"]["type"] = "kafka"
        with pytest.raises(ConnectorConfigError, match="Неизвестный тип"):
            create_connectors(cfg)

    def test_missing_file_raises(self):
        """Если файл не существует — должен быть ConnectorConfigError."""
        cfg = {
            "metrics":  {"type": "csv",  "path": "/nonexistent/metrics.csv"},
            "logs":     {"type": "file", "path": "/nonexistent/logs.txt"},
            "traces":   {"type": "file", "path": "/nonexistent/traces.json"},
            "topology": {"type": "yaml", "path": "/nonexistent/topology.yaml"},
        }
        with pytest.raises(ConnectorConfigError):
            create_connectors(cfg)

    def test_topology_yaml_alias(self):
        """type: file должен быть синонимом yaml для топологии."""
        cfg = _demo_config()
        cfg["topology"]["type"] = "file"
        *_, topc = create_connectors(cfg)
        assert isinstance(topc, BaseTopologyConnector)
