"""Фабрика коннекторов CAIRN.

Создаёт коннекторы по конфигурации из configs/default.yaml.
Пользователь управляет источниками данных исключительно через YAML,
без изменения кода.

Пример конфигурации:
    connectors:
      metrics:
        type: csv
        path: data/sample/metrics.csv
        filter_instances: []    # [] = все
        filter_metrics: []      # [] = все
      logs:
        type: file
        path: data/sample/logs.txt
      traces:
        type: file
        path: data/sample/traces.json
      topology:
        type: yaml
        path: data/sample/topology.yaml
"""

from __future__ import annotations

from typing import Any, Optional

from cairn.connectors.base import (
    BaseLogConnector,
    BaseMetricConnector,
    BaseTopologyConnector,
    BaseTraceConnector,
    ConnectorConfigError,
    ConnectorUnavailableError,
)


def _get(cfg: dict, key: str, default: Any = None) -> Any:
    return cfg.get(key, default)


def _create_metric_connector(cfg: dict) -> BaseMetricConnector:
    ctype = _get(cfg, "type", "csv")

    if ctype == "csv":
        from cairn.connectors.csv_file import CSVMetricConnector
        path = _get(cfg, "path", "data/sample/metrics.csv")
        return CSVMetricConnector(path)

    if ctype == "prometheus":
        from cairn.connectors.prometheus import PrometheusMetricConnector
        return PrometheusMetricConnector(
            url=_get(cfg, "url", "http://localhost:9090"),
            metric_names=_get(cfg, "metrics", []),
            step=_get(cfg, "step", "15s"),
            label_filter=_get(cfg, "label_filter", {}),
            instance_label=_get(cfg, "instance_label", "pod"),
        )

    raise ConnectorConfigError(
        f"Неизвестный тип метрик-коннектора: '{ctype}'. "
        "Доступные типы: csv, prometheus"
    )


def _create_log_connector(cfg: dict) -> BaseLogConnector:
    ctype = _get(cfg, "type", "file")

    if ctype == "file":
        from cairn.connectors.csv_file import FileLogConnector
        return FileLogConnector(_get(cfg, "path", "data/sample/logs.txt"))

    if ctype == "elasticsearch":
        raise ConnectorUnavailableError(
            "Elasticsearch-коннектор будет реализован в следующей версии."
        )

    raise ConnectorConfigError(
        f"Неизвестный тип лог-коннектора: '{ctype}'. "
        "Доступные типы: file, elasticsearch"
    )


def _create_trace_connector(cfg: dict) -> BaseTraceConnector:
    ctype = _get(cfg, "type", "file")

    if ctype == "file":
        from cairn.connectors.csv_file import JSONTraceConnector
        return JSONTraceConnector(_get(cfg, "path", "data/sample/traces.json"))

    if ctype == "jaeger":
        raise ConnectorUnavailableError(
            "Jaeger-коннектор будет реализован в следующей версии."
        )

    raise ConnectorConfigError(
        f"Неизвестный тип трейс-коннектора: '{ctype}'. "
        "Доступные типы: file, jaeger"
    )


def _create_topology_connector(cfg: dict) -> BaseTopologyConnector:
    ctype = _get(cfg, "type", "yaml")

    if ctype in ("yaml", "file"):
        from cairn.connectors.csv_file import YAMLTopologyConnector
        return YAMLTopologyConnector(_get(cfg, "path", "data/sample/topology.yaml"))

    raise ConnectorConfigError(
        f"Неизвестный тип топологии-коннектора: '{ctype}'. "
        "Доступные типы: yaml"
    )


def create_connectors(
    config: dict,
) -> tuple[
    BaseMetricConnector,
    BaseLogConnector,
    BaseTraceConnector,
    BaseTopologyConnector,
]:
    """Создаёт все четыре коннектора по секции connectors конфигурации.

    Parameters
    ----------
    config : dict
        Словарь из секции ``connectors`` конфигурационного файла.
        Ожидаемые ключи: metrics, logs, traces, topology.

    Returns
    -------
    (metric_connector, log_connector, trace_connector, topology_connector)

    Raises
    ------
    ConnectorConfigError
        Неизвестный тип или отсутствующий обязательный параметр.
    ConnectorUnavailableError
        Источник данных недоступен (файл не существует, сервер не отвечает).
    """
    metrics_cfg = _get(config, "metrics", {})
    logs_cfg = _get(config, "logs", {})
    traces_cfg = _get(config, "traces", {})
    topology_cfg = _get(config, "topology", {})

    return (
        _create_metric_connector(metrics_cfg),
        _create_log_connector(logs_cfg),
        _create_trace_connector(traces_cfg),
        _create_topology_connector(topology_cfg),
    )
