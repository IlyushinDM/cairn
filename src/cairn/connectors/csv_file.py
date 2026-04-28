"""CSV / file-based коннекторы для CAIRN.

Читают все четыре типа данных из локальных файлов:
  - MetricData   ← CSV (timestamp, instance, metric1, metric2, ...)
  - LogData      ← текстовый файл (timestamp | instance | level | message)
  - TraceData    ← JSON-файл (список трассировок)
  - TopologyData ← YAML-файл

Используются для демо-режима и unit-тестов.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from cairn.connectors.base import (
    BaseLogConnector,
    BaseMetricConnector,
    BaseTopologyConnector,
    BaseTraceConnector,
    ConnectorConfigError,
    InstanceInfo,
    LogData,
    LogEntry,
    MetricData,
    SpanData,
    TopologyData,
    TraceData,
)

_LEVEL_ORDER = {"DEBUG": 0, "INFO": 1, "WARN": 2, "WARNING": 2, "ERROR": 3}


class CSVMetricConnector(BaseMetricConnector):
    """Коннектор метрик из CSV-файла.

    Формат CSV:
        timestamp, instance, metric1, metric2, ...

    Parameters
    ----------
    path : str | Path
        Путь к CSV-файлу.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise ConnectorConfigError(f"CSV-файл метрик не найден: {path}")
        self._cache: dict | None = None

    # ------------------------------------------------------------------
    # Внутренние методы
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        """Загружает и кэширует весь CSV в структуру {instance → {metric → [(ts, val)]}}."""
        if self._cache is not None:
            return self._cache

        raw: dict[str, dict[str, list]] = {}
        metrics_set: list[str] = []

        with self._path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ConnectorConfigError("CSV-файл пуст или без заголовков")

            # Метрики — все столбцы кроме timestamp и instance
            metrics_set = [
                col for col in reader.fieldnames
                if col not in ("timestamp", "instance")
            ]

            for row in reader:
                ts = float(row["timestamp"])
                inst = row["instance"]
                raw.setdefault(inst, {m: [] for m in metrics_set})
                for m in metrics_set:
                    raw[inst][m].append((ts, float(row.get(m, "nan") or "nan")))

        self._cache = {"raw": raw, "metrics": metrics_set}
        return self._cache

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def available_metrics(self) -> list[str]:
        return self._load()["metrics"]

    def available_instances(self) -> list[str]:
        return sorted(self._load()["raw"].keys())

    def fetch(
        self,
        start_time: float,
        end_time: float,
        instances: Optional[list[str]] = None,
        metrics: Optional[list[str]] = None,
    ) -> MetricData:
        data = self._load()
        raw = data["raw"]
        all_metrics = data["metrics"]

        inst_names = sorted(instances or raw.keys())
        metric_names = metrics or all_metrics

        # Проверка допустимости фильтров
        unknown_inst = set(inst_names) - set(raw.keys())
        if unknown_inst:
            raise ConnectorConfigError(f"Неизвестные экземпляры: {unknown_inst}")
        unknown_m = set(metric_names) - set(all_metrics)
        if unknown_m:
            raise ConnectorConfigError(f"Неизвестные метрики: {unknown_m}")

        # Собираем общую сетку временных меток (пересечение для запрошенных экземпляров)
        ref_inst = inst_names[0]
        ref_metric = metric_names[0]
        ts_vals = [
            ts for ts, _ in raw[ref_inst][ref_metric]
            if start_time <= ts <= end_time
        ]
        if not ts_vals:
            T = 0
            timestamps = np.array([], dtype=np.float64)
            values = np.zeros((0, len(inst_names), len(metric_names)))
            return MetricData(timestamps, values, inst_names, metric_names)

        timestamps = np.array(ts_vals, dtype=np.float64)
        T = len(timestamps)
        values = np.full((T, len(inst_names), len(metric_names)), np.nan)

        ts_set = dict(zip(ts_vals, range(T)))  # timestamp → row index

        for ni, inst in enumerate(inst_names):
            for mi, met in enumerate(metric_names):
                for ts, val in raw.get(inst, {}).get(met, []):
                    if ts in ts_set:
                        values[ts_set[ts], ni, mi] = val

        return MetricData(timestamps, values, inst_names, metric_names)


class FileLogConnector(BaseLogConnector):
    """Коннектор журналов из текстового файла.

    Формат строки:
        <timestamp> | <instance> | <level> | <message>
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise ConnectorConfigError(f"Файл журналов не найден: {path}")

    def fetch(
        self,
        start_time: float,
        end_time: float,
        instances: Optional[list[str]] = None,
        min_level: str = "INFO",
    ) -> LogData:
        min_ord = _LEVEL_ORDER.get(min_level.upper(), 1)
        entries: list[LogEntry] = []

        with self._path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split("|", 3)]
                if len(parts) < 4:
                    continue
                try:
                    ts = float(parts[0])
                except ValueError:
                    continue

                if not (start_time <= ts <= end_time):
                    continue

                inst, level, msg = parts[1], parts[2].upper(), parts[3]
                if instances and inst not in instances:
                    continue
                if _LEVEL_ORDER.get(level, 0) < min_ord:
                    continue

                entries.append(LogEntry(ts, inst, level, msg))

        return LogData(entries)


class JSONTraceConnector(BaseTraceConnector):
    """Коннектор трассировок из JSON-файла.

    Формат JSON:
        [
          {
            "trace_id": "abc123",
            "spans": [
              {
                "span_id": "s1",
                "parent_span_id": null,
                "service": "frontend",
                "instance": "frontend-1",
                "operation": "GET /api",
                "start_time": 1700000000.0,
                "duration_ms": 42.5,
                "status": "OK",
                "attributes": {}
              },
              ...
            ]
          },
          ...
        ]
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise ConnectorConfigError(f"JSON-файл трассировок не найден: {path}")

    def fetch(self, start_time: float, end_time: float) -> list[TraceData]:
        with self._path.open(encoding="utf-8") as f:
            raw: list[dict] = json.load(f)

        result: list[TraceData] = []
        for trace_dict in raw:
            spans = [
                SpanData(
                    span_id=s["span_id"],
                    parent_span_id=s.get("parent_span_id"),
                    service=s["service"],
                    instance=s["instance"],
                    operation=s["operation"],
                    start_time=float(s["start_time"]),
                    duration_ms=float(s["duration_ms"]),
                    status=s.get("status", "OK"),
                    attributes=s.get("attributes", {}),
                )
                for s in trace_dict["spans"]
            ]
            if not spans:
                continue

            trace_start = min(s.start_time for s in spans)
            if start_time <= trace_start <= end_time:
                result.append(TraceData(trace_id=trace_dict["trace_id"], spans=spans))

        return result


class YAMLTopologyConnector(BaseTopologyConnector):
    """Коннектор топологии из YAML-файла.

    Формат YAML:
        instances:
          - name: frontend-1
            service: frontend
            host: node-1
            cpu_limit: 2.0
            memory_limit: 512
            version: "1.0"

        call_edges:
          - [frontend-1, order-service-1]

        colocation_groups:
          - [order-service-1, cache-service-1]

        load_balancer_groups:
          - [order-service-1, order-service-2]
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise ConnectorConfigError(f"YAML-файл топологии не найден: {path}")

    def fetch(self) -> TopologyData:
        with self._path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)

        instances = [
            InstanceInfo(
                name=inst["name"],
                service=inst["service"],
                host=inst["host"],
                cpu_limit=float(inst.get("cpu_limit", 1.0)),
                memory_limit=float(inst.get("memory_limit", 256)),
                version=str(inst.get("version", "unknown")),
            )
            for inst in data.get("instances", [])
        ]

        call_edges = [
            (str(e[0]), str(e[1]))
            for e in data.get("call_edges", [])
        ]
        colocation_groups = [
            [str(n) for n in grp]
            for grp in data.get("colocation_groups", [])
        ]
        lb_groups = [
            [str(n) for n in grp]
            for grp in data.get("load_balancer_groups", [])
        ]

        return TopologyData(
            instances=instances,
            call_edges=call_edges,
            colocation_groups=colocation_groups,
            load_balancer_groups=lb_groups,
        )
