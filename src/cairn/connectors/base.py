"""Базовые классы и унифицированные форматы данных для коннекторов CAIRN.

Все коннекторы реализуют абстрактные классы этого модуля и возвращают данные
в унифицированных форматах (MetricData, LogData, TraceData, TopologyData),
независимо от источника (CSV, Prometheus, Elasticsearch и т.д.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Унифицированные форматы данных
# ---------------------------------------------------------------------------


@dataclass
class MetricData:
    """Унифицированный формат метрик.

    Attributes
    ----------
    timestamps : np.ndarray, shape (T,), dtype float64
        Временные метки (Unix timestamp).
    values : np.ndarray, shape (T, N_instances, N_metrics)
        Значения метрик.
    instance_names : list[str]
        Имена экземпляров сервисов, длина N_instances.
    metric_names : list[str]
        Имена метрик, длина N_metrics.
    """

    timestamps: np.ndarray
    values: np.ndarray
    instance_names: list[str]
    metric_names: list[str]

    def __post_init__(self) -> None:
        T, N, M = len(self.timestamps), len(self.instance_names), len(self.metric_names)
        if self.values.shape != (T, N, M):
            raise ValueError(
                f"values.shape {self.values.shape} != (T={T}, N={N}, M={M})"
            )

    @property
    def n_timesteps(self) -> int:
        return len(self.timestamps)

    @property
    def n_instances(self) -> int:
        return len(self.instance_names)

    @property
    def n_metrics(self) -> int:
        return len(self.metric_names)

    def slice_instances(self, names: list[str]) -> "MetricData":
        """Возвращает подмножество экземпляров по именам."""
        idx = [self.instance_names.index(n) for n in names]
        return MetricData(
            timestamps=self.timestamps,
            values=self.values[:, idx, :],
            instance_names=names,
            metric_names=self.metric_names,
        )

    def slice_metrics(self, names: list[str]) -> "MetricData":
        """Возвращает подмножество метрик по именам."""
        idx = [self.metric_names.index(n) for n in names]
        return MetricData(
            timestamps=self.timestamps,
            values=self.values[:, :, idx],
            instance_names=self.instance_names,
            metric_names=names,
        )


@dataclass
class LogEntry:
    """Одна запись журнала."""
    timestamp: float
    instance_name: str
    level: str        # INFO | WARN | ERROR | DEBUG
    message: str


@dataclass
class LogData:
    """Унифицированный формат журналов."""
    entries: list[LogEntry] = field(default_factory=list)

    @property
    def timestamps(self) -> np.ndarray:
        return np.array([e.timestamp for e in self.entries], dtype=np.float64)

    @property
    def instance_names(self) -> list[str]:
        return [e.instance_name for e in self.entries]

    @property
    def messages(self) -> list[str]:
        return [e.message for e in self.entries]

    @property
    def levels(self) -> list[str]:
        return [e.level for e in self.entries]

    def filter_level(self, level: str) -> "LogData":
        return LogData([e for e in self.entries if e.level == level])

    def filter_instance(self, name: str) -> "LogData":
        return LogData([e for e in self.entries if e.instance_name == name])


@dataclass
class SpanData:
    """Один span в трассировке."""
    span_id: str
    parent_span_id: Optional[str]
    service: str
    instance: str
    operation: str
    start_time: float     # Unix timestamp
    duration_ms: float
    status: str           # OK | ERROR | TIMEOUT
    attributes: dict = field(default_factory=dict)


@dataclass
class TraceData:
    """Унифицированный формат одной трассировки."""
    trace_id: str
    spans: list[SpanData]

    @property
    def root_span(self) -> Optional[SpanData]:
        for s in self.spans:
            if s.parent_span_id is None:
                return s
        return self.spans[0] if self.spans else None

    @property
    def start_time(self) -> float:
        return min(s.start_time for s in self.spans) if self.spans else 0.0

    @property
    def services(self) -> list[str]:
        return list({s.service for s in self.spans})

    def call_path(self) -> list[tuple[str, str]]:
        """Рёбра вызовов (caller_instance, callee_instance)."""
        span_map = {s.span_id: s for s in self.spans}
        return [
            (span_map[s.parent_span_id].instance, s.instance)
            for s in self.spans
            if s.parent_span_id and s.parent_span_id in span_map
        ]


@dataclass
class InstanceInfo:
    """Метаданные одного экземпляра сервиса."""
    name: str
    service: str
    host: str
    cpu_limit: float
    memory_limit: float
    version: str = "unknown"


@dataclass
class TopologyData:
    """Топология системы: экземпляры, рёбра вызовов, группы размещения.

    Attributes
    ----------
    instances : list[InstanceInfo]
    call_edges : list[tuple[str, str]]
        (caller_instance, callee_instance)
    colocation_groups : list[list[str]]
        Группы экземпляров на одном хосте.
    load_balancer_groups : list[list[str]]
        Группы реплик одного сервиса.
    """
    instances: list[InstanceInfo]
    call_edges: list[tuple[str, str]] = field(default_factory=list)
    colocation_groups: list[list[str]] = field(default_factory=list)
    load_balancer_groups: list[list[str]] = field(default_factory=list)

    @property
    def instance_names(self) -> list[str]:
        return [i.name for i in self.instances]

    def get_instance(self, name: str) -> Optional[InstanceInfo]:
        return next((i for i in self.instances if i.name == name), None)

    def callers_of(self, instance: str) -> list[str]:
        return [s for s, d in self.call_edges if d == instance]

    def callees_of(self, instance: str) -> list[str]:
        return [d for s, d in self.call_edges if s == instance]


# ---------------------------------------------------------------------------
# Исключения
# ---------------------------------------------------------------------------


class ConnectorError(RuntimeError):
    """Базовое исключение коннектора."""


class ConnectorUnavailableError(ConnectorError):
    """Источник данных недоступен."""


class ConnectorConfigError(ConnectorError):
    """Ошибка конфигурации коннектора."""


# ---------------------------------------------------------------------------
# Абстрактные базовые классы коннекторов
# ---------------------------------------------------------------------------


class BaseMetricConnector(ABC):
    """Абстрактный коннектор метрик."""

    @abstractmethod
    def fetch(
        self,
        start_time: float,
        end_time: float,
        instances: Optional[list[str]] = None,
        metrics: Optional[list[str]] = None,
    ) -> MetricData:
        """Загружает метрики за [start_time, end_time].

        Parameters
        ----------
        instances : фильтр по экземплярам (None = все)
        metrics   : фильтр по метрикам (None = все)
        """
        ...

    @abstractmethod
    def available_metrics(self) -> list[str]: ...

    @abstractmethod
    def available_instances(self) -> list[str]: ...


class BaseLogConnector(ABC):
    """Абстрактный коннектор журналов."""

    @abstractmethod
    def fetch(
        self,
        start_time: float,
        end_time: float,
        instances: Optional[list[str]] = None,
        min_level: str = "INFO",
    ) -> LogData: ...


class BaseTraceConnector(ABC):
    """Абстрактный коннектор трассировок."""

    @abstractmethod
    def fetch(self, start_time: float, end_time: float) -> list[TraceData]: ...


class BaseTopologyConnector(ABC):
    """Абстрактный коннектор топологии."""

    @abstractmethod
    def fetch(self) -> TopologyData: ...
