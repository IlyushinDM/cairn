"""Генератор синтетических лог и трассировочных данных для демо-сценариев.

Создаёт реалистичные LogData и TraceData на основе метрических данных сценария:
- Нормальные сервисы: низкий error rate, нормальная latency
- Аномальный сервис: высокий error rate, latency spike
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np

from cairn.connectors.docker_log_connector import (
    LogData, LogTimeSeries, LogEvent,
)
from cairn.connectors.latency_trace_connector import (
    TraceData, ServiceLatency,
)


# ── Шаблоны сообщений ─────────────────────────────────────────────────────────

_NORMAL_WARNS = [
    "slow query detected ({}ms)",
    "connection pool at {}% capacity",
    "cache miss rate {}%",
    "retry attempt {} for upstream",
    "rate limit approaching: {}req/s",
]

_ANOMALY_ERRORS = [
    "connection refused: upstream {}",
    "timeout after {}ms waiting for {}",
    "circuit breaker OPEN for {}",
    "FATAL: out of memory ({}MB)",
    "panic: nil pointer dereference in {}",
    "database connection pool exhausted",
    "health check FAILED: {}",
]

_NORMAL_INFO = [
    "request processed in {}ms",
    "cache hit ratio {}%",
    "metrics collected",
    "heartbeat OK",
]


def generate_demo_log_data(
    instance_names: list[str],
    timestamps: np.ndarray,
    root_cause: Optional[str] = None,
    anomaly_start_idx: int = -1,
) -> LogData:
    """Генерирует LogData для демо-сценария.

    Args:
        instance_names: список имён сервисов
        timestamps:     временные метки из metric_data
        root_cause:     имя аномального сервиса (или None)
        anomaly_start_idx: с какого индекса начинается аномалия
    """
    rng = random.Random(42)
    T   = len(timestamps)
    if anomaly_start_idx < 0:
        anomaly_start_idx = int(T * 0.6)

    series: dict[str, LogTimeSeries] = {}
    all_events: list = []

    for svc in instance_names:
        is_anomalous = (root_cause is not None and svc == root_cause)
        ts_list     = []
        err_rate    = []
        warn_rate   = []
        total_rate  = []

        for i, t in enumerate(timestamps):
            in_anomaly = is_anomalous and i >= anomaly_start_idx

            # Базовые частоты событий
            base_total = rng.uniform(8, 15)
            base_warn  = rng.uniform(0.2, 0.8)
            base_error = rng.uniform(0.0, 0.1)

            if in_anomaly:
                spike = rng.uniform(3.0, 8.0)
                base_error = base_error * spike + rng.uniform(2.0, 5.0)
                base_warn  = base_warn  * 2.0
                base_total = base_total * 1.5

                # Добавляем error события
                msg = rng.choice(_ANOMALY_ERRORS).format(
                    svc, rng.randint(1000, 9000), "db"
                )
                all_events.append(LogEvent(
                    timestamp=float(t),
                    level="ERROR",
                    message=msg,
                    container=svc,
                ))
            elif rng.random() < 0.15:
                msg = rng.choice(_NORMAL_WARNS).format(
                    rng.randint(100, 500), rng.randint(50, 95)
                )
                all_events.append(LogEvent(
                    timestamp=float(t),
                    level="WARN",
                    message=msg,
                    container=svc,
                ))

            ts_list.append(float(t))
            err_rate.append(max(0.0, base_error + rng.gauss(0, 0.1)))
            warn_rate.append(max(0.0, base_warn  + rng.gauss(0, 0.05)))
            total_rate.append(max(0.0, base_total + rng.gauss(0, 0.5)))

        anomaly_score = float(np.mean(err_rate[-5:])) if is_anomalous else float(np.mean(err_rate))
        is_anom       = is_anomalous and anomaly_score > 1.0

        top_err = []
        if is_anom:
            top_err = [
                rng.choice(_ANOMALY_ERRORS).format(svc, rng.randint(100, 5000), "upstream"),
                f"connection timeout after {rng.randint(500,3000)}ms",
            ]
        else:
            top_err = [rng.choice(_NORMAL_WARNS).format(rng.randint(100, 300))]

        series[svc] = LogTimeSeries(
            container=svc,
            timestamps=ts_list,
            error_rate=err_rate,
            warn_rate=warn_rate,
            total_rate=total_rate,
            top_errors=top_err,
            anomaly_score=anomaly_score,
            is_anomalous=is_anom,
        )

    collect_time = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 60.0
    return LogData(series=series, all_events=all_events, collect_time=collect_time)


def generate_demo_trace_data(
    instance_names: list[str],
    timestamps: np.ndarray,
    root_cause: Optional[str] = None,
    anomaly_start_idx: int = -1,
    base_latency_ms: float = 45.0,
) -> TraceData:
    """Генерирует TraceData для демо-сценария."""
    rng = random.Random(123)
    T   = len(timestamps)
    if anomaly_start_idx < 0:
        anomaly_start_idx = int(T * 0.6)

    services: dict[str, ServiceLatency] = {}

    for svc in instance_names:
        is_anomalous = (root_cause is not None and svc == root_cause)
        p50_list = []

        for i in range(T):
            in_anomaly = is_anomalous and i >= anomaly_start_idx
            if in_anomaly:
                lat = base_latency_ms * rng.uniform(4.0, 12.0) + rng.gauss(0, 20)
            else:
                lat = base_latency_ms * rng.uniform(0.6, 1.4) + rng.gauss(0, 5)
            p50_list.append(max(1.0, lat))

        avg_p50    = float(np.mean(p50_list))
        is_slow    = is_anomalous and avg_p50 > base_latency_ms * 2.5
        anom_score = avg_p50 / base_latency_ms if is_slow else 0.0

        # Эндпоинты – имитируем REST
        short = svc.replace("cairn-", "").replace("service", "")
        endpoints = [f"/{short}/", f"/{short}/health", f"/{short}/api"]

        services[svc] = ServiceLatency(
            service=svc,
            endpoints=endpoints,
            p50_ms=p50_list,
            timestamps=[float(t) for t in timestamps],
            request_count=int(T * rng.uniform(10, 30)),
            avg_p50_ms=avg_p50,
            is_slow=is_slow,
            anomaly_score=anom_score,
        )

    collect_time = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 60.0
    return TraceData(
        services=services,
        collect_time=collect_time,
        source="demo_generator",
    )
