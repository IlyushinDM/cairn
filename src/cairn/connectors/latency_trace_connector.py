"""LatencyTraceConnector — latency трассировки через логи loadgenerator.

Online Boutique использует Locust как load generator.
Locust логирует per-endpoint статистику в формате:
  METHOD /path  <requests> <failures>(<%>) | <median_ms>

Маппинг endpoint → сервис позволяет получить latency per service.
Это практический MVP трассировок без OTel/Jaeger.

Для систем с OpenTelemetry замените _fetch_locust_logs()
на реальный Jaeger/Zipkin коннектор.
"""
from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── Маппинг endpoint → сервис ─────────────────────────────────────────────────

ENDPOINT_TO_SERVICE: dict[str, str] = {
    "/product":      "cairn-productcatalog",
    "/cart":         "cairn-cartservice",
    "/checkout":     "cairn-checkoutservice",
    "/currency":     "cairn-currencyservice",
    "/shipping":     "cairn-shippingservice",
    "/payment":      "cairn-paymentservice",
    "/recommendation": "cairn-recommendationservice",
    "/ad":           "cairn-adservice",
    "/":             "cairn-frontend",
}

# Regex для строки Locust статистики
# GET /product/9SIQT8TOJO   134 134(100.00%) |   3812
_LOCUST_RE = re.compile(
    r'(GET|POST|PUT|DELETE|PATCH)\s+(/\S*)\s+'
    r'(\d+)\s+\d+\([^)]+\)\s*\|\s*(\d+)'
)


@dataclass
class ServiceLatency:
    """Latency статистика для одного сервиса."""
    service:       str
    endpoints:     list[str]           = field(default_factory=list)
    p50_ms:        list[float]         = field(default_factory=list)
    timestamps:    list[float]         = field(default_factory=list)
    request_count: int                 = 0
    avg_p50_ms:    float               = 0.0
    is_slow:       bool                = False   # latency > baseline * threshold
    anomaly_score: float               = 0.0


@dataclass
class TraceData:
    """Результат сбора latency трассировок."""
    services:     dict[str, ServiceLatency] = field(default_factory=dict)
    collect_time: float                     = 0.0
    source:       str                       = "locust_logs"

    @property
    def n_services(self) -> int:
        return len(self.services)

    @property
    def slow_services(self) -> list[str]:
        return [name for name, s in self.services.items() if s.is_slow]

    def as_metric_array(
        self, service_names: list[str]
    ) -> np.ndarray:
        """Возвращает массив p50 latency для заданных сервисов."""
        result = np.zeros(len(service_names))
        for i, name in enumerate(service_names):
            if name in self.services:
                result[i] = self.services[name].avg_p50_ms
        return result


class LatencyTraceConnector:
    """Коннектор latency трассировок через логи loadgenerator (Locust).

    Архитектурно: реализует интерфейс TraceConnector.
    Для замены на OTel/Jaeger — переопределить _fetch_raw_latency().
    """

    def __init__(
        self,
        loadgen_container: str = "cairn-loadgenerator",
        threshold_factor:  float = 2.0,   # во сколько раз медиана > baseline
    ):
        self._container = loadgen_container
        self._threshold = threshold_factor
        self._baseline:  dict[str, float] = {}

    def fetch(self, window_sec: int = 120) -> TraceData:
        """Собирает latency данные за последние window_sec секунд."""
        data = TraceData(collect_time=time.time())

        raw = self._fetch_locust_logs(window_sec)
        if not raw:
            return data

        # Парсим строки статистики
        service_stats: dict[str, list[float]] = {}
        service_endpoints: dict[str, list[str]] = {}

        for line in raw:
            m = _LOCUST_RE.search(line)
            if not m:
                continue
            method, path, req_count, median_ms = (
                m.group(1), m.group(2),
                int(m.group(3)), float(m.group(4)),
            )
            service = self._map_endpoint(path)
            service_stats.setdefault(service, []).append(median_ms)
            service_endpoints.setdefault(service, []).append(
                f"{method} {path}"
            )

        # Строим ServiceLatency объекты
        for service, latencies in service_stats.items():
            avg = float(np.mean(latencies))
            is_slow, score = self._detect_anomaly(service, avg)

            sl = ServiceLatency(
                service       = service,
                endpoints     = list(set(service_endpoints.get(service, []))),
                p50_ms        = latencies,
                timestamps    = [data.collect_time] * len(latencies),
                request_count = len(latencies),
                avg_p50_ms    = avg,
                is_slow       = is_slow,
                anomaly_score = score,
            )
            data.services[service] = sl

        return data

    # ── Внутренние методы ─────────────────────────────────────────────────────

    def _fetch_locust_logs(self, window_sec: int) -> list[str]:
        """Читает логи loadgenerator."""
        try:
            result = subprocess.run(
                ["docker", "logs", self._container,
                 "--since", f"{window_sec}s"],
                capture_output=True, text=True, timeout=10
            )
            lines = (result.stdout + result.stderr).split("\n")
            # Фильтруем строки со статистикой (содержат "|")
            return [l for l in lines if "|" in l and _LOCUST_RE.search(l)]
        except Exception:
            return []

    def _map_endpoint(self, path: str) -> str:
        """Маппит URL path на имя сервиса."""
        for prefix, service in ENDPOINT_TO_SERVICE.items():
            if path.startswith(prefix):
                return service
        return "cairn-frontend"  # по умолчанию

    def _detect_anomaly(
        self, service: str, current_ms: float
    ) -> tuple[bool, float]:
        """Определяет latency аномалию относительно baseline."""
        if service not in self._baseline:
            self._baseline[service] = current_ms
            return False, 0.0

        baseline = self._baseline[service]
        # Обновляем baseline
        self._baseline[service] = 0.85 * baseline + 0.15 * current_ms

        if baseline < 1.0:
            return False, 0.0

        ratio = current_ms / (baseline + 1e-6)
        score = max(0.0, ratio - 1.0)   # 0 = норма, 1.0 = в 2 раза выше
        return ratio > self._threshold, score

    def is_available(self) -> tuple[bool, str]:
        """Проверяет доступность loadgenerator."""
        try:
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}",
                 self._container],
                capture_output=True, text=True, timeout=3
            )
            running = result.stdout.strip() == "true"
            return running, (
                f"loadgenerator: {'запущен' if running else 'не запущен'}"
            )
        except Exception as e:
            return False, f"docker error: {e}"


def merge_trace_anomalies_with_metrics(
    trace_data: TraceData,
    metric_scores: dict[str, float],
    boost_factor: float = 0.25,
) -> dict[str, float]:
    """Повышает metric score для сервисов с latency аномалией.

    Latency spike + метрическая аномалия = высокая уверенность первопричины.
    """
    updated = dict(metric_scores)
    for service, sl in trace_data.services.items():
        if sl.is_slow and service in updated:
            updated[service] *= (1.0 + boost_factor * sl.anomaly_score)
    return updated
