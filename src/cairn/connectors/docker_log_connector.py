"""DockerLogConnector – сбор и анализ журналов контейнеров.

MVP-подход (без тяжёлого NLP):
  - Читает docker logs --since N для каждого контейнера
  - Считает частоту ERROR/WARN/CRITICAL как временной ряд
  - Детектирует аномалии: рост ошибок относительно baseline
  - Извлекает паттерны: повторяющиеся сообщения

Интеграция с MetricData: log-аномалия повышает NLL для сервиса.

Использование:
    conn = DockerLogConnector(instance_filter=["cairn-redis", ...])
    data = conn.fetch(window_sec=300, step_sec=30)
    # data.series: {container: LogTimeSeries}
"""
from __future__ import annotations

import re
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── Уровни логов ─────────────────────────────────────────────────────────────

LOG_LEVELS = {
    "CRITICAL": 4,
    "FATAL":    4,
    "ERROR":    3,
    "ERR":      3,
    "WARN":     2,
    "WARNING":  2,
    "INFO":     1,
    "DEBUG":    0,
}

# Регулярка для парсинга уровня лога
_LEVEL_RE = re.compile(
    r'\b(CRITICAL|FATAL|ERROR|ERR|WARN(?:ING)?|INFO|DEBUG)\b',
    re.IGNORECASE
)


@dataclass
class LogEvent:
    """Одно лог-событие."""
    timestamp: float
    level:     str
    message:   str
    container: str


@dataclass
class LogTimeSeries:
    """Временной ряд частот лог-событий для одного контейнера."""
    container:    str
    timestamps:   list[float]    = field(default_factory=list)
    error_rate:   list[float]    = field(default_factory=list)   # ERROR/CRITICAL в мин
    warn_rate:    list[float]    = field(default_factory=list)    # WARN в мин
    total_rate:   list[float]    = field(default_factory=list)    # всего событий в мин
    top_errors:   list[str]      = field(default_factory=list)    # топ повторяющихся
    anomaly_score: float         = 0.0
    is_anomalous:  bool          = False


@dataclass
class LogData:
    """Результат сбора журналов."""
    series:       dict[str, LogTimeSeries]  = field(default_factory=dict)
    all_events:   list[LogEvent]            = field(default_factory=list)
    collect_time: float                     = 0.0

    @property
    def n_containers(self) -> int:
        return len(self.series)

    @property
    def anomalous_containers(self) -> list[str]:
        return [name for name, s in self.series.items() if s.is_anomalous]


class DockerLogConnector:
    """Коннектор для сбора журналов Docker-контейнеров."""

    def __init__(
        self,
        instance_filter: Optional[list[str]] = None,
        min_level: str = "WARN",               # минимальный уровень для анализа
        anomaly_threshold: float = 2.0,         # σ от baseline для детекции
    ):
        self._filter    = set(instance_filter or [])
        self._min_level = LOG_LEVELS.get(min_level.upper(), 2)
        self._threshold = anomaly_threshold
        self._baseline: dict[str, float] = {}  # baseline error rate per container

    def fetch(
        self,
        window_sec: int = 300,
        step_sec:   int = 30,
    ) -> LogData:
        """Собирает журналы за последние window_sec секунд.

        Разбивает на step_sec-интервалы для построения временного ряда.
        """
        t_end   = time.time()
        t_start = t_end - window_sec

        data = LogData(collect_time=t_end)

        # Получаем список контейнеров
        containers = self._get_containers()

        for container in containers:
            events = self._fetch_container_logs(container, since_sec=window_sec)
            if not events:
                continue

            ts = self._build_time_series(events, t_start, t_end, step_sec)
            ts.top_errors  = self._extract_top_errors(events, n=3)
            ts.anomaly_score, ts.is_anomalous = self._detect_anomaly(container, ts)

            data.series[container]  = ts
            data.all_events.extend(events)

        return data

    # ── Внутренние методы ─────────────────────────────────────────────────────

    def _get_containers(self) -> list[str]:
        try:
            result = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=5
            )
            names = [n.strip() for n in result.stdout.strip().split("\n") if n.strip()]
            if self._filter:
                names = [n for n in names if n in self._filter]
            return names
        except Exception:
            return []

    def _fetch_container_logs(
        self, container: str, since_sec: int
    ) -> list[LogEvent]:
        """Читает логи контейнера за последние since_sec секунд."""
        try:
            result = subprocess.run(
                ["docker", "logs", container,
                 "--since", f"{since_sec}s",
                 "--timestamps"],
                capture_output=True, text=True, timeout=10
            )
            # docker logs пишет в stderr тоже
            raw = result.stdout + result.stderr
        except Exception:
            return []

        events = []
        for line in raw.strip().split("\n"):
            if not line.strip():
                continue
            event = self._parse_log_line(line, container)
            if event and LOG_LEVELS.get(event.level, 0) >= self._min_level:
                events.append(event)
        return events

    def _parse_log_line(self, line: str, container: str) -> Optional[LogEvent]:
        """Парсит строку лога с временной меткой Docker."""
        # Docker timestamp format: 2024-01-15T10:30:45.123456789Z message
        ts_match = re.match(
            r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})[\.\d]*Z?\s*(.*)',
            line.strip()
        )
        if ts_match:
            ts_str  = ts_match.group(1)
            message = ts_match.group(2)
            try:
                from datetime import datetime, timezone
                dt = datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")
                ts = dt.replace(tzinfo=timezone.utc).timestamp()
            except Exception:
                ts = time.time()
        else:
            message = line.strip()
            ts = time.time()

        # Определяем уровень
        m = _LEVEL_RE.search(message)
        level = m.group(1).upper() if m else "INFO"
        if level == "ERR":
            level = "ERROR"
        if level == "WARNING":
            level = "WARN"

        return LogEvent(timestamp=ts, level=level,
                        message=message[:200], container=container)

    def _build_time_series(
        self, events: list[LogEvent],
        t_start: float, t_end: float, step_sec: int
    ) -> LogTimeSeries:
        """Строит временной ряд частот из событий."""
        n_steps = max(1, int((t_end - t_start) / step_sec))
        ts = LogTimeSeries(container=events[0].container if events else "")

        for i in range(n_steps):
            bin_start = t_start + i * step_sec
            bin_end   = bin_start + step_sec
            bin_events = [e for e in events if bin_start <= e.timestamp < bin_end]

            errors = sum(1 for e in bin_events
                        if LOG_LEVELS.get(e.level, 0) >= 3)   # ERROR+
            warns  = sum(1 for e in bin_events
                        if LOG_LEVELS.get(e.level, 0) == 2)   # WARN

            # Нормируем на минуты
            rate_factor = 60.0 / max(step_sec, 1)
            ts.timestamps.append(bin_start)
            ts.error_rate.append(errors * rate_factor)
            ts.warn_rate.append(warns * rate_factor)
            ts.total_rate.append(len(bin_events) * rate_factor)

        return ts

    def _extract_top_errors(
        self, events: list[LogEvent], n: int = 3
    ) -> list[str]:
        """Извлекает топ повторяющихся сообщений."""
        # Нормализуем сообщения – убираем числа и UUID
        normalized = []
        for e in events:
            if LOG_LEVELS.get(e.level, 0) >= 2:  # WARN+
                msg = re.sub(r'\b[0-9a-f-]{8,}\b', '<id>', e.message)
                msg = re.sub(r'\d+', 'N', msg)
                msg = msg[:100]
                normalized.append(msg)

        counter = Counter(normalized)
        return [f"{msg} (×{cnt})" for msg, cnt in counter.most_common(n)]

    def _detect_anomaly(
        self, container: str, ts: LogTimeSeries
    ) -> tuple[float, bool]:
        """Определяет аномальность по сравнению с baseline."""
        if not ts.error_rate:
            return 0.0, False

        current_rate = np.mean(ts.error_rate[-3:])  # последние 3 интервала

        if container not in self._baseline:
            # Первый раз – устанавливаем baseline
            self._baseline[container] = current_rate
            return 0.0, False

        baseline = self._baseline[container]
        # Обновляем baseline экспоненциально
        self._baseline[container] = 0.9 * baseline + 0.1 * current_rate

        if baseline < 0.01:  # нет ошибок в baseline – любая ошибка аномалия
            score = current_rate * 10
        else:
            score = (current_rate - baseline) / (baseline + 1e-6)

        is_anomalous = score > self._threshold
        return float(score), is_anomalous

    def update_baseline(self) -> None:
        """Сбрасывает baseline для переустановки."""
        self._baseline.clear()


def merge_log_anomalies_with_metrics(
    log_data: LogData,
    metric_scores: dict[str, float],
    boost_factor: float = 0.3,
) -> dict[str, float]:
    """Повышает metric score для сервисов с лог-аномалиями.

    Если сервис аномален и по метрикам, и по логам –
    это сильный сигнал первопричины.

    Args:
        log_data: результат DockerLogConnector.fetch()
        metric_scores: {service_name: ce_score}
        boost_factor: насколько поднимать score (0.3 = +30%)

    Returns:
        Обновлённый dict scores.
    """
    updated = dict(metric_scores)
    for container, log_ts in log_data.series.items():
        if log_ts.is_anomalous and container in updated:
            updated[container] *= (1.0 + boost_factor * log_ts.anomaly_score)
    return updated
