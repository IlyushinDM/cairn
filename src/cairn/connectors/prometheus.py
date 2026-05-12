"""Prometheus-коннектор для CAIRN (опциональный).

Зависимость: prometheus-api-client (pip install prometheus-api-client).
Если пакет не установлен – импорт модуля не ломает приложение,
но попытка создать экземпляр вызовет ConnectorUnavailableError.

Пример конфигурации (configs/default.yaml):
    connectors:
      metrics:
        type: prometheus
        url: http://localhost:9090
        metrics:
          - container_cpu_usage_seconds_total
          - container_memory_usage_bytes
        step: 15s
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from cairn.connectors.base import (
    BaseMetricConnector,
    ConnectorUnavailableError,
    MetricData,
)

# Попытка импорта опциональной зависимости
_PROM_AVAILABLE = False
PrometheusConnect = None  # type: ignore[assignment]
try:
    from prometheus_api_client import PrometheusConnect, MetricRangeDataFrame  # type: ignore
    _PROM_AVAILABLE = True
except ImportError:
    pass


class PrometheusMetricConnector(BaseMetricConnector):
    """Коннектор метрик из Prometheus.

    Parameters
    ----------
    url : str
        URL Prometheus (например, http://localhost:9090).
    metric_names : list[str]
        Список PromQL-метрик для опроса.
    step : str
        Шаг выборки (например, "15s", "1m").
    label_filter : dict | None
        Дополнительные label-фильтры для запросов.
        Например: {"namespace": "production"}
    instance_label : str
        Имя label, используемого как имя экземпляра (по умолчанию "pod").
    """

    def __init__(
        self,
        url: str,
        metric_names: list[str],
        step: str = "15s",
        label_filter: Optional[dict] = None,
        instance_label: str = "pod",
    ) -> None:
        if not _PROM_AVAILABLE:
            raise ConnectorUnavailableError(
                "Пакет prometheus-api-client не установлен. "
                "Установите его: pip install prometheus-api-client"
            )

        self._url = url
        self._metric_names = metric_names
        self._step = step
        self._label_filter = label_filter or {}
        self._instance_label = instance_label

        # Проверяем доступность Prometheus
        try:
            self._client = PrometheusConnect(url=url, disable_ssl=True)
            if not self._client.check_prometheus_connection():
                raise ConnectorUnavailableError(
                    f"Prometheus недоступен по адресу {url}. "
                    "Проверьте URL и что сервер запущен."
                )
        except Exception as exc:
            raise ConnectorUnavailableError(
                f"Не удалось подключиться к Prometheus ({url}): {exc}"
            ) from exc

    def available_metrics(self) -> list[str]:
        return list(self._metric_names)

    def available_instances(self) -> list[str]:
        """Возвращает все значения instance_label из первой метрики."""
        if not self._metric_names:
            return []
        label_data = self._client.get_label_values(
            label_name=self._instance_label
        )
        return sorted(label_data)

    def fetch(
        self,
        start_time: float,
        end_time: float,
        instances: Optional[list[str]] = None,
        metrics: Optional[list[str]] = None,
    ) -> MetricData:
        """Выполняет range_query к Prometheus.

        Каждая запрошенная метрика опрашивается отдельным range_query.
        Данные нормализуются на общую временну́ю сетку.
        """
        from datetime import datetime, timezone

        target_metrics = metrics or self._metric_names
        start_dt = datetime.fromtimestamp(start_time, tz=timezone.utc)
        end_dt = datetime.fromtimestamp(end_time, tz=timezone.utc)

        # Собираем данные по всем метрикам
        all_data: dict[str, dict[str, list]] = {}  # {instance: {metric: [(ts, val)]}}

        for metric in target_metrics:
            label_config = dict(self._label_filter)
            try:
                result = self._client.get_metric_range_data(
                    metric_name=metric,
                    label_config=label_config or None,
                    start_time=start_dt,
                    end_time=end_dt,
                    chunk_size=None,
                )
            except Exception as exc:
                raise ConnectorUnavailableError(
                    f"Ошибка запроса к Prometheus (метрика={metric}): {exc}"
                ) from exc

            for series in result:
                inst = series["metric"].get(self._instance_label, "unknown")
                if instances and inst not in instances:
                    continue
                all_data.setdefault(inst, {})
                all_data[inst][metric] = [
                    (float(ts), float(val))
                    for ts, val in series["values"]
                ]

        if not all_data:
            inst_names = instances or []
            return MetricData(
                timestamps=np.array([], dtype=np.float64),
                values=np.zeros((0, len(inst_names), len(target_metrics))),
                instance_names=inst_names,
                metric_names=target_metrics,
            )

        inst_names = sorted(all_data.keys())

        # Строим общую временну́ю сетку из первой доступной серии
        first_inst = inst_names[0]
        first_metric = next(iter(all_data[first_inst]))
        ts_list = [ts for ts, _ in all_data[first_inst][first_metric]]
        timestamps = np.array(ts_list, dtype=np.float64)
        T = len(timestamps)
        ts_idx = {ts: i for i, ts in enumerate(ts_list)}

        values = np.full((T, len(inst_names), len(target_metrics)), np.nan)

        for ni, inst in enumerate(inst_names):
            for mi, met in enumerate(target_metrics):
                for ts, val in all_data.get(inst, {}).get(met, []):
                    if ts in ts_idx:
                        values[ts_idx[ts], ni, mi] = val

        return MetricData(timestamps, values, inst_names, target_metrics)
