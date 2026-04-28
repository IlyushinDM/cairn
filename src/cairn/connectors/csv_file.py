"""CSV-коннектор для работы с демо-данными (data/sample/).

Читает metrics.csv, logs.txt и traces.json.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from cairn.connectors.base import BaseConnector


class CSVFileConnector(BaseConnector):
    """Коннектор к локальным файлам CSV / TXT / JSON.

    Параметры
    ----------
    metrics_path : str
        Путь к metrics.csv (колонки: timestamp, service_id, metric_name, value).
    logs_path : str
        Путь к logs.txt (каждая строка — одна запись журнала с временной меткой).
    traces_path : str
        Путь к traces.json (список словарей с полями: timestamp, path, depth).
    """

    def __init__(
        self,
        metrics_path: str = "data/sample/metrics.csv",
        logs_path: str = "data/sample/logs.txt",
        traces_path: str = "data/sample/traces.json",
    ) -> None:
        self.metrics_path = Path(metrics_path)
        self.logs_path = Path(logs_path)
        self.traces_path = Path(traces_path)

    def fetch_metrics(self, start_ts: float, end_ts: float) -> Dict[str, Any]:
        """Возвращает метрики за интервал [start_ts, end_ts].

        Формат возврата:
        {
            service_id: {metric_name: [value, ...]}
        }
        """
        result: Dict[str, Dict[str, List[float]]] = {}
        if not self.metrics_path.exists():
            return result

        with self.metrics_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = float(row.get("timestamp", 0))
                if not (start_ts <= ts <= end_ts):
                    continue
                sid = row.get("service_id", "unknown")
                name = row.get("metric_name", "unknown")
                val = float(row.get("value", 0))
                result.setdefault(sid, {}).setdefault(name, []).append(val)
        return result

    def fetch_logs(self, start_ts: float, end_ts: float) -> List[str]:
        """Возвращает строки журнала за интервал [start_ts, end_ts].

        Формат строки: «<timestamp> <level> <message>»
        """
        lines = []
        if not self.logs_path.exists():
            return lines

        with self.logs_path.open(encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if not parts:
                    continue
                try:
                    ts = float(parts[0])
                except ValueError:
                    continue
                if start_ts <= ts <= end_ts:
                    lines.append(line.strip())
        return lines

    def fetch_traces(self, start_ts: float, end_ts: float) -> List[Dict[str, Any]]:
        """Возвращает трассировки за интервал [start_ts, end_ts]."""
        if not self.traces_path.exists():
            return []

        with self.traces_path.open(encoding="utf-8") as f:
            raw: List[Dict[str, Any]] = json.load(f)

        return [
            t for t in raw
            if start_ts <= float(t.get("timestamp", 0)) <= end_ts
        ]
