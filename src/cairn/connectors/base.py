"""Базовый класс коннектора к источнику данных."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseConnector(ABC):
    """Интерфейс коннектора к источнику данных мониторинга."""

    @abstractmethod
    def fetch_metrics(self, start_ts: float, end_ts: float) -> Dict[str, Any]:
        """Загружает метрики за указанный временной интервал."""
        ...

    @abstractmethod
    def fetch_logs(self, start_ts: float, end_ts: float) -> List[str]:
        """Загружает записи журналов."""
        ...

    @abstractmethod
    def fetch_traces(self, start_ts: float, end_ts: float) -> List[Dict[str, Any]]:
        """Загружает трассировки запросов."""
        ...
