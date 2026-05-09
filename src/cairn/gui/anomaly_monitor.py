"""AnomalyMonitor — автономный мониторинг аномалий для CAIRN.

Цикл работы:
  1. Собирает метрики через LiveSystemConnector каждые poll_interval секунд
  2. Быстро оценивает аномальность через GMM (без полного анализа)
  3. Если NLL > порога — эмитирует сигнал anomaly_detected
  4. Контроллер запускает полный анализ и уведомляет оператора

Использование:
    monitor = AnomalyMonitor(connector, model, threshold=0.5)
    monitor.anomaly_detected.connect(on_anomaly)
    monitor.metrics_updated.connect(on_metrics)
    monitor.start()
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch

from PySide6.QtCore import QThread, Signal


class AnomalyMonitor(QThread):
    """Фоновый поток непрерывного мониторинга аномалий."""

    # Эмитируется при каждом обновлении метрик
    metrics_updated = Signal(object)        # MetricData

    # Эмитируется при обнаружении аномалии
    anomaly_detected = Signal(object, float)  # MetricData, nll_score

    # Статус для отображения в GUI
    status_changed = Signal(str, str)       # message, level ("ok"|"warn"|"error")

    def __init__(
        self,
        connector,
        model=None,
        poll_interval: float = 30.0,
        anomaly_threshold: float = 2.0,     # σ от baseline
        cooldown: float = 180.0,
        min_baseline_cycles: int = 5,       # циклов до первого алерта
        parent=None,
    ):
        super().__init__(parent)
        self._connector           = connector
        self._model               = model
        self._poll_interval       = poll_interval
        self._threshold           = anomaly_threshold
        self._cooldown            = cooldown
        self._min_baseline_cycles = min_baseline_cycles
        self._stop_flag           = False
        self._last_alert_time     = 0.0
        self._context_dim         = 8

        self._nll_history: list[float] = []
        self._baseline_nll: Optional[float] = None
        self._baseline_std: float = 0.0

    def set_model(self, model) -> None:
        self._model = model

    def stop(self) -> None:
        self._stop_flag = True

    def run(self) -> None:
        self.status_changed.emit("Мониторинг запущен", "ok")
        cycle = 0

        while not self._stop_flag:
            cycle += 1
            try:
                self._do_cycle(cycle)
            except Exception as e:
                self.status_changed.emit(f"Ошибка мониторинга: {e}", "warn")

            # Ждём до следующего цикла с возможностью быстро остановиться
            for _ in range(int(self._poll_interval * 10)):
                if self._stop_flag:
                    break
                time.sleep(0.1)

        self.status_changed.emit("Мониторинг остановлен", "ok")

    def _do_cycle(self, cycle: int) -> None:
        """Один цикл: сбор метрик → оценка аномальности."""
        # Сбор метрик
        now = time.time()
        try:
            md = self._connector.fetch_metrics(now - 300, now)
        except Exception as e:
            self.status_changed.emit(f"Ошибка сбора метрик: {e}", "warn")
            return

        if md.n_instances == 0 or len(md.timestamps) < 3:
            self.status_changed.emit("Нет данных от системы", "warn")
            return

        self.metrics_updated.emit(md)

        # Быстрая оценка аномальности
        nll_score = self._compute_anomaly_score(md)
        if nll_score is None:
            # Модель не загружена — используем статистический метод
            nll_score = self._statistical_anomaly_score(md)

        # Обновляем историю для адаптивного порога
        self._nll_history.append(nll_score)
        if len(self._nll_history) > 20:
            self._nll_history.pop(0)

        # Baseline устанавливается после min_baseline_cycles циклов
        n = self._min_baseline_cycles
        if len(self._nll_history) >= n and self._baseline_nll is None:
            baseline_vals      = self._nll_history[-n:]
            self._baseline_nll = float(np.median(baseline_vals))
            self._baseline_std = float(np.std(baseline_vals)) + 1e-6
            self.status_changed.emit(
                f"Baseline установлен: NLL={self._baseline_nll:.3f} "
                f"±{self._baseline_std:.3f} (мониторинг активен)", "ok"
            )

        # Вычисляем эффективный порог
        effective_threshold = self._get_effective_threshold()

        status_msg = (
            f"Цикл {cycle} | "
            f"Экз.: {md.n_instances} | "
            f"Score: {nll_score:.3f} | "
            f"Порог: {effective_threshold:.3f}"
        )

        if nll_score > effective_threshold:
            # Проверяем cooldown
            if now - self._last_alert_time >= self._cooldown:
                self._last_alert_time = now
                self.status_changed.emit(
                    f"АНОМАЛИЯ ОБНАРУЖЕНА (score={nll_score:.3f})", "error"
                )
                self.anomaly_detected.emit(md, nll_score)
            else:
                remaining = int(self._cooldown - (now - self._last_alert_time))
                self.status_changed.emit(
                    f"Аномалия (cooldown {remaining}с) | {status_msg}", "warn"
                )
        else:
            level = "ok" if nll_score < effective_threshold * 0.7 else "warn"
            self.status_changed.emit(status_msg, level)

    def _compute_anomaly_score(self, md) -> Optional[float]:
        """Быстрая оценка через GMM модели CAIRN."""
        if self._model is None:
            return None
        try:
            with torch.no_grad():
                scores = []
                W = 15  # окно StateBuilder
                F = md.n_metrics

                for ni in range(md.n_instances):
                    vals = np.nan_to_num(md.values[:, ni, :F], nan=0.0)
                    T = vals.shape[0]

                    if T >= W:
                        chunk = vals[-W:, :]
                    else:
                        chunk = np.vstack([np.zeros((W - T, F)), vals])

                    # Нормализуем в [0, 1]
                    for fi in range(F):
                        col_max = chunk[:, fi].max()
                        if col_max > 1e-6:
                            chunk[:, fi] /= col_max

                    # Дополняем до 4 метрик если нужно (модель обучена на 4)
                    expected_F = getattr(
                        self._model.state_builder, "n_metrics", 4
                    )
                    if chunk.shape[1] < expected_F:
                        pad = np.zeros((chunk.shape[0], expected_F - chunk.shape[1]))
                        chunk = np.hstack([chunk, pad])
                    elif chunk.shape[1] > expected_F:
                        chunk = chunk[:, :expected_F]

                    m_t = torch.tensor(
                        chunk, dtype=torch.float32
                    ).unsqueeze(0)  # (1, W, expected_F)

                    log_ids = torch.zeros(1, 1, dtype=torch.long)
                    log_len = torch.ones(1, dtype=torch.long)
                    dummy_d = torch.zeros(1, 16, dtype=torch.float32)  # d_met=16

                    H, C  = self._model.state_builder(m_t, log_ids, log_len, dummy_d)
                    nll   = self._model.gmm.nll(H, C)
                    scores.append(float(nll.mean()))

                return float(np.max(scores)) if scores else None
        except Exception:
            return None

    def _statistical_anomaly_score(self, md) -> float:
        """Статистическая оценка без модели — Cohen's d от baseline."""
        if len(md.timestamps) < 6:
            return 0.0

        T    = len(md.timestamps)
        half = T // 2
        scores = []

        for ni in range(md.n_instances):
            for fi in range(md.n_metrics):
                col    = np.nan_to_num(md.values[:, ni, fi], nan=0.0)
                first  = col[:half]
                second = col[half:]
                pooled = np.sqrt(
                    (first.std()**2 + second.std()**2) / 2.0 + 1e-9
                )
                effect = abs(second.mean() - first.mean()) / (pooled + 1e-9)
                scores.append(effect)

        return float(np.mean(scores)) if scores else 0.0

    def _get_effective_threshold(self) -> float:
        """Адаптивный порог: baseline + threshold*σ.
        
        Требует min_baseline_cycles циклов для установки baseline.
        До этого момента алерты не отправляются (возвращаем inf).
        """
        if len(self._nll_history) < self._min_baseline_cycles:
            return float('inf')  # Ещё нет baseline — не алертим
        if self._baseline_nll is not None:
            return self._baseline_nll + max(
                self._threshold * self._baseline_std,
                self._baseline_nll * 0.3  # минимум 30% от baseline
            )
        return self._threshold