"""Загрузчик данных для обучения CAIRN (раздел 5.2).

Иерархия:
  Incident        – одна размеченная аномалия с тензорами и метками
  CAIRNDataset    – набор инцидентов (torch.utils.data.Dataset)
  IncidentBuilder – строит Incident из MetricData + LogData + TraceData
  create_demo_dataset – создаёт датасет из демо-файлов
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Структура одного инцидента
# ---------------------------------------------------------------------------

@dataclass
class Incident:
    """Один обучающий пример – инцидент или нормальное окно.

    Атрибуты
    ----------
    metric_data  : Tensor (N, T, F)  – временные ряды метрик
    log_data     : Tensor (N, L)     – ID шаблонов журналов
    trace_data   : Tensor (N,)       – глубины вызовов
    context      : Tensor (N, C)     – контекстные векторы (или нули)
    root_cause   : int               – индекс первопричины (-1 = нормальное окно)
    fault_type   : str               – тип сбоя ("normal" для нормальных окон)
    instance_names: list[str]        – имена экземпляров (длина N)
    is_anomaly   : bool
    """
    metric_data:    torch.Tensor
    log_data:       torch.Tensor
    trace_data:     torch.Tensor
    context:        torch.Tensor
    root_cause:     int               = -1
    fault_type:     str               = "normal"
    instance_names: List[str]         = field(default_factory=list)
    is_anomaly:     bool              = False

    @property
    def n_instances(self) -> int:
        return self.metric_data.shape[0]

    @property
    def n_timesteps(self) -> int:
        return self.metric_data.shape[1]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CAIRNDataset(Dataset):
    """Набор данных для обучения CAIRN.

    Параметры
    ----------
    incidents : list[Incident]
    normal_only : bool
        Если True – возвращает только нормальные окна (для претрейна).
    anomaly_only : bool
        Если True – возвращает только аномальные окна (для основного этапа).
    """

    def __init__(
        self,
        incidents: List[Incident],
        normal_only: bool = False,
        anomaly_only: bool = False,
    ) -> None:
        if normal_only:
            self._data = [inc for inc in incidents if not inc.is_anomaly]
        elif anomaly_only:
            self._data = [inc for inc in incidents if inc.is_anomaly]
        else:
            self._data = list(incidents)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Incident:
        return self._data[idx]

    def normal_subset(self) -> "CAIRNDataset":
        return CAIRNDataset(self._data, normal_only=True)

    def anomaly_subset(self) -> "CAIRNDataset":
        return CAIRNDataset(self._data, anomaly_only=True)

    @property
    def n_normal(self) -> int:
        return sum(1 for inc in self._data if not inc.is_anomaly)

    @property
    def n_anomaly(self) -> int:
        return sum(1 for inc in self._data if inc.is_anomaly)

    def summary(self) -> str:
        return (
            f"CAIRNDataset: {len(self)} инцидентов "
            f"({self.n_normal} нормальных, {self.n_anomaly} аномальных)"
        )

    @property
    def incidents(self) -> List["Incident"]:
        """Публичный доступ к списку инцидентов."""
        return list(self._data)


def collate_incidents(batch: List[Incident]):
    """Collate-функция для DataLoader.

    Так как у инцидентов может быть одинаковое N (число сервисов),
    стекаем в батч по первому измерению.
    """
    # Стекаем тензоры по batch-измерению
    metrics  = torch.stack([inc.metric_data for inc in batch])    # (B, N, T, F)
    logs     = torch.stack([inc.log_data    for inc in batch])    # (B, N, L)
    traces   = torch.stack([inc.trace_data  for inc in batch])    # (B, N)
    contexts = torch.stack([inc.context     for inc in batch])    # (B, N, C)
    root_causes = torch.tensor([inc.root_cause for inc in batch], dtype=torch.long)
    is_anomaly  = torch.tensor([inc.is_anomaly for inc in batch], dtype=torch.bool)
    return {
        "metric_data":    metrics,
        "log_data":       logs,
        "trace_data":     traces,
        "context":        contexts,
        "root_cause":     root_causes,
        "is_anomaly":     is_anomaly,
        "fault_types":    [inc.fault_type for inc in batch],
        "instance_names": batch[0].instance_names,  # все одинаковые в батче
    }


# ---------------------------------------------------------------------------
# IncidentBuilder
# ---------------------------------------------------------------------------

class IncidentBuilder:
    """Строит Incident из MetricData + LogData + TraceData + метаданных.

    Параметры
    ----------
    window_size : int
        Число временных шагов T в одном инциденте.
    max_log_len : int
        Максимальная длина последовательности журналов.
    context_dim : int
        Размерность контекстного вектора.
    """

    def __init__(
        self,
        window_size: int = 60,
        max_log_len: int = 20,
        context_dim: int = 16,
    ) -> None:
        self.window_size = window_size
        self.max_log_len = max_log_len
        self.context_dim = context_dim

    def build(
        self,
        metric_data,            # MetricData
        log_data,               # LogData
        trace_data,             # list[TraceData]
        tokenizer,              # DrainTokenizer
        root_cause_name: Optional[str] = None,
        fault_type: str = "normal",
        t_start: int = 0,
    ) -> Incident:
        """Строит один Incident из данных за окно [t_start, t_start+window_size).

        Параметры
        ----------
        metric_data   : MetricData с >= t_start + window_size временными шагами
        log_data      : LogData за этот временной интервал
        trace_data    : list[TraceData] за этот интервал
        tokenizer     : DrainTokenizer для преобразования журналов
        root_cause_name : имя экземпляра-первопричины (или None)
        fault_type    : тип сбоя
        t_start       : начальный временной шаг
        """
        N = metric_data.n_instances
        T = self.window_size
        F = metric_data.n_metrics
        L = self.max_log_len

        # Метрики: срез окна (N, T, F)
        t_end = min(t_start + T, metric_data.n_timesteps)
        win   = metric_data.values[t_start:t_end]          # (t_len, N, F)
        if win.shape[0] < T:
            # Дополняем нулями если данных меньше окна
            pad = np.zeros((T - win.shape[0], N, F))
            win = np.concatenate([win, pad], axis=0)
        metrics_t = torch.tensor(win.transpose(1, 0, 2), dtype=torch.float32)
        metrics_t = torch.nan_to_num(metrics_t)             # (N, T, F)

        # Журналы: по экземплярам (N, L)
        log_ids = []
        for inst_name in metric_data.instance_names:
            msgs = log_data.filter_instance(inst_name).messages
            ids  = [tokenizer.transform_one(m) for m in msgs[:L]] or [0]
            ids  = ids[:L] + [0] * (L - len(ids))
            log_ids.append(ids)
        log_t = torch.tensor(log_ids, dtype=torch.long)     # (N, L)

        # Трассировки: глубина вызовов (N,)
        depths = [0] * N
        for tr in trace_data:
            for span in tr.spans:
                if span.instance in metric_data.instance_names:
                    idx = metric_data.instance_names.index(span.instance)
                    depths[idx] = max(depths[idx], 1 if span.parent_span_id else 0)
        depths_t = torch.tensor(depths, dtype=torch.long)   # (N,)

        # Контекст: агрегированные метрики окна (N, context_dim)
        # Среднее по времени нормализованное → даёт GMM дифференцирующий сигнал
        ctx_raw = np.nanmean(win, axis=0)                           # (N, F)
        ctx_raw = np.nan_to_num(ctx_raw, nan=0.0)
        # Нормируем каждый узел относительно максимума по метрикам
        row_max = np.abs(ctx_raw).max(axis=1, keepdims=True) + 1e-8
        ctx_norm = ctx_raw / row_max                               # (N, F) в [-1, 1]
        # Вписываем в context_dim (F ≤ context_dim обычно)
        if ctx_norm.shape[1] >= self.context_dim:
            ctx_np = ctx_norm[:, :self.context_dim]
        else:
            pad    = np.zeros((N, self.context_dim - ctx_norm.shape[1]))
            ctx_np = np.concatenate([ctx_norm, pad], axis=1)
        ctx_t = torch.tensor(ctx_np, dtype=torch.float32)          # (N, context_dim)

        # Индекс первопричины
        root_idx = -1
        if root_cause_name and root_cause_name in metric_data.instance_names:
            root_idx = metric_data.instance_names.index(root_cause_name)

        is_anomaly = (root_idx >= 0)

        return Incident(
            metric_data=metrics_t,
            log_data=log_t,
            trace_data=depths_t,
            context=ctx_t,
            root_cause=root_idx,
            fault_type=fault_type,
            instance_names=list(metric_data.instance_names),
            is_anomaly=is_anomaly,
        )


# ---------------------------------------------------------------------------
# Фабрика датасета из демо-данных
# ---------------------------------------------------------------------------

def create_demo_dataset(
    sample_dir: str | Path,
    window_size: int = 60,
    max_log_len: int = 20,
    stride: int = 10,
    seed: int = 42,
) -> "CAIRNDataset":
    """Создаёт датасет из демо-файлов data/sample/.

    Скользящим окном обходит:
      - нормальный период (t=0..199): создаёт нормальные инциденты
      - аномальный период (t=200..299): создаёт аномальные инциденты

    Параметры
    ----------
    sample_dir : путь к папке с metrics.csv, logs.txt, traces.json, labels.json
    window_size : размер окна T
    stride : шаг скользящего окна
    seed : для воспроизводимости перемешивания
    """
    from cairn.connectors.csv_file import (
        CSVMetricConnector, FileLogConnector,
        JSONTraceConnector,
    )
    from cairn.perception.log_encoder import DrainTokenizer

    sample_dir = Path(sample_dir)

    # Метаданные инцидентов
    labels_path = sample_dir / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}

    base_ts       = labels.get("normal_period", {}).get("start", 1_700_000_000.0)
    anomaly_start = labels.get("anomaly_period", {}).get("start", base_ts + 200)
    anomaly_end   = labels.get("anomaly_period", {}).get("end",   base_ts + 299)
    normal_end    = labels.get("normal_period",  {}).get("end",   base_ts + 199)
    root_inst     = labels.get("root_cause", {}).get("instance",  None)
    fault_type    = labels.get("root_cause", {}).get("type",      "unknown")

    metric_conn = CSVMetricConnector(sample_dir / "metrics.csv")
    log_conn    = FileLogConnector(sample_dir / "logs.txt")
    trace_conn  = JSONTraceConnector(sample_dir / "traces.json")

    # Загружаем всё единожды
    md_normal = metric_conn.fetch(base_ts, normal_end)
    md_anomaly= metric_conn.fetch(anomaly_start, anomaly_end)
    ld_all    = log_conn.fetch(base_ts, anomaly_end)
    tr_all    = trace_conn.fetch(base_ts, anomaly_end)

    # Обучаем токенизатор на всех сообщениях
    tokenizer = DrainTokenizer(sim_threshold=0.5, max_templates=300)
    tokenizer.fit_transform(ld_all.messages)

    builder = IncidentBuilder(
        window_size=window_size,
        max_log_len=max_log_len,
    )
    incidents: List[Incident] = []

    # Нормальные окна
    T_normal = md_normal.n_timesteps
    for t in range(0, T_normal - window_size + 1, stride):
        ts_win_start = base_ts + t
        ts_win_end   = ts_win_start + window_size
        ld_win = log_conn.fetch(ts_win_start, ts_win_end)
        tr_win = [tr for tr in tr_all if ts_win_start <= tr.start_time <= ts_win_end]
        inc = builder.build(
            _slice_metric_data(md_normal, t, t + window_size),
            ld_win, tr_win, tokenizer,
            root_cause_name=None,
            fault_type="normal",
            t_start=0,
        )
        incidents.append(inc)

    # Аномальные окна
    T_anomaly = md_anomaly.n_timesteps
    for t in range(0, T_anomaly - window_size + 1, stride):
        ts_win_start = anomaly_start + t
        ts_win_end   = ts_win_start + window_size
        ld_win = log_conn.fetch(ts_win_start, ts_win_end)
        tr_win = [tr for tr in tr_all if ts_win_start <= tr.start_time <= ts_win_end]
        # Только если окно полностью в аномальном периоде
        if ts_win_end > anomaly_end:
            break
        inc = builder.build(
            _slice_metric_data(md_anomaly, t, t + window_size),
            ld_win, tr_win, tokenizer,
            root_cause_name=root_inst,
            fault_type=fault_type,
            t_start=0,
        )
        incidents.append(inc)

    random.seed(seed)
    random.shuffle(incidents)
    return CAIRNDataset(incidents)


def _slice_metric_data(md, t_start: int, t_end: int):
    """Создаёт срез MetricData по временной оси."""
    from cairn.connectors.base import MetricData
    t_end = min(t_end, md.n_timesteps)
    return MetricData(
        timestamps=md.timestamps[t_start:t_end],
        values=md.values[t_start:t_end],
        instance_names=list(md.instance_names),
        metric_names=list(md.metric_names),
    )
