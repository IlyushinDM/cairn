"""Система обучения CAIRN — потери, загрузчик данных, тренер.

Публичный API:
    CAIRNLoss             — составная функция потерь (6 компонент)
    LossWeights           — веса λ₁–λ₆
    Incident              — один обучающий пример
    CAIRNDataset          — набор данных (torch.utils.data.Dataset)
    IncidentBuilder       — строит Incident из MetricData + LogData + TraceData
    create_demo_dataset   — датасет из демо-файлов data/sample/
    collate_incidents     — collate-функция для DataLoader
    CAIRNModel            — обёртка над всеми обучаемыми модулями
    CAIRNTrainer          — трёхэтапный тренер
    TrainerConfig         — гиперпараметры обучения
    compute_metrics       — AC@k, Avg@5, F1
"""

from cairn.training.loss import CAIRNLoss, LossWeights
from cairn.training.data_loader import (
    Incident,
    CAIRNDataset,
    IncidentBuilder,
    create_demo_dataset,
    collate_incidents,
)
from cairn.training.trainer import (
    CAIRNModel,
    CAIRNTrainer,
    TrainerConfig,
    compute_metrics,
)

__all__ = [
    "CAIRNLoss", "LossWeights",
    "Incident", "CAIRNDataset", "IncidentBuilder",
    "create_demo_dataset", "collate_incidents",
    "CAIRNModel", "CAIRNTrainer", "TrainerConfig", "compute_metrics",
]
