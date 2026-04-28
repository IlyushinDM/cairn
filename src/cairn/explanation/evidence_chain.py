"""Проверяемая цепочка доказательств (раздел 4.1).

Структурированная запись причинного пути от первопричины к симптомам
с аннотацией типа и силы каждой связи.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class NodeAnnotation:
    """Аннотация вершины в цепочке доказательств."""
    node_idx: int
    node_name: str
    nll: float                     # Аномальность
    dominant_metric: Optional[str] = None
    failure_type: Optional[str] = None


@dataclass
class EdgeAnnotation:
    """Аннотация ребра в цепочке доказательств."""
    src: int
    dst: int
    edge_type: str                 # "call" | "colocation" | "loadbalance" | "adaptive"
    strength: float                # Контрфактическая тестовая статистика τ(e)
    has_confounder: bool = False   # Флаг скрытого общего фактора


@dataclass
class EvidenceChain:
    """Проверяемая цепочка доказательств.

    Атрибуты
    ----------
    root_cause_idx : int
        Индекс первопричины.
    path_nodes : List[NodeAnnotation]
        Узлы от первопричины к симптомам.
    path_edges : List[EdgeAnnotation]
        Рёбра пути.
    causal_effect : float
        ПЭ(root) — снижение аномальности при нормализации первопричины.
    confidence : float
        Доля выполненных аксиом верификатора (0–1).
    confounder_warnings : List[str]
        Предупреждения о скрытых факторах.
    drift_warning : bool
        Предупреждение о дрейфе распределения.
    """

    root_cause_idx: int
    path_nodes: List[NodeAnnotation] = field(default_factory=list)
    path_edges: List[EdgeAnnotation] = field(default_factory=list)
    causal_effect: float = 0.0
    confidence: float = 1.0
    confounder_warnings: List[str] = field(default_factory=list)
    drift_warning: bool = False

    def to_dict(self) -> dict:
        return {
            "root_cause": self.root_cause_idx,
            "causal_effect": self.causal_effect,
            "confidence": self.confidence,
            "path": [
                {
                    "node": n.node_idx,
                    "name": n.node_name,
                    "nll": round(n.nll, 4),
                    "dominant_metric": n.dominant_metric,
                    "failure_type": n.failure_type,
                }
                for n in self.path_nodes
            ],
            "edges": [
                {
                    "src": e.src,
                    "dst": e.dst,
                    "type": e.edge_type,
                    "strength": round(e.strength, 4),
                    "has_confounder": e.has_confounder,
                }
                for e in self.path_edges
            ],
            "warnings": {
                "confounders": self.confounder_warnings,
                "drift": self.drift_warning,
            },
        }

    def summary(self) -> str:
        """Краткое текстовое описание цепочки."""
        path_str = " → ".join(n.node_name for n in self.path_nodes)
        lines = [
            f"Первопричина: {self.path_nodes[0].node_name if self.path_nodes else self.root_cause_idx}",
            f"Причинный эффект: {self.causal_effect:.1%}",
            f"Достоверность: {self.confidence:.0%}",
            f"Путь распространения: {path_str}",
        ]
        if self.confounder_warnings:
            for w in self.confounder_warnings:
                lines.append(f"⚠ {w}")
        if self.drift_warning:
            lines.append("⚠ Обнаружен дрейф распределения — достоверность снижена.")
        return "\n".join(lines)
