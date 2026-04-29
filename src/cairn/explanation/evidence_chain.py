"""Проверяемая цепочка доказательств и её построитель (раздел 4.1).

EvidenceChain       — структура данных: аннотированный путь от первопричины к симптомам.
EvidenceChainBuilder — строит цепочку из результатов фазы рассуждения:
    root_cause, CausalHypergraph, NLL-оценки, CE-оценки, метаданные узлов.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Аннотации элементов цепочки
# ---------------------------------------------------------------------------

@dataclass
class NodeAnnotation:
    """Аннотация вершины в цепочке доказательств."""
    node_idx: int
    node_name: str
    nll: float                          # Аномальность (выше = хуже)
    causal_effect: float = 0.0          # CE(i) — вклад в снижение аномальности
    dominant_metric: Optional[str] = None
    failure_type: Optional[str] = None  # cpu_exhaustion | latency | memory | network


@dataclass
class EdgeAnnotation:
    """Аннотация ребра в цепочке доказательств."""
    src: int
    dst: int
    edge_type: str                      # call | colocation | loadbalance | adaptive
    strength: float                     # τ(e) — контрфактическая значимость ребра
    has_confounder: bool = False        # скрытый общий фактор обнаружен


@dataclass
class EvidenceChain:
    """Проверяемая цепочка доказательств.

    Атрибуты
    ----------
    root_cause_idx : int
    path_nodes : list[NodeAnnotation]   — от первопричины к симптомам
    path_edges : list[EdgeAnnotation]   — рёбра пути
    causal_effect : float               — ПЭ(root)
    confidence : float                  — доля выполненных аксиом [0, 1]
    confounder_warnings : list[str]
    drift_warning : bool
    """

    root_cause_idx: int
    path_nodes: List[NodeAnnotation] = field(default_factory=list)
    path_edges: List[EdgeAnnotation] = field(default_factory=list)
    causal_effect: float = 0.0
    confidence: float = 1.0
    confounder_warnings: List[str] = field(default_factory=list)
    drift_warning: bool = False

    def to_dict(self) -> dict:
        """Словарь для GUI и API."""
        return {
            "root_cause": self.root_cause_idx,
            "root_name": self.path_nodes[0].node_name if self.path_nodes else str(self.root_cause_idx),
            "causal_effect": round(self.causal_effect, 4),
            "confidence": round(self.confidence, 4),
            "path": [
                {
                    "node": n.node_idx,
                    "name": n.node_name,
                    "nll": round(n.nll, 4),
                    "causal_effect": round(n.causal_effect, 4),
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
        """Краткое текстовое описание для логов и отладки."""
        root = self.path_nodes[0].node_name if self.path_nodes else str(self.root_cause_idx)
        path = " → ".join(n.node_name for n in self.path_nodes)
        lines = [
            f"Первопричина: {root}",
            f"Тип сбоя:     {self.path_nodes[0].failure_type or 'неизвестен'}",
            f"Причинный эффект: {self.causal_effect:.1%}",
            f"Достоверность: {self.confidence:.0%}",
            f"Путь распространения: {path}",
        ]
        for w in self.confounder_warnings:
            lines.append(f"⚠  {w}")
        if self.drift_warning:
            lines.append("⚠  Дрейф распределения — достоверность снижена.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# EvidenceChainBuilder
# ---------------------------------------------------------------------------

# Эвристика: какая метрика доминирует при каком типе сбоя
_FAILURE_HEURISTICS: List[Tuple[str, str, float]] = [
    ("cpu_exhaustion", "cpu",        0.7),
    ("memory_pressure","memory",     0.7),
    ("latency_spike",  "latency_ms", 0.6),
    ("overload",       "rps",        0.5),
]

_FAULT_TYPE_RECOMMENDATIONS: Dict[str, str] = {
    "cpu_exhaustion":  "Проверьте загрузку CPU, горизонтально масштабируйте сервис или оптимизируйте код.",
    "memory_pressure": "Проверьте утечки памяти, увеличьте memory limit или перезапустите сервис.",
    "latency_spike":   "Проверьте зависимости upstream, оцените очереди и тайм-ауты.",
    "overload":        "Включите circuit breaker, добавьте rate limiting или реплики.",
    "unknown":         "Проверьте метрики и журналы сервиса для определения причины.",
}


class EvidenceChainBuilder:
    """Строит проверяемую цепочку доказательств из результатов фазы рассуждения.

    Параметры
    ----------
    confounder_threshold : float
        Минимальная корреляция конфаундера для установки флага.
    max_path_depth : int
        Максимальная глубина BFS для поиска пути распространения.
    """

    def __init__(
        self,
        confounder_threshold: float = 0.3,
        max_path_depth: int = 6,
    ) -> None:
        self.confounder_threshold = confounder_threshold
        self.max_path_depth = max_path_depth

    def build(
        self,
        root_cause: int,
        causal_graph,                             # CausalHypergraph
        ce_scores: Dict[int, float],              # {node_idx: CE}
        nll_scores: Dict[int, float],             # {node_idx: NLL}
        node_names: Optional[List[str]] = None,   # [name_0, name_1, ...]
        metadata: Optional[Dict[int, dict]] = None,  # {idx: {dominant_metric, failure_type, ...}}
        confounder_flags: Optional[Dict[int, bool]] = None,
        anomaly_threshold: float = 0.5,
        verification_confidence: float = 1.0,
        drift_warning: bool = False,
    ) -> "EvidenceChain":
        """Строит EvidenceChain из компонентов фазы рассуждения.

        Параметры
        ----------
        root_cause : int
            Индекс первопричины (из CascadeFunnel).
        causal_graph : CausalHypergraph
            Граф из HypergraphBuilder.
        ce_scores : dict[int, float]
            Причинный эффект каждого узла (из CounterfactualModule.rank_candidates).
        nll_scores : dict[int, float]
            NLL каждого узла (из ConditionalGMM.nll).
        node_names : list[str] | None
            Имена узлов. Если None — берутся из causal_graph.instance_names.
        metadata : dict | None
            Доп. метаданные: {idx: {dominant_metric, failure_type}}.
        confounder_flags : dict | None
            {idx: True} если у узла обнаружен скрытый фактор.
        anomaly_threshold : float
            Порог δ для пометки узла как аномального.
        verification_confidence : float
            Достоверность из CausalGraphVerifier.
        drift_warning : bool
            Флаг дрейфа из ConditionalGMM.detect_drift.
        """
        names = node_names or causal_graph.instance_names
        meta  = metadata or {}
        cflag = confounder_flags or {}

        # ----------------------------------------------------------------
        # BFS: находим путь распространения от root_cause
        # ----------------------------------------------------------------
        adjacency = causal_graph.adjacency_matrix()   # (N, N) — неориентированный
        call_edges = {
            (e.members[0], e.members[1])
            for e in causal_graph.edges
            if e.edge_type == "call" and len(e.members) >= 2
        }

        path_nodes_idx, path_edges_raw = self._bfs_propagation(
            root_cause, adjacency, nll_scores, anomaly_threshold, call_edges
        )

        # ----------------------------------------------------------------
        # Аннотируем узлы
        # ----------------------------------------------------------------
        annotated_nodes: List[NodeAnnotation] = []
        for idx in path_nodes_idx:
            m = meta.get(idx, {})
            dominant, ftype = self._infer_fault(idx, m, nll_scores, ce_scores)
            annotated_nodes.append(NodeAnnotation(
                node_idx=idx,
                node_name=names[idx] if idx < len(names) else f"node-{idx}",
                nll=nll_scores.get(idx, 0.0),
                causal_effect=ce_scores.get(idx, 0.0),
                dominant_metric=m.get("dominant_metric", dominant),
                failure_type=m.get("failure_type", ftype),
            ))

        # ----------------------------------------------------------------
        # Аннотируем рёбра
        # ----------------------------------------------------------------
        annotated_edges: List[EdgeAnnotation] = []
        for src, dst, etype, strength in path_edges_raw:
            has_cf = cflag.get(src, False) or cflag.get(dst, False)
            annotated_edges.append(EdgeAnnotation(
                src=src, dst=dst,
                edge_type=etype,
                strength=strength,
                has_confounder=has_cf,
            ))

        # ----------------------------------------------------------------
        # Предупреждения о конфаундерах
        # ----------------------------------------------------------------
        warnings: List[str] = []
        for idx in path_nodes_idx:
            if cflag.get(idx, False):
                name = names[idx] if idx < len(names) else f"node-{idx}"
                warnings.append(
                    f"У узла '{name}' обнаружен скрытый общий фактор "
                    f"(конфаундер). Причинно-следственная связь может быть частичной."
                )

        return EvidenceChain(
            root_cause_idx=root_cause,
            path_nodes=annotated_nodes,
            path_edges=annotated_edges,
            causal_effect=ce_scores.get(root_cause, 0.0),
            confidence=verification_confidence,
            confounder_warnings=warnings,
            drift_warning=drift_warning,
        )

    def to_text(self, chain: "EvidenceChain") -> str:
        """Текстовое представление цепочки для отчёта (делегирует TemplateGenerator)."""
        from cairn.explanation.text_generator import TemplateTextGenerator
        return TemplateTextGenerator().generate(chain)

    def to_dict(self, chain: "EvidenceChain") -> dict:
        """Словарь для передачи в GUI."""
        return chain.to_dict()

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    def _bfs_propagation(
        self,
        root: int,
        adjacency,          # torch.Tensor (N, N)
        nll_scores: Dict[int, float],
        anomaly_threshold: float,
        call_edges: set,
    ) -> Tuple[List[int], List[Tuple[int, int, str, float]]]:
        """BFS от root по аномальным соседям — строит путь распространения."""
        visited   = {root}
        queue     = deque([(root, 0)])
        path_idxs = [root]
        path_edges: List[Tuple[int, int, str, float]] = []

        while queue:
            node, depth = queue.popleft()
            if depth >= self.max_path_depth:
                continue
            # Соседи с NLL выше порога (аномальные)
            neighbors = adjacency[node].nonzero(as_tuple=True)[0].tolist()
            for nbr in sorted(neighbors, key=lambda n: -nll_scores.get(n, 0.0)):
                if nbr in visited:
                    continue
                nll_nbr = nll_scores.get(nbr, 0.0)
                if nll_nbr < anomaly_threshold:
                    continue
                visited.add(nbr)
                queue.append((nbr, depth + 1))
                path_idxs.append(nbr)
                etype = "call" if (node, nbr) in call_edges or (nbr, node) in call_edges else "colocation"
                # Сила ребра: произведение NLL узлов, нормализованное
                strength = min(1.0, nll_scores.get(node, 0.0) * nll_scores.get(nbr, 0.0) / 100.0)
                path_edges.append((node, nbr, etype, round(strength, 4)))

        return path_idxs, path_edges

    def _infer_fault(
        self,
        idx: int,
        meta: dict,
        nll_scores: Dict[int, float],
        ce_scores: Dict[int, float],
    ) -> Tuple[Optional[str], Optional[str]]:
        """Эвристически определяет доминирующую метрику и тип сбоя."""
        # Если метаданные уже содержат ответ — используем их
        if "dominant_metric" in meta and "failure_type" in meta:
            return meta["dominant_metric"], meta["failure_type"]

        # Эвристика по NLL (отсутствие реальных метрик → возвращаем None)
        nll = nll_scores.get(idx, 0.0)
        ce  = ce_scores.get(idx, 0.0)

        # Смотрим на raw_metrics в metadata
        raw = meta.get("metrics", {})
        dominant: Optional[str] = None
        fault:    Optional[str] = None

        if raw:
            dominant = max(raw, key=lambda k: abs(raw[k]))
            for ftype, metric, _ in _FAILURE_HEURISTICS:
                if metric == dominant:
                    fault = ftype
                    break

        return dominant, fault
