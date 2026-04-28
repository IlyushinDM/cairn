"""Верификатор причинного графа по пяти аксиомам (раздел 3.6, таблица 2).

Аксиомы:
  1. Ацикличность             — алгоритм SCС
  2. Темпоральная согласованность — начало аномалии причины ≤ начало следствия + Δ_доп
  3. Транзитивность           — прямой эффект не пренебрежимо мал
  4. Согласованность с топологией — ребро подтверждено физической связью
  5. Монотонность вмешательства   — ПЭ(первопричина) ≥ ПЭ(промежуточных)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch


class AxiomStatus(str, Enum):
    OK = "ok"
    VIOLATED = "violated"
    WARNING = "warning"


@dataclass
class AxiomResult:
    name: str
    status: AxiomStatus
    message: str = ""
    affected_edges: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class VerificationReport:
    axiom_results: List[AxiomResult] = field(default_factory=list)
    confidence: float = 1.0   # доля выполненных аксиом

    def summary(self) -> str:
        lines = [f"Достоверность вывода: {self.confidence:.2f}"]
        for ar in self.axiom_results:
            lines.append(f"  [{ar.status.value.upper()}] {ar.name}: {ar.message}")
        return "\n".join(lines)


class CausalGraphVerifier:
    """Верификатор причинного графа (раздел 3.6).

    Параметры
    ----------
    temporal_tolerance_sec : float
        Δ_доп — допуск на задержку сбора данных (формула 3.33).
    transitivity_threshold : float
        Порог транзитивности.
    monotonicity_epsilon : float
        ε для проверки монотонности.
    """

    def __init__(
        self,
        temporal_tolerance_sec: float = 15.0,
        transitivity_threshold: float = 0.3,
        monotonicity_epsilon: float = 0.05,
    ) -> None:
        self.temporal_tolerance = temporal_tolerance_sec
        self.trans_threshold = transitivity_threshold
        self.mono_eps = monotonicity_epsilon

    def verify(
        self,
        edges: List[Tuple[int, int]],              # (src, dst) — причинные рёбра
        causal_effects: Dict[int, float],          # ПЭ(i) для каждого узла
        anomaly_times: Dict[int, float],           # t_нач(i) для каждого узла
        physical_edges: set[Tuple[int, int]],      # рёбра, подтверждённые топологией
        root_candidate: int,
        path_effects: Optional[Dict[Tuple[int, int], float]] = None,  # ПЭ(i→j)
    ) -> VerificationReport:
        """Запускает все пять проверок и возвращает отчёт."""
        results: List[AxiomResult] = []

        results.append(self._check_acyclicity(edges))
        results.append(self._check_temporal(edges, anomaly_times))
        results.append(self._check_transitivity(edges, path_effects or {}))
        results.append(self._check_topology(edges, physical_edges))
        results.append(self._check_monotonicity(edges, causal_effects, root_candidate))

        ok_count = sum(1 for r in results if r.status == AxiomStatus.OK)
        confidence = ok_count / max(len(results), 1)

        return VerificationReport(axiom_results=results, confidence=confidence)

    # ------------------------------------------------------------------
    # Аксиома 1: Ацикличность
    # ------------------------------------------------------------------
    def _check_acyclicity(self, edges: List[Tuple[int, int]]) -> AxiomResult:
        """Поиск циклов через DFS (алгоритм на основе раскраски)."""
        from collections import defaultdict
        adj: dict[int, list[int]] = defaultdict(list)
        nodes = set()
        for src, dst in edges:
            adj[src].append(dst)
            nodes |= {src, dst}

        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in nodes}
        cycle_edges: List[Tuple[int, int]] = []

        def dfs(v: int) -> bool:
            color[v] = GRAY
            for u in adj[v]:
                if color[u] == GRAY:
                    cycle_edges.append((v, u))
                    return True
                if color[u] == WHITE and dfs(u):
                    return True
            color[v] = BLACK
            return False

        has_cycle = any(dfs(n) for n in nodes if color[n] == WHITE)

        if has_cycle:
            return AxiomResult(
                "Ацикличность", AxiomStatus.VIOLATED,
                "Обнаружен цикл; ребро с наименьшей значимостью рекомендовано к удалению.",
                cycle_edges,
            )
        return AxiomResult("Ацикличность", AxiomStatus.OK, "Граф ациклический.")

    # ------------------------------------------------------------------
    # Аксиома 2: Темпоральная согласованность (формула 3.33)
    # ------------------------------------------------------------------
    def _check_temporal(
        self,
        edges: List[Tuple[int, int]],
        anomaly_times: Dict[int, float],
    ) -> AxiomResult:
        bad: List[Tuple[int, int]] = []
        for src, dst in edges:
            t_src = anomaly_times.get(src, 0.0)
            t_dst = anomaly_times.get(dst, float("inf"))
            if t_src > t_dst + self.temporal_tolerance:
                bad.append((src, dst))

        if bad:
            return AxiomResult(
                "Темпоральная согласованность", AxiomStatus.VIOLATED,
                f"Рёбра с несогласованным направлением во времени: {bad}.",
                bad,
            )
        return AxiomResult("Темпоральная согласованность", AxiomStatus.OK, "Все рёбра темпорально согласованы.")

    # ------------------------------------------------------------------
    # Аксиома 3: Транзитивность
    # ------------------------------------------------------------------
    def _check_transitivity(
        self,
        edges: List[Tuple[int, int]],
        path_effects: Dict[Tuple[int, int], float],
    ) -> AxiomResult:
        if not path_effects:
            return AxiomResult(
                "Транзитивность", AxiomStatus.WARNING,
                "Данные о причинных эффектах пути не предоставлены; проверка пропущена.",
            )
        # Проверяем пути длины 2: src→mid→dst
        from collections import defaultdict
        adj: dict[int, list[int]] = defaultdict(list)
        for src, dst in edges:
            adj[src].append(dst)

        bad: List[Tuple[int, int]] = []
        for src, mids in adj.items():
            for mid in mids:
                for dst in adj.get(mid, []):
                    pe_direct = path_effects.get((src, dst), None)
                    pe_step1 = path_effects.get((src, mid), 1.0)
                    pe_step2 = path_effects.get((mid, dst), 1.0)
                    if pe_direct is not None and pe_direct < self.trans_threshold * pe_step1 * pe_step2:
                        bad.append((src, dst))

        if bad:
            return AxiomResult(
                "Транзитивность", AxiomStatus.WARNING,
                f"Слабая транзитивность на путях: {bad}. Возможен компенсирующий механизм.",
                bad,
            )
        return AxiomResult("Транзитивность", AxiomStatus.OK, "Транзитивность соблюдена.")

    # ------------------------------------------------------------------
    # Аксиома 4: Согласованность с топологией
    # ------------------------------------------------------------------
    def _check_topology(
        self,
        edges: List[Tuple[int, int]],
        physical_edges: set[Tuple[int, int]],
    ) -> AxiomResult:
        bad = [(s, d) for s, d in edges if (s, d) not in physical_edges]
        if bad:
            return AxiomResult(
                "Согласованность с топологией", AxiomStatus.WARNING,
                f"Рёбра без физического обоснования: {bad}. Возможны недокументированные зависимости.",
                bad,
            )
        return AxiomResult("Согласованность с топологией", AxiomStatus.OK, "Все рёбра физически обоснованы.")

    # ------------------------------------------------------------------
    # Аксиома 5: Монотонность вмешательства
    # ------------------------------------------------------------------
    def _check_monotonicity(
        self,
        edges: List[Tuple[int, int]],
        causal_effects: Dict[int, float],
        root_candidate: int,
    ) -> AxiomResult:
        pe_root = causal_effects.get(root_candidate, 0.0)
        bad: List[Tuple[int, int]] = []
        for src, dst in edges:
            if src != root_candidate:
                pe_src = causal_effects.get(src, 0.0)
                if pe_src > pe_root + self.mono_eps:
                    bad.append((src, dst))

        if bad:
            return AxiomResult(
                "Монотонность", AxiomStatus.WARNING,
                f"Промежуточные узлы с ПЭ > ПЭ(первопричины): {bad}. Возможна альтернативная первопричина.",
                bad,
            )
        return AxiomResult("Монотонность", AxiomStatus.OK, "Монотонность вмешательства соблюдена.")
