"""Построение причинного гиперграфа (раздел 2.6).

Статические гиперрёбра трёх типов:
  - CALL: вызовы из трассировок
  - COLOCATION: совместное размещение на физическом узле
  - LOADBALANCE: экземпляры одного сервиса

Адаптивные гиперрёбра добавляются модулем VerifiedEdgeDiscovery (раздел 3.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

import torch


class EdgeType(str, Enum):
    CALL = "call"
    COLOCATION = "colocation"
    LOADBALANCE = "loadbalance"
    ADAPTIVE = "adaptive"


@dataclass
class HyperEdge:
    """Одно гиперребро гиперграфа G = (V, E)."""

    members: List[int]        # Индексы вершин, входящих в гиперребро
    edge_type: EdgeType
    weight: float = 1.0
    verified: bool = True     # False — исключается из причинного анализа


@dataclass
class CausalHypergraph:
    """Причинный гиперграф системы."""

    n_nodes: int
    edges: List[HyperEdge] = field(default_factory=list)
    # Метаданные вершин
    node_types: Dict[int, str] = field(default_factory=dict)  # {idx: "service"|"db"|"cache"}

    def incidence_matrix(self) -> torch.Tensor:
        """Матрица инцидентности H (строки — вершины, столбцы — рёбра).

        Используется в гиперграфовой свёртке (формула 3.24).
        """
        n_edges = len(self.edges)
        H = torch.zeros(self.n_nodes, n_edges)
        for j, edge in enumerate(self.edges):
            for v in edge.members:
                H[v, j] = 1.0
        return H

    def edge_weights(self) -> torch.Tensor:
        """Диагональ матрицы W_H (веса гиперрёбер)."""
        return torch.tensor([e.weight for e in self.edges], dtype=torch.float)

    def causal_edges(self) -> "CausalHypergraph":
        """Подграф только из верифицированных рёбер (для причинного анализа)."""
        verified = [e for e in self.edges if e.verified]
        g = CausalHypergraph(self.n_nodes, verified, self.node_types.copy())
        return g


class HypergraphBuilder:
    """Строит статический причинный гиперграф из конфигурации и трассировок.

    Параметры
    ----------
    n_nodes : int
        Число экземпляров сервисов N.
    """

    def __init__(self, n_nodes: int) -> None:
        self.n_nodes = n_nodes

    def from_topology(
        self,
        call_paths: List[List[int]],          # Каждый элемент — список индексов в одной трассировке
        colocated_groups: List[List[int]],    # Группы узлов на одном физическом сервере
        lb_groups: List[List[int]],           # Группы экземпляров одного сервиса
        node_types: Dict[int, str] | None = None,
    ) -> CausalHypergraph:
        """Строит гиперграф из трёх типов статических рёбер."""
        edges: List[HyperEdge] = []

        for path in call_paths:
            if len(path) >= 2:
                edges.append(HyperEdge(members=path, edge_type=EdgeType.CALL))

        for group in colocated_groups:
            if len(group) >= 2:
                edges.append(HyperEdge(members=group, edge_type=EdgeType.COLOCATION))

        for group in lb_groups:
            if len(group) >= 2:
                edges.append(HyperEdge(members=group, edge_type=EdgeType.LOADBALANCE))

        return CausalHypergraph(
            n_nodes=self.n_nodes,
            edges=edges,
            node_types=node_types or {},
        )
