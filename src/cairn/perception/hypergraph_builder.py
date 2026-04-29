"""Построение причинного гиперграфа из TopologyData (раздел 2.6).

Три типа статических гиперрёбер:
  - CALL        : путь вызова из трассировки
  - COLOCATION  : экземпляры на одном физическом хосте
  - LOADBALANCE : реплики одного сервиса (за балансировщиком)

Адаптивные гиперрёбра добавляются верификатором (раздел 3.4).

Интеграция:
  HypergraphBuilder.from_topology_data(topo) — строит граф из TopologyData.
  CausalHypergraph.to_pyg_tensors()          — возвращает тензоры для PyG/HGNN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch


class EdgeType(str, Enum):
    CALL = "call"
    COLOCATION = "colocation"
    LOADBALANCE = "loadbalance"
    ADAPTIVE = "adaptive"


@dataclass
class HyperEdge:
    """Одно гиперребро гиперграфа G = (V, E)."""

    members: List[int]        # Индексы вершин (порядок совпадает с instance_names)
    edge_type: EdgeType
    weight: float = 1.0
    verified: bool = True     # False → исключается из причинного анализа
    source_label: str = ""    # Человекочитаемая метка (e.g. "frontend→order")


@dataclass
class CausalHypergraph:
    """Причинный гиперграф G = (V, E_статич ∪ E_адапт).

    Атрибуты
    ----------
    n_nodes : int
    instance_names : list[str]
        Имена экземпляров (индекс в списке = индекс вершины).
    edges : list[HyperEdge]
    node_types : dict[int, str]
        {vertex_idx: "service"|"database"|"cache"|...}
    """

    n_nodes: int
    instance_names: List[str] = field(default_factory=list)
    edges: List[HyperEdge] = field(default_factory=list)
    node_types: Dict[int, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Матрицы для гиперграфовой свёртки (формула 3.24)
    # ------------------------------------------------------------------

    def incidence_matrix(self, causal_only: bool = False) -> torch.Tensor:
        """Матрица инцидентности H ∈ {0,1}^{N×M}.

        Строки — вершины, столбцы — гиперрёбра.
        Используется в формуле 3.24: X^{l+1} = D_v^{-½} H W_H D_e^{-1} H^T D_v^{-½} X^{l} Θ
        """
        edges = [e for e in self.edges if e.verified] if causal_only else self.edges
        M = len(edges)
        H = torch.zeros(self.n_nodes, M)
        for j, edge in enumerate(edges):
            for v in edge.members:
                if 0 <= v < self.n_nodes:
                    H[v, j] = 1.0
        return H

    def edge_weights(self, causal_only: bool = False) -> torch.Tensor:
        """Веса гиперрёбер W_H ∈ ℝ^M (диагональ матрицы весов)."""
        edges = [e for e in self.edges if e.verified] if causal_only else self.edges
        return torch.tensor([e.weight for e in edges], dtype=torch.float)

    def to_pyg_tensors(self) -> dict:
        """Возвращает словарь тензоров для использования в PyG/HGNN.

        Ключи:
          incidence       : Tensor (N, M) — матрица инцидентности
          edge_weights    : Tensor (M,)   — веса рёбер
          edge_type_ids   : Tensor (M,)   — int-коды типов рёбер
          n_nodes         : int
          n_edges         : int
          instance_names  : list[str]

        Только верифицированные рёбра (causal_only=True).
        """
        _type_to_id = {t: i for i, t in enumerate(EdgeType)}
        edges = [e for e in self.edges if e.verified]

        return {
            "incidence": self.incidence_matrix(causal_only=True),
            "edge_weights": self.edge_weights(causal_only=True),
            "edge_type_ids": torch.tensor(
                [_type_to_id[e.edge_type] for e in edges], dtype=torch.long
            ),
            "n_nodes": self.n_nodes,
            "n_edges": len(edges),
            "instance_names": list(self.instance_names),
        }

    def adjacency_matrix(self) -> torch.Tensor:
        """Симметричная матрица смежности A ∈ {0,1}^{N×N}.

        A[i,j] = 1 если i и j принадлежат хотя бы одному гиперребру.
        Используется в каскадной воронке для поиска соседей.
        """
        H = self.incidence_matrix(causal_only=True)  # (N, M)
        A = (H @ H.T).clamp(0, 1)
        A.fill_diagonal_(0)
        return A

    def causal_subgraph(self) -> "CausalHypergraph":
        """Возвращает подграф только из верифицированных рёбер."""
        return CausalHypergraph(
            n_nodes=self.n_nodes,
            instance_names=list(self.instance_names),
            edges=[e for e in self.edges if e.verified],
            node_types=dict(self.node_types),
        )

    def add_adaptive_edge(
        self,
        members: List[int],
        weight: float = 1.0,
        verified: bool = False,
        label: str = "",
    ) -> HyperEdge:
        """Добавляет адаптивное гиперребро (вызывается верификатором, раздел 3.4)."""
        edge = HyperEdge(
            members=members,
            edge_type=EdgeType.ADAPTIVE,
            weight=weight,
            verified=verified,
            source_label=label,
        )
        self.edges.append(edge)
        return edge

    def instance_idx(self, name: str) -> int:
        """Возвращает индекс экземпляра по имени."""
        return self.instance_names.index(name)


# ---------------------------------------------------------------------------
# HypergraphBuilder
# ---------------------------------------------------------------------------


class HypergraphBuilder:
    """Строит причинный гиперграф из TopologyData или явных списков.

    Параметры
    ----------
    n_nodes : int
        Число экземпляров сервисов N.
    """

    def __init__(self, n_nodes: int) -> None:
        self.n_nodes = n_nodes

    # ------------------------------------------------------------------
    # Основной метод — из TopologyData (интеграция с коннекторами)
    # ------------------------------------------------------------------

    @classmethod
    def from_topology_data(cls, topo: object) -> CausalHypergraph:
        """Строит гиперграф из объекта TopologyData (из cairn.connectors.base).

        Параметры
        ----------
        topo : TopologyData
            Объект топологии от коннектора (избегаем прямого импорта для
            декаплинга модулей).

        Возвращает
        ----------
        CausalHypergraph
        """
        instance_names: List[str] = topo.instance_names          # type: ignore[attr-defined]
        n_nodes = len(instance_names)
        name_to_idx = {name: i for i, name in enumerate(instance_names)}

        # Определяем типы узлов из поля service
        node_types: Dict[int, str] = {}
        for inst in topo.instances:                               # type: ignore[attr-defined]
            idx = name_to_idx[inst.name]
            svc = inst.service.lower()
            if "database" in svc or "db" in svc:
                node_types[idx] = "database"
            elif "cache" in svc or "redis" in svc or "memcached" in svc:
                node_types[idx] = "cache"
            else:
                node_types[idx] = "service"

        edges: List[HyperEdge] = []

        # Рёбра вызовов: каждое ребро (src, dst) → гиперребро из двух вершин
        for src_name, dst_name in topo.call_edges:               # type: ignore[attr-defined]
            if src_name in name_to_idx and dst_name in name_to_idx:
                edges.append(HyperEdge(
                    members=[name_to_idx[src_name], name_to_idx[dst_name]],
                    edge_type=EdgeType.CALL,
                    source_label=f"{src_name}→{dst_name}",
                ))

        # Рёбра совместного размещения
        for group in topo.colocation_groups:                      # type: ignore[attr-defined]
            idxs = [name_to_idx[n] for n in group if n in name_to_idx]
            if len(idxs) >= 2:
                edges.append(HyperEdge(
                    members=idxs,
                    edge_type=EdgeType.COLOCATION,
                    source_label="+".join(group),
                ))

        # Рёбра балансировки
        for group in topo.load_balancer_groups:                   # type: ignore[attr-defined]
            idxs = [name_to_idx[n] for n in group if n in name_to_idx]
            if len(idxs) >= 2:
                edges.append(HyperEdge(
                    members=idxs,
                    edge_type=EdgeType.LOADBALANCE,
                    source_label="+".join(group),
                ))

        return CausalHypergraph(
            n_nodes=n_nodes,
            instance_names=instance_names,
            edges=edges,
            node_types=node_types,
        )

    # ------------------------------------------------------------------
    # Метод из явных списков (обратная совместимость + тесты)
    # ------------------------------------------------------------------

    def from_topology(
        self,
        call_paths: List[List[int]],
        colocated_groups: List[List[int]],
        lb_groups: List[List[int]],
        instance_names: Optional[List[str]] = None,
        node_types: Optional[Dict[int, str]] = None,
    ) -> CausalHypergraph:
        """Строит гиперграф из явных списков индексов.

        Параметры
        ----------
        call_paths : list[list[int]]
            Каждый элемент — упорядоченный путь вызова (индексы вершин).
        colocated_groups : list[list[int]]
            Группы узлов на одном физическом сервере.
        lb_groups : list[list[int]]
            Группы реплик одного сервиса.
        instance_names : list[str] | None
            Человекочитаемые имена (если None — генерируются автоматически).
        node_types : dict[int, str] | None
        """
        names = instance_names or [f"instance-{i}" for i in range(self.n_nodes)]
        edges: List[HyperEdge] = []

        for path in call_paths:
            if len(path) >= 2:
                edges.append(HyperEdge(
                    members=path,
                    edge_type=EdgeType.CALL,
                    source_label="→".join(str(v) for v in path),
                ))

        for group in colocated_groups:
            if len(group) >= 2:
                edges.append(HyperEdge(
                    members=group,
                    edge_type=EdgeType.COLOCATION,
                ))

        for group in lb_groups:
            if len(group) >= 2:
                edges.append(HyperEdge(
                    members=group,
                    edge_type=EdgeType.LOADBALANCE,
                ))

        return CausalHypergraph(
            n_nodes=self.n_nodes,
            instance_names=names,
            edges=edges,
            node_types=node_types or {},
        )
