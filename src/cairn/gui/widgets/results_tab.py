"""Вкладка «Результаты» — таблица ранжирования + визуализация гиперграфа."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QFrame, QHBoxLayout, QHeaderView, QLabel,
    QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)


class ResultsTab(QWidget):
    """Вкладка с ранжированием первопричин и графом."""

    show_explanation = Signal(int)  # node_idx

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Левая: таблица ──────────────────────────────
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(6)

        lbl = QLabel("РАНЖИРОВАНИЕ ПЕРВОПРИЧИН")
        lbl.setObjectName("sectionTitle")
        ll.addWidget(lbl)

        self.results_table = QTableWidget(0, 6)
        self.results_table.setHorizontalHeaderLabels([
            "Ранг", "Компонент", "ПЭ", "Тип сбоя",
            "Достоверность", "Действие",
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.results_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        ll.addWidget(self.results_table)

        # Итоговая строка
        summary_frame = QFrame()
        summary_frame.setObjectName("card")
        sf_layout = QHBoxLayout(summary_frame)
        sf_layout.setContentsMargins(12, 10, 12, 10)

        self.root_label = QLabel("Первопричина не определена")
        self.root_label.setObjectName("metricValue")
        self.ce_label = QLabel("ПЭ: —")
        self.ce_label.setStyleSheet("font-size: 14px; color: #6c7a9c;")
        self.conf_label = QLabel("Достоверность: —")
        self.conf_label.setStyleSheet("font-size: 14px; color: #6c7a9c;")

        sf_layout.addWidget(self.root_label)
        sf_layout.addSpacing(24)
        sf_layout.addWidget(self.ce_label)
        sf_layout.addSpacing(24)
        sf_layout.addWidget(self.conf_label)
        sf_layout.addStretch()
        ll.addWidget(summary_frame)

        splitter.addWidget(left)

        # ── Правая: граф ────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)

        rl.addWidget(self._section_label("ГРАФ ПРИЧИННОГО РАСПРОСТРАНЕНИЯ"))
        self._graph_area = self._build_graph_area()
        rl.addWidget(self._graph_area)
        splitter.addWidget(right)

        splitter.setSizes([500, 500])
        layout.addWidget(splitter)

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("sectionTitle")
        return lbl

    def _build_graph_area(self) -> QWidget:
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            fig = Figure(figsize=(6, 5), facecolor="#161922")
            ax = fig.add_subplot(111)
            ax.set_facecolor("#161922")
            ax.set_xticks([]); ax.set_yticks([])
            ax.spines[:].set_color("#2d3348")
            ax.set_title("Запустите анализ для отображения", color="#6c7a9c", fontsize=10)
            fig.tight_layout()
            self._graph_fig = fig
            self._graph_ax = ax
            canvas = FigureCanvasQTAgg(fig)
            return canvas
        except ImportError:
            self._graph_fig = None
            self._graph_ax = None
            lbl = QLabel("Граф гиперграфа\n(требуется matplotlib + networkx)")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: #6c7a9c; border: 1px dashed #2d3348; border-radius:6px;")
            return lbl

    def show_results(self, ranked: list[tuple[int, float]], instance_names: list[str],
                     nll_scores: dict, verification_confidence: float = 1.0,
                     fault_type: str = "—"):
        """Заполняет таблицу результатами воронки."""
        self.results_table.setRowCount(0)
        for rank, (idx, ce) in enumerate(ranked, 1):
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            name = instance_names[idx] if idx < len(instance_names) else f"node-{idx}"

            self.results_table.setItem(row, 0, self._centered(str(rank)))
            self.results_table.setItem(row, 1, QTableWidgetItem(name))
            self.results_table.setItem(row, 2, self._centered(f"{ce:.3f}"))
            self.results_table.setItem(row, 3, QTableWidgetItem(fault_type if rank == 1 else "—"))
            conf_pct = f"{verification_confidence:.0%}" if rank == 1 else "—"
            self.results_table.setItem(row, 4, self._centered(conf_pct))
            self.results_table.setRowHeight(row, 30)

            if rank == 1:
                btn = QPushButton("Подробнее →")
                btn.setFixedHeight(24)
                btn.setObjectName("primaryBtn")
                btn.clicked.connect(lambda _, i=idx: self.show_explanation.emit(i))
                self.results_table.setCellWidget(row, 5, btn)

        if ranked:
            root_idx, root_ce = ranked[0]
            root_name = instance_names[root_idx] if root_idx < len(instance_names) else f"node-{root_idx}"
            self.root_label.setText(root_name)
            self.ce_label.setText(f"ПЭ: {root_ce:.3f}")
            self.conf_label.setText(f"Достоверность: {verification_confidence:.0%}")

    def draw_hypergraph(self, hypergraph, ce_scores: dict[int, float]):
        """Рисует гиперграф с окраской по ПЭ."""
        if self._graph_ax is None:
            return
        try:
            import networkx as nx
            import numpy as np
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors

            ax = self._graph_ax
            ax.clear()
            ax.set_facecolor("#161922")
            ax.set_xticks([]); ax.set_yticks([])

            G = nx.DiGraph()
            names = hypergraph.instance_names
            for name in names:
                G.add_node(name)
            for edge in hypergraph.edges:
                if edge.edge_type == "call" and len(edge.members) >= 2:
                    G.add_edge(names[edge.members[0]], names[edge.members[1]])

            pos = nx.spring_layout(G, seed=42, k=2.5)

            import matplotlib.cm as _cm
            import matplotlib.colors as _mcolors

            ce_vals = [ce_scores.get(i, None) for i in range(len(names))]
            n_known = sum(1 for v in ce_vals if v is not None)

            if n_known <= 1:
                # Если CE есть только для root — красить root красным, остальных серым
                root_idx_g = next((i for i, v in enumerate(ce_vals) if v is not None), None)
                node_colors = []
                for i in range(len(names)):
                    if i == root_idx_g:
                        node_colors.append((0.85, 0.15, 0.15, 0.9))   # красный
                    else:
                        node_colors.append((0.25, 0.35, 0.55, 0.9))   # тёмно-синий
                node_sizes = [600 if i == root_idx_g else 350 for i in range(len(names))]
            else:
                # Все узлы имеют CE — нормализованный coolwarm
                filled = [v if v is not None else 0.0 for v in ce_vals]
                ce_min  = min(filled)
                ce_max  = max(filled)
                ce_span = max(ce_max - ce_min, 1e-8)
                cmap = _cm.get_cmap("coolwarm")
                norm = _mcolors.Normalize(vmin=ce_min, vmax=ce_max)
                node_colors = [cmap(norm(v)) for v in filled]
                node_sizes = [
                    350 + 250 * max(0.0, (filled[i] - ce_min) / ce_span)
                    for i in range(len(names))
                ]

            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                                   node_size=node_sizes, alpha=0.9)
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#4a9eff",
                                   arrows=True, arrowsize=15, width=1.5, alpha=0.7)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8,
                                    font_color="#d1d5e0")
            ax.set_title("Красный = высокий ПЭ (вероятная первопричина)",
                        color="#6c7a9c", fontsize=9)
            if self._graph_fig is not None:
                self._graph_fig.tight_layout()
            if hasattr(self._graph_area, 'draw'):
                self._graph_area.draw()  # type: ignore[union-attr]
        except ImportError:
            pass

    def _centered(self, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        return item