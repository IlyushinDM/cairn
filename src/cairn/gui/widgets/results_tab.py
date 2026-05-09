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

    show_explanation         = Signal(int)  # node_idx
    counterfactual_requested = Signal(int)  # 3.3

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # ── Верхняя: таблица ────────────────────────────
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(4)

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
        self.results_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        self.results_table.setColumnWidth(5, 215)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
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

        # ── Нижняя: граф на всю ширину ──────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)

        # Заголовок с легендой
        hdr = QHBoxLayout()
        hdr.addWidget(self._section_label("ГРАФ ПРИЧИННОГО РАСПРОСТРАНЕНИЯ"))
        hdr.addStretch()
        legend = QLabel("● красный = высокий ПЭ  ● синий = низкий ПЭ")
        legend.setStyleSheet("color: #6c7a9c; font-size: 10px;")
        hdr.addWidget(legend)
        rl.addLayout(hdr)

        self._graph_area = self._build_graph_area()
        rl.addWidget(self._graph_area)
        splitter.addWidget(right)

        splitter.setSizes([320, 400])
        layout.addWidget(splitter)

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("sectionTitle")
        return lbl

    def _build_graph_area(self) -> QWidget:
        from cairn.gui.widgets.interactive_graph import InteractiveGraphWidget
        self._igraph = InteractiveGraphWidget()
        self._igraph.node_clicked.connect(lambda idx: self._on_graph_node_clicked(idx))
        return self._igraph

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
            self.results_table.setRowHeight(row, 36)

            if rank == 1:
                cell = QWidget()
                cl = QHBoxLayout(cell)
                cl.setContentsMargins(2,2,2,2); cl.setSpacing(3)
                b1 = QPushButton("Подробнее")
                b1.setFixedHeight(28); b1.setObjectName("primaryBtn")
                b1.clicked.connect(lambda _,i=idx: self.show_explanation.emit(i))
                b2 = QPushButton("Воздействие")
                b2.setFixedHeight(28)
                b2.setToolTip("Контрфактический анализ")
                b2.clicked.connect(lambda _,i=idx: self.counterfactual_requested.emit(i))
                cl.addWidget(b1); cl.addWidget(b2)
                self.results_table.setCellWidget(row, 5, cell)

        if ranked:
            root_idx, root_ce = ranked[0]
            root_name = instance_names[root_idx] if root_idx < len(instance_names) else f"node-{root_idx}"
            self.root_label.setText(root_name)
            self.ce_label.setText(f"ПЭ: {root_ce:.3f}")
            self.conf_label.setText(f"Достоверность: {verification_confidence:.0%}")

    def draw_hypergraph(self, hypergraph, ce_scores: dict[int, float],
                         root_idx: int = None):
        """Рисует интерактивный граф."""
        if not hasattr(self, "_igraph"):
            return
        names   = hypergraph.instance_names
        ce_vals = [ce_scores.get(i, 0.0) for i in range(len(names))]
        nodes   = [{"idx": i, "name": names[i], "score": ce_vals[i]}
                   for i in range(len(names))]
        edges   = []
        for edge in hypergraph.edges:
            if len(edge.members) >= 2:
                edges.append({"src": edge.members[0], "dst": edge.members[1],
                               "type": edge.edge_type})
        self._igraph.set_graph(nodes, edges, root_idx=root_idx)


    def _on_graph_node_clicked(self, node_idx: int) -> None:
        """Выделяет узел при клике по нему в графе."""
        if hasattr(self, "_igraph"):
            self._igraph.highlight_node(node_idx)

    def highlight_graph_node_from_table(self, node_idx: int) -> None:
        """Выделяет узел при выборе строки в таблице."""
        if hasattr(self, "_igraph"):
            self._igraph.highlight_node(node_idx)

    def _centered(self, text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        return item