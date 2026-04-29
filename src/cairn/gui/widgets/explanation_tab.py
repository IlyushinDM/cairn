"""Вкладка «Объяснение» — цепочка доказательств, текст, верификация."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QGroupBox, QHBoxLayout, QLabel,
    QScrollArea, QSplitter, QTextEdit,
    QVBoxLayout, QWidget,
)


class AxiomRow(QWidget):
    """Одна строка верификации аксиомы/правила."""

    def __init__(self, name: str, status: str = "—", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(10)

        icon = "✓" if status == "ok" else ("✗" if status == "fail" else "—")
        color = "#3ecf8e" if status == "ok" else ("#ff5f5f" if status == "fail" else "#6c7a9c")

        icon_lbl = QLabel(icon)
        icon_lbl.setFixedWidth(20)
        icon_lbl.setAlignment(Qt.AlignCenter)
        icon_lbl.setStyleSheet(f"color: {color}; font-size: 14px; font-weight: bold;")

        name_lbl = QLabel(name)
        name_lbl.setStyleSheet("color: #d1d5e0; font-size: 12px;")

        layout.addWidget(icon_lbl)
        layout.addWidget(name_lbl)
        layout.addStretch()


class ExplanationTab(QWidget):
    """Вкладка «Объяснение»."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        splitter = QSplitter(Qt.Horizontal)

        # ── Левая: граф цепочки + верификация ───────────
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(10)

        # Граф цепочки
        ll.addWidget(self._section_label("ЦЕПОЧКА ДОКАЗАТЕЛЬСТВ"))
        self._chain_area = self._build_chain_area()
        ll.addWidget(self._chain_area)

        # Верификация (разделена на два блока)
        verif_splitter = QSplitter(Qt.Horizontal)

        # Аксиомы графа
        axiom_group = QGroupBox("Верификатор графа")
        ag_layout = QVBoxLayout(axiom_group)
        ag_layout.setSpacing(2)
        ag_layout.setContentsMargins(8, 12, 8, 8)

        self.axiom_summary = QLabel("5 / 5 аксиом ✓")
        self.axiom_summary.setStyleSheet("font-size: 13px; font-weight: 600; color: #3ecf8e; margin-bottom: 6px;")
        ag_layout.addWidget(self.axiom_summary)

        self.axiom_rows: list[AxiomRow] = []
        axioms = ["Ацикличность", "Темпоральная согласованность",
                  "Транзитивность", "Согласованность с топологией",
                  "Монотонность вмешательства"]
        for ax in axioms:
            row = AxiomRow(ax, "—")
            self.axiom_rows.append(row)
            ag_layout.addWidget(row)
        ag_layout.addStretch()
        verif_splitter.addWidget(axiom_group)

        # Правила ALP
        alp_group = QGroupBox("Логическая верификация (ALP)")
        alp_layout = QVBoxLayout(alp_group)
        alp_layout.setSpacing(2)
        alp_layout.setContentsMargins(8, 12, 8, 8)

        self.alp_summary = QLabel("5 / 5 правил ✓")
        self.alp_summary.setStyleSheet("font-size: 13px; font-weight: 600; color: #3ecf8e; margin-bottom: 6px;")
        alp_layout.addWidget(self.alp_summary)

        self.alp_rows: list[AxiomRow] = []
        rules = ["IC1: первопричина аномальна", "IC2: CE значим",
                 "IC3: путь существует", "IC4: текст содержит имя",
                 "IC5: числа согласованы"]
        for r in rules:
            row = AxiomRow(r, "—")
            self.alp_rows.append(row)
            alp_layout.addWidget(row)
        alp_layout.addStretch()
        verif_splitter.addWidget(alp_group)

        ll.addWidget(verif_splitter)
        splitter.addWidget(left)

        # ── Правая: текстовое объяснение ────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)
        rl.addWidget(self._section_label("ТЕКСТОВОЕ ОБЪЯСНЕНИЕ"))

        self.explanation_text = QTextEdit()
        self.explanation_text.setReadOnly(True)
        self.explanation_text.setPlaceholderText(
            "Объяснение появится после завершения анализа...\n\n"
            "Запустите анализ: Панель инструментов → [Анализ]"
        )
        rl.addWidget(self.explanation_text)

        # Контр-абдуктивная гипотеза
        rl.addWidget(self._section_label("КОНТР-АБДУКТИВНАЯ ГИПОТЕЗА"))
        self.counter_text = QTextEdit()
        self.counter_text.setReadOnly(True)
        self.counter_text.setMaximumHeight(120)
        self.counter_text.setPlaceholderText("Альтернативная гипотеза...")
        rl.addWidget(self.counter_text)

        splitter.addWidget(right)
        splitter.setSizes([550, 450])
        layout.addWidget(splitter)

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("sectionTitle")
        return lbl

    def _build_chain_area(self) -> QWidget:
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            fig = Figure(figsize=(6, 3.5), facecolor="#161922")
            ax = fig.add_subplot(111)
            ax.set_facecolor("#161922")
            ax.set_xticks([]); ax.set_yticks([])
            ax.spines[:].set_color("#2d3348")
            ax.set_title("Цепочка доказательств", color="#6c7a9c", fontsize=10)
            fig.tight_layout()
            self._chain_fig = fig
            self._chain_ax = ax
            canvas = FigureCanvasQTAgg(fig)
            canvas.setMinimumHeight(220)
            return canvas
        except ImportError:
            self._chain_fig = None
            self._chain_ax = None
            lbl = QLabel("Граф цепочки доказательств\n(требуется matplotlib + networkx)")
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: #6c7a9c; border: 1px dashed #2d3348; border-radius:6px;")
            lbl.setMinimumHeight(220)
            return lbl

    def show_chain(self, chain) -> None:
        """Отображает цепочку доказательств."""
        self._draw_chain_graph(chain)
        # Текстовое объяснение
        from cairn.explanation import TemplateTextGenerator
        gen = TemplateTextGenerator()
        text = gen.generate(chain)
        self.explanation_text.setPlainText(text)

    def show_alp_result(self, result) -> None:
        """Обновляет строки ALP-верификатора."""
        ok = len(result.violated_rules)
        total = len(self.alp_rows)
        passed = total - ok
        color = "#3ecf8e" if ok == 0 else "#f6a623" if ok <= 1 else "#ff5f5f"
        self.alp_summary.setText(f"{passed} / {total} правил ✓")
        self.alp_summary.setStyleSheet(f"font-size:13px;font-weight:600;color:{color};margin-bottom:6px;")
        if result.counter_hypothesis:
            self.counter_text.setPlainText(result.counter_hypothesis)

    def show_verifier_result(self, report) -> None:
        """Обновляет строки верификатора графа."""
        statuses = [
            r.status.value for r in report.axiom_results
        ] if report.axiom_results else []
        ok_count = sum(1 for s in statuses if s == "ok")
        total = len(self.axiom_rows)
        color = "#3ecf8e" if ok_count == total else "#f6a623" if ok_count >= total - 1 else "#ff5f5f"
        self.axiom_summary.setText(f"{ok_count} / {total} аксиом ✓")
        self.axiom_summary.setStyleSheet(
            f"font-size:13px;font-weight:600;color:{color};margin-bottom:6px;"
        )
        for i, (row_widget, status) in enumerate(zip(self.axiom_rows, statuses)):
            # Пересоздаём виджет строки с нужным статусом
            parent_layout = row_widget.parent().layout()
            idx = parent_layout.indexOf(row_widget)
            axiom_name = ["Ацикличность", "Темпоральная согласованность",
                          "Транзитивность", "Согласованность с топологией",
                          "Монотонность вмешательства"][i]
            new_row = AxiomRow(axiom_name, status)
            parent_layout.removeWidget(row_widget)
            row_widget.deleteLater()
            parent_layout.insertWidget(idx, new_row)
            self.axiom_rows[i] = new_row

    def _draw_chain_graph(self, chain) -> None:
        if self._chain_ax is None:
            return
        try:
            import networkx as nx
            ax = self._chain_ax
            ax.clear()
            ax.set_facecolor("#161922")
            ax.set_xticks([]); ax.set_yticks([])

            G = nx.DiGraph()
            for node in chain.path_nodes:
                G.add_node(node.node_name, nll=node.nll, ce=node.causal_effect)
            for edge in chain.path_edges:
                src_name = chain.path_nodes[0].node_name  # упрощение
                for n in chain.path_nodes:
                    if n.node_idx == edge.src:
                        src_name = n.node_name
                for n in chain.path_nodes:
                    if n.node_idx == edge.dst:
                        G.add_edge(src_name, n.node_name,
                                   weight=edge.strength, etype=edge.edge_type)

            if len(G.nodes) == 0:
                return

            pos = nx.spring_layout(G, seed=42)
            colors = ["#ff5f5f" if i == 0 else "#4a9eff"
                      for i in range(len(G.nodes))]
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                                   node_size=700, alpha=0.9)
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#3ecf8e",
                                   arrows=True, arrowsize=20, width=2.0)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8,
                                    font_color="#ffffff", font_weight="bold")
            edge_labels = {(u, v): f"{d.get('etype','')[:4]}" for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax,
                                          font_size=7, font_color="#a0a8bc")
            ax.set_title("Красный = первопричина", color="#6c7a9c", fontsize=9)
            self._chain_fig.tight_layout()
            if hasattr(self._chain_area, 'draw'):
                self._chain_area.draw()
        except ImportError:
            pass
