"""Вкладка «Объяснение» – цепочка доказательств, текст, верификация."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QGroupBox, QHBoxLayout, QLabel,
    QScrollArea, QSplitter, QTextEdit,
    QVBoxLayout, QWidget,
)


class AxiomRow(QWidget):
    """Одна строка верификации аксиомы/правила."""

    def __init__(self, name: str, status: str = "–", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(10)

        icon = "✓" if status == "ok" else ("✗" if status == "fail" else "–")
        color = "#3ecf8e" if status == "ok" else ("#ff5f5f" if status == "fail" else "#6c7a9c")

        icon_lbl = QLabel(icon)
        icon_lbl.setFixedWidth(20)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
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

        splitter = QSplitter(Qt.Orientation.Horizontal)

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
        verif_splitter = QSplitter(Qt.Orientation.Horizontal)

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
            row = AxiomRow(ax, "–")
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
            row = AxiomRow(r, "–")
            self.alp_rows.append(row)
            alp_layout.addWidget(row)
        alp_layout.addStretch()
        verif_splitter.addWidget(alp_group)

        ll.addWidget(verif_splitter)
        splitter.addWidget(left)

        # ── Правая: вкладки объяснения ─────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        from PySide6.QtWidgets import QTabWidget as _QTW
        self._right_tabs = _QTW()
        self._right_tabs.setDocumentMode(True)

        tab1 = QWidget()
        t1l  = QVBoxLayout(tab1)
        t1l.setContentsMargins(0, 6, 0, 0)
        self.explanation_text = QTextEdit()
        self.explanation_text.setReadOnly(True)
        self.explanation_text.setPlaceholderText(
            "Объяснение появится после завершения анализа...\n\n"
            "Запустите анализ: Панель инструментов → [Анализ]"
        )
        t1l.addWidget(self.explanation_text)
        self._right_tabs.addTab(tab1, "Объяснение")

        tab2 = QWidget()
        t2l  = QVBoxLayout(tab2)
        t2l.setContentsMargins(0, 6, 0, 0)
        self.counter_text = QTextEdit()
        self.counter_text.setReadOnly(True)
        self.counter_text.setPlaceholderText(
            "Контрфактический анализ.\n"
            "Нажмите 'Что если?' в разделе Результаты."
        )
        t2l.addWidget(self.counter_text)
        self._right_tabs.addTab(tab2, "Анализ воздействия")

        rl.addWidget(self._right_tabs)
        splitter.addWidget(right)
        splitter.setSizes([550, 450])
        layout.addWidget(splitter)

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("sectionTitle")
        return lbl

    def _build_chain_area(self) -> QWidget:
        from cairn.gui.widgets.interactive_graph import InteractiveGraphWidget
        import matplotlib
        matplotlib.rcParams["font.family"] = "DejaVu Sans"
        self._chain_igraph = InteractiveGraphWidget()
        self._chain_igraph.setMinimumHeight(220)
        return self._chain_igraph

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
            parent_widget = row_widget.parent()
            from PySide6.QtWidgets import QWidget as _QW
            parent_layout = parent_widget.layout() if isinstance(parent_widget, _QW) else None
            if parent_layout is None:
                continue
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
        if not hasattr(self, "_chain_igraph"):
            return
        nodes = []
        node_map = {}
        root_idx = None
        for i, node in enumerate(chain.path_nodes):
            nodes.append({"idx": node.node_idx, "name": node.node_name,
                           "score": getattr(node, "causal_effect", 0.0) or 0.0})
            node_map[node.node_idx] = node.node_name
            if i == 0:
                root_idx = node.node_idx
        edges = []
        for edge in chain.path_edges:
            if edge.src in node_map and edge.dst in node_map:
                edges.append({"src": edge.src, "dst": edge.dst,
                               "type": getattr(edge, "edge_type", "call")})
        self._chain_igraph.set_graph(nodes, edges, root_idx=root_idx)