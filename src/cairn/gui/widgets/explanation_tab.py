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
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
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
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
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
        violated = set()
        for r in result.violated_rules:
            # violated_rules может быть строками или объектами с .name
            violated.add(str(r) if not hasattr(r, "name") else r.name)

        # Имена правил как в разметке строк
        rule_keys = ["IC1", "IC2", "IC3", "IC4", "IC5"]
        rule_names = ["IC1: первопричина аномальна", "IC2: CE значим",
                      "IC3: путь существует", "IC4: текст содержит имя",
                      "IC5: числа согласованы"]

        total  = len(self.alp_rows)
        passed = 0
        for i, (row_widget, key, name) in enumerate(zip(self.alp_rows, rule_keys, rule_names)):
            # Правило нарушено если его ключ встречается в violated
            is_violated = any(key in v for v in violated)
            status = "fail" if is_violated else "ok"
            if not is_violated:
                passed += 1

            parent_widget = row_widget.parent()
            from PySide6.QtWidgets import QWidget as _QW
            parent_layout = parent_widget.layout() if isinstance(parent_widget, _QW) else None
            if parent_layout is None:
                continue
            idx = parent_layout.indexOf(row_widget)
            new_row = AxiomRow(name, status)
            parent_layout.removeWidget(row_widget)
            row_widget.deleteLater()
            parent_layout.insertWidget(idx, new_row)
            self.alp_rows[i] = new_row

        color = "#3ecf8e" if passed == total else "#f6a623" if passed >= total - 1 else "#ff5f5f"
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
        if self._chain_ax is None:
            return
        try:
            import networkx as nx
            ax = self._chain_ax
            ax.clear()
            ax.set_facecolor("#161922")
            ax.set_xticks([]); ax.set_yticks([])

            G = nx.DiGraph()
            root_name = None

            # Добавляем узлы из chain (все, не только root)
            node_map: dict[int, str] = {}
            for node in chain.path_nodes:
                G.add_node(node.node_name)
                node_map[node.node_idx] = node.node_name
                if root_name is None:
                    root_name = node.node_name  # первый = root

            # Если chain содержит только 1 узел — пытаемся добавить соседей из ce_scores
            if hasattr(chain, "ce_scores") and len(G.nodes) <= 1:
                for idx, name in getattr(chain, "all_nodes", {}).items():
                    G.add_node(name)
                    node_map[idx] = name

            # Рёбра из chain
            for edge in chain.path_edges:
                src = node_map.get(edge.src)
                dst = node_map.get(edge.dst)
                if src and dst and src != dst:
                    G.add_edge(src, dst, etype=getattr(edge, "edge_type", "call"))

            if len(G.nodes) == 0:
                ax.set_title("Нет данных для отображения", color="#6c7a9c", fontsize=10)
                if hasattr(self._chain_area, "draw"):
                    self._chain_area.draw()
                return

            pos = nx.spring_layout(G, seed=42, k=2.0)
            node_list = list(G.nodes)
            colors = ["#ff5f5f" if n == root_name else "#4a9eff"
                      for n in node_list]
            sizes  = [800 if n == root_name else 500 for n in node_list]

            nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=node_list,
                                   node_color=colors, node_size=sizes, alpha=0.9)
            if G.number_of_edges() > 0:
                nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#3ecf8e",
                                       arrows=True, arrowsize=20, width=2.0)
                edge_labels = {(u, v): d.get("etype", "")[:4]
                               for u, v, d in G.edges(data=True)}
                nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax,
                                             font_size=7, font_color="#a0a8bc")
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8,
                                    font_color="#ffffff", font_weight="bold")
            title = f"Красный = первопричина ({root_name})" if root_name else "Цепочка доказательств"
            ax.set_title(title, color="#6c7a9c", fontsize=9)
            if self._chain_fig is not None:
                self._chain_fig.tight_layout()
            if hasattr(self._chain_area, "draw"):
                self._chain_area.draw()  # type: ignore[union-attr]
        except ImportError:
            pass