"""Вкладка «Данные» — таблицы метрик, экземпляров, временной ряд."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QHBoxLayout, QHeaderView, QLabel,
    QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)


class DataTab(QWidget):
    """Вкладка с таблицами метрик и экземпляров + график временных рядов."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # ── Верхняя часть: таблицы ──────────────────────
        tables_widget = QWidget()
        tables_layout = QHBoxLayout(tables_widget)
        tables_layout.setContentsMargins(0, 0, 0, 0)
        tables_layout.setSpacing(10)

        # Таблица метрик
        metrics_frame = QWidget()
        mf_layout = QVBoxLayout(metrics_frame)
        mf_layout.setContentsMargins(0, 0, 0, 0)
        mf_layout.setSpacing(6)
        mf_layout.addWidget(self._section_label("МЕТРИКИ"))

        self.metrics_table = QTableWidget(0, 7)
        self.metrics_table.setHorizontalHeaderLabels(
            ["", "Экземпляр", "Метрика", "Мин", "Макс", "Среднее", "σ"]
        )
        self.metrics_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.metrics_table.setColumnWidth(0, 32)
        self.metrics_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.metrics_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for c in range(3, 7):
            self.metrics_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        mf_layout.addWidget(self.metrics_table)
        tables_layout.addWidget(metrics_frame)

        # Таблица экземпляров
        instances_frame = QWidget()
        if_layout = QVBoxLayout(instances_frame)
        if_layout.setContentsMargins(0, 0, 0, 0)
        if_layout.setSpacing(6)
        if_layout.addWidget(self._section_label("ЭКЗЕМПЛЯРЫ СЕРВИСОВ"))

        self.instances_table = QTableWidget(0, 6)
        self.instances_table.setHorizontalHeaderLabels(
            ["", "Имя", "Сервис", "Хост", "CPU", "Версия"]
        )
        self.instances_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.instances_table.setColumnWidth(0, 32)
        self.instances_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.instances_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for c in range(3, 6):
            self.instances_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.instances_table.setAlternatingRowColors(True)
        self.instances_table.verticalHeader().setVisible(False)
        self.instances_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        if_layout.addWidget(self.instances_table)
        tables_layout.addWidget(instances_frame, stretch=1)

        splitter.addWidget(tables_widget)

        # ── Нижняя часть: график ─────────────────────────
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(6)
        chart_layout.addWidget(self._section_label("ВРЕМЕННЫЕ РЯДЫ"))

        self._chart_area = self._build_chart_area()
        chart_layout.addWidget(self._chart_area)
        splitter.addWidget(chart_widget)

        splitter.setSizes([400, 300])
        layout.addWidget(splitter)

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("sectionTitle")
        return lbl

    def _build_chart_area(self) -> QWidget:
        """Строит canvas для matplotlib или заглушку."""
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            import matplotlib.pyplot as plt

            fig = Figure(figsize=(10, 3), facecolor="#161922")
            ax = fig.add_subplot(111)
            ax.set_facecolor("#161922")
            ax.tick_params(colors="#6c7a9c")
            ax.spines[:].set_color("#2d3348")
            ax.set_xlabel("Время (с)", color="#6c7a9c", fontsize=10)
            ax.set_ylabel("Значение", color="#6c7a9c", fontsize=10)
            ax.set_title("Загрузите данные для отображения", color="#6c7a9c", fontsize=10)
            fig.tight_layout()

            self._fig = fig
            self._ax = ax
            canvas = FigureCanvasQTAgg(fig)
            canvas.setMinimumHeight(180)
            return canvas
        except ImportError:
            placeholder = QLabel("График временных рядов\n(требуется matplotlib)")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: #6c7a9c; font-size: 13px; border: 1px dashed #2d3348; border-radius: 6px;")
            placeholder.setMinimumHeight(180)
            self._fig = None
            self._ax = None
            return placeholder

    def load_metric_data(self, metric_data) -> None:
        """Заполняет таблицу метрик из MetricData."""
        import numpy as np
        self.metrics_table.setRowCount(0)
        for ni, inst in enumerate(metric_data.instance_names):
            for mi, metric in enumerate(metric_data.metric_names):
                vals = metric_data.values[:, ni, mi]
                vals = vals[~np.isnan(vals)]
                row = self.metrics_table.rowCount()
                self.metrics_table.insertRow(row)
                cb_item = QTableWidgetItem()
                cb_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
                cb_item.setCheckState(Qt.CheckState.Checked)
                self.metrics_table.setItem(row, 0, cb_item)
                self.metrics_table.setItem(row, 1, QTableWidgetItem(inst))
                self.metrics_table.setItem(row, 2, QTableWidgetItem(metric))
                self.metrics_table.setItem(row, 3, QTableWidgetItem(f"{vals.min():.3f}" if len(vals) else "—"))
                self.metrics_table.setItem(row, 4, QTableWidgetItem(f"{vals.max():.3f}" if len(vals) else "—"))
                self.metrics_table.setItem(row, 5, QTableWidgetItem(f"{vals.mean():.3f}" if len(vals) else "—"))
                self.metrics_table.setItem(row, 6, QTableWidgetItem(f"{vals.std():.3f}" if len(vals) else "—"))

    def load_topology(self, topo) -> None:
        """Заполняет таблицу экземпляров из TopologyData."""
        self.instances_table.setRowCount(0)
        for inst in topo.instances:
            row = self.instances_table.rowCount()
            self.instances_table.insertRow(row)
            cb_item = QTableWidgetItem()
            cb_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            cb_item.setCheckState(Qt.CheckState.Checked)
            self.instances_table.setItem(row, 0, cb_item)
            self.instances_table.setItem(row, 1, QTableWidgetItem(inst.name))
            self.instances_table.setItem(row, 2, QTableWidgetItem(inst.service))
            self.instances_table.setItem(row, 3, QTableWidgetItem(inst.host))
            self.instances_table.setItem(row, 4, QTableWidgetItem(str(inst.cpu_limit)))
            self.instances_table.setItem(row, 5, QTableWidgetItem(inst.version))

    def plot_series(self, timestamps, values, labels: list[str]) -> None:
        """Рисует временные ряды на canvas."""
        if self._ax is None:
            return
        self._ax.clear()
        self._ax.set_facecolor("#161922")
        colors = ["#4a9eff", "#3ecf8e", "#f6a623", "#ff5f5f", "#a78bfa"]
        for i, (vals, label) in enumerate(zip(values, labels)):
            self._ax.plot(timestamps, vals, label=label,
                         color=colors[i % len(colors)], linewidth=1.5)
        self._ax.legend(fontsize=9, labelcolor="#a0a8bc",
                        facecolor="#1e2130", edgecolor="#2d3348")
        self._ax.tick_params(colors="#6c7a9c")
        self._ax.spines[:].set_color("#2d3348")
        self._ax.set_xlabel("Время (с)", color="#6c7a9c", fontsize=9)
        if self._fig is not None:
            self._fig.tight_layout()
        if hasattr(self._chart_area, 'draw'):
            self._chart_area.draw()  # type: ignore[union-attr]
