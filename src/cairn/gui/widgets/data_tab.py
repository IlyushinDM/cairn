"""Вкладка «Данные» – таблицы метрик, экземпляров, временной ряд."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel,
    QPushButton, QScrollBar, QSplitter, QTableWidget, QTableWidgetItem,
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
        self._splitter = splitter

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
        self.metrics_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        # Кнопка разворачивания
        self._btn_expand = QPushButton("Развернуть")
        self._btn_expand.setFixedHeight(28)
        self._btn_expand.setCheckable(True)
        self._btn_expand.toggled.connect(self._toggle_expand)
        mf_layout.addWidget(self._btn_expand)
        mf_layout.addWidget(self.metrics_table)
        self._last_md = None
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
        self.instances_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        if_layout.addWidget(self.instances_table)
        tables_layout.addWidget(instances_frame, stretch=1)

        splitter.addWidget(tables_widget)

        # ── Нижняя часть: график ─────────────────────────
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(4)

        # Заголовок + выбор метрики (2.3: рядом с графиком)
        chart_hdr = QHBoxLayout()
        chart_hdr.addWidget(self._section_label("ВРЕМЕННЫЕ РЯДЫ"))
        chart_hdr.addStretch()
        chart_hdr.addWidget(QLabel("Метрика:"))
        self._metric_combo = QComboBox()
        self._metric_combo.setFixedWidth(130)
        self._metric_combo.setFixedHeight(26)
        self._metric_combo.currentTextChanged.connect(self._on_metric_changed)
        chart_hdr.addWidget(self._metric_combo)
        chart_layout.addLayout(chart_hdr)

        # Фильтр сервисов (2.1: включить/выключить видимость)
        self._service_filter_row = QHBoxLayout()
        self._service_filter_row.addWidget(QLabel("Сервисы:"))
        self._service_checks: dict[str, QCheckBox] = {}
        self._service_filter_row.addStretch()
        chart_layout.addLayout(self._service_filter_row)

        self._chart_area = self._build_chart_area()
        chart_layout.addWidget(self._chart_area)
        splitter.addWidget(chart_widget)

        splitter.setSizes([400, 300])
        layout.addWidget(splitter)

    def _toggle_expand(self, expanded: bool) -> None:
        self._btn_expand.setText("Свернуть" if expanded else "Развернуть")
        if hasattr(self, '_splitter'):
            if expanded:
                # Полностью разворачиваем – убираем нижний виджет
                self._splitter.setSizes([10000, 0])
                # 2.4: растягиваем колонки по содержимому
                for c in range(self.metrics_table.columnCount()):
                    self.metrics_table.resizeColumnToContents(c)
                self.metrics_table.horizontalHeader().setSectionResizeMode(
                    1, QHeaderView.ResizeMode.Stretch)
                self.metrics_table.horizontalHeader().setSectionResizeMode(
                    2, QHeaderView.ResizeMode.Stretch)
            else:
                self._splitter.setSizes([400, 300])

    def _on_metric_changed(self, metric: str) -> None:
        if getattr(self, '_last_md', None) is not None:
            self._plot_filtered()

    def _plot_filtered(self) -> None:
        """Строит временной ряд с учётом выбранной метрики и фильтра сервисов."""
        import numpy as np
        md = getattr(self, '_last_md', None)
        if md is None:
            return
        try:
            metric = self._metric_combo.currentText()
            mi = md.metric_names.index(metric) if metric in md.metric_names else 0
            timestamps = md.timestamps
            series, labels = [], []
            for ni, inst in enumerate(md.instance_names):
                # Проверяем чекбокс видимости
                cb = self._service_checks.get(inst)
                if cb is not None and not cb.isChecked():
                    continue
                vals = md.values[:, ni, mi]
                nans = np.isnan(vals)
                if nans.all():
                    continue
                idx_arr = np.arange(len(vals))
                vals = np.interp(idx_arr, idx_arr[~nans], vals[~nans])
                series.append(vals)
                labels.append(inst)
            if series:
                self.plot_series(timestamps, series, labels)
        except Exception:
            pass

    def _plot_from_md(self, md, metric: str = None) -> None:
        """Точка входа для построения графика (вызывается из main_window)."""
        self._last_md = md
        if metric and hasattr(self, '_metric_combo'):
            idx = self._metric_combo.findText(metric)
            if idx >= 0:
                self._metric_combo.blockSignals(True)
                self._metric_combo.setCurrentIndex(idx)
                self._metric_combo.blockSignals(False)
        self._plot_filtered()

    def load_metric_data(self, metric_data) -> None:
        """Заполняет таблицу метрик + обновляет комбо метрик и фильтр сервисов."""
        import numpy as np
        self._last_md = metric_data

        # ── Заполняем таблицу метрик ──────────────────────────────────────
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
                def _ro(t):
                    it = QTableWidgetItem(t)
                    it.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                    return it
                self.metrics_table.setItem(row, 1, _ro(inst))
                self.metrics_table.setItem(row, 2, _ro(metric))
                self.metrics_table.setItem(row, 3, _ro(f"{vals.min():.3f}" if len(vals) else "–"))
                self.metrics_table.setItem(row, 4, _ro(f"{vals.max():.3f}" if len(vals) else "–"))
                self.metrics_table.setItem(row, 5, _ro(f"{vals.mean():.3f}" if len(vals) else "–"))
                self.metrics_table.setItem(row, 6, _ro(f"{vals.std():.3f}" if len(vals) else "–"))

        # ── Обновляем список метрик в комбо из реальных данных ────────────
        if hasattr(self, '_metric_combo'):
            self._metric_combo.blockSignals(True)
            self._metric_combo.clear()
            self._metric_combo.addItems(metric_data.metric_names)
            self._metric_combo.blockSignals(False)

        # ── Обновляем чекбоксы фильтра сервисов ──────────────────────────
        if hasattr(self, '_service_filter_row') and hasattr(self, '_service_checks'):
            for cb in self._service_checks.values():
                self._service_filter_row.removeWidget(cb)
                cb.deleteLater()
            self._service_checks.clear()
            for inst in metric_data.instance_names:
                cb = QCheckBox(inst)
                cb.setChecked(True)
                cb.stateChanged.connect(lambda _: self._plot_filtered())
                self._service_checks[inst] = cb
                self._service_filter_row.insertWidget(
                    self._service_filter_row.count() - 1, cb)

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
