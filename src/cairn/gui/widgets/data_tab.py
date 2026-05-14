"""Вкладка «Данные»: таблицы метрик/экземпляров + график временных рядов.

Исправления:
- п.2: селектор метрики (QComboBox), легенда слева с прокруткой,
       постоянные цвета сервисов, автоотображение при загрузке
- п.1 (предыдущей итерации): убраны флажки из таблиц
- п.2 (предыдущей итерации): горизонтальный Splitter для таблиц
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFrame,
    QHBoxLayout, QHeaderView, QLabel,
    QPushButton, QScrollArea, QSplitter,
    QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

# Фиксированная палитра — не зависит от видимости сервисов
_PALETTE = [
    "#4a9eff", "#3ecf8e", "#f6a623", "#ff5f5f",
    "#a78bfa", "#f97316", "#06b6d4", "#ec4899",
    "#84cc16", "#eab308",
]


class LegendItem(QWidget):
    """Одна строка в легенде: флажок + цветная полоска + название."""

    visibility_changed = Signal(int, bool)   # service_idx, visible

    def __init__(self, idx: int, name: str, color: str, parent=None):
        super().__init__(parent)
        self._idx = idx
        row = QHBoxLayout(self)
        row.setContentsMargins(4, 2, 4, 2)
        row.setSpacing(6)

        self._cb = QCheckBox()
        self._cb.setChecked(True)
        self._cb.toggled.connect(lambda v: self.visibility_changed.emit(idx, v))
        row.addWidget(self._cb)

        dot = QLabel()
        dot.setFixedSize(12, 12)
        dot.setStyleSheet(f"background:{color}; border-radius:2px;")
        row.addWidget(dot)

        lbl = QLabel(name)
        lbl.setToolTip(name)
        row.addWidget(lbl, stretch=1)


class MetricLegend(QWidget):
    """Прокручиваемая легенда сервисов слева от графика."""

    visibility_changed = Signal(int, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(200)
        self.setObjectName("metricLegend")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        hdr = QLabel("Сервисы")
        hdr.setObjectName("sectionTitle")
        hdr.setContentsMargins(6, 6, 6, 4)
        outer.addWidget(hdr)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameStyle(0)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._inner = QWidget()
        self._inner_layout = QVBoxLayout(self._inner)
        self._inner_layout.setContentsMargins(0, 0, 0, 0)
        self._inner_layout.setSpacing(0)
        self._inner_layout.addStretch()

        self._scroll.setWidget(self._inner)
        outer.addWidget(self._scroll)

    def set_services(self, names: list[str]) -> None:
        # Очищаем
        while self._inner_layout.count() > 1:
            item = self._inner_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, name in enumerate(names):
            color = _PALETTE[i % len(_PALETTE)]
            item = LegendItem(i, name, color)
            item.visibility_changed.connect(self.visibility_changed)
            self._inner_layout.insertWidget(self._inner_layout.count() - 1, item)


class DetachedTableWindow(QWidget):
    """Таблица в отдельном окне — п.4."""

    def __init__(self, title: str, source_table: QTableWidget, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle(title)
        self.resize(600, 400)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Клонируем таблицу (копируем данные)
        tbl = QTableWidget(source_table.rowCount(),
                           source_table.columnCount())

        # Заголовки
        headers = [source_table.horizontalHeaderItem(c).text()
                   if source_table.horizontalHeaderItem(c) else ""
                   for c in range(source_table.columnCount())]
        tbl.setHorizontalHeaderLabels(headers)
        tbl.verticalHeader().setVisible(False)
        tbl.setAlternatingRowColors(True)
        tbl.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        tbl.horizontalHeader().setStretchLastSection(True)

        # Копируем данные
        for r in range(source_table.rowCount()):
            for c in range(source_table.columnCount()):
                item = source_table.item(r, c)
                if item:
                    new_item = QTableWidgetItem(item.text())
                    new_item.setForeground(item.foreground())
                    tbl.setItem(r, c, new_item)

        layout.addWidget(tbl)




class DataTab(QWidget):
    """Вкладка «Данные»."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._metric_data = None
        self._visible_services: set[int] | None = None
        self._fig = None
        self._ax  = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Главный сплиттер: таблицы сверху, график снизу
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setObjectName("mainSplitter")

        # ── Верхняя часть: таблицы ───────────────────────────────────────
        tables_widget  = QWidget()
        tables_layout  = QVBoxLayout(tables_widget)
        tables_layout.setContentsMargins(0, 0, 0, 0)
        tables_layout.setSpacing(4)

        # Горизонтальный сплиттер для двух таблиц (п.2 предыдущей итерации)
        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.setHandleWidth(8)  # шире разделитель между таблицами

        # Таблица метрик
        metrics_frame = QWidget()
        mf_layout = QVBoxLayout(metrics_frame)
        mf_layout.setContentsMargins(0, 0, 0, 0)
        mf_layout.setSpacing(4)
        # п.4: строка с заголовком и кнопкой «В окне»
        mf_hdr = QHBoxLayout()
        mf_hdr.addWidget(self._section_label("МЕТРИКИ"))
        mf_hdr.addStretch()
        btn_metrics_pop = QPushButton("⬡ В окне")
        btn_metrics_pop.setFixedHeight(22)
        btn_metrics_pop.setToolTip("Открыть таблицу метрик в отдельном окне")
        btn_metrics_pop.setStyleSheet("font-size: 11px; padding: 0 6px;")
        btn_metrics_pop.clicked.connect(
            lambda: self._open_in_window("Метрики", self.metrics_table))
        mf_hdr.addWidget(btn_metrics_pop)
        mf_layout.addLayout(mf_hdr)

        self.metrics_table = QTableWidget(0, 6)
        self.metrics_table.setHorizontalHeaderLabels(
            ["Экземпляр", "Метрика", "Мин", "Макс", "Среднее", "σ"]
        )
        self.metrics_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.metrics_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        for c in range(2, 6):
            self.metrics_table.horizontalHeader().setSectionResizeMode(
                c, QHeaderView.ResizeMode.ResizeToContents)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.metrics_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        mf_layout.addWidget(self.metrics_table)
        h_split.addWidget(metrics_frame)

        # Таблица экземпляров
        instances_frame = QWidget()
        if_layout = QVBoxLayout(instances_frame)
        if_layout.setContentsMargins(0, 0, 0, 0)
        if_layout.setSpacing(4)
        # п.4: строка с заголовком и кнопкой «В окне»
        if_hdr = QHBoxLayout()
        if_hdr.addWidget(self._section_label("ЭКЗЕМПЛЯРЫ СЕРВИСОВ"))
        if_hdr.addStretch()
        btn_inst_pop = QPushButton("⬡ В окне")
        btn_inst_pop.setFixedHeight(22)
        btn_inst_pop.setToolTip("Открыть таблицу экземпляров в отдельном окне")
        btn_inst_pop.setStyleSheet("font-size: 11px; padding: 0 6px;")
        btn_inst_pop.clicked.connect(
            lambda: self._open_in_window("Экземпляры сервисов", self.instances_table))
        if_hdr.addWidget(btn_inst_pop)
        if_layout.addLayout(if_hdr)

        self.instances_table = QTableWidget(0, 5)
        self.instances_table.setHorizontalHeaderLabels(
            ["Имя", "Сервис", "Хост", "CPU", "Версия"]
        )
        self.instances_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.instances_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch)
        for c in range(2, 5):
            self.instances_table.horizontalHeader().setSectionResizeMode(
                c, QHeaderView.ResizeMode.ResizeToContents)
        self.instances_table.setAlternatingRowColors(True)
        self.instances_table.verticalHeader().setVisible(False)
        self.instances_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.instances_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        if_layout.addWidget(self.instances_table)
        h_split.addWidget(instances_frame)

        h_split.setSizes([500, 300])
        tables_layout.addWidget(h_split)
        splitter.addWidget(tables_widget)

        # ── Нижняя часть: селектор + легенда + график ────────────────────
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(0, 4, 0, 0)
        chart_layout.setSpacing(4)

        # Строка: заголовок + комбобокс выбора метрики
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(self._section_label("ВРЕМЕННЫЕ РЯДЫ"))
        ctrl_row.addStretch()
        ctrl_row.addWidget(QLabel("Метрика:"))
        self._metric_combo = QComboBox()
        self._metric_combo.setMinimumWidth(160)
        self._metric_combo.setFixedHeight(26)
        self._metric_combo.currentTextChanged.connect(self._on_metric_changed)
        ctrl_row.addWidget(self._metric_combo)
        chart_layout.addLayout(ctrl_row)

        # Строка: легенда слева + canvas справа
        chart_row = QHBoxLayout()
        chart_row.setSpacing(0)

        self._legend = MetricLegend()
        self._legend.visibility_changed.connect(self._on_visibility_changed)
        chart_row.addWidget(self._legend)

        # Разделитель
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFixedWidth(1)
        sep.setStyleSheet("background: #3f3f46;")
        chart_row.addWidget(sep)

        self._canvas_container = QWidget()
        self._canvas_layout = QVBoxLayout(self._canvas_container)
        self._canvas_layout.setContentsMargins(0, 0, 0, 0)
        self._canvas_widget = self._build_canvas()
        self._canvas_layout.addWidget(self._canvas_widget)
        chart_row.addWidget(self._canvas_container, stretch=1)

        chart_layout.addLayout(chart_row, stretch=1)
        splitter.addWidget(chart_widget)
        splitter.setSizes([280, 300])

        layout.addWidget(splitter)

    # ── Вспомогательные ──────────────────────────────────────────────────

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("sectionTitle")
        return lbl

    def _build_canvas(self) -> QWidget:
        """Строит matplotlib canvas или заглушку."""
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            fig = Figure(figsize=(10, 3), facecolor="none")
            ax  = fig.add_subplot(111)
            ax.set_facecolor("none")
            ax.tick_params(colors="#6c7a9c")
            ax.spines[:].set_color("#2d3348")
            ax.set_xlabel("Время (с)", color="#6c7a9c", fontsize=10)
            ax.set_ylabel("Значение",  color="#6c7a9c", fontsize=10)
            ax.set_title("Загрузите данные для отображения",
                         color="#6c7a9c", fontsize=10)
            fig.tight_layout(pad=1.5)

            self._fig = fig
            self._ax  = ax
            canvas = FigureCanvasQTAgg(fig)
            canvas.setMinimumHeight(160)
            return canvas
        except ImportError:
            self._fig = None
            self._ax  = None
            lbl = QLabel("График временных рядов\n(требуется matplotlib)")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(
                "color:#6c7a9c; font-size:13px; border:1px dashed #2d3348;")
            lbl.setMinimumHeight(160)
            return lbl

    # ── Загрузка данных ───────────────────────────────────────────────────

    def load_metric_data(self, metric_data) -> None:
        """Заполняет таблицу метрик и обновляет комбобокс."""
        self._metric_data = metric_data
        self._visible_services = set(range(metric_data.n_instances))

        # Таблица
        self.metrics_table.setRowCount(0)
        for ni, inst in enumerate(metric_data.instance_names):
            for mi, metric in enumerate(metric_data.metric_names):
                vals = metric_data.values[:, ni, mi]
                vals = vals[~np.isnan(vals)]
                row  = self.metrics_table.rowCount()
                self.metrics_table.insertRow(row)
                self.metrics_table.setItem(row, 0, QTableWidgetItem(inst))
                self.metrics_table.setItem(row, 1, QTableWidgetItem(metric))
                self.metrics_table.setItem(row, 2, QTableWidgetItem(
                    f"{vals.min():.3f}" if len(vals) else "—"))
                self.metrics_table.setItem(row, 3, QTableWidgetItem(
                    f"{vals.max():.3f}" if len(vals) else "—"))
                self.metrics_table.setItem(row, 4, QTableWidgetItem(
                    f"{vals.mean():.3f}" if len(vals) else "—"))
                self.metrics_table.setItem(row, 5, QTableWidgetItem(
                    f"{vals.std():.3f}" if len(vals) else "—"))

        # Легенда
        self._legend.set_services(metric_data.instance_names)
        self._visible_services = set(range(metric_data.n_instances))

        # Комбобокс метрик
        self._metric_combo.blockSignals(True)
        self._metric_combo.clear()
        for m in metric_data.metric_names:
            self._metric_combo.addItem(m)
        self._metric_combo.blockSignals(False)

        # Сразу строим график для первой метрики
        self._draw_current_metric()

    def load_topology(self, topo) -> None:
        """Заполняет таблицу экземпляров."""
        self.instances_table.setRowCount(0)
        for inst in topo.instances:
            row = self.instances_table.rowCount()
            self.instances_table.insertRow(row)
            self.instances_table.setItem(row, 0, QTableWidgetItem(inst.name))
            self.instances_table.setItem(row, 1, QTableWidgetItem(inst.service))
            self.instances_table.setItem(row, 2, QTableWidgetItem(inst.host))
            self.instances_table.setItem(row, 3, QTableWidgetItem(
                str(getattr(inst, "cpu_limit", "—"))))
            self.instances_table.setItem(row, 4, QTableWidgetItem(
                getattr(inst, "version", "—")))

    # ── Обработчики событий ───────────────────────────────────────────────

    def _on_metric_changed(self, metric_name: str) -> None:
        self._draw_current_metric()

    def _on_visibility_changed(self, svc_idx: int, visible: bool) -> None:
        if self._visible_services is None:
            return
        if visible:
            self._visible_services.add(svc_idx)
        else:
            self._visible_services.discard(svc_idx)
        self._draw_current_metric()

    # ── Отрисовка ─────────────────────────────────────────────────────────

    def _draw_current_metric(self) -> None:
        """Перерисовывает график для текущей метрики и видимых сервисов."""
        if self._metric_data is None or self._ax is None:
            return
        metric_name = self._metric_combo.currentText()
        if not metric_name:
            return

        # Найти индекс метрики
        names = list(self._metric_data.metric_names)
        if metric_name not in names:
            return
        mi = names.index(metric_name)

        timestamps = self._metric_data.timestamps
        self._ax.clear()
        self._ax.set_facecolor(self._ax.get_facecolor())  # сохраняем цвет

        has_data = False
        for ni, inst in enumerate(self._metric_data.instance_names):
            if self._visible_services is not None and ni not in self._visible_services:
                continue
            vals  = self._metric_data.values[:, ni, mi]
            color = _PALETTE[ni % len(_PALETTE)]
            self._ax.plot(timestamps, vals, label=inst,
                          color=color, linewidth=1.5)
            has_data = True

        if not has_data:
            self._ax.set_title("Все сервисы скрыты",
                               color="#6c7a9c", fontsize=10)
        else:
            self._ax.set_title(metric_name, color="#a0a8bc",
                               fontsize=10, pad=4)

        self._ax.tick_params(colors="#6c7a9c")
        self._ax.spines[:].set_color("#2d3348")
        self._ax.set_xlabel("Время (с)", color="#6c7a9c", fontsize=9)
        self._ax.set_ylabel("Значение",  color="#6c7a9c", fontsize=9)

        if self._fig is not None:
            self._fig.tight_layout(pad=1.5)
        if hasattr(self._canvas_widget, "draw"):
            self._canvas_widget.draw()

    def _open_in_window(self, title: str, table: QTableWidget) -> None:
        """Открывает таблицу в отдельном плавающем окне."""
        win = DetachedTableWindow(title, table, parent=self)
        win.show()
        win.raise_()
        self._detached_windows = getattr(self, "_detached_windows", [])
        self._detached_windows.append(win)

    def plot_series(self, timestamps, values, labels: list[str]) -> None:
        """Внешний вызов отрисовки (для live-режима)."""
        if self._ax is None:
            return
        self._ax.clear()
        self._ax.set_facecolor("none")
        for i, (vals, label) in enumerate(zip(values, labels)):
            color = _PALETTE[i % len(_PALETTE)]
            self._ax.plot(timestamps, vals, label=label,
                          color=color, linewidth=1.5)
        self._ax.tick_params(colors="#6c7a9c")
        self._ax.spines[:].set_color("#2d3348")
        self._ax.set_xlabel("Время (с)", color="#6c7a9c", fontsize=9)
        if self._fig is not None:
            self._fig.tight_layout(pad=1.5)
        if hasattr(self._canvas_widget, "draw"):
            self._canvas_widget.draw()
