"""Вкладка 'Журналы' – отображение лог-аномалий по контейнерам.

Показывает:
  - Таблицу контейнеров с частотой ERROR/WARN и статусом аномалии
  - График частоты ошибок во времени
  - Топ повторяющихся ошибок для выбранного контейнера
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel,
    QListWidget, QListWidgetItem, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)


class LogsTab(QWidget):
    """Вкладка журналов контейнеров."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Заголовок со статусом
        hdr = QHBoxLayout()
        self._status_label = QLabel("Журналы не загружены. Подключите систему.")
        self._status_label.setStyleSheet("color: #858585; font-size: 11px;")
        hdr.addWidget(self._status_label)
        hdr.addStretch()
        layout.addLayout(hdr)

        splitter = QSplitter(Qt.Orientation.Vertical)

        # ── Верхняя: таблица контейнеров ─────────────────────────────────
        top = QWidget()
        tl  = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(4)

        lbl = QLabel("СТАТУС ЖУРНАЛОВ")
        lbl.setObjectName("sectionTitle")
        tl.addWidget(lbl)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels([
            "Контейнер", "ERROR/мин", "WARN/мин",
            "Аномалия", "Последние ошибки",
        ])
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        for c in (1, 2, 3):
            hh.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.itemSelectionChanged.connect(self._on_row_selected)
        tl.addWidget(self._table)
        splitter.addWidget(top)

        # ── Нижняя: детали выбранного контейнера ─────────────────────────
        bottom = QWidget()
        bl = QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(4)

        self._detail_label = QLabel("Топ повторяющихся ошибок:")
        self._detail_label.setObjectName("sectionTitle")
        bl.addWidget(self._detail_label)

        self._error_list = QListWidget()
        self._error_list.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self._error_list.setStyleSheet("""
            QListWidget {
                background: #1e1e1e;
                border: 1px solid #3f3f46;
                font-family: "Consolas", monospace;
                font-size: 11px;
            }
            QListWidget::item { padding: 3px 6px; }
        """)
        bl.addWidget(self._error_list)
        splitter.addWidget(bottom)

        splitter.setSizes([350, 200])
        layout.addWidget(splitter)

        self._log_data = None

    # ── Публичный API ─────────────────────────────────────────────────────

    def load_log_data(self, log_data) -> None:
        """Отображает данные от DockerLogConnector."""
        self._log_data = log_data
        self._table.setRowCount(0)

        n_anomalous = len(log_data.anomalous_containers)
        n_total     = log_data.n_containers
        self._status_label.setText(
            f"Контейнеров: {n_total} | "
            f"Аномалий: {n_anomalous} | "
            f"Время сбора: {log_data.collect_time:.0f}с"
        )
        color = "#f44747" if n_anomalous > 0 else "#4ec9b0"
        self._status_label.setStyleSheet(f"color: {color}; font-size: 11px;")

        for name, ts in sorted(log_data.series.items()):
            row = self._table.rowCount()
            self._table.insertRow(row)

            # Контейнер
            item_name = QTableWidgetItem(name)
            item_name.setFlags(Qt.ItemFlag.ItemIsSelectable |
                               Qt.ItemFlag.ItemIsEnabled)
            self._table.setItem(row, 0, item_name)

            # ERROR rate
            err_rate = (sum(ts.error_rate) / len(ts.error_rate)
                        if ts.error_rate else 0.0)
            item_err = QTableWidgetItem(f"{err_rate:.2f}")
            item_err.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_err.setFlags(Qt.ItemFlag.ItemIsSelectable |
                              Qt.ItemFlag.ItemIsEnabled)
            if err_rate > 1.0:
                item_err.setForeground(QColor("#f44747"))
            self._table.setItem(row, 1, item_err)

            # WARN rate
            warn_rate = (sum(ts.warn_rate) / len(ts.warn_rate)
                         if ts.warn_rate else 0.0)
            item_warn = QTableWidgetItem(f"{warn_rate:.2f}")
            item_warn.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_warn.setFlags(Qt.ItemFlag.ItemIsSelectable |
                               Qt.ItemFlag.ItemIsEnabled)
            if warn_rate > 5.0:
                item_warn.setForeground(QColor("#cca700"))
            self._table.setItem(row, 2, item_warn)

            # Аномалия
            anom_text = f"{'ДА' if ts.is_anomalous else 'нет'}"
            item_anom = QTableWidgetItem(anom_text)
            item_anom.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_anom.setFlags(Qt.ItemFlag.ItemIsSelectable |
                               Qt.ItemFlag.ItemIsEnabled)
            if ts.is_anomalous:
                item_anom.setForeground(QColor("#f44747"))
                item_anom.setText(f"ДА  score={ts.anomaly_score:.2f}")
            self._table.setItem(row, 3, item_anom)

            # Последние ошибки (превью)
            preview = ts.top_errors[0][:60] if ts.top_errors else "–"
            item_err_prev = QTableWidgetItem(preview)
            item_err_prev.setFlags(Qt.ItemFlag.ItemIsSelectable |
                                   Qt.ItemFlag.ItemIsEnabled)
            self._table.setItem(row, 4, item_err_prev)

            # Подсвечиваем строку при аномалии
            if ts.is_anomalous:
                for c in range(5):
                    it = self._table.item(row, c)
                    if it:
                        it.setBackground(QColor("#3d1a1a"))

    def _on_row_selected(self) -> None:
        """Показывает топ ошибок для выбранного контейнера."""
        rows = self._table.selectedItems()
        if not rows or self._log_data is None:
            return
        row  = rows[0].row()
        name = self._table.item(row, 0).text()
        ts   = self._log_data.series.get(name)
        if ts is None:
            return

        self._error_list.clear()
        self._detail_label.setText(f"Топ ошибок: {name}")

        if not ts.top_errors:
            item = QListWidgetItem("Нет повторяющихся ошибок")
            item.setForeground(QColor("#858585"))
            self._error_list.addItem(item)
        else:
            for err_msg in ts.top_errors:
                item = QListWidgetItem(err_msg)
                item.setForeground(QColor("#f44747"))
                self._error_list.addItem(item)
