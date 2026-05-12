"""Вкладка 'Трассировки' – latency per service из loadgenerator."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel,
    QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)


class TracesTab(QWidget):
    """Вкладка latency трассировок."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        hdr = QHBoxLayout()
        self._status_label = QLabel(
            "Трассировки не загружены. Подключите систему."
        )
        self._status_label.setStyleSheet("color: #858585; font-size: 11px;")
        hdr.addWidget(self._status_label)

        src_label = QLabel("Источник: Locust loadgenerator logs")
        src_label.setStyleSheet("color: #3f3f46; font-size: 10px;")
        hdr.addStretch()
        hdr.addWidget(src_label)
        layout.addLayout(hdr)

        # Таблица latency
        lbl = QLabel("LATENCY PER SERVICE (p50, мс)")
        lbl.setObjectName("sectionTitle")
        layout.addWidget(lbl)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels([
            "Сервис", "p50 (мс)", "Запросов",
            "Аномалия", "Endpoints",
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
        layout.addWidget(self._table)

        # Подсказка
        hint = QLabel(
            "Примечание: для полных распределённых трассировок "
            "требуется OpenTelemetry инструментирование сервисов."
        )
        hint.setStyleSheet("color: #3f3f46; font-size: 10px; padding: 4px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

    def load_trace_data(self, trace_data) -> None:
        """Отображает данные от LatencyTraceConnector."""
        self._table.setRowCount(0)

        n_slow = len(trace_data.slow_services)
        n_total = trace_data.n_services
        self._status_label.setText(
            f"Сервисов: {n_total} | "
            f"Slow: {n_slow} | "
            f"Источник: {trace_data.source}"
        )
        color = "#f44747" if n_slow > 0 else "#4ec9b0"
        self._status_label.setStyleSheet(f"color: {color}; font-size: 11px;")

        # Сортируем по latency убыванию
        sorted_services = sorted(
            trace_data.services.items(),
            key=lambda x: x[1].avg_p50_ms,
            reverse=True,
        )

        for name, sl in sorted_services:
            row = self._table.rowCount()
            self._table.insertRow(row)

            def _ro(text, align=None):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemFlag.ItemIsSelectable |
                              Qt.ItemFlag.ItemIsEnabled)
                if align:
                    item.setTextAlignment(align)
                return item

            self._table.setItem(row, 0, _ro(name))

            # p50 latency
            p50_item = _ro(
                f"{sl.avg_p50_ms:.0f}",
                Qt.AlignmentFlag.AlignCenter
            )
            if sl.avg_p50_ms > 1000:
                p50_item.setForeground(QColor("#f44747"))
            elif sl.avg_p50_ms > 500:
                p50_item.setForeground(QColor("#cca700"))
            else:
                p50_item.setForeground(QColor("#4ec9b0"))
            self._table.setItem(row, 1, p50_item)

            self._table.setItem(row, 2, _ro(
                str(sl.request_count), Qt.AlignmentFlag.AlignCenter))

            # Аномалия
            anom_text = (
                f"SLOW  ×{1+sl.anomaly_score:.1f}"
                if sl.is_slow else "норма"
            )
            anom_item = _ro(anom_text, Qt.AlignmentFlag.AlignCenter)
            if sl.is_slow:
                anom_item.setForeground(QColor("#f44747"))
            self._table.setItem(row, 3, anom_item)

            # Endpoints
            ep_text = ", ".join(sl.endpoints[:3])
            if len(sl.endpoints) > 3:
                ep_text += f" (+{len(sl.endpoints)-3})"
            self._table.setItem(row, 4, _ro(ep_text))

            # Подсветка строки
            if sl.is_slow:
                for c in range(5):
                    it = self._table.item(row, c)
                    if it:
                        it.setBackground(QColor("#3d1a1a"))
