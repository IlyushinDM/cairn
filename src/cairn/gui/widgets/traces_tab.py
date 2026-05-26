"""Вкладка 'Трассировки' – latency per service из loadgenerator."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QPushButton, QHBoxLayout, QHeaderView, QLabel,
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
        thdr = QHBoxLayout()
        lbl = QLabel("ЗАДЕРЖКА ПО СЕРВИСАМ (p50, мс)")
        lbl.setObjectName("sectionTitle")
        thdr.addWidget(lbl)
        thdr.addStretch()
        btn_tpop = QPushButton("⬡ В окне")
        btn_tpop.setFixedHeight(22)
        btn_tpop.setStyleSheet("font-size: 11px; padding: 0 6px;")
        btn_tpop.clicked.connect(lambda: self._open_table_in_window())
        thdr.addWidget(btn_tpop)
        layout.addLayout(thdr)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels([
            "Сервис", "p50 (мс)", "Запросов",
            "Аномалия", "Маршруты",
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

    def _open_table_in_window(self) -> None:
        """Открывает таблицу трассировок в отдельном окне."""
        from PySide6.QtWidgets import (
            QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem)
        dlg = QDialog(self)
        dlg.setWindowTitle("Трассировки – задержка по сервисам")
        dlg.resize(750, 400)
        _lyt = QVBoxLayout(dlg)
        tbl = QTableWidget(self._table.rowCount(), self._table.columnCount())
        headers = [self._table.horizontalHeaderItem(c).text()
                   if self._table.horizontalHeaderItem(c) else ""
                   for c in range(self._table.columnCount())]
        tbl.setHorizontalHeaderLabels(headers)
        tbl.verticalHeader().setVisible(False)
        tbl.horizontalHeader().setStretchLastSection(True)
        tbl.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        tbl.setAlternatingRowColors(True)
        for r in range(self._table.rowCount()):
            for c in range(self._table.columnCount()):
                it = self._table.item(r, c)
                if it:
                    tbl.setItem(r, c, QTableWidgetItem(it.text()))
        _lyt.addWidget(tbl)
        # show() вместо exec() – не блокирует основное окно
        dlg.setAttribute(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.setModal(False)
        dlg.show()
        dlg.raise_()

    def load_trace_data(self, trace_data) -> None:
        """Отображает данные от LatencyTraceConnector."""
        self._table.setRowCount(0)

        n_slow = len(trace_data.slow_services)
        n_total = trace_data.n_services
        self._status_label.setText(
            f"Сервисов: {n_total} | "
            f"Медленных: {n_slow} | "
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
                f"МЕДЛЕННО  ×{1+sl.anomaly_score:.1f}"
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

            # Подсветка строки – только foreground, без hardcoded фона
            if sl.is_slow:
                for c in range(5):
                    it = self._table.item(row, c)
                    if it:
                        it.setForeground(QColor("#f44747"))
