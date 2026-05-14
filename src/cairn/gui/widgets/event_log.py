"""Журнал событий CAIRN – фиксирует инциденты и действия системы.

Показывает оператору что происходило в системе с временными метками.
Новые события добавляются сверху.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QVBoxLayout, QWidget,
)


class EventLevel(str, Enum):
    INFO    = "INFO"
    WARNING = "WARN"
    ERROR   = "ERROR"
    SUCCESS = "OK"


LEVEL_COLORS = {
    EventLevel.INFO:    "#858585",
    EventLevel.WARNING: "#cca700",
    EventLevel.ERROR:   "#f44747",
    EventLevel.SUCCESS: "#4ec9b0",
}

LEVEL_ICONS = {
    EventLevel.INFO:    "·",
    EventLevel.WARNING: "▲",
    EventLevel.ERROR:   "●",
    EventLevel.SUCCESS: "✓",
}


class EventLogWidget(QWidget):
    """Виджет журнала событий."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._events: list[dict] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Заголовок
        hdr = QHBoxLayout()
        lbl = QLabel("ЖУРНАЛ СОБЫТИЙ")
        lbl.setObjectName("sectionTitle")
        hdr.addWidget(lbl)
        hdr.addStretch()

        btn_clear = QPushButton("Очистить")
        btn_clear.setFixedHeight(22)
        btn_clear.setFixedWidth(80)
        btn_clear.clicked.connect(self.clear)
        hdr.addWidget(btn_clear)
        layout.addLayout(hdr)

        # Список событий
        self._list = QListWidget()
        self._list.setAlternatingRowColors(False)
        self._list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self._list.setObjectName("eventLogList")
        self._list.setFont(__import__("PySide6.QtGui", fromlist=["QFont"]).QFont(
            "Consolas", 10))
        layout.addWidget(self._list)

    def add_event(
        self,
        message: str,
        level: EventLevel = EventLevel.INFO,
        service: str | None = None,
        score: float | None = None,
    ) -> None:
        """Добавляет событие в журнал."""
        ts = datetime.now().strftime("%H:%M:%S")
        icon = LEVEL_ICONS[level]
        color = LEVEL_COLORS[level]

        parts = [f"{ts}  {icon}  {message}"]
        if service:
            parts.append(f"  [{service}]")
        if score is not None:
            parts.append(f"  score={score:.3f}")

        text = "".join(parts)

        item = QListWidgetItem(text)
        item.setForeground(QColor(color))
        font = QFont("Consolas", 11)
        item.setFont(font)

        # Вставляем сверху
        self._list.insertItem(0, item)
        self._events.insert(0, {
            "ts": ts, "level": level, "message": message,
            "service": service, "score": score,
        })

        # Ограничиваем размер журнала
        if self._list.count() > 200:
            self._list.takeItem(self._list.count() - 1)

    def add_anomaly(self, service: str, score: float, fault_type: str) -> None:
        self.add_event(
            f"АНОМАЛИЯ: {service} – тип: {fault_type}",
            level=EventLevel.ERROR,
            service=service,
            score=score,
        )

    def add_analysis_result(self, root_name: str, score: float,
                             fault_type: str, confidence: float) -> None:
        self.add_event(
            f"Первопричина: {root_name} ({fault_type}) "
            f"– уверенность {confidence:.0%}",
            level=EventLevel.SUCCESS,
            service=root_name,
            score=score,
        )

    def add_info(self, message: str) -> None:
        self.add_event(message, level=EventLevel.INFO)

    def add_warning(self, message: str, service: str | None = None) -> None:
        self.add_event(message, level=EventLevel.WARNING, service=service)

    def clear(self) -> None:
        self._list.clear()
        self._events.clear()

    def export_text(self) -> str:
        """Экспортирует журнал в текст."""
        lines = []
        for ev in reversed(self._events):
            line = f"[{ev['ts']}] {ev['level'].value}  {ev['message']}"
            if ev.get("service"):
                line += f"  [{ev['service']}]"
            if ev.get("score") is not None:
                line += f"  score={ev['score']:.3f}"
            lines.append(line)
        return "\n".join(lines)
