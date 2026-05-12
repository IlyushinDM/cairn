"""Activity Bar – вертикальная панель иконок в стиле VS Code.

Верхняя группа: действия (Load, Analyze, Train)
Средняя группа: панели (Sources, Modules, Connect)
Нижняя: Settings
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import (
    QFrame, QSizePolicy, QToolButton, QVBoxLayout, QWidget,
)


def _make_icon(name: str, color: str = "#858585", size: int = 22):
    try:
        from cairn.gui.icons import icon as _icon
        return _icon(name, color=color, size=size)
    except Exception:
        return None


class ActivityButton(QToolButton):
    """Кнопка activity bar – иконка с tooltip."""

    def __init__(self, tooltip: str, icon_name: str,
                 checkable: bool = False, parent=None):
        super().__init__(parent)
        self._icon_name  = icon_name
        self._checkable  = checkable
        self.setToolTip(tooltip)
        self.setCheckable(checkable)
        self.setFixedSize(48, 48)
        self.setIconSize(QSize(22, 22))
        self._refresh_icon(False)

    def _refresh_icon(self, active: bool) -> None:
        color = "#cccccc" if active else "#858585"
        ic = _make_icon(self._icon_name, color=color)
        if ic:
            self.setIcon(ic)
        else:
            self.setText(self._icon_name[:2].upper())

    def setChecked(self, v: bool) -> None:
        super().setChecked(v)
        self._refresh_icon(v)

    def set_color(self, color: str) -> None:
        ic = _make_icon(self._icon_name, color=color)
        if ic:
            self.setIcon(ic)


class ActivityBar(QWidget):
    """VS Code-style activity bar."""

    # Сигналы действий
    load_requested    = Signal()
    analyze_requested = Signal()
    train_requested   = Signal()
    # Сигналы панелей
    panel_requested   = Signal(str)  # "sources"|"modules"|"connect"|"settings"|"none"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(48)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setObjectName("activityBarWidget")
        self.setStyleSheet("""
            QWidget#activityBarWidget {
                background: #1e1e1e;
                border-right: 1px solid #2d2d2d;
            }
            QToolButton {
                background: transparent;
                border: none;
                border-left: 2px solid transparent;
                padding: 0;
            }
            QToolButton:hover { background: rgba(255,255,255,0.06); }
            QToolButton:checked {
                border-left: 2px solid #cccccc;
                background: rgba(255,255,255,0.05);
            }
            QToolButton:disabled { opacity: 0.3; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Группа 1: Действия ────────────────────────────────────────────
        self.btn_load    = ActivityButton("Загрузить данные (Ctrl+O)", "load")
        self.btn_analyze = ActivityButton("Запустить анализ", "analyze")
        self.btn_train   = ActivityButton("Обучить модель", "train")
        self.btn_analyze.setEnabled(False)
        self.btn_train.setEnabled(False)

        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_analyze)
        layout.addWidget(self.btn_train)

        # Разделитель
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet("background:#2d2d2d; margin: 4px 8px;")
        sep1.setFixedHeight(1)
        layout.addWidget(sep1)

        # ── Группа 2: Панели (Sources / Modules / Connect) ────────────────
        self.btn_sources = ActivityButton("Источники данных", "sources",
                                          checkable=True)
        self.btn_modules = ActivityButton("Модули", "modules",
                                          checkable=True)
        self.btn_connect = ActivityButton("Подключить систему", "connect")
        self.btn_log     = ActivityButton("Журнал событий", "log", checkable=True)

        layout.addWidget(self.btn_sources)
        layout.addWidget(self.btn_modules)
        layout.addWidget(self.btn_connect)
        layout.addWidget(self.btn_log)

        layout.addStretch()

        # ── Нижняя: Settings ──────────────────────────────────────────────
        self.btn_settings = ActivityButton("Настройки", "settings")
        layout.addWidget(self.btn_settings)

        # Подключаем сигналы
        self.btn_load.clicked.connect(self.load_requested)
        self.btn_analyze.clicked.connect(self.analyze_requested)
        self.btn_train.clicked.connect(self.train_requested)

        self.btn_sources.toggled.connect(self._on_sources_toggled)
        self.btn_modules.toggled.connect(self._on_modules_toggled)
        self.btn_connect.clicked.connect(
            lambda: self.panel_requested.emit("connect"))
        self.btn_log.toggled.connect(
            lambda on: self._toggle("log", on))
        self.btn_settings.clicked.connect(
            lambda: self.panel_requested.emit("settings"))

    def _toggle(self, panel: str, on: bool) -> None:
        """Переключает боковую панель – только одна активна."""
        panel_btns = {
            "sources": self.btn_sources,
            "modules": self.btn_modules,
            "log":     self.btn_log,
        }
        if on:
            for name, btn in panel_btns.items():
                if name != panel:
                    btn.blockSignals(True)
                    btn.setChecked(False)
                    btn.blockSignals(False)
            self.panel_requested.emit(panel)
        else:
            self.panel_requested.emit("none")

    def _on_sources_toggled(self, checked: bool) -> None:
        self._toggle("sources", checked)

    def _on_modules_toggled(self, checked: bool) -> None:
        self._toggle("modules", checked)

    def set_analyze_enabled(self, enabled: bool) -> None:
        self.btn_analyze.setEnabled(enabled)

    def set_train_enabled(self, enabled: bool) -> None:
        self.btn_train.setEnabled(enabled)

    def set_connect_status(self, connected: bool) -> None:
        color = "#4ec9b0" if connected else "#858585"
        self.btn_connect.set_color(color)