"""Activity Bar – вертикальная панель иконок в стиле VS Code.

Верхняя группа: действия (Load, Analyze, Train)
Средняя группа: панели (Sources, Modules, Connect, Log)
Нижняя: Settings

Исправления:
- п.5: hover подсвечивает иконку (opacity), а не фон квадратиком
- п.6: правая граница – тонкая линия по всей высоте
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import (
    QFrame, QSizePolicy, QToolButton, QVBoxLayout, QWidget,
)


def _make_icon(name: str, color: str = "#858585", size: int = 22):
    """Загружает иконку из SVG с заменой цвета.
    Если SVG-рендеринг не работает – возвращает None (кнопка покажет текст).
    """
    try:
        from cairn.gui.icons import icon as _icon
        ic = _icon(name, color=color, size=size)
        # Проверяем что иконка реально создалась (не пустая)
        return ic if not ic.isNull() else None
    except Exception:
        return None


class ActivityButton(QToolButton):
    """Кнопка activity bar – иконка с tooltip.

    п.5: hover меняет яркость самой иконки, не рисует квадрат вокруг.
    При подключении (set_color) сохраняет акцентный цвет и при hover делает его ярче.
    """

    def __init__(self, tooltip: str, icon_name: str,
                 checkable: bool = False, parent=None):
        super().__init__(parent)
        self._icon_name    = icon_name
        self._checkable    = checkable
        self._accent_color: str | None = None   # None = обычный режим
        self.setToolTip(tooltip)
        self.setCheckable(checkable)
        self.setFixedSize(48, 48)
        self.setIconSize(QSize(24, 24))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh_icon(active=False, hover=False)

    # ── Цветовая логика ───────────────────────────────────────────────────

    def _icon_color(self, active: bool, hover: bool) -> str:
        """Вычисляет цвет иконки в зависимости от состояния."""
        if self._accent_color:
            # Акцентный режим (например, подключено): при hover чуть светлее
            return "#ffffff" if hover else self._accent_color
        if active or self.isChecked():
            return "#ffffff" if hover else "#cccccc"
        return "#aaaaaa" if hover else "#606060"

    def _refresh_icon(self, active: bool = False, hover: bool = False) -> None:
        color = self._icon_color(active, hover)
        ic = _make_icon(self._icon_name, color=color, size=24)
        if ic:
            self.setIcon(ic)
        else:
            self.setText(self._icon_name[:2].upper())

    # ── Переопределяем события для hover-эффекта на иконке ───────────────

    def enterEvent(self, event) -> None:
        self._refresh_icon(active=self.isChecked(), hover=True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._refresh_icon(active=self.isChecked(), hover=False)
        super().leaveEvent(event)

    def setChecked(self, v: bool) -> None:
        super().setChecked(v)
        self._refresh_icon(active=v, hover=False)

    def set_color(self, color: str) -> None:
        """Устанавливает акцентный цвет (например, при подключении)."""
        self._accent_color = color if color != "#858585" else None
        self._refresh_icon(active=self.isChecked(), hover=False)


class ActivityBar(QWidget):
    """VS Code-style activity bar.

    п.6: правая граница – тонкая линия 1px по всей высоте.
    п.5: кнопки без квадратного hover-фона.
    """

    # Сигналы действий
    load_requested       = Signal()
    analyze_requested    = Signal()
    train_requested      = Signal()
    disconnect_requested = Signal()        # п.1: отключение от живой системы
    # Сигналы панелей
    panel_requested      = Signal(str)  # "sources"|"modules"|"connect"|"settings"|"none"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(48)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setObjectName("activityBarWidget")

        # Стиль управляется через QSS-файл темы (dark_theme.qss / light_theme.qss)
        # НЕ используем inline setStyleSheet – он перекрывает глобальный QSS

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
        sep1.setStyleSheet("background:#333333; margin: 4px 8px;")
        sep1.setFixedHeight(1)
        layout.addWidget(sep1)

        # ── Группа 2: Панели ──────────────────────────────────────────────
        self.btn_sources = ActivityButton("Источники данных", "sources",
                                          checkable=True)
        self.btn_modules = ActivityButton("Модули", "modules",
                                          checkable=True)
        self.btn_connect = ActivityButton("Подключить систему", "connect")
        self.btn_disconnect = ActivityButton("Отключиться", "connect")
        self.btn_disconnect.setVisible(False)   # скрыта пока не подключено
        self.btn_disconnect.setToolTip("Отключиться от живой системы")

        self.btn_log     = ActivityButton("Журнал событий", "log", checkable=True)

        layout.addWidget(self.btn_sources)
        layout.addWidget(self.btn_modules)
        layout.addWidget(self.btn_connect)
        layout.addWidget(self.btn_disconnect)
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
        self.btn_disconnect.clicked.connect(self.disconnect_requested)
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
        """Меняет цвет иконки подключения."""
        color = "#4ec9b0" if connected else "#858585"
        self.btn_connect.set_color(color)

    def set_disconnect_visible(self, visible: bool) -> None:
        """п.1: показывает/скрывает кнопку отключения."""
        self.btn_disconnect.setVisible(visible)
        self.btn_connect.setVisible(not visible)
        if visible:
            self.btn_disconnect.set_color("#f44747")
            self.btn_disconnect.setToolTip("Отключиться от живой системы")

    def showEvent(self, event) -> None:
        """Перезагружаем иконки после того как виджет показан."""
        super().showEvent(event)
        for btn in self.findChildren(ActivityButton):
            btn._refresh_icon(active=btn.isChecked(), hover=False)
