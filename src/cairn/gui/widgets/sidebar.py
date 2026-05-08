"""Боковая панель CAIRN: источники данных и переключение модулей."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)


class StatusIndicator(QLabel):
    """Цветной индикатор статуса подключения (●)."""

    def __init__(self, parent=None):
        super().__init__("●", parent)
        self.setFixedWidth(16)
        self.set_unknown()

    def set_ok(self):      self.setObjectName("statusGood"); self._refresh()
    def set_error(self):   self.setObjectName("statusBad");  self._refresh()
    def set_unknown(self): self.setObjectName("statusWarn"); self._refresh()
    def _refresh(self):
        self.style().unpolish(self)
        self.style().polish(self)


class DataSourceSection(QWidget):
    """Одна секция источника данных (Метрики / Журналы / Трассировки)."""

    configure_requested = Signal(str)   # тип источника

    def __init__(self, title: str, options: list[str], parent=None):
        super().__init__(parent)
        self._title = title
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)

        hdr = QHBoxLayout()
        lbl = QLabel(title)
        lbl.setStyleSheet("font-weight: 600; font-size: 12px;")
        self.status = StatusIndicator()
        hdr.addWidget(lbl)
        hdr.addStretch()
        hdr.addWidget(self.status)
        layout.addLayout(hdr)

        self.combo = QComboBox()
        self.combo.addItems(options)
        layout.addWidget(self.combo)

        btn = QPushButton("Настроить")
        btn.setFixedHeight(28)
        btn.clicked.connect(lambda: self.configure_requested.emit(self._title))
        layout.addWidget(btn)


class ModuleCheckBox(QCheckBox):
    """Чекбокс модуля с опциональной подсказкой."""

    def __init__(self, text: str, enabled: bool = True, planned: bool = False, parent=None):
        super().__init__(text, parent)
        if planned:
            self.setEnabled(False)
            self.setToolTip("В разработке — модуль планируется в следующей версии")
        elif not enabled:
            self.setEnabled(False)
        else:
            self.setChecked(True)


class Sidebar(QWidget):
    """Боковая панель с секциями источников данных и списком модулей."""

    configure_source = Signal(str)
    module_toggled   = Signal(str, bool)
    connect_live     = Signal()   # запрос подключения живой системы

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Кнопка подключения живой системы ──────────────
        btn_live = QPushButton("⚡  Подключить систему")
        btn_live.setObjectName("primaryBtn")
        btn_live.setFixedHeight(34)
        btn_live.setToolTip(
            "Подключить любую живую систему через конфиг-файл\n"
            "(Prometheus, CSV и другие источники метрик)"
        )
        btn_live.clicked.connect(self.connect_live)
        root.addWidget(btn_live)

        self._live_status = QLabel("Не подключено")
        self._live_status.setStyleSheet("color: #6c7a9c; font-size: 10px; padding: 2px 8px;")
        root.addWidget(self._live_status)

        sep0 = QFrame()
        sep0.setFrameShape(QFrame.Shape.HLine)
        sep0.setStyleSheet("background-color: #2d3348;")
        sep0.setFixedHeight(1)
        root.addWidget(sep0)

        # ── Источники данных ──────────────────────────────
        src_header = QLabel("ИСТОЧНИКИ ДАННЫХ")
        src_header.setObjectName("sidebarHeader")
        root.addWidget(src_header)

        src_scroll = QScrollArea()
        src_scroll.setWidgetResizable(True)
        src_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        src_scroll.setFrameShape(QFrame.Shape.NoFrame)
        src_scroll.setMaximumHeight(240)

        src_container = QWidget()
        src_layout = QVBoxLayout(src_container)
        src_layout.setContentsMargins(10, 8, 10, 8)
        src_layout.setSpacing(12)

        self.metrics_src = DataSourceSection(
            "Метрики", ["CSV-файл", "Prometheus", "—"]
        )
        self.log_src = DataSourceSection(
            "Журналы", ["Текстовый файл", "Elasticsearch", "—"]
        )
        self.trace_src = DataSourceSection(
            "Трассировки", ["JSON-файл", "Jaeger", "—"]
        )

        for src in (self.metrics_src, self.log_src, self.trace_src):
            src.configure_requested.connect(self.configure_source)
            src_layout.addWidget(src)

        src_layout.addStretch()
        src_scroll.setWidget(src_container)
        root.addWidget(src_scroll)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #2d3348;")
        sep.setFixedHeight(1)
        root.addWidget(sep)

        # ── Модули ────────────────────────────────────────
        mod_header = QLabel("МОДУЛИ")
        mod_header.setObjectName("sidebarHeader")
        root.addWidget(mod_header)

        mod_scroll = QScrollArea()
        mod_scroll.setWidgetResizable(True)
        mod_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        mod_scroll.setFrameShape(QFrame.Shape.NoFrame)

        mod_container = QWidget()
        mod_layout = QVBoxLayout(mod_container)
        mod_layout.setContentsMargins(10, 8, 10, 8)
        mod_layout.setSpacing(2)

        # Включённые
        enabled_group = QGroupBox("Активные")
        eg_layout = QVBoxLayout(enabled_group)
        eg_layout.setSpacing(4)
        eg_layout.setContentsMargins(8, 12, 8, 8)

        self._enabled_modules: dict[str, ModuleCheckBox] = {}
        enabled = [
            ("metric_enc",      "Кодировщик метрик"),
            ("log_enc",         "Кодировщик журналов"),
            ("trace_enc",       "Кодировщик трассировок"),
            ("cond_gmm",        "Условная модель нормы"),
            ("vgae",            "Обнаружение скрытых факторов"),
            ("cf_module",       "Контрфактическое вмешательство"),
            ("funnel",          "Каскадная воронка"),
            ("graph_verifier",  "Верификатор графа"),
            ("alp_verifier",    "Логическая верификация"),
            ("template_gen",    "Шаблонный генератор"),
        ]
        for key, label in enabled:
            cb = ModuleCheckBox(label, enabled=True)
            cb.toggled.connect(lambda checked, k=key: self.module_toggled.emit(k, checked))
            self._enabled_modules[key] = cb
            eg_layout.addWidget(cb)

        mod_layout.addWidget(enabled_group)

        # Отключаемые
        opt_group = QGroupBox("Опциональные")
        og_layout = QVBoxLayout(opt_group)
        og_layout.setSpacing(4)
        og_layout.setContentsMargins(8, 12, 8, 8)

        self._optional_modules: dict[str, ModuleCheckBox] = {}
        optional = [
            ("ssm_branch",   "Ветвь спектр. анализа"),
            ("drift_detect", "Обнаружение дрейфа"),
            ("indep_loss",   "Ограничение независимости"),
        ]
        for key, label in optional:
            cb = ModuleCheckBox(label, enabled=True)
            cb.setChecked(False)
            cb.toggled.connect(lambda checked, k=key: self.module_toggled.emit(k, checked))
            self._optional_modules[key] = cb
            og_layout.addWidget(cb)

        mod_layout.addWidget(opt_group)

        # Запланированные
        plan_group = QGroupBox("В разработке")
        pg_layout = QVBoxLayout(plan_group)
        pg_layout.setSpacing(4)
        pg_layout.setContentsMargins(8, 12, 8, 8)

        planned = [
            "Локальная языковая модель",
            "Облачная ЯМ с RAG",
            "Физическая модель задержки",
            "Коннектор Prometheus",
            "Коннектор Elasticsearch",
            "Коннектор Jaeger",
        ]
        for label in planned:
            cb = ModuleCheckBox(label, planned=True)
            pg_layout.addWidget(cb)

        mod_layout.addWidget(plan_group)
        mod_layout.addStretch()
        mod_scroll.setWidget(mod_container)
        root.addWidget(mod_scroll)

    def set_live_status(self, name: str, ok: bool) -> None:
        """Обновляет статус подключённой живой системы."""
        if ok:
            self._live_status.setText(f"✓ {name}")
            self._live_status.setStyleSheet(
                "color: #3ecf8e; font-size: 10px; padding: 2px 8px; font-weight: 600;"
            )
        else:
            self._live_status.setText("Не подключено")
            self._live_status.setStyleSheet("color: #6c7a9c; font-size: 10px; padding: 2px 8px;")

    def show_section(self, section: str) -> None:
        """Показывает нужную секцию (sources или modules)."""
        # Прокручиваем к нужному разделу или просто показываем
        # В текущей реализации обе секции видны всегда
        pass

    def set_source_status(self, source: str, ok: bool):
        """Устанавливает статус подключения для источника данных."""
        mapping = {
            "Метрики": self.metrics_src.status,
            "Журналы": self.log_src.status,
            "Трассировки": self.trace_src.status,
        }
        if source in mapping:
            if ok:
                mapping[source].set_ok()
            else:
                mapping[source].set_error()
