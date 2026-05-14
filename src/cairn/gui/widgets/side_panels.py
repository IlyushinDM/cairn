"""Боковые панели CAIRN — Sources и Modules как отдельные виджеты.

Каждая панель показывается независимо при нажатии на иконку в ActivityBar.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)


class SidePanelBase(QWidget):
    """Базовый класс боковой панели."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("sidePanel")
        self.setMinimumWidth(200)
        self.setMaximumWidth(500)
        self.resize(240, self.height())
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Expanding)
        self.setObjectName("sidePanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Заголовок панели
        hdr = QLabel(title.upper())
        hdr.setObjectName("sidebarHeader")
        hdr.setObjectName("sidebarHeader")
        layout.addWidget(hdr)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 8, 8, 8)
        self._content_layout.setSpacing(8)
        scroll.setWidget(self._content)
        layout.addWidget(scroll)

    def _add_section(self, title: str) -> QGroupBox:
        grp = QGroupBox(title)
        grp.setObjectName("sideGroupBox")
        return grp


class SourcesPanel(SidePanelBase):
    """Панель 'Источники данных'."""

    configure_source = Signal(str)

    def __init__(self, parent=None):
        super().__init__("Источники данных", parent)
        self._build()

    def _build(self) -> None:
        cl = self._content_layout

        for src_title, options in [
            ("Метрики",     ["CSV-файл", "Prometheus", "docker_stats", "—"]),
            ("Журналы",     ["Текстовый файл", "Elasticsearch", "—"]),
            ("Трассировки", ["JSON-файл", "Jaeger", "—"]),
        ]:
            grp = QGroupBox(src_title)
            gl  = QVBoxLayout(grp)
            gl.setSpacing(4)
            gl.setContentsMargins(0, 12, 0, 0)

            combo = QComboBox()
            combo.addItems(options)
            gl.addWidget(combo)

            btn = QPushButton("Настроить")
            btn.setFixedHeight(26)
            btn.clicked.connect(
                lambda _, t=src_title: self.configure_source.emit(t))
            gl.addWidget(btn)
            cl.addWidget(grp)

        cl.addStretch()

    def set_status(self, source: str, ok: bool) -> None:
        pass   # TODO: цветной индикатор


class ModulesPanel(SidePanelBase):
    """Панель 'Модули'."""

    module_toggled = Signal(str, bool)

    def __init__(self, parent=None):
        super().__init__("Модули", parent)
        self._checkboxes: dict[str, QCheckBox] = {}
        self._build()

    def _build(self) -> None:
        cl = self._content_layout

        # ── Активные модули (влияют на анализ) ───────────────────────────
        active = [
            ("graph_verifier", "Топологическая корректировка",
             "Корректирует скоры с учётом каскадного эффекта. "
             "Upstream-сервисы получают бонус."),
            ("cf_module",      "Контрфактический модуль",
             "VGAE + GNN для причинно-следственного вывода. "
             "Отключение → только NLL-ранжирование."),
            ("alp_verifier",   "Логическая верификация",
             "Проверяет цепочку доказательств через ALP. "
             "Отключение → без верификации корректности."),
        ]

        # ── Опциональные (расширенный режим) ──────────────────────────
        optional = [
            ("ssm_branch",   "Спектральный анализ (SSM)",
             "Добавляет спектральную ветвь в Metric Encoder. "
             "Улучшает детекцию периодических аномалий."),
            ("drift_detect", "Детекция дрейфа",
             "Мониторит дрейф распределения GMM. "
             "Полезно при долгосрочном мониторинге."),
        ]

        # ── В разработке ───────────────────────────────────────────────
        planned = [
            ("log_enc",      "Кодировщик журналов (LLM)"),
            ("trace_enc",    "Кодировщик трассировок (GCN)"),
            ("indep_loss",   "Ограничение независимости L_нез"),
            ("modal_attn",   "Мультимодальное внимание"),
        ]

        # Активные секции
        for section_title, items, is_checked in [
            ("Активные",     active,    True),
            ("Опциональные", optional,  False),
        ]:
            grp = QGroupBox(section_title)
            gl  = QVBoxLayout(grp)
            gl.setSpacing(4)
            gl.setContentsMargins(0, 12, 0, 0)
            for key, label, tooltip in items:
                cb = QCheckBox(label)
                cb.setChecked(is_checked)
                cb.setToolTip(tooltip)
                cb.toggled.connect(
                    lambda v, k=key: self.module_toggled.emit(k, v))
                self._checkboxes[key] = cb
                gl.addWidget(cb)
            cl.addWidget(grp)

        grp_plan = QGroupBox("В разработке")
        gl_plan  = QVBoxLayout(grp_plan)
        gl_plan.setContentsMargins(0, 12, 0, 0)
        for key, label in planned:
            cb = QCheckBox(label)
            cb.setEnabled(False)
            cb.setToolTip("Планируется в следующей версии")
            gl_plan.addWidget(cb)
        cl.addWidget(grp_plan)
        cl.addStretch()
