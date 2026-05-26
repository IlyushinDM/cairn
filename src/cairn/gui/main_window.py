"""Главное окно CAIRN — точка входа графического интерфейса.

Архитектура: CAIRNMainWindow (View) ↔ CAIRNController (Presenter/ViewModel).
Окно только маршрутизирует события и обновляет виджеты;
вся бизнес-логика — в контроллере.

Совместимость: PySide6 >= 6.0 (используются новые enum-пространства имён).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, cast

from PySide6.QtCore import QSize, Qt, Slot
from PySide6.QtGui import (
    QAction, QColor, QFont, QIcon, QKeySequence, QPainter, QPixmap,
)
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QSplitter, QTabWidget, QToolBar, QVBoxLayout, QWidget,
)

from cairn.gui.controller import CAIRNController
from cairn.gui.widgets.sidebar import Sidebar
from cairn.gui.widgets.data_tab import DataTab
from cairn.gui.widgets.training_tab import TrainingTab
from cairn.gui.widgets.results_tab import ResultsTab
from cairn.gui.widgets.explanation_tab import ExplanationTab


# ---------------------------------------------------------------------------
# Иконка-пирамида (рисуется программно — без внешних ресурсов)
# ---------------------------------------------------------------------------

def _make_cairn_icon(size: int = 64, color: str = "#4a9eff") -> QIcon:
    """Рисует силуэт каменной пирамиды (cairn)."""
    px = QPixmap(size, size)
    px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    c = QColor(color)
    cx = size // 2
    stones = [
        (cx - size // 2 + 4, size - size // 6 - 2, size - 8,          size // 6, 4),
        (cx - size // 3 + 2, size - size // 3 - 2, size * 2 // 3 - 4, size // 6, 4),
        (cx - size // 5,     size - size // 2 - 2, size * 2 // 5,     size // 6, 4),
        (cx - size // 8,     size - size * 2 // 3 - 2, size // 4,     size // 7, 3),
    ]
    for x, y, w, h, rx in stones:
        p.setBrush(c)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(x, y, w, h, rx, rx)
    p.end()
    return QIcon(px)


def _apply_titlebar_theme_to(widget, theme: str) -> None:
    """Применяет светлый/тёмный titlebar к любому окну через Windows DWM API."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        hwnd = int(widget.winId())
        dark = ctypes.c_int(1 if theme == "dark" else 0)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, 20, ctypes.byref(dark), ctypes.sizeof(dark)
        )
    except Exception:
        pass



def _get_app() -> QApplication:
    """Возвращает текущий QApplication с правильным типом для Pylance."""
    return cast(QApplication, QApplication.instance())


# ---------------------------------------------------------------------------
# Главное окно
# ---------------------------------------------------------------------------

class CAIRNMainWindow(QMainWindow):
    """Главное окно CAIRN."""

    TAB_DATA     = 0
    TAB_TRAINING = 1
    TAB_RESULTS  = 2   # обновляется в _build_central
    TAB_EXPLAIN  = 3   # обновляется в _build_central

    def __init__(self, config_path: Optional[str | Path] = None, parent=None):
        super().__init__(parent)
        # Инициализируем _logger до любых вызовов
        import logging as _logging
        self._logger = _logging.getLogger("cairn.gui")
        # Инициализируем тему до _setup_style
        self._current_theme = "dark"
        # Тема может быть задана через параметр конструктора
        if hasattr(self, "_init_theme"):
            self._current_theme = self._init_theme
        # Live-состояние
        self._live_connector = None

        self._ctrl = CAIRNController(
            config_path=Path(config_path) if config_path else None,
            parent=self,
        )
        self._setup_window()
        self._setup_style()
        self._build_ui()
        self._connect_signals()
        self._install_titlebar_filter()
        self._update_status()

    # ── Инициализация ─────────────────────────────────────────────────

    def _setup_window(self) -> None:
        self.setWindowTitle("CAIRN — Система анализа первопричин сбоев")
        self.setWindowIcon(_make_cairn_icon())
        self.setMinimumSize(1000, 700)
        self.resize(1400, 900)

    def _setup_style(self) -> None:
        styles_dir = Path(__file__).parent / "styles"
        qss_path = styles_dir / "dark_theme.qss"
        # matplotlib цвета задаём в _build_canvas каждого виджета напрямую,
        # а не через rcParams — rcParams.update триггерит QFont::setPointSize <= 0
        # через FigureCanvasQTAgg при инициализации
        # Явно задаём шрифт приложения ДО загрузки QSS.
        from PySide6.QtGui import QFont as _QFont
        app_font = _QFont("Segoe UI", 10) if sys.platform == "win32" else _QFont("SF Pro Text", 10)
        if app_font.pointSize() > 0:
            _get_app().setFont(app_font)

        # Загружаем QSS только если ещё не загружен (run_gui.py мог загрузить раньше)
        if not _get_app().styleSheet() and qss_path.exists():
            _get_app().setStyleSheet(qss_path.read_text(encoding="utf-8"))

    def _build_ui(self) -> None:
        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()

    def _build_menu(self) -> None:
        mb = self.menuBar()

        # Файл
        file_m = mb.addMenu("Файл")

        act = QAction("Загрузить данные…", self)
        act.setShortcut(QKeySequence("Ctrl+O"))
        act.triggered.connect(self._on_load_data_confirmed)
        file_m.addAction(act)

        act_demo = QAction("Демо-режим…", self)
        act_demo.setShortcut(QKeySequence("Ctrl+D"))
        act_demo.triggered.connect(self._on_demo)
        file_m.addAction(act_demo)
        file_m.addSeparator()

        act = QAction("Экспорт результатов…", self)
        act.triggered.connect(self._on_export)
        file_m.addAction(act)

        file_m.addSeparator()

        act = QAction("Выход", self)
        act.setShortcut(QKeySequence("Ctrl+Q"))
        act.triggered.connect(self.close)
        file_m.addAction(act)

        # Настройки
        settings_m = mb.addMenu("Настройки")

        act = QAction("Параметры модели…", self)
        act.triggered.connect(self._on_settings)
        settings_m.addAction(act)

        settings_m.addSeparator()

        act = QAction("Переключить тему", self)
        act.triggered.connect(self._on_toggle_theme)
        settings_m.addAction(act)

        # Справка
        help_m = mb.addMenu("Справка")

        act = QAction("О программе", self)
        act.triggered.connect(self._on_about)
        help_m.addAction(act)

    def _build_toolbar(self) -> None:
        """Toolbar убран — все действия перенесены в Activity Bar."""
        # Сохраняем заглушки для совместимости с другими методами
        from PySide6.QtWidgets import QToolBar
        from PySide6.QtGui import QAction
        self._act_load     = QAction("Загрузить данные", self)
        self._act_analyze  = QAction("Анализ", self)
        self._act_train    = QAction("Обучить", self)
        self._act_settings = QAction("Настройки", self)
        self._act_export   = QAction("Экспорт", self)
        self._act_demo     = QAction("Демо", self)
        self._act_load.triggered.connect(self._on_load_data_confirmed)
        self._act_analyze.triggered.connect(self._on_analyze_confirmed)
        self._act_train.triggered.connect(self._on_train_confirmed)
        self._act_settings.triggered.connect(self._on_settings)
        self._act_analyze.setEnabled(False)
        self._act_train.setEnabled(False)

    def _build_central(self) -> None:
        """Layout: ActivityBar | SidePanel | Tabs"""
        from cairn.gui.widgets.activity_bar import ActivityBar
        from cairn.gui.widgets.side_panels import SourcesPanel, ModulesPanel

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root)

        # ── Activity Bar ──────────────────────────────────────────────────
        self._activity_bar = ActivityBar()
        root_layout.addWidget(self._activity_bar)

        # п.2: явный разделитель для гарантированной видимости границы
        _sep = QFrame()
        _sep.setFrameShape(QFrame.Shape.VLine)
        _sep.setFixedWidth(1)
        _sep.setStyleSheet("background: #474747; border: none;")
        root_layout.addWidget(_sep)

        # Подключаем действия
        self._activity_bar.load_requested.connect(self._on_load_data_confirmed)
        if hasattr(self._activity_bar, "disconnect_requested"):
            self._activity_bar.disconnect_requested.connect(self._on_disconnect)
        self._activity_bar.analyze_requested.connect(self._on_analyze_confirmed)
        self._activity_bar.train_requested.connect(self._on_train_confirmed)
        self._activity_bar.panel_requested.connect(self._on_panel_requested)

        # ── Боковые панели (скрыты по умолчанию) ─────────────────────────
        self._sources_panel = SourcesPanel()
        self._sources_panel.configure_source.connect(self._on_configure_source)
        self._sources_panel.setVisible(False)

        self._modules_panel = ModulesPanel()
        self._modules_panel.module_toggled.connect(self._on_module_toggled)
        self._modules_panel.setVisible(False)

        # Совместимость: sidebar для старого кода
        from cairn.gui.widgets.sidebar import Sidebar
        self.sidebar = Sidebar()
        self.sidebar.setVisible(False)
        self.sidebar.configure_source.connect(self._on_configure_source)

        # QSplitter для боковых панелей + основной контент
        self._content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._content_splitter.setHandleWidth(4)
        self._content_splitter.addWidget(self._sources_panel)
        self._content_splitter.addWidget(self._modules_panel)
        self._content_splitter.addWidget(self.sidebar)

        # ── Вкладки ───────────────────────────────────────────────────────
        self.tabs         = QTabWidget()
        self.tabs.setObjectName("mainTabs")
        # Стиль вкладок управляется через QSS темы
        self.data_tab        = DataTab()
        self.training_tab    = TrainingTab()
        # Загружаем config чтобы кнопка "Сбросить" знала defaults
        if hasattr(self._ctrl, "_config") and self._ctrl._config is not None:
            self.training_tab.load_config(self._ctrl._config)
        self.results_tab     = ResultsTab()
        self.explanation_tab = ExplanationTab()

        from cairn.gui.widgets.logs_tab import LogsTab
        from cairn.gui.widgets.traces_tab import TracesTab
        self.logs_tab   = LogsTab()
        self.traces_tab = TracesTab()

        self.tabs.addTab(self.data_tab,        "Данные")
        self.tabs.addTab(self.logs_tab,        "Журналы")
        self.tabs.addTab(self.traces_tab,      "Трассировки")
        self.tabs.addTab(self.results_tab,     "Результаты")
        self.tabs.addTab(self.explanation_tab, "Объяснение")
        self.tabs.addTab(self.training_tab,    "Обучение")

        self.TAB_DATA    = 0
        self.TAB_LOGS    = 1
        self.TAB_TRACES  = 2
        self.TAB_RESULTS = 3
        self.TAB_EXPLAIN = 4
        self.TAB_TRAIN   = 5

        # Журнал событий — отдельная боковая панель
        from cairn.gui.widgets.event_log import EventLogWidget
        self._event_log = EventLogWidget()
        self._event_log.setMinimumWidth(240)
        self._event_log.setMaximumWidth(500)
        self._event_log.resize(320, self._event_log.height())
        self._event_log.setVisible(False)
        self._content_splitter.addWidget(self._event_log)
        self._content_splitter.addWidget(self.tabs)
        root_layout.addWidget(self._content_splitter, stretch=1)

        # Инициализируем после создания
        self._event_log.add_info("CAIRN запущен.")

    def _on_panel_requested(self, panel: str) -> None:
        """Показывает нужную боковую панель."""
        self._sources_panel.setVisible(panel == "sources")
        self._modules_panel.setVisible(panel == "modules")
        if hasattr(self, "_event_log"):
            self._event_log.setVisible(panel == "log")
        if panel == "connect":
            self._on_connect_live()
        elif panel == "settings":
            self._on_settings()

    def _build_statusbar(self) -> None:
        sb = self.statusBar()

        self._gpu_label = QLabel()
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                self._gpu_label.setText(f"GPU: ✓ {name}")
                self._gpu_label.setObjectName("statusGood")
            else:
                self._gpu_label.setText("GPU: ✗ CPU-режим")
                self._gpu_label.setObjectName("statusWarn")
        except ImportError:
            self._gpu_label.setText("GPU: —")

        self._model_label = QLabel("Модель: не загружена")
        self._model_label.setObjectName("statusBad")
        self._data_label = QLabel("Данные: не загружены")
        self._data_label.setObjectName("statusWarn")

        sb.addWidget(QLabel("  "))
        sb.addWidget(self._gpu_label)
        sb.addWidget(QLabel("  |  "))
        sb.addWidget(self._model_label)
        sb.addWidget(QLabel("  |  "))
        sb.addWidget(self._data_label)
        sb.addPermanentWidget(QLabel("CAIRN v0.1  "))

    def _install_titlebar_filter(self) -> None:
        """Перехватывает показ любого QDialog и применяет тему titlebar."""
        from PySide6.QtCore import QEvent, QObject

        theme_ref = [self._current_theme]

        class _TitlebarFilter(QObject):
            def eventFilter(self, obj, event):
                if event.type() == QEvent.Type.WinIdChange:
                    from PySide6.QtWidgets import QDialog, QMainWindow
                    from PySide6.QtWidgets import QDialog, QMainWindow
                    # QDialog — всегда применяем (даже с parent)
                    # QMainWindow — только главное окно
                    # Исключаем внутренние Qt-виджеты (не Dialog/MainWindow)
                    if isinstance(obj, (QDialog, QMainWindow)):
                        _apply_titlebar_theme_to(obj, theme_ref[0])
                return False

        self._titlebar_filter = _TitlebarFilter(self)
        _get_app().installEventFilter(self._titlebar_filter)
        # Обновляем тему при переключении
        orig_toggle = self._toggle_theme
        def patched_toggle():
            orig_toggle()
            theme_ref[0] = self._current_theme
        self._toggle_theme = patched_toggle

    def _connect_signals(self) -> None:
        # Контроллер → окно
        self._ctrl.data_loaded.connect(self._on_data_loaded)
        self._ctrl.training_progress.connect(self.training_tab.on_progress)
        self._ctrl.training_loss.connect(self.training_tab.on_loss_updated)
        self._ctrl.training_finished.connect(self._on_training_finished)
        self._ctrl.analysis_complete.connect(self._on_analysis_complete)
        self._ctrl.error.connect(self._on_error)

        # Боковая панель → контроллер
        self.sidebar.configure_source.connect(self._on_configure_source)
        self.sidebar.module_toggled.connect(self._ctrl.on_module_toggled)
        self.sidebar.module_toggled.connect(
            lambda k, e: self._update_status(
                f"Модуль «{k}» {'включён' if e else 'отключён'}"
            )
        )

        # Вкладки → контроллер
        self.training_tab.start_requested.connect(self._on_train)
        self.training_tab.stop_requested.connect(self._ctrl.stop_training)
        self.results_tab.show_explanation.connect(self._on_show_explanation)
        self.results_tab.counterfactual_requested.connect(self._on_counterfactual)
        self.results_tab.compare_modes_requested.connect(self._on_compare_modes)

    # ── Слоты ─────────────────────────────────────────────────────────

    @Slot()
    def _on_load_data(self) -> None:
        sample_dir = Path("data/sample")
        if not (sample_dir / "metrics.csv").exists():
            reply = self._ask_yes_no("Данные не найдены", "Демо-данные не найдены. Сгенерировать автоматически?")
            if reply:
                import subprocess
                try:
                    self._update_status("Генерация демо-данных…")
                    subprocess.run(
                        [sys.executable, "scripts/generate_demo_data.py"],
                        check=True,
                    )
                except Exception as e:
                    QMessageBox.critical(self, "Ошибка генерации", str(e))
                    return
            else:
                return

        self._update_status("Загрузка данных…")
        self._ctrl._last_scenario_dir = str(sample_dir)
        self._ctrl.load_demo_data(sample_dir)

    @Slot()
    def _on_data_loaded(self) -> None:
        md   = self._ctrl.get_metric_data()
        topo = self._ctrl.get_topology()
        if md:
            self.data_tab.load_metric_data(md)
            self._data_label.setText(
                f"Данные: {md.n_instances} экз., {md.n_metrics} метрик"
            )
            self._data_label.setObjectName("statusGood")
            self._refresh_label(self._data_label)
        if topo:
            self.data_tab.load_topology(topo)

        # ── Журналы и Трассировки ─────────────────────────────────────────
        # Пробуем взять реальные данные из контроллера,
        # если их нет — генерируем синтетические из metric_data
        log_data   = getattr(self._ctrl, "_log_data",   None)
        trace_data = getattr(self._ctrl, "_trace_data", None)

        # Конвертируем старый формат (FileLogConnector → DockerLogConnector)
        # FileLogConnector возвращает LogData(entries=[...])
        # DockerLogConnector возвращает LogData(series={...})
        if log_data is not None and not hasattr(log_data, "series"):
            log_data = self._convert_file_log_data(log_data, md)

        # Конвертируем старый формат (JSONTraceConnector → LatencyTraceConnector)
        # JSONTraceConnector возвращает list[TraceData(spans=[...])]
        # LatencyTraceConnector возвращает TraceData(services={...})
        if trace_data is not None and isinstance(trace_data, list):
            trace_data = self._convert_json_trace_data(trace_data)

        if (log_data is None or trace_data is None) and md is not None:
            try:
                from cairn.connectors.demo_log_trace_generator import (
                    generate_demo_log_data, generate_demo_trace_data,
                )
                import json as _json
                from pathlib import Path as _Path

                # Пытаемся узнать root_cause из labels.json
                root_cause: str | None = None
                sc_dir = getattr(self._ctrl, "_last_scenario_dir", None)
                if sc_dir:
                    lbl_path = _Path(sc_dir) / "labels.json"
                    if lbl_path.exists():
                        lbl = _json.loads(lbl_path.read_text(encoding="utf-8"))
                        root_cause = lbl.get("root_cause_instance")

                names = md.instance_names
                ts    = md.timestamps
                anom_idx = int(len(ts) * 0.6)

                if log_data is None:
                    log_data = generate_demo_log_data(
                        names, ts,
                        root_cause=root_cause,
                        anomaly_start_idx=anom_idx,
                    )
                if trace_data is None:
                    trace_data = generate_demo_trace_data(
                        names, ts,
                        root_cause=root_cause,
                        anomaly_start_idx=anom_idx,
                    )
            except Exception as e:
                self._logger.debug(f"Demo log/trace generation: {e}")

        if log_data is not None and hasattr(self, "logs_tab"):
            try:
                self.logs_tab.load_log_data(log_data)
                self.sidebar.set_source_status("Журналы", ok=True)
            except Exception as e:
                self._logger.debug(f"logs_tab.load_log_data: {e}")

        if trace_data is not None and hasattr(self, "traces_tab"):
            try:
                self.traces_tab.load_trace_data(trace_data)
                self.sidebar.set_source_status("Трассировки", ok=True)
            except Exception as e:
                self._logger.debug(f"traces_tab.load_trace_data: {e}")

        self.sidebar.set_source_status("Метрики", ok=True)

        self._act_train.setEnabled(True)
        if hasattr(self, "_activity_bar"):
            self._activity_bar.set_analyze_enabled(True)
            self._activity_bar.set_train_enabled(True)
        self.tabs.setCurrentIndex(self.TAB_DATA)
        self._update_status("Данные загружены успешно")

    @Slot()
    def _on_train(self) -> None:
        if not self._ctrl.has_data:
            reply = self._ask_yes_no("Нет данных", "Данные не загружены. Загрузить демо-данные и начать обучение?")
            if reply:
                self._on_load_data()
            return

        self.tabs.setCurrentIndex(self.TAB_TRAINING)
        self.training_tab.set_training(True)
        self._model_label.setText("Модель: обучается…")
        self._model_label.setObjectName("statusWarn")
        self._refresh_label(self._model_label)
        self._update_status("Обучение запущено…")
        self._ctrl.start_training()

    @Slot(dict)
    def _on_training_finished(self, history: dict) -> None:
        self.training_tab.set_training(False)
        self._model_label.setText("Модель: обучена ✓")
        self._model_label.setObjectName("statusGood")
        self._refresh_label(self._model_label)
        self._act_analyze.setEnabled(True)
        self._update_status("Обучение завершено")
        QMessageBox.information(
            self,
            "Обучение завершено",
            f"Претрейн:  {len(history.get('pretrain_loss', []))} эп.\n"
            f"Основное:  {len(history.get('main_loss', []))} эп.\n"
            f"Файнтюн:   {len(history.get('finetune_loss', []))} эп.",
        )

    @Slot()
    def _on_analyze(self) -> None:
        if not self._ctrl.has_model:
            QMessageBox.warning(
                self, "Модель не готова", "Сначала обучите модель."
            )
            return
        self._update_status("Анализ первопричины…")
        self._ctrl.start_analysis()

    @Slot(list)
    def _on_analysis_complete(self, results: list) -> None:
        if not results:
            self._update_status("Анализ: результаты не получены")
            return

        hg         = self._ctrl.hypergraph
        names      = hg.instance_names if hg else []
        ranked     = [(r["idx"], r["ce"]) for r in results]
        nll_scores = {r["idx"]: r["nll"] for r in results}
        fault_type = results[0].get("fault_type", "—")
        confidence = results[0].get("confidence", 0.8)

        self.results_tab.show_results(ranked, names, nll_scores, confidence, fault_type)
        if hg:
            ce_scores = {r["idx"]: r["ce"] for r in results}
            self.results_tab.draw_hypergraph(hg, ce_scores)

        chain = self._ctrl.last_chain
        alp   = self._ctrl.last_alp
        if chain:
            self.explanation_tab.show_chain(chain)
        if alp:
            self.explanation_tab.show_alp_result(alp)

        self._act_export.setEnabled(True)
        self.tabs.setCurrentIndex(self.TAB_RESULTS)
        root_name = results[0]["name"] if results else "—"
        self._update_status(f"Анализ завершён. Первопричина: {root_name}")

    @Slot(int)
    def _ask_yes_no(self, title: str, text: str) -> bool:
        """QMessageBox с русскими кнопками Да/Нет."""
        from PySide6.QtWidgets import QMessageBox
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setText(text)
        box.setIcon(QMessageBox.Icon.Question)
        yes = box.addButton("Да",     QMessageBox.ButtonRole.YesRole)
        box.addButton("Нет", QMessageBox.ButtonRole.NoRole)
        box.exec()
        return box.clickedButton() is yes

    def _show_info(self, title: str, text: str) -> None:
        """QMessageBox информация с русской кнопкой OK."""
        from PySide6.QtWidgets import QMessageBox
        box = QMessageBox(self)
        box.setWindowTitle(title)
        box.setText(text)
        box.setIcon(QMessageBox.Icon.Information)
        box.addButton("ОК", QMessageBox.ButtonRole.AcceptRole)
        box.exec()

    def _apply_theme(self, theme: str) -> None:
        """Загружает и применяет тему через QApplication."""
        filename = "dark_theme.qss" if theme == "dark" else "light_theme.qss"
        theme_path = Path(__file__).parent / "styles" / filename
        if not theme_path.exists():
            return
        qss = theme_path.read_text(encoding="utf-8")
        _get_app().setStyleSheet(qss)
        self._current_theme = theme
        self.tabs.setStyleSheet("")
        # Даём Qt время обновить QPalette перед перерисовкой графов
        _get_app().processEvents()
        self._repaint_plots(theme)
        # Принудительно обновляем QGraphicsScene фоны
        self._refresh_graph_backgrounds()
        # п.7: светлый/тёмный titlebar на Windows
        self._set_titlebar_theme(theme)

    def _set_titlebar_theme(self, theme: str) -> None:
        """Устанавливает цвет заголовка окна через Windows API."""
        if sys.platform != "win32":
            return
        try:
            import ctypes
            hwnd = int(self.winId())
            # DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            dark = ctypes.c_int(1 if theme == "dark" else 0)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, 20, ctypes.byref(dark), ctypes.sizeof(dark)
            )
        except Exception:
            pass

    def _repaint_plots(self, theme: str) -> None:
        """Перекрашивает matplotlib-графики при смене темы (п.6)."""
        if theme == "light":
            bg, fg, grid = "#f5f5f5", "#333333", "#cccccc"
        else:
            bg, fg, grid = "#161922", "#6c7a9c", "#2d3348"
        # data_tab
        if hasattr(self, "data_tab") and hasattr(self.data_tab, "_ax"):
            ax = self.data_tab._ax
            if ax is not None:
                ax.set_facecolor(bg)
                ax.tick_params(colors=fg)
                for s in ax.spines.values():
                    s.set_color(grid)
                ax.set_xlabel(ax.get_xlabel(), color=fg)
                ax.set_ylabel(ax.get_ylabel(), color=fg)
                if self.data_tab._fig is not None:
                    self.data_tab._fig.patch.set_facecolor(bg)
                    try:
                        self.data_tab._fig.canvas.draw()
                    except Exception:
                        pass
        # training_tab loss_chart
        if hasattr(self, "training_tab"):
            lc = getattr(self.training_tab, "loss_chart", None)
            if lc and hasattr(lc, "_ax") and lc._ax is not None:
                lc._ax.set_facecolor(bg)
                lc._ax.tick_params(colors=fg)
                for s in lc._ax.spines.values():
                    s.set_color(grid)
                if lc._fig is not None:
                    lc._fig.patch.set_facecolor(bg)
                    try:
                        lc._fig.canvas.draw()
                    except Exception:
                        pass
        # InteractiveGraphWidget — обновляем цвета подписей узлов
        if hasattr(self, "results_tab"):
            igw = getattr(self.results_tab, "_igraph", None)
            if igw is not None:
                try:
                    for node in igw._nodes.values():
                        node.refresh_label_color()
                    igw._scene.update()
                except Exception:
                    pass
        # ChainGraphWidget — обновляем фон и цвета узлов/рёбер
        if hasattr(self, "explanation_tab"):
            cw = getattr(self.explanation_tab, "_chain_widget", None)
            if cw is not None:
                try:
                    cw.refresh_theme()
                except Exception:
                    pass

    def _refresh_graph_backgrounds(self) -> None:
        """Принудительно обновляет фон всех QGraphicsView после смены темы."""
        from PySide6.QtWidgets import QGraphicsView
        for view in self.findChildren(QGraphicsView):
            view.viewport().update()
            view.update()

    def _toggle_theme(self) -> None:
        """Переключает между тёмной и светлой темой."""
        new_theme = "light" if self._current_theme == "dark" else "dark"
        self._apply_theme(new_theme)
        # Обновляем цвет стрелок в QProxyStyle
        from PySide6.QtWidgets import QApplication as _QApp
        style = _QApp.instance().style()
        if hasattr(style, "set_theme"):
            style.set_theme(new_theme)
        label = "Светлая тема" if new_theme == "light" else "Тёмная тема"
        self._update_status(f"Тема изменена: {label}")

    def _convert_file_log_data(self, old_log_data, metric_data):
        """Конвертирует LogData(entries) → LogData(series) для LogsTab."""
        try:
            from cairn.connectors.docker_log_connector import (
                LogData, LogTimeSeries,
            )
            import numpy as np
            from collections import defaultdict

            # Группируем entries по instance_name
            by_instance: dict = defaultdict(list)
            for entry in old_log_data.entries:
                name = getattr(entry, "instance_name", getattr(entry, "instance", "unknown"))
                by_instance[name].append(entry)

            series = {}
            for name, entries in by_instance.items():
                timestamps  = sorted({e.timestamp for e in entries})
                error_rate  = []
                warn_rate   = []
                total_rate  = []
                top_errors  = []

                for t in timestamps:
                    window = [e for e in entries
                              if abs(e.timestamp - t) < 5]
                    errors = [e for e in window
                              if getattr(e, "level", "") in ("ERROR", "CRITICAL")]
                    warns  = [e for e in window
                              if getattr(e, "level", "") == "WARN"]
                    error_rate.append(len(errors) / max(len(window), 1) * 10)
                    warn_rate.append(len(warns)  / max(len(window), 1) * 10)
                    total_rate.append(float(len(window)))

                # Топ ошибок
                error_msgs = [e.message for e in entries
                              if getattr(e, "level", "") in ("ERROR", "CRITICAL")]
                top_errors = list(dict.fromkeys(error_msgs))[:3]

                avg_err     = float(np.mean(error_rate)) if error_rate else 0.0
                is_anomalous = avg_err > 0.5

                series[name] = LogTimeSeries(
                    container=name,
                    timestamps=timestamps,
                    error_rate=error_rate,
                    warn_rate=warn_rate,
                    total_rate=total_rate,
                    top_errors=top_errors,
                    anomaly_score=avg_err,
                    is_anomalous=is_anomalous,
                )

            collect_time = getattr(old_log_data, "collect_time", 60.0)
            return LogData(series=series, collect_time=collect_time)
        except Exception as e:
            self._logger.debug(f"_convert_file_log_data: {e}")
            return None

    def _convert_json_trace_data(self, trace_list: list):
        """Конвертирует list[TraceData(spans)] → TraceData(services) для TracesTab."""
        try:
            from cairn.connectors.latency_trace_connector import (
                TraceData, ServiceLatency,
            )
            import numpy as np
            from collections import defaultdict

            # Группируем spans по instance
            by_instance: dict = defaultdict(list)
            for trace in trace_list:
                for span in getattr(trace, "spans", []):
                    inst = getattr(span, "instance", None) or getattr(span, "service", "unknown")
                    by_instance[inst].append(span)

            services = {}
            all_durations = [s.duration_ms for spans in by_instance.values()
                             for s in spans]
            baseline = float(np.median(all_durations)) if all_durations else 100.0

            for inst, spans in by_instance.items():
                durations  = [s.duration_ms for s in spans]
                timestamps = [s.start_time  for s in spans]
                endpoints  = list(dict.fromkeys(
                    getattr(s, "operation", "/") for s in spans
                ))[:5]
                avg_p50    = float(np.mean(durations)) if durations else 0.0
                is_slow    = avg_p50 > baseline * 2.0
                anom_score = avg_p50 / max(baseline, 1.0) if is_slow else 0.0

                services[inst] = ServiceLatency(
                    service=inst,
                    endpoints=endpoints,
                    p50_ms=durations,
                    timestamps=timestamps,
                    request_count=len(spans),
                    avg_p50_ms=avg_p50,
                    is_slow=is_slow,
                    anomaly_score=anom_score,
                )

            return TraceData(services=services, source="json_traces")
        except Exception as e:
            self._logger.debug(f"_convert_json_trace_data: {e}")
            return None

    def _on_disconnect(self) -> None:
        """Безопасно отключается от живой системы."""
        if not hasattr(self, "_live_connector") or self._live_connector is None:
            return

        name = getattr(self._live_connector, "system_name", "система")
        r = self._ask_yes_no(
            "Отключение",
            f"Отключиться от «{name}»?\n\nСбор метрик будет остановлен."
        )
        if not r:
            return

        # Останавливаем мониторинг
        if hasattr(self, "_anomaly_monitor") and self._anomaly_monitor:
            try:
                self._anomaly_monitor.stop()
            except Exception:
                pass

        # Останавливаем live workers
        for attr in ("_live_worker", "_log_worker", "_trace_worker"):
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    w.stop()
                    w.wait(2000)
                except Exception:
                    pass
            setattr(self, attr, None)

        self._live_connector = None
        setattr(self._ctrl, "_is_live_mode", False)

        # Очищаем данные в GUI
        try:
            if hasattr(self, "data_tab"):
                self.data_tab.metrics_table.setRowCount(0)
                self.data_tab.instances_table.setRowCount(0)
                self.data_tab._metric_combo.clear()
                self.data_tab._legend.set_services([])
                if self.data_tab._ax is not None:
                    self.data_tab._ax.clear()
                    self.data_tab._ax.set_title(
                        "Данные очищены — переподключитесь",
                        color="#6c7a9c", fontsize=10
                    )
                    if hasattr(self.data_tab._canvas_widget, "draw"):
                        self.data_tab._canvas_widget.draw()
            if hasattr(self, "results_tab"):
                self.results_tab.root_label.setText("Первопричина не определена")
                self.results_tab.ce_label.setText("ПЭ: —")
                self.results_tab.conf_label.setText("Достоверность: —")
        except Exception:
            pass

        if hasattr(self, "_activity_bar"):
            self._activity_bar.set_connect_status(False)
            self._activity_bar.set_disconnect_visible(False)

        self._update_status(f"Отключено от {name}")

    def _on_compare_modes(self) -> None:
        """B2: Открывает диалог ablation-сравнения режимов.

        Требует: загруженная модель (_model) и гиперграф (_hypergraph).
        В демо-режиме они доступны после запуска анализа.
        """
        from cairn.gui.widgets.comparison_dialog import ComparisonDialog
        try:
            # Если диалог уже открыт — поднимаем его
            if hasattr(self, "_comparison_dlg") and self._comparison_dlg is not None:
                self._comparison_dlg.show()
                self._comparison_dlg.raise_()
                return
            self._comparison_dlg = ComparisonDialog(self._ctrl, parent=self)
            self._comparison_dlg.finished.connect(
                lambda: setattr(self, "_comparison_dlg", None))
            self._comparison_dlg.show()
            self._comparison_dlg.raise_()
        except Exception as e:
            self._logger.debug(f"ComparisonDialog error: {e}")
            import traceback; traceback.print_exc()
            self._show_info("Сравнение режимов",
                f"Ошибка: {e}. Убедитесь что анализ был выполнен.")

    def _on_module_toggled(self, key: str, enabled: bool) -> None:
        """Переключает модуль и обновляет статус в журнале."""
        self._ctrl.on_module_toggled(key, enabled)
        state = "включён" if enabled else "отключён"
        if hasattr(self, "_event_log"):
            self._event_log.add_info(f"Модуль '{key}' {state}")
        self._update_status(f"Модуль {key}: {state}")

    def _on_show_explanation(self, node_idx: int) -> None:
        chain = self._ctrl.last_chain
        alp   = self._ctrl.last_alp
        if chain:
            self.explanation_tab.show_chain(chain)
        if alp:
            self.explanation_tab.show_alp_result(alp)
        self.tabs.setCurrentIndex(self.TAB_EXPLAIN)

    @Slot()
    def _on_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Экспортировать результаты",
            "cairn_results",
            "JSON-отчёт (*.json);;PNG-граф (*.png);;Все файлы (*)",
        )
        if not path:
            return
        try:
            if path.endswith(".png"):
                ax = getattr(self.results_tab, "_graph_ax", None)
                self._ctrl.export_graph_png(path, ax)
            else:
                if not path.endswith(".json"):
                    path += ".json"
                self._ctrl.export_json(path)
            self._update_status(f"Экспортировано: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка экспорта", str(e))

    @Slot()
    def _on_settings(self) -> None:
        from cairn.gui.widgets.settings_dialog import SettingsDialog
        dlg = SettingsDialog(self._ctrl._config, parent=self)
        dlg.config_saved.connect(self._ctrl.save_config)
        dlg.exec()

    @Slot()
    def _on_toggle_theme(self) -> None:
        """Переключатель темы — делегирует в _toggle_theme."""
        self._toggle_theme()

    @Slot(str)
    def _on_configure_source(self, source_name: str) -> None:
        QMessageBox.information(
            self,
            f"Источник: {source_name}",
            f"Демо-режим: данные из папки data/sample/.\n"
            f"Настройте источник «{source_name}» в configs/default.yaml.",
        )

    @Slot(str, str)
    def _on_error(self, title: str, msg: str) -> None:
        QMessageBox.critical(self, title, msg)
        self._update_status(f"Ошибка: {title}")

    @Slot()
    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "О программе CAIRN",
            "<h2>CAIRN v0.1</h2>"
            "<p><b>Causal Attentive Intervention Reasoning Network</b></p>"
            "<p>Система прогнозирования и устранения причин возникновения<br>"
            "нештатных событий в микросервисных системах.</p>"
            "<p>Дипломная работа, СПбГУТ, 2026.</p>"
            "<hr>"
            "<p><small>PyTorch · PyTorch Geometric · PySide6 6.x</small></p>",
        )

    # ── Вспомогательные ───────────────────────────────────────────────

    def _update_status(self, msg: str = "") -> None:
        if msg:
            self.statusBar().showMessage(msg, 5000)

    def _refresh_label(self, label: QLabel) -> None:
        label.style().unpolish(label)
        label.style().polish(label)

    # ── Закрытие ──────────────────────────────────────────────────────

    @Slot()
    def _on_demo(self) -> None:
        """Демонстрационный сценарий: выбор → загрузка → анализ."""
        from cairn.gui.widgets.demo_dialog import ScenarioDialog
        from cairn.training import CAIRNModel, CAIRNLoss, CAIRNTrainer, TrainerConfig
        from cairn.perception import StateBuilder, HypergraphBuilder
        from cairn.reasoning import ConditionalGMM, ConfoundedVGAE, CounterfactualModule
        import torch

        data_dir = Path("data/sample")
        dlg = ScenarioDialog(data_dir, parent=self)
        if dlg.exec() != dlg.DialogCode.Accepted or dlg.selected_scenario is None:
            return

        sc_dir  = dlg.scenario_dir
        sc_info = dlg.scenario_info
        if sc_dir is None or sc_info is None:
            return

        self._update_status(f"Демо: загрузка сценария '{sc_info['name']}'…")

        try:
            # Загружаем или строим модель
            model_path = data_dir / "demo_model.pt"
            D, CTX, F = 32, 8, 4

            from cairn.connectors.csv_file import YAMLTopologyConnector
            topo = YAMLTopologyConnector(sc_dir / "topology.yaml").fetch()
            hg   = HypergraphBuilder.from_topology_data(topo)

            model = CAIRNModel(
                state_builder=StateBuilder(
                    n_metrics=F, log_vocab_size=300,
                    state_dim=D, context_dim=CTX,
                    d_met=16, d_log=8, d_tr=8,
                    d_ssm=8, d_brk=8, ssm_state_dim=16, window=15,
                    context_raw_dim=16,
                ),
                gmm=ConditionalGMM(state_dim=D, context_dim=CTX, n_components=3),
                vgae=ConfoundedVGAE(state_dim=D, n_confounders=2, confounder_dim=8),
                cf_module=CounterfactualModule(state_dim=D, n_conv_layers=1),
            )

            if model_path.exists():
                state = torch.load(str(model_path), map_location="cpu", weights_only=True)
                if "model_state" in state:
                    model.load_state_dict(state["model_state"], strict=False)
                self._update_status("Демо: модель загружена из чекпоинта")
            else:
                self._update_status("Демо: чекпоинт не найден — быстрое обучение 1 эп.")
                from cairn.training import create_demo_dataset, TrainerConfig
                ds  = create_demo_dataset(sc_dir, window_size=30, stride=15)
                cfg = TrainerConfig(
                    pretrain_epochs=1, main_epochs=1, finetune_epochs=1,
                    freeze_epochs=0, patience=999, log_every=999,
                    device="cpu", checkpoint_dir="/tmp/cairn_demo", save_every=999,
                )
                CAIRNTrainer(model, CAIRNLoss(), hg, cfg).train(ds)

            # Передаём модель в контроллер
            self._ctrl._model      = model
            self._ctrl._hypergraph = hg

            # Загружаем данные сценария в GUI
            self._ctrl._last_scenario_dir = str(sc_dir)
            self._ctrl.load_demo_data(sc_dir)

            # Добавляем метаданные первопричины в контроллер
            self._ctrl._demo_root_hint = sc_info["root"]
            self._ctrl._demo_fault_hint = sc_info["fault"]

            self._model_label.setText("Модель: демо ✓")
            self._model_label.setObjectName("statusGood")
            self._refresh_label(self._model_label)
            self._act_analyze.setEnabled(True)
            self._act_export.setEnabled(False)

            # Автоматически запускаем анализ
            self._update_status(f"Демо '{sc_info['name']}': анализ…")
            self._ctrl.start_analysis()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка демо", str(e))
            self._update_status("Ошибка демо")

    def closeEvent(self, event) -> None:
        try:
            self._ctrl.stop_training()
        except RuntimeError:
            pass
        event.accept()


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

    def _on_load_data_confirmed(self) -> None:
        """Открывает ScenarioDialog для выбора демо-сценария."""
        from cairn.gui.widgets.demo_dialog import ScenarioDialog
        data_dir = Path("data/sample")
        dlg = ScenarioDialog(data_dir, parent=self)
        if dlg.exec() != dlg.DialogCode.Accepted or dlg.selected_scenario is None:
            return
        sc_dir  = dlg.scenario_dir
        sc_info = dlg.scenario_info
        if sc_dir:
            self._update_status(f"Загрузка демо: {sc_info['name'] if sc_info else sc_dir.name}…")
            self._ctrl._last_scenario_dir = str(sc_dir)
            self._ctrl.load_demo_data(sc_dir)

    def _on_analyze_confirmed(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        r = self._ask_yes_no("Запустить анализ", "Запустить анализ первопричин на загруженных данных?")
        if r:
            self._ctrl.start_analysis()

    def _on_train_confirmed(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        r = self._ask_yes_no("Обучить модель", "Начать обучение модели CAIRN? Это может занять несколько минут.")
        if r:
            self._on_train()

    def _on_connect_live(self) -> None:
        from cairn.gui.widgets.connect_dialog import ConnectDialog
        dlg = ConnectDialog(self)
        dlg.connected.connect(self._on_live_connected)
        dlg.exec()

    def _on_live_connected(self, connector) -> None:
        """Обработчик успешного подключения живой системы."""
        from PySide6.QtCore import QThread, Signal as _Signal
        from PySide6.QtWidgets import QMessageBox
        from cairn.perception import HypergraphBuilder

        self._live_connector = connector
        if hasattr(self, "_activity_bar"):
            self._activity_bar.set_connect_status(True)
            self._activity_bar.set_disconnect_visible(True)

        # Явное уведомление
        window_sec = connector._cfg.get("metrics", {}).get("window_sec", 60)
        msg = QMessageBox(self)
        msg.setWindowTitle("Подключение установлено")
        msg.setText(
            f"<b>{connector.system_name}</b> подключён.<br><br>"
            f"Сбор метрик займёт ~{window_sec:.0f} секунд.<br>"
            f"Прогресс отображается в строке статуса."
        )
        msg.setIcon(QMessageBox.Icon.Information)
        msg.addButton("ОК", QMessageBox.ButtonRole.AcceptRole)
        msg.exec()

        # Шаг 1: Топология (быстро — сразу в DataTab)
        self._update_status(f"Загружаю топологию {connector.system_name}...")
        import traceback as _tb
        topo = None
        try:
            topo       = connector.fetch_topology()
            hypergraph = HypergraphBuilder.from_topology_data(topo)
            self._ctrl._topo_data  = topo
            self._ctrl._hypergraph = hypergraph
            self.data_tab.load_topology(topo)
            n = len(topo.instances) if hasattr(topo, "instances") else "?"
            self._update_status(f"Топология загружена: {n} сервисов")
            self._logger.debug(f"Topology OK: {n} instances")
        except Exception as e:
            self._logger.debug(f"Topology ERROR: {e}")
            _tb.print_exc()
            self._update_status(f"Ошибка топологии: {e}")

        # Шаг 2: Авто-загрузка модели
        try:
            self._auto_load_model()
            self._logger.debug("Model: loaded OK")
        except Exception as e:
            self._logger.debug(f"Model load ERROR: {e}")
            _tb.print_exc()

        # Шаг 3: Метрики в фоне с прогресс-таймером
        self._update_status(
            f"Сбор метрик... (~{window_sec:.0f}с) | "
            f"Готовность через ~{window_sec:.0f}с"
        )
        self._start_progress_timer(window_sec)

        class _MetricWorker(QThread):
            done  = _Signal(object)
            error = _Signal(str)
            def __init__(self, conn):
                super().__init__()
                self._conn = conn
            def run(self):
                import time as _t
                now = _t.time()
                try:
                    md = self._conn.fetch_metrics(now - 300, now)
                    self.done.emit(md)
                except Exception as exc:
                    self.error.emit(str(exc))

        def _on_done(md):
            self._stop_progress_timer()
            try:
                self._ctrl.load_live_data(
                    md,
                    self._ctrl._topo_data,
                    self._ctrl._hypergraph,
                )
            except Exception as e:
                self._ctrl._metric_data = md

            self.data_tab.load_metric_data(md)
            try:
                self.data_tab._plot_from_md(md)
            except Exception:
                pass

            # Обновляем статус модели
            if self._ctrl._model is not None:
                self._model_label.setText("Модель: загружена (авто)")
                self._model_label.setObjectName("statusGood")
                self._refresh_label(self._model_label)

            self._data_label.setText(
                f"Данные: {md.n_instances} экз., {md.n_metrics} метрик"
            )
            self._data_label.setObjectName("statusGood")
            self._refresh_label(self._data_label)

            # Активируем кнопку анализа
            self._act_analyze.setEnabled(True)
            if hasattr(self, "_activity_bar"):
                self._activity_bar.set_analyze_enabled(True)

            self._update_status(
                f"{connector.system_name}: {md.n_instances} экз., "
                f"{md.n_metrics} метрик, {len(md.timestamps)} точек — готово"
            )
            self.tabs.setCurrentIndex(self.TAB_DATA)
            self._start_live_refresh(connector)
            self._start_anomaly_monitor(connector)

        def _on_error(msg_text):
            self._stop_progress_timer()
            self._update_status(f"Ошибка метрик: {msg_text}")

        worker = _MetricWorker(connector)
        worker.done.connect(_on_done)
        worker.error.connect(_on_error)
        self._live_worker = worker
        worker.start()

    def _auto_load_model(self) -> None:
        """Авто-загружает pre-trained модель если доступна."""
        from pathlib import Path
        model_path = Path("data/sample/demo_model.pt")
        if self._ctrl._model is None and model_path.exists():
            try:
                self._ctrl._load_checkpoint_silent(str(model_path))
                self._model_label.setText("Модель: загружена (авто)")
                self._model_label.setObjectName("statusGood")
                self._refresh_label(self._model_label)
                self._act_analyze.setEnabled(True)
                if hasattr(self, "_activity_bar"):
                    self._activity_bar.set_analyze_enabled(True)
            except Exception as e:
                self._update_status(f"Авто-загрузка модели не удалась: {e}")

    def _start_progress_timer(self, total_sec: float) -> None:
        """Показывает обратный отсчёт в статусбаре."""
        from PySide6.QtCore import QTimer
        self._progress_elapsed = 0
        self._progress_total   = int(total_sec)
        self._progress_timer   = QTimer(self)
        self._progress_timer.setInterval(1000)  # каждую секунду

        def _tick():
            self._progress_elapsed += 1
            remaining = max(0, self._progress_total - self._progress_elapsed)
            pct = min(100, int(self._progress_elapsed / self._progress_total * 100))
            bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
            self._update_status(
                f"Сбор метрик [{bar}] {pct}% — осталось ~{remaining}с"
            )
            if self._progress_elapsed >= self._progress_total:
                self._progress_timer.stop()

        self._progress_timer.timeout.connect(_tick)
        self._progress_timer.start()

    def _stop_progress_timer(self) -> None:
        if hasattr(self, "_progress_timer") and self._progress_timer is not None:
            self._progress_timer.stop()
            self._progress_timer = None

    def _start_live_refresh(self, connector) -> None:
        from PySide6.QtCore import QTimer, QThread, Signal as _Signal
        if hasattr(self, "_live_timer") and self._live_timer is not None:
            self._live_timer.stop()
        window_sec  = int(connector._cfg.get("metrics", {}).get("window_sec", 60))
        interval_ms = window_sec * 1000
        self._live_refresh_running = False

        def _do_refresh():
            if getattr(self, "_live_refresh_running", False):
                return
            self._live_refresh_running = True

            class _RefreshWorker(QThread):
                done = _Signal(object)
                def __init__(self, conn):
                    super().__init__()
                    self._conn = conn
                def run(self):
                    import time as _t
                    now = _t.time()
                    try:
                        md = self._conn.fetch_metrics(now - 300, now)
                        self.done.emit(md)
                    except Exception:
                        pass

            def _on_refresh_done(md):
                self._live_refresh_running = False
                self._ctrl._metric_data = md
                try:
                    combo  = getattr(self.data_tab, "_metric_combo", None)
                    metric = combo.currentText() if combo else "cpu_pct"
                    self.data_tab._plot_from_md(md, metric)
                except Exception:
                    pass
                self._update_status(
                    f"Live: {connector.system_name} — "
                    f"{md.n_instances} экз., {len(md.timestamps)} точек"
                )

            def _on_finished():
                self._live_refresh_running = False
                self._live_refresh_worker = None

            w = _RefreshWorker(connector)
            w.done.connect(_on_refresh_done)
            w.finished.connect(_on_finished)
            self._live_refresh_worker = w
            w.start()

        self._live_timer = QTimer(self)
        self._live_timer.setInterval(interval_ms)
        self._live_timer.timeout.connect(_do_refresh)
        self._live_timer.start()
        self._update_status(
            f"Live: {connector.system_name} — обновление каждые {window_sec}с"
        )

    def _on_results_row_selected(self) -> None:
        rows = self.results_tab.results_table.selectedItems()
        if not rows:
            return
        row = rows[0].row()
        results = getattr(self._ctrl, "_last_results", None) or []
        if row < len(results):
            node_idx = results[row].get("idx", -1)
            self.results_tab.highlight_graph_node_from_table(node_idx)

    def _on_counterfactual(self, node_idx: int) -> None:
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QLabel, QDialogButtonBox
        results = getattr(self._ctrl, "_last_results", None) or []
        node_entry = next((r for r in results if r.get("idx") == node_idx), None)
        if not node_entry:
            self._update_status("Нет данных — сначала запустите анализ")
            return
        node_name   = node_entry.get("name", f"node-{node_idx}")
        ce          = node_entry.get("ce", 0.0)
        nll         = node_entry.get("nll", 0.0)
        conf        = node_entry.get("confidence", 0.0)
        dom         = node_entry.get("dominant_metric") or "—"
        delta_nll   = abs(ce)
        improvement = min(99.0, delta_nll / (abs(nll) + 1e-6) * 100)
        lines = [
            f"Что если бы {node_name} работал нормально?",
            "-" * 48, "",
            "Текущее состояние:",
            f"  CE:                     {ce:+.4f}",
            f"  NLL:                    {nll:.4f}",
            f"  Уверенность:            {conf:.1%}",
            f"  Доминирующая метрика:   {dom}", "",
            "Прогноз после устранения:",
            f"  Снижение NLL:           {delta_nll:.4f}",
            f"  Улучшение состояния:    ~{improvement:.1f}%", "",
        ]
        if improvement > 50:
            lines.append(f"Вывод: устранение {node_name} существенно улучшит систему.")
        elif improvement > 20:
            lines.append(f"Вывод: {node_name} — значимая причина деградации.")
        else:
            lines.append(f"Вывод: умеренное влияние. Проверьте соседние сервисы.")
        text = "\n".join(lines)
        try:
            self.explanation_tab.counter_text.setPlainText(text)
            if hasattr(self.explanation_tab, "_right_tabs"):
                self.explanation_tab._right_tabs.setCurrentIndex(1)
        except Exception:
            pass
        self.tabs.setCurrentIndex(self.TAB_EXPLAIN)

    def _toggle_sidebar(self, checked: bool) -> None:
        self.sidebar.setVisible(checked)
        if hasattr(self, "_act_sidebar"):
            self._act_sidebar.setText("Панель" if checked else "Панель")
        if hasattr(self, "_main_splitter"):
            self._main_splitter.setSizes([260, 1140] if checked else [0, 1400])

    def _plot_metric_series(self, md, metric: str = "latency_ms") -> None:
        try:
            self.data_tab._plot_from_md(md, metric)
        except Exception:
            pass
    def _start_anomaly_monitor(self, connector) -> None:
        """Запускает автономный мониторинг аномалий."""
        from cairn.gui.anomaly_monitor import AnomalyMonitor

        # Останавливаем предыдущий монитор если был
        if hasattr(self, "_anomaly_monitor") and self._anomaly_monitor is not None:
            self._anomaly_monitor.stop()
            self._anomaly_monitor.wait(3000)

        poll_interval = connector._cfg.get("metrics", {}).get("window_sec", 30)
        model = getattr(self._ctrl, "_model", None)

        # DockerLogConnector для параллельного сбора журналов
        try:
            from cairn.connectors.docker_log_connector import DockerLogConnector
            inst_filter = list(
                connector._cfg.get("metrics", {}).get("instance_filter", [])
            )
            self._log_connector = DockerLogConnector(
                instance_filter=inst_filter or None,
                min_level="WARN",
            )
        except Exception:
            self._log_connector = None

        monitor = AnomalyMonitor(
            connector=connector,
            model=model,
            poll_interval=float(poll_interval),
            anomaly_threshold=2.0,
            cooldown=180.0,
            min_baseline_cycles=5,
        )
        monitor.metrics_updated.connect(self._on_monitor_metrics)
        monitor.anomaly_detected.connect(self._on_anomaly_detected)
        monitor.status_changed.connect(self._on_monitor_status)
        self._anomaly_monitor = monitor
        monitor.start()
        self._update_status(
            f"Мониторинг активен: {connector.system_name} "
            f"(опрос каждые {poll_interval:.0f}с)"
        )

    def _on_monitor_metrics(self, md) -> None:
        """Обновляет данные при получении новых метрик от монитора."""
        self._ctrl._metric_data = md
        try:
            combo  = getattr(self.data_tab, "_metric_combo", None)
            metric = combo.currentText() if combo else None
            self.data_tab._plot_from_md(md, metric)
        except Exception:
            pass

        # Параллельно собираем журналы (в фоне)
        if hasattr(self, "_log_connector"):
            from PySide6.QtCore import QThread, Signal as _Signal

            class _LogWorker(QThread):
                done = _Signal(object)
                def __init__(self, conn):
                    super().__init__()
                    self._conn = conn
                def run(self):
                    try:
                        data = self._conn.fetch(window_sec=120, step_sec=30)
                        self.done.emit(data)
                    except Exception:
                        pass

            def _on_log_done(log_data):
                if hasattr(self, "logs_tab"):
                    self.logs_tab.load_log_data(log_data)
                # Если есть лог-аномалии — добавляем в журнал событий
                if hasattr(self, "_event_log"):
                    for name in log_data.anomalous_containers:
                        ts = log_data.series[name]
                        self._event_log.add_warning(
                            f"Лог-аномалия: рост ошибок "
                            f"(score={ts.anomaly_score:.2f})",
                            service=name,
                        )
                self._log_worker = None

            if not hasattr(self, "_log_worker") or self._log_worker is None:
                w = _LogWorker(self._log_connector)
                w.done.connect(_on_log_done)
                self._log_worker = w
                w.start()

        # Сбор latency трассировок из loadgenerator
        if not hasattr(self, "_trace_connector"):
            try:
                from cairn.connectors.latency_trace_connector import (
                    LatencyTraceConnector)
                self._trace_connector = LatencyTraceConnector()
            except Exception:
                self._trace_connector = None

        if self._trace_connector is not None:
            class _TraceWorker(QThread):
                done = _Signal(object)
                def __init__(self, conn):
                    super().__init__()
                    self._conn = conn
                def run(self):
                    try:
                        self.done.emit(self._conn.fetch(window_sec=120))
                    except Exception:
                        pass

            def _on_trace_done(trace_data):
                if hasattr(self, "traces_tab"):
                    self.traces_tab.load_trace_data(trace_data)
                if hasattr(self, "_event_log"):
                    for svc in trace_data.slow_services:
                        sl = trace_data.services[svc]
                        self._event_log.add_warning(
                            f"Latency spike: p50={sl.avg_p50_ms:.0f}мс "
                            f"(×{1+sl.anomaly_score:.1f} от нормы)",
                            service=svc,
                        )
                self._trace_worker = None

            if not hasattr(self, "_trace_worker") or self._trace_worker is None:
                w = _TraceWorker(self._trace_connector)
                w.done.connect(_on_trace_done)
                self._trace_worker = w
                w.start()

    def _on_monitor_status(self, message: str, level: str) -> None:
        """Обновляет строку статуса монитора."""
        self._update_status(message)
        if level == "error" and hasattr(self, "_event_log"):
            self._event_log.add_warning(message)
        # Меняем цвет статусбара по уровню
        colors = {"ok": "#007acc", "warn": "#b8860b", "error": "#8b1a1a"}
        color  = colors.get(level, "#007acc")
        self.statusBar().setStyleSheet(
            f"QStatusBar {{ background: {color}; color: #ffffff; }}"
        )

    def _on_anomaly_detected(self, md, nll_score: float) -> None:
        """Реагирует на обнаруженную аномалию — автозапуск анализа."""
        from PySide6.QtWidgets import QMessageBox

        # Обновляем данные
        self._ctrl._metric_data = md
        self.data_tab.load_metric_data(md)

        # Журнал событий
        if hasattr(self, "_event_log"):
            self._event_log.add_anomaly(
                service    = "система",
                score      = nll_score,
                fault_type = "auto-detected",
            )
            # Показываем последнее событие в статусбаре
            self._update_status(
                f"АНОМАЛИЯ: score={nll_score:.3f} — см. Журнал событий"
            )

        # Автоматически запускаем анализ если модель загружена
        if self._ctrl._model is not None:
            self._update_status(
                f"АНОМАЛИЯ (score={nll_score:.3f}) — запускаю анализ..."
            )
            self._ctrl.start_analysis()
            # Переключаемся на результаты
            self.tabs.setCurrentIndex(self.TAB_RESULTS)
        else:
            # Модель не загружена — просто уведомляем
            self.tabs.setCurrentIndex(self.TAB_DATA)
            msg = QMessageBox(self)
            msg.setWindowTitle("Аномалия обнаружена")
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText(
                f"<b>Обнаружена аномалия!</b><br><br>"
                f"Score: <b>{nll_score:.3f}</b><br>"
                f"Экземпляров: {md.n_instances}<br><br>"
                f"Загрузите модель для автоматического анализа.<br>"
                f"Или запустите анализ вручную."
            )
            msg.setStandardButtons(
                QMessageBox.StandardButton.Ok |
                QMessageBox.StandardButton.Open
            )
            msg.button(QMessageBox.StandardButton.Open).setText("Запустить анализ")
            result = msg.exec()
            if result == QMessageBox.StandardButton.Open:
                self._ctrl.start_analysis()

    
def main(config_path: Optional[str] = None, theme: str = "dark") -> None:
    """Запускает CAIRN GUI."""
    import os as _os, threading as _threading
    if sys.platform == "win32":
        _os.environ.setdefault("QT_FONT_DPI", "96")
    _IGNORE = b"WM_DESTROY"
    try:
        _r, _w = _os.pipe()
        _orig_fd = _os.dup(2)
        _os.dup2(_w, 2)
        _os.close(_w)
        def _stderr_filter(_r=_r, _orig=_orig_fd, _ign=_IGNORE):
            _buf = b""
            _nl  = b"\n"
            while True:
                try:
                    _chunk = _os.read(_r, 256)
                except OSError:
                    break
                if not _chunk:
                    break
                _buf += _chunk
                while _nl in _buf:
                    _line, _buf = _buf.split(_nl, 1)
                    if _ign not in _line:
                        _os.write(_orig, _line + _nl)
        _threading.Thread(target=_stderr_filter, daemon=True).start()
    except Exception:
        pass
    import argparse as _ap
    parser = _ap.ArgumentParser(description="CAIRN GUI", add_help=True)
    parser.add_argument("--config", default=config_path or "configs/default.yaml",
                        help="Путь к конфигурации")
    parser.add_argument("--theme", default=theme, choices=["dark", "light"],
                        help="Тема интерфейса (dark/light)")
    args, _ = parser.parse_known_args()

    app = QApplication.instance() or QApplication(sys.argv)
    assert isinstance(app, QApplication)
    app.setApplicationName("CAIRN")
    app.setOrganizationName("SPbGUT")
    app.setFont(
        QFont("Segoe UI", 10) if sys.platform == "win32"
        else QFont("SF Pro Text", 10)
    )

    # Применяем тему ДО создания окна
    from cairn.gui.styles import load_theme as _lt
    app.setStyleSheet(_lt(args.theme))

    window = CAIRNMainWindow(config_path=args.config)
    window._init_theme = args.theme
    window._current_theme = args.theme
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
