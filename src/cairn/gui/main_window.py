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
    QApplication, QFileDialog, QHBoxLayout, QLabel, QMainWindow,
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
    TAB_RESULTS  = 2
    TAB_EXPLAIN  = 3

    def __init__(self, config_path: Optional[str | Path] = None, parent=None):
        super().__init__(parent)
        self._ctrl = CAIRNController(
            config_path=Path(config_path) if config_path else None,
            parent=self,
        )
        self._setup_window()
        self._setup_style()
        self._build_ui()
        self._connect_signals()
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
        # Set matplotlib dark defaults
        try:
            import matplotlib
            matplotlib.rcParams.update({
                "axes.facecolor":   "#161922",
                "figure.facecolor": "#161922",
                "axes.edgecolor":   "#2d3348",
                "text.color":       "#a0a8bc",
                "axes.labelcolor":  "#6c7a9c",
                "xtick.color":      "#6c7a9c",
                "ytick.color":      "#6c7a9c",
            })
        except Exception:
            pass
        if qss_path.exists():
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

        # Подключаем действия
        self._activity_bar.load_requested.connect(self._on_load_data_confirmed)
        self._activity_bar.analyze_requested.connect(self._on_analyze_confirmed)
        self._activity_bar.train_requested.connect(self._on_train_confirmed)
        self._activity_bar.panel_requested.connect(self._on_panel_requested)

        # ── Боковые панели (скрыты по умолчанию) ─────────────────────────
        self._sources_panel = SourcesPanel()
        self._sources_panel.configure_source.connect(self._on_configure_source)
        self._sources_panel.setVisible(False)
        root_layout.addWidget(self._sources_panel)

        self._modules_panel = ModulesPanel()
        self._modules_panel.module_toggled.connect(
            lambda k, v: self._ctrl.toggle_module(k, v)
            if hasattr(self._ctrl, "toggle_module") else None
        )
        self._modules_panel.setVisible(False)
        root_layout.addWidget(self._modules_panel)

        # Совместимость: sidebar для старого кода
        from cairn.gui.widgets.sidebar import Sidebar
        self.sidebar = Sidebar()
        self.sidebar.setVisible(False)
        self.sidebar.configure_source.connect(self._on_configure_source)

        # ── Вкладки ───────────────────────────────────────────────────────
        self.tabs         = QTabWidget()
        self.data_tab        = DataTab()
        self.training_tab    = TrainingTab()
        self.results_tab     = ResultsTab()
        self.explanation_tab = ExplanationTab()

        self.tabs.addTab(self.data_tab,        "Данные")
        self.tabs.addTab(self.results_tab,     "Результаты")
        self.tabs.addTab(self.explanation_tab, "Объяснение")
        self.tabs.addTab(self.training_tab,    "Обучение")

        self.TAB_DATA    = 0
        self.TAB_RESULTS = 1
        self.TAB_EXPLAIN = 2
        self.TAB_TRAIN   = 3

        # Журнал событий — отдельная боковая панель
        from cairn.gui.widgets.event_log import EventLogWidget
        self._event_log = EventLogWidget()
        self._event_log.setFixedWidth(320)
        self._event_log.setVisible(False)
        root_layout.addWidget(self._event_log)

        root_layout.addWidget(self.tabs, stretch=1)

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

    # ── Слоты ─────────────────────────────────────────────────────────

    @Slot()
    def _on_load_data(self) -> None:
        sample_dir = Path("data/sample")
        if not (sample_dir / "metrics.csv").exists():
            reply = QMessageBox.question(
                self,
                "Данные не найдены",
                "Демо-данные не найдены. Сгенерировать автоматически?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
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

        self.sidebar.set_source_status("Метрики",     ok=True)
        self.sidebar.set_source_status("Журналы",     ok=True)
        self.sidebar.set_source_status("Трассировки", ok=True)

        self._act_train.setEnabled(True)
        if hasattr(self, "_activity_bar"):
            self._activity_bar.set_analyze_enabled(True)
            self._activity_bar.set_train_enabled(True)
        self.tabs.setCurrentIndex(self.TAB_DATA)
        self._update_status("Данные загружены успешно")

    @Slot()
    def _on_train(self) -> None:
        if not self._ctrl.has_data:
            reply = QMessageBox.question(
                self,
                "Нет данных",
                "Данные не загружены. Загрузить демо-данные и начать обучение?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
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
        styles_dir = Path(__file__).parent / "styles"
        app = _get_app()
        current = app.styleSheet()
        name = "light" if "dark" in current else "dark"
        qss_path = styles_dir / f"{name}_theme.qss"
        if qss_path.exists():
            app.setStyleSheet(qss_path.read_text(encoding="utf-8"))
        self._update_status(f"Тема: {name}")

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
        self._ctrl.stop_training()
        event.accept()


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

    def _on_load_data_confirmed(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        r = QMessageBox.question(self, "Загрузить данные",
            "Выберите директорию с данными (metrics.csv + topology.yaml)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        if r == QMessageBox.StandardButton.Yes:
            self._on_load_data()

    def _on_analyze_confirmed(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        r = QMessageBox.question(self, "Запустить анализ",
            "Запустить анализ первопричин на загруженных данных?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        if r == QMessageBox.StandardButton.Yes:
            self._ctrl.start_analysis()

    def _on_train_confirmed(self) -> None:
        from PySide6.QtWidgets import QMessageBox
        r = QMessageBox.question(self, "Обучить модель",
            "Начать обучение модели CAIRN? Это может занять несколько минут.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        if r == QMessageBox.StandardButton.Yes:
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
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
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
            print(f"[DEBUG] Topology OK: {n} instances")
        except Exception as e:
            print(f"[DEBUG] Topology ERROR: {e}")
            _tb.print_exc()
            self._update_status(f"Ошибка топологии: {e}")

        # Шаг 2: Авто-загрузка модели
        try:
            self._auto_load_model()
            print("[DEBUG] Model: loaded OK")
        except Exception as e:
            print(f"[DEBUG] Model load ERROR: {e}")
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

        monitor = AnomalyMonitor(
            connector=connector,
            model=model,
            poll_interval=float(poll_interval),
            anomaly_threshold=0.5,
            cooldown=120.0,
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
        # Обновляем только график — таблицу не перезагружаем каждый раз
        try:
            combo  = getattr(self.data_tab, "_metric_combo", None)
            metric = combo.currentText() if combo else None
            self.data_tab._plot_from_md(md, metric)
        except Exception:
            pass

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

    
def main(config_path: Optional[str] = None) -> None:
    """Запускает CAIRN GUI."""
    app = QApplication.instance() or QApplication(sys.argv)
    assert isinstance(app, QApplication)
    app.setApplicationName("CAIRN")
    app.setOrganizationName("SPbGUT")
    app.setFont(
        QFont("Segoe UI", 10) if sys.platform == "win32"
        else QFont("SF Pro Text", 10)
    )
    window = CAIRNMainWindow(config_path=config_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()