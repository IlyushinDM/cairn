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
    QApplication, QFileDialog, QLabel, QMainWindow,
    QMessageBox, QSplitter, QTabWidget, QToolBar,
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
        act.triggered.connect(self._on_load_data)
        file_m.addAction(act)

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
        tb = QToolBar("Главная панель")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(tb)

        self._act_load = QAction("📂  Загрузить данные", self)
        self._act_load.triggered.connect(self._on_load_data)
        tb.addAction(self._act_load)
        tb.addSeparator()

        self._act_train = QAction("🧠  Обучить модель", self)
        self._act_train.triggered.connect(self._on_train)
        self._act_train.setEnabled(False)
        tb.addAction(self._act_train)

        self._act_analyze = QAction("🔍  Анализ", self)
        self._act_analyze.triggered.connect(self._on_analyze)
        self._act_analyze.setEnabled(False)
        tb.addAction(self._act_analyze)

        tb.addSeparator()

        self._act_export = QAction("💾  Экспорт", self)
        self._act_export.triggered.connect(self._on_export)
        self._act_export.setEnabled(False)
        tb.addAction(self._act_export)

        tb.addSeparator()

        self._act_demo = QAction("🎓  Демо", self)
        self._act_demo.setToolTip("Запустить демонстрационный сценарий (для защиты диплома)")
        self._act_demo.triggered.connect(self._on_demo)
        tb.addAction(self._act_demo)

    def _build_central(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        self.sidebar = Sidebar()
        splitter.addWidget(self.sidebar)

        self.tabs         = QTabWidget()
        self.data_tab        = DataTab()
        self.training_tab    = TrainingTab()
        self.results_tab     = ResultsTab()
        self.explanation_tab = ExplanationTab()

        self.tabs.addTab(self.data_tab,        "📊  Данные")
        self.tabs.addTab(self.training_tab,    "🧠  Обучение")
        self.tabs.addTab(self.results_tab,     "📋  Результаты")
        self.tabs.addTab(self.explanation_tab, "💡  Объяснение")

        splitter.addWidget(self.tabs)
        splitter.setSizes([260, 1140])
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

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

            # Читаем arch_config из чекпоинта — не хардкодим параметры
            n_components   = 3
            n_confounders  = 2
            confounder_dim = 8
            context_raw_dim = CTX

            ckpt = None
            if model_path.exists():
                ckpt = torch.load(str(model_path), map_location="cpu", weights_only=True)
                if "arch_config" in ckpt:
                    A = ckpt["arch_config"]
                    D              = A.get("state_dim",      D)
                    CTX            = A.get("context_dim",    CTX)
                    F              = A.get("n_metrics",      F)
                    n_components   = A.get("n_components",   n_components)
                    n_confounders  = A.get("n_confounders",  n_confounders)
                    confounder_dim = A.get("confounder_dim", confounder_dim)
                # context_raw_dim всегда из весов
                _key = "model_state" if "model_state" in ckpt else "model_state_dict"
                _st  = ckpt.get(_key, {})
                _w   = _st.get("state_builder.context_builder.proj.0.weight")
                if _w is not None:
                    context_raw_dim = _w.shape[1]

            model = CAIRNModel(
                state_builder=StateBuilder(
                    n_metrics=F, log_vocab_size=300,
                    state_dim=D, context_dim=CTX,
                    d_met=16, d_log=8, d_tr=8,
                    d_ssm=8, d_brk=8, ssm_state_dim=16, window=15,
                    context_raw_dim=context_raw_dim,
                ),
                gmm=ConditionalGMM(state_dim=D, context_dim=CTX, n_components=n_components),
                vgae=ConfoundedVGAE(state_dim=D, n_confounders=n_confounders, confounder_dim=confounder_dim),
                cf_module=CounterfactualModule(state_dim=D, n_conv_layers=1),
            )

            if ckpt is not None:
                _key = "model_state" if "model_state" in ckpt else "model_state_dict"
                if _key in ckpt:
                    model.load_state_dict(ckpt[_key], strict=False)
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
            self._ctrl._demo_sc_dir    = sc_dir   # путь к выбранному сценарию

            self._model_label.setText("Модель: демо ✓")
            self._model_label.setObjectName("statusGood")
            self._refresh_label(self._model_label)
            self._act_analyze.setEnabled(True)
            self._act_export.setEnabled(False)

            # Автоматически запускаем анализ
            # Сохраняем ссылку на воркер чтобы GC не удалил C++ объект раньше времени
            self._demo_worker_ref = None
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

def main(config_path: Optional[str] = None) -> None:
    """Запускает CAIRN GUI."""
    app = QApplication.instance() or QApplication(sys.argv)
    assert isinstance(app, QApplication)
    app.setApplicationName("CAIRN")
    app.setOrganizationName("SPbGUT")
    app.setFont(
        QFont("Segoe UI", 10) if sys.platform == "win32"
        else QFont("Arial", 10)
    )
    window = CAIRNMainWindow(config_path=config_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()