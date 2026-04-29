"""Главное окно приложения CAIRN.

Структура:
  MenuBar → ToolBar → QSplitter(Sidebar | TabWidget) → StatusBar

Вкладки: Данные / Обучение / Результаты / Объяснение
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QSize, Slot
from PySide6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QLabel, QMainWindow,
    QMessageBox, QSplitter, QStatusBar, QTabWidget,
    QToolBar, QWidget,
)

from cairn.gui.styles import load_theme
from cairn.gui.widgets.sidebar import Sidebar
from cairn.gui.widgets.data_tab import DataTab
from cairn.gui.widgets.training_tab import TrainingTab, TrainingWorker
from cairn.gui.widgets.results_tab import ResultsTab
from cairn.gui.widgets.explanation_tab import ExplanationTab
from cairn.gui.widgets.settings_dialog import SettingsDialog


# ---------------------------------------------------------------------------
# Иконка-пирамида (cairo cairn)
# ---------------------------------------------------------------------------

def _make_cairn_icon(size: int = 64, accent: str = "#4a9eff") -> QIcon:
    """Рисует простую пиксельную пирамиду — символ каменного ориентира."""
    px = QPixmap(size, size)
    px.fill(QColor(0, 0, 0, 0))
    painter = QPainter(px)
    painter.setRenderHint(QPainter.Antialiasing)

    # Фон — тёмный круг
    painter.setBrush(QColor("#1e2130"))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(2, 2, size - 4, size - 4)

    # Пирамида из трёх ярусов
    painter.setBrush(QColor(accent))
    cx = size // 2
    # Основание
    painter.drawRect(cx - 18, size - 18, 36, 8)
    # Средний ярус
    painter.drawRect(cx - 12, size - 28, 24, 8)
    # Верхушка
    painter.drawRect(cx - 6, size - 38, 12, 8)
    # Пик
    painter.drawRect(cx - 2, size - 46, 4, 6)

    painter.end()
    return QIcon(px)


# ---------------------------------------------------------------------------
# Главное окно
# ---------------------------------------------------------------------------

class CAIRNMainWindow(QMainWindow):
    """Главное окно CAIRN."""

    def __init__(self, config=None):
        super().__init__()
        self._config = config
        self._trainer = None
        self._worker  = None
        self._dataset = None
        self._hypergraph = None
        self._current_theme = "dark"

        self.setWindowTitle("CAIRN — Causal Attentive Intervention Reasoning Network")
        self.setMinimumSize(1000, 700)
        self.resize(1400, 900)
        self.setWindowIcon(_make_cairn_icon())

        self._apply_theme("dark")
        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()

    # ── Тема ──────────────────────────────────────────────────────────────

    def _apply_theme(self, name: str):
        self._current_theme = name
        qss = load_theme(name)
        QApplication.instance().setStyleSheet(qss)

    # ── Меню ──────────────────────────────────────────────────────────────

    def _build_menu(self):
        mb = self.menuBar()

        # Файл
        file_menu = mb.addMenu("Файл")
        act_load  = QAction("Загрузить данные…", self)
        act_load.setShortcut("Ctrl+O")
        act_load.triggered.connect(self._on_load_data)
        act_export = QAction("Экспорт результатов…", self)
        act_export.triggered.connect(self._on_export)
        act_quit   = QAction("Выход", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_load)
        file_menu.addAction(act_export)
        file_menu.addSeparator()
        file_menu.addAction(act_quit)

        # Настройки
        settings_menu = mb.addMenu("Настройки")
        act_settings = QAction("Параметры модели…", self)
        act_settings.triggered.connect(self._on_settings)
        act_dark  = QAction("Тёмная тема",  self)
        act_dark.triggered.connect(lambda: self._apply_theme("dark"))
        act_light = QAction("Светлая тема", self)
        act_light.triggered.connect(lambda: self._apply_theme("light"))
        settings_menu.addAction(act_settings)
        settings_menu.addSeparator()
        settings_menu.addAction(act_dark)
        settings_menu.addAction(act_light)

        # Справка
        help_menu = mb.addMenu("Справка")
        act_about = QAction("О программе", self)
        act_about.triggered.connect(self._on_about)
        help_menu.addAction(act_about)

    # ── Панель инструментов ───────────────────────────────────────────────

    def _build_toolbar(self):
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

    # ── Центральная область ───────────────────────────────────────────────

    def _build_central(self):
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)

        # Боковая панель
        self.sidebar = Sidebar()
        self.sidebar.configure_source.connect(self._on_configure_source)
        self.sidebar.module_toggled.connect(self._on_module_toggled)
        splitter.addWidget(self.sidebar)

        # Вкладки
        self.tabs = QTabWidget()
        self.tab_data        = DataTab()
        self.tab_training    = TrainingTab()
        self.tab_results     = ResultsTab()
        self.tab_explanation = ExplanationTab()

        self.tabs.addTab(self.tab_data,        "📊  Данные")
        self.tabs.addTab(self.tab_training,    "🧠  Обучение")
        self.tabs.addTab(self.tab_results,     "📋  Результаты")
        self.tabs.addTab(self.tab_explanation, "💡  Объяснение")

        # Сигналы вкладок
        self.tab_training.start_requested.connect(self._on_train)
        self.tab_training.stop_requested.connect(self._on_stop_train)
        self.tab_results.show_explanation.connect(self._on_show_explanation)

        splitter.addWidget(self.tabs)
        splitter.setSizes([240, 1160])
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

    # ── Строка состояния ──────────────────────────────────────────────────

    def _build_statusbar(self):
        sb = self.statusBar()

        # GPU
        self._gpu_label = QLabel()
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                self._gpu_label.setText(f"GPU: ✓ {gpu_name}")
                self._gpu_label.setObjectName("statusGood")
            else:
                self._gpu_label.setText("GPU: ✗ CPU-режим")
                self._gpu_label.setObjectName("statusWarn")
        except ImportError:
            self._gpu_label.setText("GPU: —")

        # Модель
        self._model_label = QLabel("Модель: не загружена")
        self._model_label.setObjectName("statusBad")

        # Данные
        self._data_label = QLabel("Данные: не загружены")
        self._data_label.setObjectName("statusWarn")

        sb.addWidget(QLabel("  "))
        sb.addWidget(self._gpu_label)
        sb.addWidget(QLabel("  |  "))
        sb.addWidget(self._model_label)
        sb.addWidget(QLabel("  |  "))
        sb.addWidget(self._data_label)
        sb.addPermanentWidget(QLabel("CAIRN v0.1  "))

    # ── Обработчики ───────────────────────────────────────────────────────

    @Slot()
    def _on_load_data(self):
        """Загружает демо-данные или выбирает папку с данными."""
        demo_dir = Path("data/sample")
        if not (demo_dir / "metrics.csv").exists():
            reply = QMessageBox.question(
                self, "CAIRN",
                "Демо-данные не найдены. Сгенерировать?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                try:
                    import subprocess, sys
                    subprocess.run([sys.executable, "scripts/generate_demo_data.py"], check=True)
                    self.statusBar().showMessage("Демо-данные сгенерированы.", 3000)
                except Exception as e:
                    QMessageBox.critical(self, "Ошибка", str(e))
                    return
            else:
                return

        try:
            from cairn.connectors.csv_file import (
                CSVMetricConnector, YAMLTopologyConnector,
            )
            from cairn.perception import HypergraphBuilder

            BASE_TS = 1_700_000_000.0
            metric_conn = CSVMetricConnector(demo_dir / "metrics.csv")
            md = metric_conn.fetch(BASE_TS, BASE_TS + 299)

            topo_conn = YAMLTopologyConnector(demo_dir / "topology.yaml")
            topo = topo_conn.fetch()

            self._hypergraph = HypergraphBuilder.from_topology_data(topo)
            self._metric_data = md
            self._topology = topo

            # Заполняем вкладку Данные
            self.tab_data.load_metric_data(md)
            self.tab_data.load_topology(topo)

            # Обновляем статус
            n = md.n_instances
            self._data_label.setText(f"Данные: {n} экземпляров, {md.n_metrics} метрик")
            self._data_label.setObjectName("statusGood")
            self._data_label.style().unpolish(self._data_label)
            self._data_label.style().polish(self._data_label)

            self.sidebar.set_source_status("Метрики", True)
            self.sidebar.set_source_status("Журналы", True)
            self.sidebar.set_source_status("Трассировки", True)

            self._act_train.setEnabled(True)
            self._act_analyze.setEnabled(True)
            self.tabs.setCurrentWidget(self.tab_data)
            self.statusBar().showMessage(f"Загружено {n} экземпляров сервисов.", 4000)

        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки", str(e))

    @Slot()
    def _on_train(self):
        """Запускает обучение в фоновом потоке."""
        if self._config is None:
            from cairn.config import CAIRNConfig
            self._config = CAIRNConfig()

        try:
            from cairn.training import (
                CAIRNLoss, CAIRNModel, CAIRNTrainer, TrainerConfig,
                create_demo_dataset,
            )
            from cairn.perception import StateBuilder
            from cairn.reasoning import ConditionalGMM, ConfoundedVGAE, CounterfactualModule

            dataset = create_demo_dataset(Path("data/sample"), window_size=30, stride=10)

            mc = self._config.model
            model = CAIRNModel(
                state_builder=StateBuilder(
                    n_metrics=4, log_vocab_size=300,
                    state_dim=mc.state_dim, context_dim=mc.context_dim,
                    d_met=mc.metric_dim, d_log=mc.log_dim, d_tr=mc.trace_dim,
                    d_ssm=mc.metric_dim // 2, d_brk=mc.metric_dim // 2,
                    ssm_state_dim=mc.state_dim // 2, window=mc.breakpoint_window,
                ),
                gmm=ConditionalGMM(mc.state_dim, mc.context_dim, mc.gmm_components),
                vgae=ConfoundedVGAE(mc.state_dim, mc.latent_confounders, mc.confounder_dim),
                cf_module=CounterfactualModule(mc.state_dim, mc.hypergraph_layers),
            )

            tc = self._config.training
            trainer_cfg = TrainerConfig(
                pretrain_epochs=tc.pretrain_epochs,
                main_epochs=tc.main_epochs,
                finetune_epochs=tc.finetune_epochs,
                freeze_epochs=tc.freeze_epochs,
                lr=tc.lr,
            )

            if self._hypergraph is None:
                from cairn.connectors.csv_file import YAMLTopologyConnector
                from cairn.perception import HypergraphBuilder
                topo = YAMLTopologyConnector("data/sample/topology.yaml").fetch()
                self._hypergraph = HypergraphBuilder.from_topology_data(topo)

            self._trainer = CAIRNTrainer(model, CAIRNLoss(), self._hypergraph, trainer_cfg)
            self._dataset = dataset

            self._worker = TrainingWorker(self._trainer, dataset, trainer_cfg)
            self._worker.progress.connect(self.tab_training.on_progress)
            self._worker.loss_updated.connect(self.tab_training.on_loss_updated)
            self._worker.finished.connect(self._on_train_finished)
            self._worker.error.connect(self._on_train_error)

            self.tab_training.set_training(True)
            self.tabs.setCurrentWidget(self.tab_training)
            self._model_label.setText("Модель: обучается…")
            self._model_label.setObjectName("statusWarn")
            self._model_label.style().unpolish(self._model_label)
            self._model_label.style().polish(self._model_label)

            self._worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось запустить обучение:\n{e}")

    @Slot()
    def _on_stop_train(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self.tab_training.set_training(False)
            self.statusBar().showMessage("Обучение остановлено.", 3000)

    @Slot(dict)
    def _on_train_finished(self, history: dict):
        self.tab_training.set_training(False)
        self._model_label.setText("Модель: обучена ✓")
        self._model_label.setObjectName("statusGood")
        self._model_label.style().unpolish(self._model_label)
        self._model_label.style().polish(self._model_label)
        self._act_export.setEnabled(True)
        self.statusBar().showMessage("Обучение завершено.", 5000)

    @Slot(str)
    def _on_train_error(self, msg: str):
        self.tab_training.set_training(False)
        self._model_label.setText("Модель: ошибка")
        self._model_label.setObjectName("statusBad")
        QMessageBox.critical(self, "Ошибка обучения", msg)

    @Slot()
    def _on_analyze(self):
        """Запускает анализ первопричин на текущих данных."""
        if self._trainer is None:
            QMessageBox.information(self, "CAIRN",
                "Сначала обучите модель или загрузите чекпоинт.")
            return

        try:
            import torch
            from cairn.reasoning import CascadeFunnel, CausalGraphVerifier
            from cairn.explanation import EvidenceChainBuilder, ALPVerifier, TemplateTextGenerator

            self.statusBar().showMessage("Анализ…")

            # Forward pass на первом аномальном инциденте
            hg = self._hypergraph
            model = self._trainer.model
            gmm   = model.gmm
            cf    = model.cf_module

            # Берём первый аномальный инцидент
            anom = [inc for inc in self._dataset if inc.is_anomaly]
            if not anom:
                QMessageBox.warning(self, "CAIRN", "В датасете нет аномалий.")
                return

            incident = anom[0]
            model.eval()
            with torch.no_grad():
                outputs = model(incident, hg)
            H, C = outputs["H"], outputs["C"]
            nll = gmm.nll(H, C)

            adj = hg.adjacency_matrix()
            adj_norm = adj / adj.sum(1, keepdim=True).clamp(min=1)
            funnel = CascadeFunnel(l0_top_k=H.shape[0], l1_top_k=3, l2_top_k=1)
            ranked = funnel.run(nll, H, adj_norm, cf, gmm, C, hg)

            # Верификатор графа
            root_idx = ranked[0][0] if ranked else 0
            edges    = [(e.members[0], e.members[1]) for e in hg.edges if len(e.members) >= 2]
            verifier = CausalGraphVerifier()
            report   = verifier.verify(
                edges=edges,
                causal_effects={i: nll[i].item() for i in range(H.shape[0])},
                anomaly_times={i: float(i) for i in range(H.shape[0])},
                physical_edges=set(edges),
                root_candidate=root_idx,
            )

            # Цепочка доказательств
            nll_scores = {i: nll[i].item() for i in range(H.shape[0])}
            ce_scores  = dict(ranked)
            chain_builder = EvidenceChainBuilder()
            chain = chain_builder.build(
                root_cause=root_idx,
                causal_graph=hg,
                ce_scores=ce_scores,
                nll_scores=nll_scores,
                anomaly_threshold=0.0,
                metadata={root_idx: {"failure_type": incident.fault_type}},
                verification_confidence=report.confidence,
            )

            # ALP верификация
            gen  = TemplateTextGenerator()
            text = gen.generate(chain)
            alp  = ALPVerifier(anomaly_threshold=0.0, ce_threshold=0.0)
            alp_result = alp.verify(chain, text)

            # Отображение
            self.tab_results.show_results(
                ranked, hg.instance_names, nll_scores, report.confidence, incident.fault_type
            )
            self.tab_results.draw_hypergraph(hg, ce_scores)
            self.tab_explanation.show_chain(chain)
            self.tab_explanation.show_verifier_result(report)
            self.tab_explanation.show_alp_result(alp_result)

            self.tabs.setCurrentWidget(self.tab_results)
            self.statusBar().showMessage(
                f"Анализ завершён. Первопричина: {hg.instance_names[root_idx]} "
                f"(ПЭ={ranked[0][1]:.3f}, достоверность={report.confidence:.0%})", 8000
            )

        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Ошибка анализа",
                                 f"{e}\n\n{traceback.format_exc()[:500]}")

    @Slot(int)
    def _on_show_explanation(self, node_idx: int):
        self.tabs.setCurrentWidget(self.tab_explanation)

    @Slot()
    def _on_export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Экспорт результатов", "results.json", "JSON (*.json)"
        )
        if path:
            try:
                import json
                data = {"status": "Экспорт реализуется на следующем этапе"}
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                self.statusBar().showMessage(f"Экспортировано: {path}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", str(e))

    @Slot()
    def _on_settings(self):
        if self._config is None:
            from cairn.config import CAIRNConfig
            self._config = CAIRNConfig()
        dlg = SettingsDialog(self._config, self)
        dlg.exec()

    @Slot(str)
    def _on_configure_source(self, source: str):
        QMessageBox.information(
            self, f"Настройка: {source}",
            f"Конфигурация источника «{source}» доступна в configs/default.yaml.\n\n"
            "Полный диалог настройки источников будет реализован в следующей версии."
        )

    @Slot(str, bool)
    def _on_module_toggled(self, key: str, enabled: bool):
        state = "включён" if enabled else "отключён"
        self.statusBar().showMessage(f"Модуль '{key}' {state}.", 2000)

    @Slot()
    def _on_about(self):
        QMessageBox.about(
            self, "О программе CAIRN",
            "<h3>CAIRN v0.1</h3>"
            "<p><b>C</b>ausal <b>A</b>ttentive <b>I</b>ntervention <b>R</b>easoning <b>N</b>etwork</p>"
            "<p>Система поиска первопричин сбоев в микросервисных системах "
            "на основе причинно-следственного вывода и интерпретируемых нейросетевых архитектур.</p>"
            "<p>Дипломная работа, СПбГУТ, 2026.</p>"
            "<hr>"
            "<p><small>PySide6 · PyTorch · PyTorch Geometric · NetworkX · Matplotlib</small></p>"
        )


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CAIRN")
    app.setOrganizationName("СПбГУТ")

    try:
        from cairn.config import load_config
        cfg = load_config("configs/default.yaml")
    except Exception:
        cfg = None

    window = CAIRNMainWindow(config=cfg)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
