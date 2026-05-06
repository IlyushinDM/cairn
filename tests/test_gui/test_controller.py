"""Тесты CAIRNController и GUI-интеграции.

Запускаются без отображения окна (QApplication в offscreen-режиме).
Тест на полный цикл: загрузка → обучение (1 эпоха) → анализ.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# offscreen rendering — не нужен экран
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest

SAMPLE_DIR = Path(__file__).parent.parent.parent / "data" / "sample"


# ---------------------------------------------------------------------------
# QApplication (одна на весь модуль)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


# ---------------------------------------------------------------------------
# TestModuleConfig
# ---------------------------------------------------------------------------

class TestModuleConfig:

    def test_defaults(self):
        from cairn.gui.controller import ModuleConfig
        mc = ModuleConfig()
        assert mc.ssm_branch   is False
        assert mc.drift_detect is False
        assert mc.indep_loss   is False

    def test_apply_known_key(self):
        from cairn.gui.controller import ModuleConfig
        mc = ModuleConfig()
        mc.apply("ssm_branch", True)
        assert mc.ssm_branch is True

    def test_apply_unknown_key_safe(self):
        from cairn.gui.controller import ModuleConfig
        mc = ModuleConfig()
        mc.apply("nonexistent_key", True)   # не должно падать

    def test_to_dict(self):
        from cairn.gui.controller import ModuleConfig
        mc = ModuleConfig()
        d = mc.to_dict()
        assert "ssm_branch" in d
        assert "indep_loss" in d
        assert "drift_detect" in d


# ---------------------------------------------------------------------------
# TestCAIRNController — unit-тесты без реальных данных
# ---------------------------------------------------------------------------

class TestCAIRNController:

    @pytest.fixture
    def ctrl(self, qapp):
        from cairn.gui.controller import CAIRNController
        return CAIRNController(config_path=Path("configs/demo.yaml"))

    def test_init_state(self, ctrl):
        assert not ctrl.has_data
        assert not ctrl.has_model
        assert ctrl.hypergraph is None
        assert ctrl.model is None
        assert ctrl.last_chain is None

    def test_module_toggled_no_model(self, ctrl):
        """Переключение модуля без модели не должно падать."""
        ctrl.on_module_toggled("ssm_branch", True)
        ctrl.on_module_toggled("indep_loss", False)
        ctrl.on_module_toggled("drift_detect", True)

    def test_error_signal_emitted_on_analysis_without_model(self, ctrl, qapp):
        """Попытка анализа без модели должна эмитить error, не падать."""
        errors = []
        ctrl.error.connect(lambda t, m: errors.append((t, m)))
        ctrl.start_analysis()
        qapp.processEvents()
        assert len(errors) == 1

    def test_module_config_to_dict(self, ctrl):
        d = ctrl._modules.to_dict()
        assert isinstance(d, dict)
        assert "ssm_branch" in d


# ---------------------------------------------------------------------------
# TestControllerWithDemoData — полный цикл на демо-данных
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (SAMPLE_DIR / "metrics.csv").exists(),
    reason="Демо-данные отсутствуют. Запустите: python scripts/generate_demo_data.py",
)
class TestControllerWithDemoData:

    @pytest.fixture(scope="class")
    def ctrl(self, qapp):
        from cairn.gui.controller import CAIRNController
        c = CAIRNController(config_path=Path("configs/demo.yaml"))
        return c

    def test_load_demo_data(self, ctrl, qapp):
        loaded = []
        ctrl.data_loaded.connect(lambda: loaded.append(True))
        ctrl.load_demo_data(SAMPLE_DIR)
        qapp.processEvents()
        assert loaded, "Сигнал data_loaded не был эмитирован"
        assert ctrl.has_data

    def test_hypergraph_created(self, ctrl):
        assert ctrl.hypergraph is not None
        assert ctrl.hypergraph.n_nodes == 5

    def test_metric_data_loaded(self, ctrl):
        md = ctrl.get_metric_data()
        assert md is not None
        assert md.n_instances == 5

    def test_topology_loaded(self, ctrl):
        topo = ctrl.get_topology()
        assert topo is not None
        assert len(topo.instances) == 5

    def test_training_one_epoch(self, ctrl, qapp):
        """Запускает обучение в потоке и ждёт завершения (таймаут 60 с)."""
        from PySide6.QtCore import QEventLoop, QTimer
        from cairn.training import TrainerConfig

        # Переопределяем конфиг на 1 эпоху каждого этапа
        import cairn.training.trainer as trainer_mod
        original_pretrain = TrainerConfig.__dataclass_fields__["pretrain_epochs"].default
        losses_received    = []
        progress_received  = []
        history_received   = []

        ctrl.training_loss.connect(lambda d: losses_received.append(d))
        ctrl.training_progress.connect(lambda *a: progress_received.append(a))

        loop = QEventLoop()
        ctrl.training_finished.connect(lambda h: (history_received.append(h), loop.quit()))
        ctrl.error.connect(lambda t, m: (pytest.fail(f"{t}: {m}"), loop.quit()))

        # Принудительно устанавливаем 1 эпоху через патч конфига
        orig_build = ctrl._build_trainer
        def patched_build():
            trainer, dataset = orig_build()
            trainer.cfg.pretrain_epochs  = 1
            trainer.cfg.main_epochs      = 1
            trainer.cfg.finetune_epochs  = 1
            return trainer, dataset
        ctrl._build_trainer = patched_build

        ctrl.start_training()

        # Таймаут 60 секунд
        QTimer.singleShot(60_000, loop.quit)
        loop.exec()

        assert history_received, "Обучение не завершилось в 60 с"
        h = history_received[0]
        assert "pretrain_loss" in h or "main_loss" in h

    def test_model_available_after_training(self, ctrl):
        assert ctrl.has_model
        assert ctrl.model is not None

    def test_analysis_sync(self, ctrl):
        """Синхронный вызов ядра анализа (в тестовом потоке)."""
        results = ctrl._run_analysis_core()
        assert isinstance(results, list)
        assert len(results) >= 1
        r = results[0]
        assert "name" in r and "ce" in r and "nll" in r

    def test_chain_built(self, ctrl):
        ctrl._run_analysis_core()
        assert ctrl.last_chain is not None
        assert len(ctrl.last_chain.path_nodes) >= 1

    def test_export_json(self, ctrl, tmp_path):
        ctrl._run_analysis_core()
        out = tmp_path / "report.json"
        ctrl.export_json(out)
        assert out.exists()
        import json
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "results" in data
        assert "chain" in data
        assert "modules" in data

    def test_export_png(self, ctrl, tmp_path, qapp):
        """PNG-экспорт требует matplotlib — пропускаем если недоступен."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib.figure import Figure
            fig = Figure()
            ax  = fig.add_subplot(111)
            ax.plot([1, 2, 3], [1, 2, 3])
            out = tmp_path / "graph.png"
            ctrl.export_graph_png(out, ax)
            assert out.exists()
            assert out.stat().st_size > 1000
        except ImportError:
            pytest.skip("matplotlib недоступен")

    def test_module_toggle_affects_loss_weight(self, ctrl):
        """Снятие чекбокса indep_loss должно обнулить lambda_nez."""
        ctrl.on_module_toggled("indep_loss", False)
        if ctrl._trainer:
            assert ctrl._trainer.loss_fn.w.lambda_nez == 0.0

        ctrl.on_module_toggled("indep_loss", True)
        if ctrl._trainer:
            assert ctrl._trainer.loss_fn.w.lambda_nez == 0.5
