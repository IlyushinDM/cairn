"""Вкладка «Обучение» — п.11.

Изменения:
- Убраны иконки ▶ и ⏹ у кнопок (только текст)
- Добавлена панель быстрых настроек обучения прямо во вкладке
- Настройки синхронизированы с SettingsDialog (при открытии берут текущие значения)
- Кнопка «Сбросить» возвращает настройки к значениям из config
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QDoubleSpinBox, QFormLayout, QFrame, QGroupBox,
    QHBoxLayout, QLabel, QProgressBar, QPushButton,
    QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)


class TrainingWorker(QThread):
    """QThread-воркер для фонового обучения CAIRN."""

    progress     = Signal(int, int, int, str)
    loss_updated = Signal(dict)
    finished     = Signal(dict)
    error        = Signal(str)

    def __init__(self, trainer, dataset, config, parent=None):
        super().__init__(parent)
        self.trainer = trainer
        self.dataset = dataset
        self.config  = config
        self._stop   = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            original_log = self.trainer._log
            stage_epochs = {
                1: self.config.pretrain_epochs,
                2: self.config.main_epochs,
                3: self.config.finetune_epochs,
            }
            stage_names = {1: "Претрейн", 2: "Основное", 3: "Файнтюн"}
            current_stage = [1]

            def patched_log(msg: str):
                original_log(msg)
                if "эп." in msg:
                    try:
                        ep    = int(msg.split("эп.")[1].split(":")[0].strip())
                        stage = current_stage[0]
                        total = stage_epochs[stage]
                        self.progress.emit(ep, total, stage, stage_names[stage])
                    except Exception:
                        pass
                if "loss=" in msg:
                    try:
                        val = float(msg.split("loss=")[1].strip())
                        self.loss_updated.emit({"loss": val})
                    except Exception:
                        pass

            self.trainer._log = patched_log
            for stage in (1, 2, 3):
                if self._stop:
                    break
                current_stage[0] = stage
            history = self.trainer.train(self.dataset)
            if not self._stop:
                self.finished.emit(history)
        except Exception as e:
            self.error.emit(str(e))


class LossChart(QWidget):
    """Простой виджет с графиком функции потерь через matplotlib."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._history: dict[str, list[float]] = {
            "pretrain": [], "main": [], "finetune": [],
        }
        self._canvas = self._build_canvas()
        layout.addWidget(self._canvas)

    def _build_canvas(self):
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            fig = Figure(figsize=(8, 3), facecolor="#161922")
            self._ax = fig.add_subplot(111)
            self._ax.set_facecolor("#161922")
            self._ax.tick_params(colors="#6c7a9c")
            self._ax.spines[:].set_color("#2d3348")
            self._ax.set_xlabel("Эпоха",  color="#6c7a9c", fontsize=9)
            self._ax.set_ylabel("Потеря", color="#6c7a9c", fontsize=9)
            self._ax.set_title("Функция потерь", color="#a0a8bc", fontsize=10)
            fig.tight_layout()
            self._fig = fig
            canvas = FigureCanvasQTAgg(fig)
            canvas.setMinimumHeight(200)
            return canvas
        except ImportError:
            self._fig = None
            self._ax  = None
            lbl = QLabel("График потерь\n(требуется matplotlib)")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(
                "color: #6c7a9c; border: 1px dashed #2d3348;"
            )
            lbl.setMinimumHeight(200)
            return lbl

    def append_loss(self, stage: str, value: float):
        if stage in self._history:
            self._history[stage].append(value)
        if self._ax is None:
            return
        self._ax.clear()
        self._ax.set_facecolor("#161922")
        colors = {"pretrain": "#4a9eff", "main": "#3ecf8e", "finetune": "#f6a623"}
        labels = {"pretrain": "Претрейн", "main": "Основное", "finetune": "Файнтюн"}
        for key, vals in self._history.items():
            if vals:
                self._ax.plot(vals, label=labels[key],
                              color=colors[key], linewidth=1.5)
        self._ax.legend(fontsize=9, labelcolor="#a0a8bc",
                        facecolor="#1e2130", edgecolor="#2d3348")
        self._ax.tick_params(colors="#6c7a9c")
        self._ax.spines[:].set_color("#2d3348")
        self._ax.set_xlabel("Эпоха",  color="#6c7a9c", fontsize=9)
        self._ax.set_ylabel("Потеря", color="#6c7a9c", fontsize=9)
        if self._fig is not None:
            self._fig.tight_layout()
        if hasattr(self._canvas, "draw"):
            self._canvas.draw()

    def reset(self):
        self._history = {"pretrain": [], "main": [], "finetune": []}


class TrainingSettingsPanel(QGroupBox):
    """Панель быстрых настроек обучения внутри вкладки.

    п.11: настройки обучения доступны прямо во вкладке, без открытия Settings.
    Синхронизируется с config при загрузке; кнопка «Сбросить» восстанавливает
    исходные значения из config.
    """

    def __init__(self, config=None, parent=None):
        super().__init__("Параметры обучения", parent)
        self._config   = config
        self._defaults = {}

        form = QFormLayout(self)
        form.setSpacing(8)
        form.setContentsMargins(12, 16, 12, 12)

        # ── Поля ──────────────────────────────────────────────────────────
        self._pretrain_epochs = self._spin(50,  1, 500)
        self._main_epochs     = self._spin(100, 1, 1000)
        self._finetune_epochs = self._spin(30,  1, 500)
        self._lr              = self._dspin(1e-3, 1e-6, 1.0, 6, 1e-4)
        self._batch_size      = self._spin(32,  1, 512)
        self._patience        = self._spin(10,  1, 100)

        form.addRow("Эпохи претрейна:",  self._pretrain_epochs)
        form.addRow("Эпохи основного:",  self._main_epochs)
        form.addRow("Эпохи файнтюна:",   self._finetune_epochs)
        form.addRow("Learning rate:",     self._lr)
        form.addRow("Batch size:",        self._batch_size)
        form.addRow("Early-stop (patience):", self._patience)

        # Кнопка сброса
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background: #3f3f46; margin: 4px 0;")
        form.addRow(sep)

        reset_btn = QPushButton("Сбросить к значениям по умолчанию")
        reset_btn.setFixedHeight(26)
        reset_btn.clicked.connect(self._reset)
        form.addRow(reset_btn)

        # Загружаем значения из config
        if config is not None:
            self.load_from_config(config)

    # ── Вспомогательные ──────────────────────────────────────────────────

    @staticmethod
    def _spin(val: int, lo: int, hi: int) -> QSpinBox:
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(val)
        w.setFixedHeight(26)
        return w

    @staticmethod
    def _dspin(val: float, lo: float, hi: float,
               decimals: int, step: float) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(decimals)
        w.setSingleStep(step)
        w.setValue(val)
        w.setFixedHeight(26)
        return w

    def load_from_config(self, config) -> None:
        """Загружает значения из объекта конфига и запоминает как defaults."""
        tc = getattr(config, "training", None)
        if tc is None:
            return
        vals = {
            "pretrain": getattr(tc, "pretrain_epochs", 50),
            "main":     getattr(tc, "main_epochs",     100),
            "finetune": getattr(tc, "finetune_epochs", 30),
            "lr":       getattr(tc, "lr",              1e-3),
            "batch":    getattr(tc, "batch_size",      32),
            "patience": getattr(tc, "patience",        10),
        }
        self._defaults = vals
        self._apply(vals)

    def _apply(self, vals: dict) -> None:
        self._pretrain_epochs.setValue(int(vals.get("pretrain", 50)))
        self._main_epochs.setValue(int(vals.get("main",     100)))
        self._finetune_epochs.setValue(int(vals.get("finetune", 30)))
        self._lr.setValue(float(vals.get("lr",      1e-3)))
        self._batch_size.setValue(int(vals.get("batch",   32)))
        self._patience.setValue(int(vals.get("patience", 10)))

    def _reset(self) -> None:
        """Сбрасывает настройки к значениям из config."""
        if self._defaults:
            self._apply(self._defaults)

    def get_overrides(self) -> dict:
        """Возвращает текущие значения для передачи в trainer."""
        return {
            "pretrain_epochs":  self._pretrain_epochs.value(),
            "main_epochs":      self._main_epochs.value(),
            "finetune_epochs":  self._finetune_epochs.value(),
            "lr":               self._lr.value(),
            "batch_size":       self._batch_size.value(),
            "patience":         self._patience.value(),
        }


class TrainingTab(QWidget):
    """Вкладка «Обучение» — п.11."""

    start_requested = Signal()
    stop_requested  = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ── Управление (п.11: без иконок у кнопок) ───────────────────────
        ctrl = QHBoxLayout()

        self.btn_start = QPushButton("Начать обучение")   # убрана ▶
        self.btn_start.setObjectName("primaryBtn")
        self.btn_start.setFixedHeight(32)
        self.btn_start.clicked.connect(self.start_requested)

        self.btn_stop = QPushButton("Остановить")         # убрана ⏹
        self.btn_stop.setObjectName("dangerBtn")
        self.btn_stop.setFixedHeight(32)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_requested)

        self.stage_label = QLabel("Этап: —")
        self.stage_label.setStyleSheet("color: #6c7a9c; font-size: 12px;")

        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)
        ctrl.addSpacing(20)
        ctrl.addWidget(self.stage_label)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        # ── Прогресс ──────────────────────────────────────────────────────
        prog_row = QHBoxLayout()
        self.epoch_label = QLabel("Эпоха 0 / 0")
        self.epoch_label.setFixedWidth(120)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        prog_row.addWidget(self.epoch_label)
        prog_row.addWidget(self.progress_bar)
        layout.addLayout(prog_row)

        # ── Настройки + График (горизонтальный split) ─────────────────────
        from PySide6.QtWidgets import QSplitter
        h_split = QSplitter(Qt.Orientation.Horizontal)

        # Левая часть — настройки обучения (п.11)
        self.settings_panel = TrainingSettingsPanel()
        h_split.addWidget(self.settings_panel)

        # Правая часть — график + таблица в вертикальном сплиттере (п.5)
        right_vsplit = QSplitter(Qt.Orientation.Vertical)

        chart_w = QWidget()
        chart_wl = QVBoxLayout(chart_w)
        chart_wl.setContentsMargins(0, 4, 0, 0)  # п.5: убрать лишний отступ сверху
        chart_wl.setSpacing(4)
        chart_lbl = QLabel("ФУНКЦИЯ ПОТЕРЬ")
        chart_lbl.setObjectName("sectionTitle")
        chart_wl.addWidget(chart_lbl)
        self.loss_chart = LossChart()
        chart_wl.addWidget(self.loss_chart)
        right_vsplit.addWidget(chart_w)

        table_w = QWidget()
        table_wl = QVBoxLayout(table_w)
        table_wl.setContentsMargins(0, 4, 0, 0)
        table_wl.setSpacing(4)
        lbl2 = QLabel("КОМПОНЕНТЫ ПОТЕРЬ")
        lbl2.setObjectName("sectionTitle")
        table_wl.addWidget(lbl2)

        self.loss_table = QTableWidget(6, 3)
        self.loss_table.setHorizontalHeaderLabels(["Компонент", "Значение", "Описание"])
        self.loss_table.verticalHeader().setVisible(False)
        # п.5: таблица растягивается, нет фиксированной высоты
        self.loss_table.horizontalHeader().setStretchLastSection(True)
        self.loss_table.setAlternatingRowColors(True)

        components = [
            ("L_ПЭ",  "—", "Ранжирование причинных эффектов"),
            ("L_УМ",  "—", "Условная модель нормального состояния"),
            ("L_ВАК", "—", "Вариационный автокодировщик"),
            ("L_нез", "—", "Ограничение независимости"),
            ("L_КР",  "—", "Контрастное разделение"),
            ("L_реб", "—", "Штраф за необоснованные рёбра"),
        ]
        for row, (name, val, desc) in enumerate(components):
            self.loss_table.setItem(row, 0, QTableWidgetItem(name))
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.loss_table.setItem(row, 1, item)
            self.loss_table.setItem(row, 2, QTableWidgetItem(desc))
            self.loss_table.setRowHeight(row, 26)
        table_wl.addWidget(self.loss_table)
        right_vsplit.addWidget(table_w)
        right_vsplit.setSizes([220, 180])

        h_split.addWidget(right_vsplit)
        h_split.setSizes([280, 500])
        layout.addWidget(h_split, stretch=1)

    def load_config(self, config) -> None:
        """Загружает настройки из config в панель настроек (п.11)."""
        self.settings_panel.load_from_config(config)

    @Slot(int, int, int, str)
    def on_progress(self, epoch: int, total: int, stage: int, stage_name: str):
        self.epoch_label.setText(f"Эпоха {epoch} / {total}")
        pct = int(epoch / max(total, 1) * 100)
        self.progress_bar.setValue(pct)
        self.stage_label.setText(f"Этап {stage}/3 — {stage_name}")

    @Slot(dict)
    def on_loss_updated(self, losses: dict):
        mapping = {
            "L_pe": 0, "L_um": 1, "L_vak": 2,
            "L_nez": 3, "L_kr": 4, "L_reb": 5,
        }
        for key, row in mapping.items():
            if key in losses:
                item = self.loss_table.item(row, 1)
                if item:
                    item.setText(f"{losses[key]:.4f}")
        if "loss" in losses:
            self.loss_chart.append_loss("main", losses["loss"])

    def set_training(self, active: bool):
        self.btn_start.setEnabled(not active)
        self.btn_stop.setEnabled(active)
        if active:
            self.progress_bar.setValue(0)
            self.loss_chart.reset()
