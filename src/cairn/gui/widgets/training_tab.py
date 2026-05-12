"""Вкладка «Обучение» с QThread для фоновой тренировки."""

from __future__ import annotations

from PySide6.QtCore import QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)


class TrainingWorker(QThread):
    """QThread-воркер для фонового обучения CAIRN."""

    progress     = Signal(int, int, int, str)     # epoch, total, stage, stage_name
    loss_updated = Signal(dict)                   # {component: value}
    finished     = Signal(dict)                   # history
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
            # Патчим метод логирования для эмита сигналов
            original_log = self.trainer._log

            stage_epochs = {
                1: self.config.pretrain_epochs,
                2: self.config.main_epochs,
                3: self.config.finetune_epochs,
            }
            stage_names = {1: "Претрейн", 2: "Основное", 3: "Файнтюн"}
            current_stage = [1]
            current_epoch = [0]

            def patched_log(msg: str):
                original_log(msg)
                # Парсим эпоху из сообщения
                if "эп." in msg:
                    try:
                        ep = int(msg.split("эп.")[1].split(":")[0].strip())
                        stage = current_stage[0]
                        total = stage_epochs[stage]
                        self.progress.emit(ep, total, stage, stage_names[stage])
                    except Exception:
                        pass
                # Парсим потери
                if "loss=" in msg:
                    try:
                        val = float(msg.split("loss=")[1].strip())
                        self.loss_updated.emit({"loss": val})
                    except Exception:
                        pass

            self.trainer._log = patched_log

            # Этапы
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

        self._history: dict[str, list[float]] = {"pretrain": [], "main": [], "finetune": []}
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
            self._ax.set_xlabel("Эпоха", color="#6c7a9c", fontsize=9)
            self._ax.set_ylabel("Потеря", color="#6c7a9c", fontsize=9)
            self._ax.set_title("Функция потерь", color="#a0a8bc", fontsize=10)
            fig.tight_layout()
            self._fig = fig
            canvas = FigureCanvasQTAgg(fig)
            canvas.setMinimumHeight(200)
            return canvas
        except ImportError:
            self._fig = None
            self._ax = None
            lbl = QLabel("График потерь\n(требуется matplotlib)")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("color: #6c7a9c; border: 1px dashed #2d3348; border-radius:6px;")
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
                self._ax.plot(vals, label=labels[key], color=colors[key], linewidth=1.5)
        self._ax.legend(fontsize=9, labelcolor="#a0a8bc",
                       facecolor="#1e2130", edgecolor="#2d3348")
        self._ax.tick_params(colors="#6c7a9c")
        self._ax.spines[:].set_color("#2d3348")
        self._ax.set_xlabel("Эпоха", color="#6c7a9c", fontsize=9)
        self._ax.set_ylabel("Потеря", color="#6c7a9c", fontsize=9)
        if self._fig is not None:
            self._fig.tight_layout()
        if hasattr(self._canvas, 'draw'):
            self._canvas.draw()  # type: ignore[union-attr]

    def reset(self):
        self._history = {"pretrain": [], "main": [], "finetune": []}


class TrainingTab(QWidget):
    """Вкладка «Обучение»."""

    start_requested = Signal()
    stop_requested  = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # ── Управление ──
        ctrl = QHBoxLayout()
        self.btn_start = QPushButton("▶  Начать обучение")
        self.btn_start.setObjectName("primaryBtn")
        self.btn_start.setFixedHeight(36)
        self.btn_start.clicked.connect(self.start_requested)

        self.btn_stop = QPushButton("⏹  Остановить")
        self.btn_stop.setObjectName("dangerBtn")
        self.btn_stop.setFixedHeight(36)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_requested)

        self.stage_label = QLabel("Этап: –")
        self.stage_label.setStyleSheet("color: #6c7a9c; font-size: 12px;")

        ctrl.addWidget(self.btn_start)
        ctrl.addWidget(self.btn_stop)
        ctrl.addSpacing(20)
        ctrl.addWidget(self.stage_label)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        # ── Прогресс ──
        prog_layout = QHBoxLayout()
        self.epoch_label = QLabel("Эпоха 0 / 0")
        self.epoch_label.setFixedWidth(120)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        prog_layout.addWidget(self.epoch_label)
        prog_layout.addWidget(self.progress_bar)
        layout.addLayout(prog_layout)

        # ── График ──
        lbl = QLabel("ФУНКЦИЯ ПОТЕРЬ")
        lbl.setObjectName("sectionTitle")
        layout.addWidget(lbl)

        self.loss_chart = LossChart()
        layout.addWidget(self.loss_chart)

        # ── Таблица компонентов ──
        lbl2 = QLabel("КОМПОНЕНТЫ ПОТЕРЬ")
        lbl2.setObjectName("sectionTitle")
        layout.addWidget(lbl2)

        self.loss_table = QTableWidget(6, 3)
        self.loss_table.setHorizontalHeaderLabels(["Компонент", "Значение", "Описание"])
        self.loss_table.verticalHeader().setVisible(False)
        self.loss_table.setMaximumHeight(170)
        self.loss_table.horizontalHeader().setStretchLastSection(True)
        self.loss_table.setAlternatingRowColors(True)

        components = [
            ("L_ПЭ",  "–", "Ранжирование причинных эффектов"),
            ("L_УМ",  "–", "Условная модель нормального состояния"),
            ("L_ВАК", "–", "Вариационный автокодировщик"),
            ("L_нез", "–", "Ограничение независимости"),
            ("L_КР",  "–", "Контрастное разделение"),
            ("L_реб", "–", "Штраф за необоснованные рёбра"),
        ]
        for row, (name, val, desc) in enumerate(components):
            self.loss_table.setItem(row, 0, QTableWidgetItem(name))
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.loss_table.setItem(row, 1, item)
            self.loss_table.setItem(row, 2, QTableWidgetItem(desc))
            self.loss_table.setRowHeight(row, 26)
        layout.addWidget(self.loss_table)

    @Slot(int, int, int, str)
    def on_progress(self, epoch: int, total: int, stage: int, stage_name: str):
        self.epoch_label.setText(f"Эпоха {epoch} / {total}")
        pct = int(epoch / max(total, 1) * 100)
        self.progress_bar.setValue(pct)
        self.stage_label.setText(f"Этап {stage}/3 – {stage_name}")
        # Обновляем график
        key_map = {1: "pretrain", 2: "main", 3: "finetune"}

    @Slot(dict)
    def on_loss_updated(self, losses: dict):
        # Обновляем таблицу компонентов
        mapping = {
            "L_pe": 0, "L_um": 1, "L_vak": 2,
            "L_nez": 3, "L_kr": 4, "L_reb": 5,
        }
        for key, row in mapping.items():
            if key in losses:
                item = self.loss_table.item(row, 1)
                if item:
                    item.setText(f"{losses[key]:.4f}")

        # Обновляем график
        if "loss" in losses:
            self.loss_chart.append_loss("main", losses["loss"])

    def set_training(self, active: bool):
        self.btn_start.setEnabled(not active)
        self.btn_stop.setEnabled(active)
        if active:
            self.progress_bar.setValue(0)
            self.loss_chart.reset()
