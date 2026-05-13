"""Диалог настроек CAIRN с вкладками по секциям конфигурации."""

from __future__ import annotations

from typing import Any

import yaml
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QScrollArea, QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)


class _BoundedSpinBox(QSpinBox):
    """QSpinBox с блёкнущими стрелками на границах диапазона.

    п.8: стрелочки визуально неактивны когда значение на границе.
    Нельзя выйти за range с клавиатуры или колесика.
    """

    def __init__(self, lo: int, hi: int, value: int, step: int = 1):
        super().__init__()
        self.setRange(lo, hi)
        self.setSingleStep(step)
        self.setValue(value)
        self.setFixedHeight(26)
        self.valueChanged.connect(self._update_arrow_style)
        self._update_arrow_style(value)

    def _update_arrow_style(self, v: int) -> None:
        # Не используем inline setStyleSheet — он скрывает нативные стрелки Qt.
        # Вместо этого меняем enabled-состояние кнопок через setProperty.
        self.setProperty("atMin", v <= self.minimum())
        self.setProperty("atMax", v >= self.maximum())
        # Принудительно обновляем стиль (QSS может читать свойства)
        self.style().unpolish(self)
        self.style().polish(self)

    def wheelEvent(self, event) -> None:
        """Колесико не выходит за диапазон."""
        delta = self.singleStep() if event.angleDelta().y() > 0 else -self.singleStep()
        self.setValue(max(self.minimum(), min(self.maximum(), self.value() + delta)))
        event.accept()


class _BoundedDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox с блёкнущими стрелками на границах.

    п.8: то же что _BoundedSpinBox, но для float.
    """

    def __init__(self, lo: float, hi: float, value: float,
                 decimals: int = 3, step: float = 0.001):
        super().__init__()
        self.setRange(lo, hi)
        self.setDecimals(decimals)
        self.setSingleStep(step)
        self.setValue(value)
        self.setFixedHeight(26)
        self.valueChanged.connect(self._update_arrow_style)
        self._update_arrow_style(value)

    def _update_arrow_style(self, v: float) -> None:
        self.setProperty("atMin", v <= self.minimum() + 1e-12)
        self.setProperty("atMax", v >= self.maximum() - 1e-12)
        self.style().unpolish(self)
        self.style().polish(self)

    def wheelEvent(self, event) -> None:
        delta = self.singleStep() if event.angleDelta().y() > 0 else -self.singleStep()
        self.setValue(max(self.minimum(), min(self.maximum(), self.value() + delta)))
        event.accept()


def _spin(value: float, lo: float, hi: float, decimals: int = 0,
          step: float = 1) -> _BoundedDoubleSpinBox | _BoundedSpinBox:
    """Фабрика SpinBox с ограничениями диапазона (п.8)."""
    if decimals > 0:
        return _BoundedDoubleSpinBox(lo, hi, value, decimals, step)
    return _BoundedSpinBox(int(lo), int(hi), int(value), int(step))


class SettingsDialog(QDialog):
    """Диалог настроек с вкладками: Модель / Обучение / Воронка / Верификатор."""

    config_saved = Signal(dict)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки CAIRN")
        self.setMinimumSize(700, 560)
        self.resize(760, 620)
        self._config = config
        self._widgets: dict[str, Any] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        layout.setSpacing(12)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_model_tab(),    "Модель")
        self.tabs.addTab(self._build_training_tab(), "Обучение")
        self.tabs.addTab(self._build_funnel_tab(),   "Воронка")
        self.tabs.addTab(self._build_verifier_tab(), "Верификатор")
        layout.addWidget(self.tabs)

        # Кнопки
        btn_row = QHBoxLayout()
        reset_btn = QPushButton("↺  Сбросить по умолчанию")
        reset_btn.clicked.connect(self._reset_defaults)
        save_btn = QPushButton("Сохранить конфигурацию")
        save_btn.setObjectName("primaryBtn")
        save_btn.clicked.connect(self._save)
        cancel_btn = QPushButton("Отмена")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(reset_btn)
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

    def _scroll_wrap(self, inner: QWidget) -> QScrollArea:
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameStyle(0)  # QFrame.Shape.NoFrame
        sa.setWidget(inner)
        return sa

    def _build_model_tab(self) -> QWidget:
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        mc = self._config.model
        groups = {
            "Размерности векторов": [
                ("state_dim",   "Размерность состояния d",   mc.state_dim,   16, 512,  0),
                ("context_dim", "Размерность контекста d_c", mc.context_dim, 4,  64,   0),
                ("metric_dim",  "Размерность метрик d_met",  mc.metric_dim,  8,  256,  0),
                ("log_dim",     "Размерность журналов d_log",mc.log_dim,     4,  128,  0),
                ("trace_dim",   "Размерность трассировок",   mc.trace_dim,   4,  64,   0),
            ],
            "GMM и конфаундеры": [
                ("gmm_components",    "Компонент GMM D",           mc.gmm_components,    2, 20,  0),
                ("latent_confounders","Скрытых факторов K",        mc.latent_confounders,1, 10,  0),
                ("confounder_dim",    "Размерность фактора d_z",   mc.confounder_dim,    8, 128, 0),
            ],
            "Архитектура": [
                ("hypergraph_layers","Слоёв гиперграфа",         mc.hypergraph_layers, 1, 5,  0),
                ("attention_heads",  "Голов внимания",            mc.attention_heads,   1, 16, 0),
                ("attention_layers", "Слоёв внимания",            mc.attention_layers,  1, 8,  0),
                ("breakpoint_window","Окно разрыва W",            mc.breakpoint_window, 10, 300, 0),
            ],
        }

        for group_name, fields in groups.items():
            grp = QGroupBox(group_name)
            form = QFormLayout(grp)
            form.setSpacing(8)
            form.setContentsMargins(12, 16, 12, 12)
            for key, label, val, lo, hi, dec in fields:
                w = _spin(val, lo, hi, dec)
                self._widgets[f"model.{key}"] = w
                form.addRow(label + ":", w)
            layout.addWidget(grp)

        layout.addStretch()
        return self._scroll_wrap(inner)

    def _build_training_tab(self) -> QWidget:
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        tc = self._config.training
        lw = self._config.training.loss_weights

        groups = {
            "Этапы обучения (эпох)": [
                ("pretrain_epochs",  "Этап 1 (претрейн)",  tc.pretrain_epochs,  1, 500, 0),
                ("main_epochs",      "Этап 2 (основное)",  tc.main_epochs,      1, 500, 0),
                ("finetune_epochs",  "Этап 3 (файнтюн)",   tc.finetune_epochs,  1, 200, 0),
                ("freeze_epochs",    "Заморозка (этап 2)", tc.freeze_epochs,    0, 50,  0),
            ],
            "Оптимизация": [
                ("lr",           "Скорость обучения",  tc.lr,          1e-5, 1.0, 6, 1e-4),
                ("batch_size",   "Размер батча",        tc.batch_size,  1, 64,  0),
                ("margin",       "Отступ L_ПЭ",         tc.margin,      0.0, 2.0, 3, 0.05),
                ("tcd_margin",   "Отступ L_КР δ",       tc.tcd_margin,  0.0, 2.0, 3, 0.05),
                ("beta_kl",      "KL-коэф. β_u",        tc.beta_kl,     0.0, 5.0, 2, 0.1),
                ("beta_kl_z",    "KL-коэф. β_z",        tc.beta_kl_z,   0.0, 2.0, 3, 0.01),
            ],
            "Веса компонентов потерь λ": [
                ("lambda_pe",  "λ₁ (L_ПЭ)",  lw.lambda_pe,  0.0, 5.0, 2, 0.1),
                ("lambda_um",  "λ₂ (L_УМ)",  lw.lambda_um,  0.0, 5.0, 2, 0.1),
                ("lambda_vak", "λ₃ (L_ВАК)", lw.lambda_vak, 0.0, 5.0, 2, 0.1),
                ("lambda_nez", "λ₄ (L_нез)", lw.lambda_nez, 0.0, 5.0, 2, 0.1),
                ("lambda_kr",  "λ₅ (L_КР)",  lw.lambda_kr,  0.0, 5.0, 2, 0.1),
                ("lambda_reb", "λ₆ (L_реб)", lw.lambda_reb, 0.0, 5.0, 2, 0.1),
            ],
        }

        for group_name, fields in groups.items():
            grp = QGroupBox(group_name)
            form = QFormLayout(grp)
            form.setSpacing(8)
            form.setContentsMargins(12, 16, 12, 12)
            for key, label, val, lo, hi, dec, *rest in fields:
                step = rest[0] if rest else 1
                w = _spin(val, lo, hi, dec, step)
                self._widgets[f"training.{key}"] = w
                form.addRow(label + ":", w)
            layout.addWidget(grp)

        layout.addStretch()
        return self._scroll_wrap(inner)

    def _build_funnel_tab(self) -> QWidget:
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        fc = self._config.funnel
        grp = QGroupBox("Каскадная воронка 500→30→5→1")
        form = QFormLayout(grp)
        form.setSpacing(10)
        form.setContentsMargins(12, 16, 12, 12)

        fields = [
            ("l0_top_k",   "L0: top-k кандидатов",  fc.l0_top_k,   1, 500, 0),
            ("l1_top_k",   "L1: top-k кандидатов",  fc.l1_top_k,   1, 100, 0),
            ("l2_top_k",   "L2: top-k кандидатов",  fc.l2_top_k,   1, 20,  0),
            ("local_hops", "Шагов локального BFS",  fc.local_hops,  1, 5,   0),
            ("alpha_init", "Начальный α (L0)",      fc.alpha_init,  0.0, 1.0, 3, 0.05),
        ]
        for key, label, val, lo, hi, dec, *rest in fields:
            step = rest[0] if rest else 1
            w = _spin(val, lo, hi, dec, step)
            self._widgets[f"funnel.{key}"] = w
            form.addRow(label + ":", w)

        layout.addWidget(grp)
        layout.addStretch()
        return self._scroll_wrap(inner)

    def _build_verifier_tab(self) -> QWidget:
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        vc = self._config.verifier
        grp = QGroupBox("Параметры верификатора (5 аксиом)")
        form = QFormLayout(grp)
        form.setSpacing(10)
        form.setContentsMargins(12, 16, 12, 12)

        fields = [
            ("temporal_tolerance_sec", "Допуск Δ темпорал. (с)",   vc.temporal_tolerance_sec, 0.0, 300.0, 1, 1.0),
            ("transitivity_threshold", "Порог транзитивности",      vc.transitivity_threshold, 0.0, 1.0, 3, 0.05),
            ("monotonicity_epsilon",   "ε монотонности",            vc.monotonicity_epsilon,   0.0, 0.5, 3, 0.01),
            ("permutation_tests",      "Тестов перестановки K",     vc.permutation_tests,      1, 100, 0, 1),
            ("edge_significance_threshold", "Порог значимости θ_е", vc.edge_significance_threshold, 0.0, 0.5, 3, 0.01),
            ("confounder_threshold",   "Порог конфаундера θ_с",    vc.confounder_threshold, 0.0, 1.0, 3, 0.05),
        ]
        for key, label, val, lo, hi, dec, step in fields:
            w = _spin(val, lo, hi, dec, step)
            self._widgets[f"verifier.{key}"] = w
            form.addRow(label + ":", w)

        layout.addWidget(grp)
        layout.addStretch()
        return self._scroll_wrap(inner)

    def _reset_defaults(self):
        """Сбрасывает значения к конфигурации по умолчанию."""
        from cairn.config import CAIRNConfig
        defaults = CAIRNConfig()
        mapping = {
            "model": defaults.model,
            "training": defaults.training,
            "funnel": defaults.funnel,
            "verifier": defaults.verifier,
        }
        for full_key, widget in self._widgets.items():
            section, attr = full_key.split(".", 1)
            obj = mapping.get(section)
            if obj is None:
                continue
            # Вложенный ключ (например, training.lambda_pe)
            parts = attr.split(".")
            val = obj
            try:
                for p in parts:
                    val = getattr(val, p)
                if hasattr(widget, 'setValue'):
                    widget.setValue(val)
            except AttributeError:
                pass

    def _save(self):
        """Применяет изменения к конфигурации и эмитирует сигнал."""
        mapping = {
            "model": self._config.model,
            "training": self._config.training,
            "funnel": self._config.funnel,
            "verifier": self._config.verifier,
        }
        for full_key, widget in self._widgets.items():
            section, attr = full_key.split(".", 1)
            obj = mapping.get(section)
            if obj is None:
                continue
            try:
                val = widget.value()
                setattr(obj, attr, val)
            except Exception:
                pass

        # Сохраняем в YAML
        try:
            import yaml
            from pathlib import Path
            cfg_path = Path("configs/default.yaml")
            cfg_path.parent.mkdir(exist_ok=True)
            # Простой дамп текущей конфигурации
            data = {
                "model":    dict(vars(self._config.model)),
                "training": dict(vars(self._config.training)),
                "funnel":   dict(vars(self._config.funnel)),
                "verifier": dict(vars(self._config.verifier)),
            }
            with cfg_path.open("w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True)
            QMessageBox.information(self, "CAIRN", f"Конфигурация сохранена: {cfg_path}")
        except Exception as e:
            QMessageBox.warning(self, "CAIRN", f"Не удалось сохранить: {e}")

        self.config_saved.emit({})
        self.accept()
