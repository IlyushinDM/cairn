"""Диалог выбора демонстрационного сценария (для кнопки «Демо»)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup, QDialog, QDialogButtonBox, QGroupBox, QLabel,
    QRadioButton, QVBoxLayout, QWidget,
)


SCENARIO_INFO = {
    "1": {
        "name":        "CPU Exhaustion",
        "root":        "order-service-1",
        "fault":       "cpu_exhaustion",
        "description": "Перегрузка CPU в order-service-1.\n"
                       "Эффект: деградация задержки вниз по цепочке вызовов.",
        "dir":         "scenario_1",
    },
    "2": {
        "name":        "Memory Leak",
        "root":        "cache-service-1",
        "fault":       "memory_pressure",
        "description": "Утечка памяти в cache-service-1.\n"
                       "Эффект: влияние на соседний order-service через совместное размещение.",
        "dir":         "scenario_2",
    },
    "3": {
        "name":        "Network Delay",
        "root":        "frontend-1",
        "fault":       "latency_spike",
        "description": "Сетевая задержка на пути frontend-1 → order-service-1.\n"
                       "Эффект: рост P99-латентности по всей цепочке.",
        "dir":         "scenario_3",
    },
    "4": {
        "name":        "Payment Overload",
        "root":        "payment-service-1",
        "fault":       "overload",
        "description": "Перегрузка payment-service-1.\n"
                       "Эффект: деградация по всей цепочке оплаты.",
        "dir":         "scenario_4",
    },
    "5": {
        "name":        "Database Bottleneck",
        "root":        "database-1",
        "fault":       "cpu_exhaustion",
        "description": "Bottleneck в центральном database-1.\n"
                       "Эффект: деградация всех зависимых сервисов.",
        "dir":         "scenario_5",
    },
}


class ScenarioDialog(QDialog):
    """Диалог выбора демонстрационного сценария."""

    def __init__(self, data_dir: Path, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._data_dir = data_dir
        self._selected: Optional[str] = None

        self.setWindowTitle("Выбор демонстрационного сценария")
        self.setMinimumWidth(520)
        self.resize(540, 680)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        title = QLabel("Выберите сценарий для демонстрации:")
        title.setStyleSheet("font-size: 14px; font-weight: 600; margin-bottom: 4px;")
        layout.addWidget(title)

        self._radios: dict[str, QRadioButton] = {}
        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)

        for key, info in SCENARIO_INFO.items():
            sc_dir = data_dir / info["dir"]
            has_data = (sc_dir / "metrics.csv").exists()

            group = QGroupBox(f"Сценарий {key}: {info['name']}")
            gl = QVBoxLayout(group)
            gl.setSpacing(4)
            gl.setContentsMargins(10, 8, 10, 8)

            radio = QRadioButton(f"Первопричина: {info['root']}  [{info['fault']}]")
            if key == "1":
                radio.setChecked(True)
            if not has_data:
                radio.setEnabled(False)
                radio.setText(radio.text() + "  ⚠ данные не найдены")
            self._radios[key] = radio
            self._btn_group.addButton(radio)
            gl.addWidget(radio)

            radio.setStyleSheet(
                "QRadioButton{spacing:8px;}"
                "QRadioButton::indicator{width:16px;height:16px;border-radius:8px;}"
                "QRadioButton::indicator:unchecked{background:#2d3348;border:2px solid #6c7a9c;}"
                "QRadioButton::indicator:checked{background:#4a9eff;border:2px solid #4a9eff;}"
            )
            desc = QLabel(info["description"])
            desc.setWordWrap(True)
            desc.setStyleSheet("color: #8892a4; font-size: 11px; margin-left: 22px;")
            gl.addWidget(desc)

            if not has_data:
                hint = QLabel("  Запустите: python scripts/generate_demo_data.py")
                hint.setStyleSheet("color: #f6a623; font-size: 10px; margin-left: 22px;")
                gl.addWidget(hint)

            layout.addWidget(group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_ok(self) -> None:
        for key, radio in self._radios.items():
            if radio.isChecked():
                self._selected = key
                break
        self.accept()

    @property
    def selected_scenario(self) -> Optional[str]:
        return self._selected

    @property
    def scenario_dir(self) -> Optional[Path]:
        if self._selected:
            return self._data_dir / SCENARIO_INFO[self._selected]["dir"]
        return None

    @property
    def scenario_info(self) -> Optional[dict]:
        return SCENARIO_INFO.get(self._selected) if self._selected else None
