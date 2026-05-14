"""Диалог подключения живой системы к CAIRN через LiveSystemConnector."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QGroupBox, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QVBoxLayout, QWidget,
)

from cairn.connectors.live_connector import LiveSystemConnector, discover_connector_configs


class ConnectWorker(QThread):
    """Фоновый поток проверки доступности системы."""
    result = Signal(bool, str)  # ok, message

    def __init__(self, connector: LiveSystemConnector):
        super().__init__()
        self._connector = connector

    def run(self):
        ok, msg = self._connector.is_available()
        self.result.emit(ok, msg)


class ConnectDialog(QDialog):
    """Диалог выбора и подключения живой системы.

    Архитектурный принцип:
        CAIRN не знает о конкретных системах.
        Пользователь выбирает .yaml-файл конфига → создаётся LiveSystemConnector.
        Этот коннектор реализует единый интерфейс BaseMetricConnector.
    """

    connected = Signal(object)  # LiveSystemConnector

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Подключить живую систему")
        self.setMinimumWidth(520)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # ── Заголовок ──────────────────────────────────────────────────────
        title = QLabel("Выберите конфиг системы для подключения")
        title.setStyleSheet("font-size: 14px; font-weight: 600;")
        layout.addWidget(title)

        hint = QLabel(
            "CAIRN подключается к любой системе через конфиг-файл (.yaml).\n"
            "Каждый файл описывает источник метрик и топологию."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #8892a4; font-size: 11px;")
        layout.addWidget(hint)

        # ── Выбор конфига ──────────────────────────────────────────────────
        cfg_group = QGroupBox("Конфигурация коннектора")
        cgl       = QVBoxLayout(cfg_group)

        combo_row = QHBoxLayout()
        combo_row.setSpacing(6)

        # п.9: одинаковая высота у всех элементов строки
        ELEM_H = 28

        self._combo = QComboBox()
        self._combo.setMinimumWidth(280)
        self._combo.setFixedHeight(ELEM_H)
        self._refresh_combo()
        self._combo.currentIndexChanged.connect(self._on_combo_changed)
        combo_row.addWidget(self._combo, stretch=1)

        btn_browse = QPushButton("Обзор")
        btn_browse.setFixedHeight(ELEM_H)
        btn_browse.setFixedWidth(72)
        btn_browse.clicked.connect(self._browse)
        combo_row.addWidget(btn_browse)

        # п.9: понятный значок «обновить» вместо арабоподобного ↻
        btn_refresh = QPushButton("Обновить список")
        btn_refresh.setFixedHeight(ELEM_H)
        btn_refresh.setToolTip("Обновить список конфигов")
        btn_refresh.clicked.connect(self._refresh_combo)
        combo_row.addWidget(btn_refresh)
        cgl.addLayout(combo_row)

        # Описание выбранного конфига
        self._desc_label = QLabel("")
        self._desc_label.setWordWrap(True)
        self._desc_label.setStyleSheet("color: #6c7a9c; font-size: 11px; margin-top: 4px;")
        cgl.addWidget(self._desc_label)

        layout.addWidget(cfg_group)

        # ── Статус подключения ─────────────────────────────────────────────
        status_group = QGroupBox("Статус")
        sgl = QVBoxLayout(status_group)

        self._status_label = QLabel("Выберите конфиг для проверки подключения")
        self._status_label.setStyleSheet("font-size: 12px;")
        self._status_label.setWordWrap(True)
        sgl.addWidget(self._status_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)   # indeterminate
        self._progress.setVisible(False)
        sgl.addWidget(self._progress)

        btn_check = QPushButton("Проверить подключение")
        btn_check.clicked.connect(self._check_connection)
        sgl.addWidget(btn_check)

        layout.addWidget(status_group)

        # ── Кнопки ────────────────────────────────────────────────────────
        self._btn_box = QDialogButtonBox()
        self._ok_btn = self._btn_box.addButton("Подключить", QDialogButtonBox.ButtonRole.AcceptRole)
        self._btn_box.addButton("Отмена",     QDialogButtonBox.ButtonRole.RejectRole)
        self._ok_btn.setEnabled(False)
        self._btn_box.accepted.connect(self._on_connect)
        self._btn_box.rejected.connect(self.reject)
        layout.addWidget(self._btn_box)

        self._connector: Optional[LiveSystemConnector] = None
        self._worker:    Optional[ConnectWorker]        = None

        # Обновляем описание для первого элемента
        self._on_combo_changed(0)

    def _refresh_combo(self) -> None:
        """Обновляет список конфигов из configs/connectors/."""
        self._combo.clear()
        configs = discover_connector_configs("configs/connectors")
        for cfg_path in configs:
            self._combo.addItem(cfg_path.stem.replace("_", " ").title(), str(cfg_path))
        if not configs:
            self._combo.addItem("(нет конфигов – нажмите Обзор…)", "")
        self._combo.update()

    def _browse(self) -> None:
        """Открывает диалог выбора файла."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбор конфига коннектора",
            "configs/connectors",
            "YAML конфиги (*.yaml *.yml)"
        )
        if path:
            self._combo.addItem(Path(path).stem, path)
            self._combo.setCurrentIndex(self._combo.count() - 1)

    def _on_combo_changed(self, idx: int) -> None:
        """Обновляет описание при смене конфига."""
        path = self._combo.itemData(idx)
        if not path:
            return
        try:
            import yaml
            with open(path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            sys_info = cfg.get("system", {})
            name  = sys_info.get("name", "–")
            desc  = sys_info.get("description", "")
            src   = cfg.get("metrics", {}).get("source", "–")
            self._desc_label.setText(
                f"{name} | Источник: {src}\n{desc}"
            )
        except Exception:
            self._desc_label.setText("")

    def _check_connection(self) -> None:
        """Асинхронно проверяет доступность системы."""
        path = self._combo.currentData()
        if not path:
            return
        try:
            self._connector = LiveSystemConnector(path)
        except Exception as e:
            self._status_label.setText(f"❌ Ошибка конфига: {e}")
            self._status_label.setStyleSheet("color: #ff5f5f; font-size: 12px;")
            return

        self._progress.setVisible(True)
        self._status_label.setText("Проверяю подключение…")
        self._status_label.setStyleSheet("color: #6c7a9c; font-size: 12px;")
        self._ok_btn.setEnabled(False)

        self._worker = ConnectWorker(self._connector)
        self._worker.result.connect(self._on_check_result)
        self._worker.start()

    def _on_check_result(self, ok: bool, msg: str) -> None:
        self._progress.setVisible(False)
        if ok:
            self._status_label.setText(f"✓ {msg}")
            self._status_label.setStyleSheet("color: #3ecf8e; font-size: 12px;")
            self._ok_btn.setEnabled(True)
        else:
            self._status_label.setText(f"❌ {msg}")
            self._status_label.setStyleSheet("color: #ff5f5f; font-size: 12px;")

    def _on_connect(self) -> None:
        if self._connector:
            self.connected.emit(self._connector)
        self.accept()

    @property
    def connector(self) -> Optional[LiveSystemConnector]:
        return self._connector
