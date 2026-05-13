"""Запуск графического интерфейса CAIRN.

Использование:
    python scripts/run_gui.py
    python scripts/run_gui.py --config configs/demo.yaml
    python scripts/run_gui.py --theme light
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="CAIRN GUI")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--theme",  default="dark", choices=["dark", "light"])
    args = parser.parse_args()

    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QFont
    except ImportError:
        print("Ошибка: PySide6 не установлен.")
        print("Установите: pip install PySide6")
        sys.exit(1)

    # Подавляем QFont::setPointSize <= 0 на Windows с HiDPI.
    # AA_UseHighDpiPixmaps deprecated в Qt6 (включён по умолчанию).
    # Единственный рабочий способ: задать QT_FONT_DPI явно до QApplication.
    if sys.platform == "win32":
        # Фиксируем DPI шрифтов = 96 (стандарт). Qt6 сам занимается масштабированием.
        os.environ.setdefault("QT_FONT_DPI", "96")

    app = QApplication(sys.argv)
    app.setApplicationName("CAIRN")
    app.setOrganizationName("СПбГУТ")

    # Явный шрифт ДО загрузки QSS — критично для Windows
    font = QFont("Segoe UI", 10) if sys.platform == "win32" else QFont("SF Pro Text", 10)
    if font.pointSize() > 0:
        app.setFont(font)

    from cairn.gui.main_window import CAIRNMainWindow
    from cairn.gui.styles import load_theme
    from cairn.gui.cairn_style import CAIRNStyle

    # Применяем кастомный стиль (рисует стрелки SpinBox в нужном цвете)
    cairn_proxy = CAIRNStyle("Fusion")
    cairn_proxy.set_theme(args.theme)
    app.setStyle(cairn_proxy)
    app.setStyleSheet(load_theme(args.theme))

    window = CAIRNMainWindow(config_path=args.config)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
