"""Запуск графического интерфейса CAIRN.

Использование:
    python scripts/run_gui.py
    python scripts/run_gui.py --config configs/demo.yaml
    python scripts/run_gui.py --theme light
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="CAIRN GUI")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--theme", default="dark", choices=["dark", "light"])
    args = parser.parse_args()

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("Ошибка: PySide6 не установлен.")
        print("Установите: pip install PySide6")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setApplicationName("CAIRN")
    app.setOrganizationName("СПбГУТ")

    from cairn.gui.main_window import CAIRNMainWindow
    from cairn.gui.styles import load_theme

    app.setStyleSheet(load_theme(args.theme))

    window = CAIRNMainWindow(config_path=args.config)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
