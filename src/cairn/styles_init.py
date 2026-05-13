"""Загрузчик тем CAIRN.

При загрузке QSS заменяет плейсхолдеры __ARROW_*__ на реальные
абсолютные пути к SVG-файлам стрелок (в папке styles/arrows/).
Это единственный надёжный способ задать ::up-arrow / ::down-arrow
в QSS без Qt Resource System (.qrc).
"""
from __future__ import annotations

from pathlib import Path

_STYLES_DIR = Path(__file__).parent


def load_theme(theme: str = "dark") -> str:
    """Загружает QSS-тему и подставляет пути к SVG-стрелкам."""
    filename = "dark_theme.qss" if theme == "dark" else "light_theme.qss"
    qss_path = _STYLES_DIR / filename
    if not qss_path.exists():
        return ""

    qss = qss_path.read_text(encoding="utf-8")

    # Папка со стрелками
    arrows_dir = _STYLES_DIR / "arrows"

    # Qt требует forward slash даже на Windows, без пробелов
    # Если в пути есть пробелы/спецсимволы — экранируем через repr не нужно,
    # Qt сам обрабатывает путь. Главное — forward slash.
    def arrow_url(name: str) -> str:
        p = (arrows_dir / name).resolve()
        # Конвертируем в строку с forward slash
        return str(p).replace("\\", "/")

    qss = qss.replace("__ARROW_UP__",          arrow_url("arrow_up.svg"))
    qss = qss.replace("__ARROW_DOWN__",         arrow_url("arrow_down.svg"))
    qss = qss.replace("__ARROW_UP_LIGHT__",     arrow_url("arrow_up_light.svg"))
    qss = qss.replace("__ARROW_DOWN_LIGHT__",   arrow_url("arrow_down_light.svg"))

    return qss
