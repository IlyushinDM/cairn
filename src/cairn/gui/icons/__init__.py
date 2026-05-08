"""Иконки CAIRN GUI — SVG файлы в стиле VS Code.

Использование:
    from cairn.gui.icons import icon
    btn.setIcon(icon("analyze"))
"""
from __future__ import annotations
from pathlib import Path
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor
from PySide6.QtCore import Qt, QSize
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

ICONS_DIR = Path(__file__).parent


def icon(name: str, color: str = "#c8ccd4", size: int = 20) -> QIcon:
    """Возвращает QIcon из SVG-файла с заданным цветом."""
    svg_path = ICONS_DIR / f"{name}.svg"
    if not svg_path.exists():
        return QIcon()

    # Читаем SVG и заменяем цвет
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_text = svg_text.replace('currentColor', color)
    svg_bytes = svg_text.encode("utf-8")

    renderer = QSvgRenderer(svg_bytes)
    pixmap   = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter  = QPainter(pixmap)
    renderer.render(painter)
    painter.end()

    return QIcon(pixmap)
