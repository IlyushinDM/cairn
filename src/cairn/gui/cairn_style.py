"""QProxyStyle для CAIRN – рисует стрелки SpinBox в нужном цвете.

Проблема: QSS на Windows переключает QSpinBox в режим custom drawing,
но не рисует стрелки – только пустые прямоугольники.
Решение: QProxyStyle.drawPrimitive перехватывает рисование стрелок
и рисует их в нужном цвете через QPainter.

Использование:
    app.setStyle(CAIRNStyle("Fusion"))
"""
from __future__ import annotations

from PySide6.QtCore    import Qt, QPoint
from PySide6.QtGui     import QColor, QPainter, QPolygon
from PySide6.QtWidgets import QProxyStyle, QStyle, QStyleOption


class CAIRNStyle(QProxyStyle):
    """Тонкая обёртка над Fusion/Windows стилем.

    Перекрашивает стрелки QSpinBox в светлый цвет (#cccccc)
    чтобы они были видны на тёмном фоне.
    """

    _ARROW_COLOR_DARK  = QColor("#cccccc")
    _ARROW_COLOR_LIGHT = QColor("#444444")
    _current_theme: str = "dark"

    def set_theme(self, theme: str) -> None:
        self._current_theme = theme

    def _arrow_color(self) -> QColor:
        return (self._ARROW_COLOR_DARK
                if self._current_theme == "dark"
                else self._ARROW_COLOR_LIGHT)

    def drawPrimitive(
        self,
        element: QStyle.PrimitiveElement,
        option:  QStyleOption,
        painter: QPainter,
        widget=None,
    ) -> None:
        arrow_elements = {
            QStyle.PrimitiveElement.PE_IndicatorArrowUp,
            QStyle.PrimitiveElement.PE_IndicatorArrowDown,
            QStyle.PrimitiveElement.PE_IndicatorArrowLeft,
            QStyle.PrimitiveElement.PE_IndicatorArrowRight,
        }

        if element not in arrow_elements:
            super().drawPrimitive(element, option, painter, widget)
            return

        r   = option.rect
        cx  = r.center().x()
        cy  = r.center().y()
        s   = min(r.width(), r.height()) // 3  # половина основания

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._arrow_color())

        if element == QStyle.PrimitiveElement.PE_IndicatorArrowUp:
            pts = QPolygon([
                QPoint(cx,     cy - s),
                QPoint(cx - s, cy + s),
                QPoint(cx + s, cy + s),
            ])
        elif element == QStyle.PrimitiveElement.PE_IndicatorArrowDown:
            pts = QPolygon([
                QPoint(cx,     cy + s),
                QPoint(cx - s, cy - s),
                QPoint(cx + s, cy - s),
            ])
        elif element == QStyle.PrimitiveElement.PE_IndicatorArrowLeft:
            pts = QPolygon([
                QPoint(cx - s, cy),
                QPoint(cx + s, cy - s),
                QPoint(cx + s, cy + s),
            ])
        else:  # Right
            pts = QPolygon([
                QPoint(cx + s, cy),
                QPoint(cx - s, cy - s),
                QPoint(cx - s, cy + s),
            ])

        painter.drawPolygon(pts)
        painter.restore()
