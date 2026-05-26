"""ChainGraphWidget – интерактивный граф цепочки доказательств.

Возможности:
  - Перетаскивание узлов мышью
  - Зум колесом мыши
  - Легенда вне графа (над ним)
  - Кнопки: Скрыть подписи / По размеру / Открыть в окне
  - Светлый/тёмный фон по теме приложения
"""
from __future__ import annotations
import math
from typing import Optional

from PySide6.QtCore  import Qt, Signal, QPointF
from PySide6.QtGui   import QBrush, QColor, QFont, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import (
    QDialog, QGraphicsEllipseItem, QGraphicsItem,
    QGraphicsLineItem, QGraphicsScene, QGraphicsTextItem,
    QGraphicsView, QHBoxLayout, QLabel, QPushButton,
    QToolTip, QVBoxLayout, QWidget,
)

NODE_R = 30

_VALID_EDGE_TYPES = {"call", "async", "sync", "data", "event", "dep", "rpc"}
# Атрибуты networkx которые могут попасть в edge_type
_INTERNAL_ATTRS  = {"none", "color", "colour", "weight", "style",
                     "width", "label", "shape", "arrow"}

def _sanitize_edge_label(edge_type: str) -> str:
    """Возвращает читаемую метку ребра (фильтрует служебные networkx атрибуты)."""
    s = str(edge_type or "").strip().lower()
    if not s or s in _INTERNAL_ATTRS:
        return "call"
    for valid in _VALID_EDGE_TYPES:
        if s.startswith(valid):
            return valid[:4]
    return s[:4] or "call"




def _current_app_theme() -> str:
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app:
        for w in app.topLevelWidgets():
            if hasattr(w, "_current_theme"):
                return w._current_theme
    return "dark"


def _is_light() -> bool:
    return _current_app_theme() == "light"


def _bg_color() -> QColor:
    return QColor("#f2f2f2") if _is_light() else QColor("#161922")



def _text_color() -> QColor:
    return QColor("#111111") if _is_light() else QColor("#ffffff")


def _edge_color() -> QColor:
    return QColor("#207a50") if _is_light() else QColor("#3ecf8e")


def _edge_label_color() -> QColor:
    return QColor("#333333") if _is_light() else QColor("#a0a8bc")


# Цвета узлов
C_ROOT   = QColor("#ff4444")
C_CHAIN  = QColor("#4a9eff")


class _Node(QGraphicsEllipseItem):
    def __init__(self, idx: int, name: str, color: QColor, parent=None):
        r = NODE_R
        super().__init__(-r, -r, 2*r, 2*r, parent)
        self._idx   = idx
        self._name  = name
        self._color = color
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(2)
        self._apply_brush()
        self.setToolTip(name)

        self._lbl = QGraphicsTextItem(name, self)
        self._lbl.setDefaultTextColor(_text_color())
        f = QFont(); f.setPointSize(8); f.setBold(True)
        self._lbl.setFont(f)
        self._center()

    def _apply_brush(self) -> None:
        g = QRadialGradient(0, -NODE_R * .3, NODE_R * 1.2)
        g.setColorAt(0.0, self._color.lighter(130))
        g.setColorAt(1.0, self._color.darker(120))
        self.setBrush(QBrush(g))
        self.setPen(QPen(self._color.darker(150), 1.5))

    def _center(self) -> None:
        br = self._lbl.boundingRect()
        self._lbl.setPos(-br.width()/2, -br.height()/2)

    def set_label_visible(self, v: bool) -> None:
        self._lbl.setVisible(v)

    def refresh_colors(self) -> None:
        self._lbl.setDefaultTextColor(_text_color())

    def hoverEnterEvent(self, event) -> None:
        """Показываем tooltip с именем при скрытых подписях."""
        if not self._lbl.isVisible():
            pos = event.screenPos()
            # PySide6: screenPos() возвращает QPointF или QPoint
            if hasattr(pos, "toPoint"):
                pos = pos.toPoint()
            QToolTip.showText(pos, self._name)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        QToolTip.hideText()
        super().hoverLeaveEvent(event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            sc = self.scene()
            if sc and hasattr(sc, "_widget"):
                sc._widget._update_edges()
        return super().itemChange(change, value)


class _Edge(QGraphicsLineItem):
    def __init__(self, src: _Node, dst: _Node, label: str = "", parent=None):
        super().__init__(parent)
        self.src    = src
        self.dst    = dst
        self.label  = label
        self._pen   = QPen(_edge_color(), 2.0)
        self.setPen(self._pen)
        self.setZValue(1)

        # Подпись типа ребра
        self._lbl = QGraphicsTextItem(label, self)
        self._lbl.setDefaultTextColor(_edge_label_color())
        f = QFont(); f.setPointSize(7)
        self._lbl.setFont(f)
        self.update_pos()

    def update_pos(self) -> None:
        sp, dp = self.src.pos(), self.dst.pos()
        dx, dy = dp.x()-sp.x(), dp.y()-sp.y()
        dist = math.sqrt(dx*dx + dy*dy) or 1
        sx = sp.x() + dx/dist * NODE_R
        sy = sp.y() + dy/dist * NODE_R
        ex = dp.x() - dx/dist * NODE_R
        ey = dp.y() - dy/dist * NODE_R
        self.setLine(sx, sy, ex, ey)
        # Подпись – по центру ребра
        self._lbl.setPos((sx+ex)/2 + 4, (sy+ey)/2 - 10)

    def refresh_colors(self) -> None:
        self._pen = QPen(_edge_color(), 2.0)
        self.setPen(self._pen)
        self._lbl.setDefaultTextColor(_edge_label_color())


class _Scene(QGraphicsScene):
    def __init__(self, widget, parent=None):
        super().__init__(parent)
        self._widget = widget
        self.setBackgroundBrush(QBrush(_bg_color()))

    def drawBackground(self, painter, rect) -> None:
        self.setBackgroundBrush(QBrush(_bg_color()))
        super().drawBackground(painter, rect)


class _View(QGraphicsView):
    def __init__(self, scene: _Scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def drawBackground(self, painter, rect) -> None:
        """Фон обновляется при каждом рендере – учитывает смену темы."""
        self.setBackgroundBrush(QBrush(_bg_color()))
        super().drawBackground(painter, rect)

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1/1.15
        self.scale(factor, factor)

    def fit(self) -> None:
        items = self.scene().items()
        if not items:
            return
        self.fitInView(self.scene().itemsBoundingRect().adjusted(-20,-20,20,20),
                       Qt.AspectRatioMode.KeepAspectRatio)


class ChainGraphWidget(QWidget):
    """Интерактивный граф цепочки доказательств с легендой вне графа."""

    def __init__(self, parent=None, show_detach_btn: bool = True):
        super().__init__(parent)
        self._show_detach_btn = show_detach_btn
        self._nodes: list[_Node] = []
        self._edges: list[_Edge] = []
        self._labels_visible = True
        self._last_chain = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # ── Тулбар + легенда ─────────────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(6)

        self._btn_labels = QPushButton("Скрыть подписи")
        self._btn_labels.setFixedHeight(24)
        self._btn_labels.setCheckable(True)
        self._btn_labels.setStyleSheet("font-size: 11px; padding: 0 8px;")
        self._btn_labels.clicked.connect(self._toggle_labels)
        top.addWidget(self._btn_labels)

        btn_fit = QPushButton("По размеру")
        btn_fit.setFixedHeight(24)
        btn_fit.setStyleSheet("font-size: 11px; padding: 0 8px;")
        btn_fit.clicked.connect(self._fit)
        top.addWidget(btn_fit)

        if show_detach_btn:
            btn_pop = QPushButton("В окне")
            btn_pop.setFixedHeight(24)
            btn_pop.setStyleSheet("font-size: 11px; padding: 0 8px;")
            btn_pop.clicked.connect(self._detach)
            top.addWidget(btn_pop)

        top.addStretch()

        # Легенда ВНЕ графа, справа от кнопок
        for color, text in [("#ff4444", "Первопричина"),
                             ("#4a9eff", "Каскадный эффект"),
                             ("#3ecf8e", "Ребро: call/color/dep")]:
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 14px;")
            dot.setFixedWidth(18)
            top.addWidget(dot)
            lbl = QLabel(text)
            lbl.setStyleSheet("font-size: 11px;")
            top.addWidget(lbl)
            top.addSpacing(8)

        layout.addLayout(top)

        # ── Граф ─────────────────────────────────────────────────────────
        self._scene = _Scene(self)
        self._view  = _View(self._scene, self)
        layout.addWidget(self._view, stretch=1)

    # ── Публичный API ─────────────────────────────────────────────────────

    def show_chain(self, chain) -> None:
        """Строит граф из объекта EvidenceChain."""
        self._last_chain = chain
        self._scene.clear()
        self._nodes.clear()
        self._edges.clear()
        self._scene.setBackgroundBrush(QBrush(_bg_color()))

        # Строим граф
        import math as _math

        n = len(chain.path_nodes)
        if n == 0:
            return

        # Размещаем узлы по кругу / в линию
        if n == 1:
            positions = [(0.0, 0.0)]
        else:
            R = max(120, n * 50)
            positions = [
                (R * _math.cos(2*_math.pi*i/n - _math.pi/2),
                 R * _math.sin(2*_math.pi*i/n - _math.pi/2))
                for i in range(n)
            ]

        # Карта name → node
        name_map: dict[str, _Node] = {}
        for i, node in enumerate(chain.path_nodes):
            color = C_ROOT if i == 0 else C_CHAIN
            gnode = _Node(i, node.node_name, color)
            gnode.setPos(QPointF(*positions[i]))
            gnode.set_label_visible(self._labels_visible)
            self._scene.addItem(gnode)
            self._nodes.append(gnode)
            name_map[node.node_name] = gnode

        # Рёбра
        for edge in chain.path_edges:
            src_name = chain.path_nodes[0].node_name
            dst_name = None
            for n_obj in chain.path_nodes:
                if n_obj.node_idx == edge.src:
                    src_name = n_obj.node_name
                if n_obj.node_idx == edge.dst:
                    dst_name = n_obj.node_name
            if dst_name and src_name in name_map and dst_name in name_map:
                gedge = _Edge(name_map[src_name], name_map[dst_name],
                              label=_sanitize_edge_label(getattr(edge, "edge_type", "")))
                self._scene.addItem(gedge)
                self._edges.append(gedge)

        self._view.fit()
        # Принудительное обновление фона (palette обновляется после QSS)
        self._scene.update()

    def clear(self) -> None:
        self._scene.clear()
        self._nodes.clear()
        self._edges.clear()
        self._last_chain = None

    def refresh_theme(self) -> None:
        """Обновляет цвета при смене темы."""
        self._scene.setBackgroundBrush(QBrush(_bg_color()))
        for node in self._nodes:
            node.refresh_colors()
        for edge in self._edges:
            edge.refresh_colors()
        self._scene.update()

    # ── Внутренние ────────────────────────────────────────────────────────

    def _update_edges(self) -> None:
        for edge in self._edges:
            edge.update_pos()

    def _toggle_labels(self, checked: bool) -> None:
        self._labels_visible = not checked
        self._btn_labels.setText(
            "Показать подписи" if checked else "Скрыть подписи")
        for node in self._nodes:
            node.set_label_visible(self._labels_visible)

    def _fit(self) -> None:
        self._view.fit()

    def _detach(self) -> None:
        if self._last_chain is None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Цепочка доказательств")
        dlg.resize(900, 550)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(4, 4, 4, 4)
        sub = ChainGraphWidget(show_detach_btn=False)
        sub.show_chain(self._last_chain)
        layout.addWidget(sub)
        dlg.setAttribute(__import__("PySide6.QtCore", fromlist=["Qt"]).Qt.WidgetAttribute.WA_DeleteOnClose)
        dlg.setModal(False)
        dlg.show()
        dlg.raise_()
        # Повторный fit после show – виджет теперь имеет правильный размер
        from PySide6.QtCore import QTimer
        QTimer.singleShot(100, sub._view.fit)
