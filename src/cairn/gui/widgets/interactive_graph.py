"""Интерактивный граф для CAIRN GUI.

Возможности:
  - Pan (перетаскивание мышью)
  - Zoom (колесо мыши)
  - Клик по узлу → сигнал node_clicked
  - Переключение подписей (кнопка)
  - Открытие в отдельном окне
  - Выделение узла программно (из таблицы)
  - Легенда цветов
  - Tooltip при наведении (если подписи скрыты)

Использование:
    graph = InteractiveGraphWidget()
    graph.set_graph(nodes, edges, root_idx)
    graph.node_clicked.connect(lambda idx: ...)
    graph.highlight_node(idx)
"""
from __future__ import annotations

import math
from typing import Optional

from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import (
    QBrush, QColor, QFont, QPainter, QPen, QWheelEvent,
    QLinearGradient, QRadialGradient,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem,
    QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsTextItem,
    QGraphicsView, QHBoxLayout, QLabel, QPushButton,
    QToolTip, QVBoxLayout, QWidget, QDialog,
)


# ── Цветовая схема ─────────────────────────────────────────────────────────────
COLORS = {
    "root":      QColor("#ff4444"),   # красный – первопричина
    "high":      QColor("#ff8c00"),   # оранжевый – высокий ПЭ
    "medium":    QColor("#f0c040"),   # жёлтый – средний ПЭ
    "low":       QColor("#4a9eff"),   # синий – низкий ПЭ
    "normal":    QColor("#2d3d5a"),   # тёмно-синий – нормальный
    "selected":  QColor("#ffffff"),   # белый – выделен
    "edge_call": QColor("#3ecf8e"),   # зелёный – вызов
    "edge_colo": QColor("#f6a623"),   # оранжевый – совместное размещение
    "bg":        QColor("#161922"),   # фон
    "text_light":QColor("#ffffff"),
    "text_dark": QColor("#111111"),
}

NODE_RADIUS = 28


class GraphNode(QGraphicsEllipseItem):
    """Узел графа с интерактивностью."""

    def __init__(self, node_idx: int, name: str, color: QColor,
                 score: float = 0.0, parent=None):
        r = NODE_RADIUS
        super().__init__(-r, -r, 2*r, 2*r, parent)
        self.node_idx  = node_idx
        self.node_name = name
        self.score     = score
        self._base_color = color
        self._selected   = False
        self._label_visible = True

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(2)
        self.setToolTip(f"{name}\nScore: {score:.4f}")

        self._apply_color(color)

        # Подпись
        self._label = QGraphicsTextItem(name, self)
        self._label.setDefaultTextColor(COLORS["text_light"])
        font = QFont()
        font.setPointSize(8)
        font.setWeight(QFont.Weight.Bold)
        self._label.setFont(font)
        self._center_label()

    def _center_label(self) -> None:
        br = self._label.boundingRect()
        self._label.setPos(-br.width()/2, -br.height()/2)

    def _apply_color(self, color: QColor) -> None:
        grad = QRadialGradient(0, -NODE_RADIUS*0.3, NODE_RADIUS*1.2)
        light = color.lighter(130)
        grad.setColorAt(0.0, light)
        grad.setColorAt(1.0, color.darker(120))
        self.setBrush(QBrush(grad))
        pen_color = COLORS["selected"] if self._selected else color.darker(150)
        pen_width  = 3 if self._selected else 1.5
        self.setPen(QPen(pen_color, pen_width))

    def set_selected_highlight(self, selected: bool) -> None:
        self._selected = selected
        self._apply_color(self._base_color)
        self.update()

    def set_label_visible(self, visible: bool) -> None:
        self._label_visible = visible
        self._label.setVisible(visible)

    def itemChange(self, change, value):
        """Обновляет рёбра при перемещении узла."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            scene = self.scene()
            if scene and hasattr(scene, "_graph_widget"):
                scene._graph_widget._update_edges()
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event) -> None:
        self.setPen(QPen(COLORS["selected"], 3))
        if not self._label_visible:
            QToolTip.showText(
                event.screenPos(),
                f"{self.node_name}\nScore: {self.score:.4f}"
            )
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        pen_color = COLORS["selected"] if self._selected else self._base_color.darker(150)
        pen_width  = 3 if self._selected else 1.5
        self.setPen(QPen(pen_color, pen_width))
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            scene = self.scene()
            if scene and hasattr(scene, "_graph_widget"):
                scene._graph_widget.node_clicked.emit(self.node_idx)
        super().mousePressEvent(event)


class GraphEdge(QGraphicsLineItem):
    """Ребро графа со стрелкой."""

    def __init__(self, src_node: GraphNode, dst_node: GraphNode,
                 edge_type: str = "call", parent=None):
        super().__init__(parent)
        self.src = src_node
        self.dst = dst_node
        color = COLORS["edge_call"] if edge_type == "call" else COLORS["edge_colo"]
        style = Qt.PenStyle.SolidLine if edge_type == "call" else Qt.PenStyle.DashLine
        self.setPen(QPen(color, 1.8, style))
        self.setZValue(1)
        self._color = color
        self.update_position()

    def update_position(self) -> None:
        sp = self.src.pos()
        dp = self.dst.pos()
        dx = dp.x() - sp.x()
        dy = dp.y() - sp.y()
        dist = math.sqrt(dx*dx + dy*dy) or 1
        # Укорачиваем линию на радиус узла
        sx = sp.x() + dx/dist * NODE_RADIUS
        sy = sp.y() + dy/dist * NODE_RADIUS
        ex = dp.x() - dx/dist * NODE_RADIUS
        ey = dp.y() - dy/dist * NODE_RADIUS
        self.setLine(sx, sy, ex, ey)


class GraphScene(QGraphicsScene):
    """Сцена графа."""
    def __init__(self, graph_widget, parent=None):
        super().__init__(parent)
        self._graph_widget = graph_widget
        self.setBackgroundBrush(QBrush(COLORS["bg"]))


class InteractiveGraphView(QGraphicsView):
    """Вид с pan и zoom."""

    def __init__(self, scene: GraphScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(COLORS["bg"]))
        self._zoom = 1.0

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1/1.15
        self._zoom = max(0.2, min(5.0, self._zoom * factor))
        self.resetTransform()
        self.scale(self._zoom, self._zoom)

    def reset_zoom(self) -> None:
        self._zoom = 1.0
        self.resetTransform()
        self.fitInView(self.scene().sceneRect().adjusted(-20, -20, 20, 20),
                       Qt.AspectRatioMode.KeepAspectRatio)


class InteractiveGraphWidget(QWidget):
    """Виджет интерактивного графа с панелью управления."""

    node_clicked = Signal(int)  # node_idx

    def __init__(self, parent=None):
        super().__init__(parent)
        self._nodes: dict[int, GraphNode] = {}
        self._edges: list[GraphEdge] = []
        self._labels_visible = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # ── Панель управления ─────────────────────────────────────────────
        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)

        self._btn_labels = QPushButton("Скрыть подписи")
        self._btn_labels.setFixedHeight(24)
        self._btn_labels.clicked.connect(self._toggle_labels)

        self._btn_fit = QPushButton("По размеру")
        self._btn_fit.setFixedHeight(24)
        self._btn_fit.clicked.connect(self._fit_view)

        self._btn_window = QPushButton("Открыть окно ↗")
        self._btn_window.setFixedHeight(24)
        self._btn_window.clicked.connect(self._open_in_window)

        # Легенда
        legend = QLabel(
            "<span style='color:#ff4444'>●</span> root &nbsp;"
            "<span style='color:#ff8c00'>●</span> высокий &nbsp;"
            "<span style='color:#f0c040'>●</span> средний &nbsp;"
            "<span style='color:#4a9eff'>●</span> низкий &nbsp;"
            "– вызов &nbsp;"
            "<span style='color:#f6a623'>- -</span> совм.размещение"
        )
        legend.setStyleSheet("color:#6c7a9c; font-size:9px;")
        legend.setTextFormat(Qt.TextFormat.RichText)

        ctrl.addWidget(self._btn_labels)
        ctrl.addWidget(self._btn_fit)
        ctrl.addWidget(self._btn_window)
        ctrl.addStretch()
        ctrl.addWidget(legend)
        layout.addLayout(ctrl)

        # ── Сцена и вид ───────────────────────────────────────────────────
        self._scene = GraphScene(self)
        self._view  = InteractiveGraphView(self._scene)
        layout.addWidget(self._view)

    # ── Построение графа ──────────────────────────────────────────────────

    def _update_edges(self) -> None:
        """Обновляет позиции всех рёбер."""
        for edge in self._edges:
            edge.update_position()

    def set_graph(self, nodes: list[dict], edges: list[dict],
                  root_idx: Optional[int] = None) -> None:
        """Строит граф из данных.

        nodes: [{"idx": int, "name": str, "score": float}, ...]
        edges: [{"src": int, "dst": int, "type": str}, ...]
        """
        self._scene.clear()
        self._nodes = {}
        self._edges = []

        if not nodes:
            return

        # Позиции: окружность или spring-layout
        n = len(nodes)
        positions = self._layout(nodes, edges)

        # Нормализуем скоры для цвета
        scores = [nd.get("score", 0.0) for nd in nodes]
        s_min  = min(scores)
        s_max  = max(scores)
        s_span = s_max - s_min

        # Используем ранговую нормализацию когда значения близки
        use_rank = s_span < 0.1 * (abs(s_max) + 1e-6)
        if use_rank:
            sorted_scores = sorted(range(len(scores)), key=lambda i: scores[i])
            rank_of = {i: sorted_scores.index(i) for i in range(len(scores))}
            n_nodes = max(len(scores) - 1, 1)

        for ni, nd in enumerate(nodes):
            idx   = nd["idx"]
            name  = nd["name"]
            score = nd.get("score", 0.0)

            if use_rank:
                t = rank_of[ni] / n_nodes
            else:
                t = (score - s_min) / max(s_span, 1e-8)

            if idx == root_idx:
                color = COLORS["root"]
            elif t > 0.66:
                color = COLORS["high"]
            elif t > 0.33:
                color = COLORS["medium"]
            else:
                color = COLORS["low"]

            node = GraphNode(idx, name, color, score)
            pos  = positions.get(idx, (0.0, 0.0))
            node.setPos(pos[0], pos[1])
            self._scene.addItem(node)
            self._nodes[idx] = node

        for ed in edges:
            src_node = self._nodes.get(ed["src"])
            dst_node = self._nodes.get(ed["dst"])
            if src_node and dst_node:
                edge = GraphEdge(src_node, dst_node, ed.get("type", "call"))
                self._scene.addItem(edge)
                self._edges.append(edge)

        self._view.reset_zoom()

    def _layout(self, nodes: list[dict], edges: list[dict]) -> dict[int, tuple]:
        """Простой круговой layout с попыткой spring-layout через networkx."""
        positions = {}
        n = len(nodes)
        try:
            import networkx as nx
            G = nx.DiGraph()
            for nd in nodes:
                G.add_node(nd["idx"])
            for ed in edges:
                G.add_edge(ed["src"], ed["dst"])
            pos = nx.spring_layout(G, seed=42, k=3.0, iterations=50)
            scale = max(120, n * 60)
            for idx, (x, y) in pos.items():
                positions[idx] = (x * scale, -y * scale)
        except ImportError:
            # Fallback: круговой layout
            radius = max(100, n * 40)
            for i, nd in enumerate(nodes):
                angle = 2 * math.pi * i / n - math.pi/2
                positions[nd["idx"]] = (
                    radius * math.cos(angle),
                    radius * math.sin(angle),
                )
        return positions

    def highlight_node(self, node_idx: int) -> None:
        """Выделяет узел (вызывается из таблицы)."""
        for idx, node in self._nodes.items():
            node.set_selected_highlight(idx == node_idx)

    def _toggle_labels(self) -> None:
        self._labels_visible = not self._labels_visible
        for node in self._nodes.values():
            node.set_label_visible(self._labels_visible)
        self._btn_labels.setText(
            "Показать подписи" if not self._labels_visible else "Скрыть подписи"
        )

    def _fit_view(self) -> None:
        self._view.reset_zoom()

    def _open_in_window(self) -> None:
        """Открывает граф в отдельном окне."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Граф причинного распространения")
        dlg.resize(900, 650)
        dlg_layout = QVBoxLayout(dlg)
        dlg_layout.setContentsMargins(8, 8, 8, 8)

        # Создаём второй вид на ту же сцену
        view2 = InteractiveGraphView(self._scene)
        view2.fitInView(self._scene.sceneRect().adjusted(-20,-20,20,20),
                        Qt.AspectRatioMode.KeepAspectRatio)

        hint = QLabel("Колесо мыши – zoom | Перетаскивание – pan | Клик по узлу – выделить")
        hint.setStyleSheet("color:#6c7a9c; font-size:10px;")
        dlg_layout.addWidget(hint)
        dlg_layout.addWidget(view2)
        dlg.exec()