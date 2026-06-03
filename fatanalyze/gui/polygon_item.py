"""Editable polygon :class:`QGraphicsPolygonItem` for ROI drawing.

The polygon keeps a list of draggable :class:`_VertexHandle` items.  The
user can:

- left-click on empty area to add a vertex
- drag a vertex handle to refine the shape
- right-click to remove the last vertex
- double-click to close the polygon (caller listens to signal)

A polygon with ``< 3`` vertices is treated as still-being-drawn.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsSceneMouseEvent,
)


Vertex = Tuple[float, float]

HANDLE_RADIUS = 2.5
HANDLE_BRUSH = QBrush(QColor(255, 255, 0, 200))
HANDLE_PEN = QPen(QColor(0, 0, 0, 255), 1.0)


class _PolygonSignals(QObject):
    """QObject that carries the signals for :class:`PolygonItem`."""
    vertex_added = Signal(int)
    vertex_removed = Signal(int)
    vertices_changed = Signal()


class _VertexHandle(QGraphicsEllipseItem):
    """A draggable vertex handle that updates its parent polygon on move."""

    def __init__(self, x: float, y: float, index: int,
                 polygon: PolygonItem) -> None:
        super().__init__(x - HANDLE_RADIUS, y - HANDLE_RADIUS,
                         2 * HANDLE_RADIUS, 2 * HANDLE_RADIUS)
        self._index = index
        self._polygon = polygon
        self.setBrush(HANDLE_BRUSH)
        self.setPen(HANDLE_PEN)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)
        self.setZValue(11)

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            self._polygon._handle_moving(self._index, value)
        elif change == QGraphicsItem.ItemPositionHasChanged:
            self._polygon._handle_moved(self._index)
        return super().itemChange(change, value)


class PolygonItem(QGraphicsPathItem):
    """An editable 2D polygon rendered as a closed :class:`QPainterPath`."""

    def __init__(self, color: QColor = QColor(0, 255, 0),
                 parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self._vertices: List[QPointF] = []
        self._handles: List[_VertexHandle] = []
        self._color = color
        self._closed = False
        self.signals = _PolygonSignals()
        self._is_dragging = False
        self._drag_positions: List[QPointF] = []

        pen = QPen(self._color, 2.0)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(
            QColor(self._color.red(), self._color.green(), self._color.blue(), 60)))
        self.setZValue(10)
        self.setFlag(QGraphicsItem.ItemIsFocusable, True)
        self.setAcceptHoverEvents(True)

    # -- vertex API ----------------------------------------------------

    def add_vertex(self, x: float, y: float) -> int:
        pt = QPointF(x, y)
        self._vertices.append(pt)
        idx = len(self._vertices) - 1
        handle = _VertexHandle(x, y, idx, self)
        scene = self.scene()
        if scene is not None:
            scene.addItem(handle)
        else:
            handle.setParentItem(self)
        self._handles.append(handle)
        self._rebuild_path()
        self.signals.vertex_added.emit(idx)
        return idx

    def remove_last_vertex(self) -> Optional[int]:
        if not self._vertices:
            return None
        handle = self._handles.pop()
        scene = self.scene()
        if scene is not None:
            scene.removeItem(handle)
        self._vertices.pop()
        if len(self._vertices) < 3:
            self._closed = False
        self._rebuild_path()
        idx = len(self._vertices)
        self.signals.vertex_removed.emit(idx)
        return idx

    def clear(self) -> None:
        for h in self._handles:
            scene = h.scene()
            if scene is not None:
                scene.removeItem(h)
        self._handles.clear()
        self._vertices.clear()
        self._rebuild_path()
        self.signals.vertices_changed.emit()

    def close(self) -> None:
        if len(self._vertices) < 3:
            return
        self._closed = True
        self._rebuild_path()

    @property
    def is_closed(self) -> bool:
        return self._closed

    def vertex_count(self) -> int:
        return len(self._vertices)

    def set_color(self, color: QColor) -> None:
        self._color = color
        pen = QPen(self._color, 2.0)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(
            QColor(self._color.red(), self._color.green(), self._color.blue(), 60)))

    def set_vertices(self, vertices: Sequence[Vertex]) -> None:
        self.clear()
        for x, y in vertices:
            self.add_vertex(x, y)
        if len(self._vertices) >= 3:
            self._closed = True
            self._rebuild_path()

    def get_vertices(self) -> List[Vertex]:
        return [(float(v.x()), float(v.y())) for v in self._vertices]

    def vertex_at(self, scene_pt: QPointF, tolerance: float = 8.0) -> Optional[int]:
        """Return the index of a vertex near *scene_pt*, or None."""
        for i, h in enumerate(self._handles):
            if h.sceneBoundingRect().contains(scene_pt):
                return i
        return None

    # -- handle drag ---------------------------------------------------

    def _handle_moving(self, index: int, new_pos: QPointF) -> None:
        self._is_dragging = True
        self._vertices[index] = new_pos
        self._rebuild_path()

    def _handle_moved(self, index: int) -> None:
        self._is_dragging = False
        self.signals.vertices_changed.emit()

    # -- path building ------------------------------------------------

    def _rebuild_path(self) -> None:
        if not self._vertices:
            self.setPath(QPainterPath())
            return
        path = QPainterPath(self._vertices[0])
        for v in self._vertices[1:]:
            path.lineTo(v)
        if self._closed and len(self._vertices) >= 3:
            path.closeSubpath()
        self.setPath(path)

    def boundingRect(self) -> QRectF:
        return super().boundingRect().adjusted(
            -HANDLE_RADIUS, -HANDLE_RADIUS,
            HANDLE_RADIUS, HANDLE_RADIUS)


__all__ = ["PolygonItem"]
