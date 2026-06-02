"""Editable polygon :class:`QGraphicsPolygonItem` for ROI drawing.

The polygon keeps a list of :class:`QGraphicsEllipseItem` vertex handles
sized ~5 px. The user can:
- left-click on empty area to add a vertex (caller decides when to "close")
- drag a vertex handle to refine
- right-click to remove the last vertex
- the caller calls :meth:`PolygonItem.polygon` to get the final vertex list
  and the polygon is then detached from the scene.

A polygon with ``< 3`` vertices is treated as still-being-drawn (no fill,
no border close path).

Note: :class:`QGraphicsItem` is *not* a :class:`QObject` in PySide6, so this
class cannot expose :class:`Signal` directly. Instead it embeds a tiny
``QObject`` helper (``_Signals``) that callers can connect to via
:attr:`signals`. The signals are emitted on vertex add/remove/change.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
)


Vertex = Tuple[float, float]

HANDLE_RADIUS = 5.0
HANDLE_BRUSH = QBrush(QColor(255, 255, 0, 200))
HANDLE_PEN = QPen(QColor(0, 0, 0, 255), 1.0)


class _PolygonSignals(QObject):
    """QObject that carries the signals for :class:`PolygonItem`."""
    vertex_added = Signal(int)
    vertex_removed = Signal(int)
    vertices_changed = Signal()


class PolygonItem(QGraphicsPathItem):
    """An editable 2D polygon rendered as a closed :class:`QPainterPath`."""

    def __init__(self, color: QColor = QColor(0, 255, 0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self._vertices: List[QPointF] = []
        self._handles: List[QGraphicsEllipseItem] = []
        self._color = color
        self._closed = False
        self.signals = _PolygonSignals()

        pen = QPen(self._color, 2.0)
        pen.setCosmetic(True)
        self.setPen(pen)
        brush = QBrush(QColor(self._color.red(), self._color.green(), self._color.blue(), 60))
        self.setBrush(brush)
        self.setZValue(10)
        self.setFlag(QGraphicsItem.ItemIsFocusable, True)

    # -- vertex API ----------------------------------------------------

    def add_vertex(self, x: float, y: float) -> int:
        """Append a new vertex at scene coords ``(x, y)``; returns the new index."""
        pt = QPointF(x, y)
        self._vertices.append(pt)
        handle = QGraphicsEllipseItem(
            pt.x() - HANDLE_RADIUS, pt.y() - HANDLE_RADIUS,
            2 * HANDLE_RADIUS, 2 * HANDLE_RADIUS,
        )
        handle.setBrush(HANDLE_BRUSH)
        handle.setPen(HANDLE_PEN)
        handle.setFlag(QGraphicsItem.ItemIsMovable, True)
        handle.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)
        handle.setZValue(11)
        # Add to scene (NOT as child of self) so it can be moved independently
        scene = self.scene()
        if scene is not None:
            scene.addItem(handle)
        else:
            handle.setParentItem(self)
        self._handles.append(handle)
        self._rebuild_path()
        idx = len(self._vertices) - 1
        self.signals.vertex_added.emit(idx)
        return idx

    def remove_last_vertex(self) -> Optional[int]:
        """Pop the last vertex (and its handle). Returns the removed index or None."""
        if not self._vertices:
            return None
        handle = self._handles.pop()
        scene = self.scene()
        if scene is not None:
            scene.removeItem(handle)
        self._vertices.pop()
        # Going below 3 vertices means the polygon is no longer closeable
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
        """Mark the polygon as closed (must have >= 3 vertices)."""
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
        """Change the polygon's outline + fill color in place."""
        self._color = color
        pen = QPen(self._color, 2.0)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(QColor(
            self._color.red(), self._color.green(), self._color.blue(), 60,
        )))

    def get_vertices(self) -> List[Vertex]:
        return [(float(v.x()), float(v.y())) for v in self._vertices]

    # -- handle drag ---------------------------------------------------

    def itemChange(self, change, value):  # type: ignore[override]
        if change == QGraphicsItem.ItemScenePositionHasChanged:
            for i, h in enumerate(self._handles):
                # When a child handle moves, its scenePos is the value
                self._vertices[i] = h.scenePos()
            self._rebuild_path()
            self.signals.vertices_changed.emit()
        return super().itemChange(change, value)

    # -- path building -------------------------------------------------

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

    def boundingRect(self) -> QRectF:  # type: ignore[override]
        return super().boundingRect().adjusted(-HANDLE_RADIUS, -HANDLE_RADIUS,
                                               HANDLE_RADIUS, HANDLE_RADIUS)


__all__ = ["PolygonItem"]
