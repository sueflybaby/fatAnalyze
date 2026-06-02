"""2D CT-slice viewer with window/level and pan/zoom.

The view is a thin :class:`QGraphicsView` that renders a single axial slice
of a 3D :class:`SimpleITK.Image` as an 8-bit grayscale :class:`QImage`.
W/L is applied at render time only — the underlying HU array is untouched.

Mouse bindings (when *not* in polygon-drawing mode):
- Middle-mouse drag: pan
- Wheel: zoom
- Right-mouse drag: adjust W/L (drag right = widen, drag up = raise center)
- ``0`` key: reset view
- Pixel readout (HU) is shown in the status bar on mouse-move.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import SimpleITK as sitk

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


# W/L presets commonly used in CT viewing
WL_PRESETS: dict[str, tuple[float, float]] = {
    "Soft Tissue": (400.0, 40.0),
    "Bone": (1800.0, 400.0),
    "Lung": (1500.0, -600.0),
    "Liver": (150.0, 30.0),
    "Psoas": (400.0, 40.0),
}


def _numpy_slice(image: sitk.Image, z: int) -> np.ndarray:
    """Return a 2D ``(H, W)`` ``int16`` numpy array for axial slice ``z``."""
    arr = sitk.GetArrayFromImage(image)  # (z, y, x)
    return arr[z, :, :].astype(np.int16, copy=False)


def _apply_window_level(slice_arr: np.ndarray, window: float, level: float) -> np.ndarray:
    """Map HU → 0..255 uint8 using ``[level - window/2, level + window/2]``."""
    lo = level - window / 2.0
    hi = level + window / 2.0
    if hi == lo:
        return np.zeros_like(slice_arr, dtype=np.uint8)
    clipped = np.clip(slice_arr, lo, hi)
    return ((clipped - lo) * 255.0 / (hi - lo)).astype(np.uint8)


class SliceView(QGraphicsView):
    """Display one axial slice; emits pixel-HU and (x, y) signals on hover."""

    pixel_hovered = Signal(int, int, float)  # (x, y, hu_value)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)

        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)

        # Image state
        self._image: Optional[sitk.Image] = None
        self._z_index: int = 0
        self._window: float = 400.0
        self._level: float = 40.0
        self._arr_shape: Optional[Tuple[int, int]] = None  # (H, W)

        # W/L drag state
        self._wl_dragging = False
        self._wl_anchor: Optional[QPointF] = None
        self._wl_window_anchor: float = self._window
        self._wl_level_anchor: float = self._level

    # -- public API ----------------------------------------------------

    def set_image(self, image: sitk.Image) -> None:
        """Bind the viewer to a new 3D volume; resets z to 0 and applies default W/L."""
        self._image = image
        self._z_index = image.GetDepth() // 2  # middle slice by default
        self._arr_shape = (image.GetHeight(), image.GetWidth())
        self._render_slice()

    def set_slice(self, z: int) -> None:
        """Jump to axial slice ``z`` (clamped to volume)."""
        if self._image is None:
            return
        depth = self._image.GetDepth()
        self._z_index = max(0, min(z, depth - 1))
        self._render_slice()

    @property
    def z_index(self) -> int:
        return self._z_index

    @property
    def slice_count(self) -> int:
        return self._image.GetDepth() if self._image is not None else 0

    def set_window_level(self, window: float, level: float) -> None:
        self._window = max(1.0, float(window))
        self._level = float(level)
        self._render_slice()

    def apply_wl_preset(self, name: str) -> None:
        if name not in WL_PRESETS:
            return
        w, l = WL_PRESETS[name]
        self.set_window_level(w, l)

    def get_window_level(self) -> tuple[float, float]:
        return self._window, self._level

    def fit_to_window(self) -> None:
        self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    # -- rendering -----------------------------------------------------

    def _render_slice(self) -> None:
        if self._image is None:
            return
        slice_arr = _numpy_slice(self._image, self._z_index)
        self._arr_shape = (int(slice_arr.shape[0]), int(slice_arr.shape[1]))
        display = _apply_window_level(slice_arr, self._window, self._level)
        h, w = display.shape
        # QImage needs (w, h) row-major, 8-bit grayscale
        qimg = QImage(display.data, w, h, w, QImage.Format_Grayscale8)
        # The QImage references display.data; copy to a QPixmap so the buffer
        # can be freed safely (display is a local).
        pix = QPixmap.fromImage(qimg.copy())
        self._pixmap_item.setPixmap(pix)
        self._scene.setSceneRect(QRectF(pix.rect()))
        # First-time fit
        if self.transform().isIdentity():
            self.fit_to_window()

    # -- mouse handling ------------------------------------------------

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        if self._image is None:
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 1.0 / 1.25
        self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self._image is None:
            return super().mousePressEvent(event)
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self._pan_pressed = True
            super().mousePressEvent(event)
        elif event.button() == Qt.RightButton:
            self._wl_dragging = True
            self._wl_anchor = event.position()
            self._wl_window_anchor = self._window
            self._wl_level_anchor = self._level
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._image is None:
            return super().mouseMoveEvent(event)
        if self._wl_dragging and self._wl_anchor is not None:
            dx = event.position().x() - self._wl_anchor.x()
            dy = event.position().y() - self._wl_anchor.y()
            new_w = max(1.0, self._wl_window_anchor + dx * 2.0)
            new_l = self._wl_level_anchor - dy * 2.0
            self.set_window_level(new_w, new_l)
        # Pixel readout (always)
        scene_pt = self.mapToScene(event.position().toPoint())
        x, y = int(scene_pt.x()), int(scene_pt.y())
        if self._arr_shape is not None and 0 <= x < self._arr_shape[1] and 0 <= y < self._arr_shape[0]:
            arr = _numpy_slice(self._image, self._z_index)
            self.pixel_hovered.emit(x, y, float(arr[y, x]))
        else:
            self.pixel_hovered.emit(x, y, float("nan"))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.NoDrag)
        elif event.button() == Qt.RightButton:
            self._wl_dragging = False
            self._wl_anchor = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() == Qt.Key_0:
            self.fit_to_window()


__all__ = ["SliceView", "WL_PRESETS"]
