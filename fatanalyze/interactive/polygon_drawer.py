"""Interactive 2D polygon drawer using ``matplotlib.widgets.PolygonSelector``."""
from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.widgets import Button, PolygonSelector

from fatanalyze.interactive.polygon_utils import (
    empty_mask_like,
    rasterize_polygon,
)
from fatanalyze.interactive.user_roi import UserROI
from fatanalyze.viz.overlay import _window

logger = logging.getLogger(__name__)


_DEFAULT_WINDOW: Tuple[float, float] = (40, 400)   # soft-tissue (center, width)


def _default_z_index(image: sitk.Image) -> int:
    return image.GetSize()[2] // 2


class MatplotlibPolygonDrawer:
    """Block on a matplotlib figure; user clicks points to define a polygon.

    On finish, the polygon is rasterized to a 3D sitk.Image mask and returned
    inside a :class:`UserROI`. The figure uses a soft-tissue CT window.

    Parameters
    ----------
    image : sitk.Image
        3D CT volume.
    z_index : int, optional
        Axial slice index to draw on. Defaults to the middle slice.
    name : str
        Display name for the ROI.
    preset : str
        Target preset (e.g. ``"iliopsoas_left"``). Validated at construction.
    title : str, optional
        Figure title.
    color : str
        Polygon edge color.
    window : (center, width)
        Soft-tissue display window.
    """

    def __init__(
        self,
        image: sitk.Image,
        z_index: Optional[int] = None,
        name: str = "roi",
        preset: str = "iliopsoas_left",
        title: Optional[str] = None,
        color: str = "#9467bd",
        window: Tuple[float, float] = _DEFAULT_WINDOW,
    ) -> None:
        self.image = image
        self.z_index = z_index if z_index is not None else _default_z_index(image)
        self.name = name
        self.preset = preset
        self.title = title
        self.color = color
        self.window = window
        self._polygon_verts: Optional[Sequence[Tuple[float, float]]] = None

        UserROI(   # validates preset early
            name=self.name, preset=self.preset, mask=empty_mask_like(self.image),
            z_index=self.z_index, n_points=0, color=self.color,
        )
        self._build_figure()

    def _build_figure(self) -> None:
        arr = sitk.GetArrayFromImage(self.image)
        z = max(0, min(self.z_index, arr.shape[0] - 1))
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.imshow(_window(arr[z], self.window), cmap="gray", origin="lower")
        self.ax.set_title(self.title or f"Draw polygon on slice z={z} (preset={self.preset})")
        self.ax.set_xlabel("x (px)")
        self.ax.set_ylabel("y (px)")

        self.selector = PolygonSelector(
            self.ax, onselect=self._on_select, useblit=True,
            props=dict(color=self.color, linewidth=2, alpha=0.8),
        )
        ax_finish = self.fig.add_axes((0.81, 0.01, 0.17, 0.05))
        self.btn_finish = Button(ax_finish, "Finish", color="#eeeeee", hovercolor="#cccccc")
        self.btn_finish.on_clicked(self._on_finish)
        self.fig.text(
            0.02, 0.02,
            "Click to add vertices, drag to move, press Esc or 'Finish' to close polygon.",
            fontsize=8, color="gray",
        )

    def _on_select(self, verts: Sequence[Tuple[float, float]]) -> None:
        self._polygon_verts = list(verts)

    def _on_finish(self, _event) -> None:
        plt.close(self.fig)

    def run(self) -> UserROI:
        """Block until the user finalizes the polygon (or closes the figure)."""
        plt.show()
        verts = self._polygon_verts
        if not verts or len(verts) < 3:
            return UserROI(
                name=self.name, preset=self.preset, mask=empty_mask_like(self.image),
                z_index=self.z_index, n_points=0,
                empty_warning=True, color=self.color,
                notes="Closed without drawing a polygon (need >=3 vertices).",
            )
        mask = rasterize_polygon(self.image, self.z_index, verts)
        return UserROI(
            name=self.name, preset=self.preset, mask=mask,
            z_index=self.z_index, n_points=len(verts), color=self.color,
        )


def draw_roi_2d(
    image: sitk.Image,
    z_index: Optional[int] = None,
    name: str = "roi",
    preset: str = "iliopsoas_left",
    title: Optional[str] = None,
    color: str = "#9467bd",
) -> UserROI:
    """One-call helper: open the polygon drawer and return a :class:`UserROI`."""
    return MatplotlibPolygonDrawer(
        image, z_index=z_index, name=name, preset=preset,
        title=title, color=color,
    ).run()


__all__ = ["MatplotlibPolygonDrawer", "draw_roi_2d"]
