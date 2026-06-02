"""Interactive user-ROI helpers: draw a polygon, analyze, visualize."""
from __future__ import annotations

from fatanalyze.interactive.analyze import analyze_user_roi
from fatanalyze.interactive.polygon_drawer import (
    MatplotlibPolygonDrawer,
    draw_roi_2d,
)
from fatanalyze.interactive.polygon_utils import (
    empty_mask_like,
    rasterize_polygon,
)
from fatanalyze.interactive.user_roi import UserROI, load_user_roi
from fatanalyze.interactive.viz import plot_user_roi


__all__ = [
    "UserROI",
    "MatplotlibPolygonDrawer",
    "analyze_user_roi",
    "draw_roi_2d",
    "empty_mask_like",
    "load_user_roi",
    "plot_user_roi",
    "rasterize_polygon",
]
