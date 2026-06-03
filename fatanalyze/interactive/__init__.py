"""Interactive user-ROI helpers: draw a polygon, analyze, visualize.

.. note::
   ``MatplotlibPolygonDrawer``, ``draw_roi_2d``, and ``plot_user_roi``
   are **not** re-exported here because they pull in ``matplotlib``,
   which causes import errors in the frozen GUI app on some Windows
   locales. Imports them directly from their modules when needed:

   .. code:: python

       from fatanalyze.interactive.polygon_drawer import MatplotlibPolygonDrawer
       from fatanalyze.interactive.viz import plot_user_roi
"""
from __future__ import annotations

from fatanalyze.interactive.analyze import analyze_user_roi
from fatanalyze.interactive.polygon_utils import (
    empty_mask_like,
    rasterize_polygon,
)
from fatanalyze.interactive.user_roi import UserROI, load_user_roi


__all__ = [
    "UserROI",
    "analyze_user_roi",
    "empty_mask_like",
    "load_user_roi",
    "rasterize_polygon",
]
