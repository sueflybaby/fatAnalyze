"""ROI dataclass used by the GUI to track user-drawn regions.

An :class:`ROI` is a 2D polygon (list of (x, y) pixel coordinates) drawn on a
single axial slice, tagged with a preset (e.g. ``iliopsoas_left``) and a name.
Once rasterized, the binary mask is cached in :attr:`mask` for analysis and
re-use. The ROI is also serializable to a NIfTI sidecar via
:mod:`fatanalyze.interactive.user_roi`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import SimpleITK as sitk


Vertex = Tuple[float, float]


@dataclass
class ROI:
    """A user-drawn region of interest.

    Attributes
    ----------
    name : str
        User-facing label (e.g. ``"L-Psoas"``).
    preset : str
        Key into :mod:`fatanalyze.config.targets`. Drives HU ranges and
        which clinical metrics are computed (``iliopsoas_left`` /
        ``iliopsoas_right`` trigger :func:`psoas_imat_fraction`).
    z_index : int
        Axial slice index on which the polygon was drawn.
    vertices : list[tuple[float, float]]
        Polygon vertices in pixel coordinates (x, y).
    status : str
        One of ``"drawn"`` (newly added), ``"analyzed"`` (results computed),
        ``"exported"`` (written to CSV / NIfTI).
    mask : sitk.Image, optional
        3D binary mask of the same size as the source volume; populated
        on rasterize.
    result : dict, optional
        Cached :func:`fatanalyze.interactive.analyze.analyze_user_roi`
        output, populated by :class:`MetricsRunner`.
    """

    name: str
    preset: str
    z_index: int
    vertices: List[Vertex] = field(default_factory=list)
    status: str = "drawn"
    mask: Optional[sitk.Image] = None
    result: Optional[Dict[str, Any]] = None

    @property
    def is_closed(self) -> bool:
        """A polygon with at least 3 vertices is considered closed (and analyzable)."""
        return len(self.vertices) >= 3

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary (no mask / result blobs)."""
        return {
            "name": self.name,
            "preset": self.preset,
            "z_index": int(self.z_index),
            "n_vertices": self.n_vertices,
            "status": self.status,
        }


__all__ = ["ROI", "Vertex"]
