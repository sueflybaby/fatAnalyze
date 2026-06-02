"""Polygon-to-mask rasterization and empty-mask helpers for user ROIs."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import SimpleITK as sitk
from matplotlib.path import Path as MplPath


def empty_mask_like(image: sitk.Image) -> sitk.Image:
    """Return a 3D zero mask sharing geometry (size/spacing/origin/direction) with ``image``."""
    size = image.GetSize()
    arr = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)
    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(image)
    return img


def rasterize_polygon(
    image: sitk.Image,
    z_index: int,
    vertices_xy: Sequence[Sequence[float]],
) -> sitk.Image:
    """Rasterize a 2D polygon onto a single axial slice of ``image``.

    Parameters
    ----------
    image : sitk.Image
        3D CT volume. Used for size/spacing/origin/direction.
    z_index : int
        Axial slice index (0-based, SimpleITK z order).
    vertices_xy : sequence of (x, y)
        Polygon vertices in matplotlib display space (x = column, y = row).
        Must contain at least 3 distinct points.

    Returns
    -------
    sitk.Image
        3D binary mask: non-zero only on ``z_index`` inside the polygon.
    """
    size_x, size_y, size_z = image.GetSize()
    if not (0 <= z_index < size_z):
        raise ValueError(f"z_index {z_index} out of range [0, {size_z})")
    verts = np.asarray(vertices_xy, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 2 or len(verts) < 3:
        raise ValueError("vertices_xy must be shape (N, 2) with N >= 3")

    xs = np.arange(size_x)
    ys = np.arange(size_y)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    poly_path = MplPath(verts)
    inside = poly_path.contains_points(points).reshape(size_y, size_x)

    arr = np.zeros((size_z, size_y, size_x), dtype=np.uint8)
    arr[z_index] = inside.astype(np.uint8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    return out


__all__ = ["empty_mask_like", "rasterize_polygon"]
