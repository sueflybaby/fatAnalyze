"""Axial slice overlay: CT + binary mask."""
from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# Default colormap + alpha for masks
_DEFAULT_MASK_COLOR = "#ff7f0e"
_DEFAULT_WINDOW = (40, 400)   # (center, width) soft-tissue window


def _window(arr: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
    lo = window[0] - window[1] / 2.0
    hi = window[0] + window[1] / 2.0
    clipped = np.clip(arr, lo, hi)
    return (clipped - lo) / (hi - lo)


def plot_overlay(
    image: sitk.Image,
    mask: sitk.Image,
    slice_index: Optional[int] = None,
    title: Optional[str] = None,
    color: str = _DEFAULT_MASK_COLOR,
    window: Tuple[float, float] = _DEFAULT_WINDOW,
    ax: Optional[Axes] = None,
):
    """Plot one axial slice with a mask overlay.

    Parameters
    ----------
    image : sitk.Image
        3D CT volume.
    mask : sitk.Image
        3D binary mask, same physical space as ``image``.
    slice_index : int, optional
        Z index to display. Defaults to the slice with the largest mask area.
    title : str, optional
    color : str
        Matplotlib color for the mask.
    window : (center, width)
        Soft-tissue display window.
    """
    img_arr = sitk.GetArrayFromImage(image)         # (z, y, x)
    mask_arr = sitk.GetArrayFromImage(mask) > 0

    if slice_index is None:
        areas = mask_arr.sum(axis=(1, 2))
        if areas.max() == 0:
            slice_index = img_arr.shape[0] // 2
        else:
            slice_index = int(np.argmax(areas))

    slice_index = max(0, min(slice_index, img_arr.shape[0] - 1))

    img_slice = img_arr[slice_index]
    mask_slice = mask_arr[slice_index]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    ax.imshow(_window(img_slice, window), cmap="gray", origin="lower")
    if mask_slice.any():
        # RGBA overlay: color where mask is True, transparent elsewhere
        rgba = np.zeros((*mask_slice.shape, 4))
        rgb = mcolors.to_rgb(color)
        rgba[mask_slice] = (*rgb, 0.4)
        ax.imshow(rgba, origin="lower")

    ax.set_title(
        title or f"slice z={slice_index} "
        f"(mask area = {int(mask_slice.sum())} px)",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)
    fig.tight_layout()
    return fig


__all__ = ["plot_overlay"]
