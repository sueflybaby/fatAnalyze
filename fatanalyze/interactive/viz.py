"""Two-panel visualization for a user ROI: CT + polygon overlay | HU histogram."""
from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib import colors as mcolors
from matplotlib.figure import Figure

from fatanalyze.interactive.user_roi import UserROI
from fatanalyze.viz.histogram_plot import plot_histogram
from fatanalyze.viz.overlay import _window


def plot_user_roi(
    image: sitk.Image,
    user_roi: UserROI,
    analysis: Optional[Dict[str, Any]] = None,
    window: tuple = (40, 400),
) -> Figure:
    """Two-panel figure: axial CT with polygon overlay | HU histogram.

    Parameters
    ----------
    image : sitk.Image
        Source 3D CT volume.
    user_roi : UserROI
        Drawn ROI.
    analysis : dict, optional
        Output of :func:`fatanalyze.interactive.analyze_user_roi`. If given,
        the histogram is colored with the preset's HU intervals.
    window : (center, width)
        Soft-tissue display window.
    """
    fig, (ax_ct, ax_hist) = plt.subplots(1, 2, figsize=(14, 6))
    arr = sitk.GetArrayFromImage(image)
    z = max(0, min(user_roi.z_index, arr.shape[0] - 1))
    ax_ct.imshow(_window(arr[z], window), cmap="gray", origin="lower")

    mask_arr = sitk.GetArrayFromImage(user_roi.mask) > 0
    if mask_arr[z].any():
        rgba = np.zeros((*mask_arr[z].shape, 4))
        rgb = mcolors.to_rgb(user_roi.color)
        rgba[mask_arr[z]] = (*rgb, 0.45)
        ax_ct.imshow(rgba, origin="lower")

    n_pts = user_roi.n_points
    ax_ct.set_title(
        f"{user_roi.name}  (preset={user_roi.preset}, z={z}, "
        f"{user_roi.n_voxels} px, {user_roi.area_cm2:.1f} cm^2, n={n_pts})"
    )
    ax_ct.set_xlabel("x"); ax_ct.set_ylabel("y")
    ax_ct.grid(False)

    if analysis is not None and analysis["n_voxels"] > 0:
        from fatanalyze.roi.extractor import extract_hu
        hu = extract_hu(image, user_roi.mask)
        plot_histogram(
            hu, target_name=user_roi.preset, ax=ax_hist,
            title=f"{user_roi.name} HU histogram",
        )
    else:
        ax_hist.text(0.5, 0.5, "no voxels", ha="center", va="center",
                     transform=ax_hist.transAxes, color="gray")
        ax_hist.set_title(f"{user_roi.name} HU histogram (empty)")

    fig.tight_layout()
    return fig


__all__ = ["plot_user_roi"]
