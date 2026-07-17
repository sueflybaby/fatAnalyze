"""HU histogram plot for a single ROI."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as mpatches
from matplotlib.axes import Axes

from fatanalyze.config import load_default_config


# Soft "color-blind safe" palette for the HU intervals
_RANGE_COLORS = {
    "fat_severe":   "#d62728",   # red
    "fat_moderate": "#ff7f0e",   # orange
    "fat_mild":     "#bcbd22",   # olive
    "normal":       "#2ca02c",   # green
    "fat":          "#d62728",
    "mixed":        "#ff7f0e",
    "imat":         "#d62728",
    "low_density":  "#ff7f0e",
    "normal_muscle": "#2ca02c",
}


def _bar_patches(patches_):
    """ax.hist returns either a BarContainer or a list of BarContainers.

    Normalize to a flat list of individual bars.
    """
    if isinstance(patches_, list):
        out = []
        for item in patches_:
            if hasattr(item, "patches"):
                out.extend(item.patches)
            else:
                out.append(item)
        return out
    return list(patches_.patches)


def plot_histogram(
    hu: np.ndarray,
    target_name: str,
    title: Optional[str] = None,
    config: Optional[Dict] = None,
    ax: Optional[Axes] = None,
    histogram_range: Tuple[int, int] = (-200, 400),
    bin_width: int = 5,
):
    """Plot a single-ROI HU histogram with clinical interval bands."""
    cfg = config if config is not None else load_default_config()
    target_cfg = cfg.get("targets", {}).get(target_name, {})
    hu_ranges = target_cfg.get("hu_ranges", {})

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    if hu.size == 0:
        ax.set_title(title or f"{target_name} (empty)")
        ax.text(0.5, 0.5, "no voxels", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return fig

    bin_edges = np.arange(
        float(histogram_range[0]),
        float(histogram_range[1]) + bin_width,
        float(bin_width),
    )
    counts, _, patches_obj = ax.hist(
        hu, bins=bin_edges, color="#7f7f7f", edgecolor="white", alpha=0.7,
        label="HU",
    )

    bar_list = _bar_patches(patches_obj)

    # Color the bins that fall inside any named interval
    legend_handles = []
    legend_labels = []
    for range_name, bounds in hu_ranges.items():
        lo, hi = bounds
        for p in bar_list:
            left = p.get_x()
            right = left + p.get_width()
            if right < lo or left > hi:
                continue
            color = _RANGE_COLORS.get(range_name, "#9467bd")
            p.set_facecolor(color)
        patch = mpatches.Patch(
            color=_RANGE_COLORS.get(range_name, "#9467bd"),
            label=f"{range_name} [{lo}, {hi}]",
        )
        legend_handles.append(patch)
        legend_labels.append(f"{range_name} [{lo}, {hi}]")

    mean_hu = float(hu.mean())
    median_hu = float(np.median(hu))
    ax.axvline(mean_hu, color="black", linestyle="-", linewidth=1.5)
    ax.axvline(median_hu, color="black", linestyle="--", linewidth=1.0)
    legend_handles.extend([
        plt.Line2D([0], [0], color="black", linestyle="-", linewidth=1.5),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.0),
    ])
    legend_labels.extend([f"mean {mean_hu:.1f}", f"median {median_hu:.1f}"])

    ratios = {}
    for range_name, bounds in hu_ranges.items():
        lo, hi = bounds
        mask = (hu >= lo) & (hu <= hi)
        ratios[range_name] = float(mask.sum()) / float(hu.size)
    text = f"voxels: {hu.size}\nmean: {mean_hu:.1f}\n"
    for k, v in ratios.items():
        text += f"{k}: {v*100:.1f}%\n"

    ax.text(0.98, 0.98, text, transform=ax.transAxes, va="top", ha="right",
            family="monospace", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    ax.set_title(title or f"{target_name} HU histogram")
    ax.set_xlabel("HU")
    ax.set_ylabel("Voxel count")
    ax.legend(legend_handles, legend_labels, loc="upper left",
              fontsize=8, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def draw_histogram_bars(
    ax: Axes,
    bin_centers: np.ndarray,
    counts: np.ndarray,
    *,
    xlabel: str,
    title: str,
    mean: Optional[float] = None,
    median: Optional[float] = None,
    color: str = "#888",
    threshold_lines: Optional[Dict[str, float]] = None,
    threshold_color: str = "red",
):
    """Draw histogram bars plus mean/median and optional threshold lines.

    Shared by the GUI results panel (CT and MR branches) so the two code
    paths do not duplicate the same matplotlib calls. ``bin_centers`` and
    ``counts`` are 1-D arrays of equal length; ``mean``/``median`` of ``None``
    or ``NaN`` are skipped.
    """
    if len(bin_centers) >= 2 and len(counts) == len(bin_centers):
        width = float(bin_centers[1] - bin_centers[0])
    else:
        width = 1.0
    ax.bar(bin_centers, counts, width=width,
           color=color, edgecolor="black", linewidth=0.3)

    if threshold_lines:
        for label, value in threshold_lines.items():
            ax.axvline(value, color=threshold_color, linestyle="--", linewidth=0.8)
    if mean is not None and not np.isnan(mean):
        ax.axvline(mean, color="blue", linestyle="-", linewidth=1.0,
                   label=f"mean={mean:.1f}")
    if median is not None and not np.isnan(median):
        ax.axvline(median, color="green", linestyle="-", linewidth=1.0,
                   label=f"median={median:.1f}")
    if mean is not None or median is not None:
        ax.legend(loc="upper right", fontsize=8)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Voxel count")


__all__ = ["plot_histogram", "draw_histogram_bars"]
