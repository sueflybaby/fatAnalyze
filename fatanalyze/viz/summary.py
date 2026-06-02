"""Single-page summary: histograms + overlays + clinical metric table."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.figure import Figure

from fatanalyze.analysis.histogram import HistogramResult
from fatanalyze.viz.histogram_plot import plot_histogram
from fatanalyze.viz.overlay import plot_overlay


def _metric_rows(
    results: Dict[str, HistogramResult],
    clinical: Dict[str, Any],
) -> List[List[str]]:
    """Build a list-of-list textual table for the summary."""
    rows: List[List[str]] = [["target", "n", "vol (ml)", "mean HU", "median HU"]]
    for name, r in results.items():
        vol = "-" if r.volume_ml is None else f"{r.volume_ml:.1f}"
        rows.append([
            name, str(r.n_voxels), vol,
            f"{r.mean_hu:.1f}" if r.n_voxels else "-",
            f"{r.median_hu:.1f}" if r.n_voxels else "-",
        ])
    rows.append([])
    rows.append(["clinical", "value", "suspicion?"])
    for k, v in clinical.items():
        if isinstance(v, dict):
            sus = v.get("fatty_liver_suspicion", v.get("fatty_pancreas_suspicion",
                    v.get("myosteatosis_flag", "?")))
            val = v.get("ratio", v.get("diff", v.get("imat_fraction", "?")))
            rows.append([k, f"{val}", str(sus)])
        else:
            rows.append([k, str(v), ""])
    return rows


def plot_summary(
    image: sitk.Image,
    hu_arrays: Dict[str, np.ndarray],
    masks: Dict[str, sitk.Image],
    results: Dict[str, HistogramResult],
    clinical: Optional[Dict[str, Any]] = None,
    l3_slice_index: Optional[int] = None,
) -> Figure:
    """Compose a 2-row summary figure: histograms on top, overlays below.

    The bottom row uses ``l3_slice_index`` for the psoas overlay and the
    argmax slice for organ overlays.
    """
    clinical = clinical or {}
    targets = list(hu_arrays.keys())
    n = len(targets)
    fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    # Top row: histograms
    for i, name in enumerate(targets):
        plot_histogram(hu_arrays[name], target_name=name, ax=axes[0, i])

    # Bottom row: overlays (psoas uses L3 index, others use argmax)
    for i, name in enumerate(targets):
        if name not in masks:
            axes[1, i].text(0.5, 0.5, "no mask", ha="center", va="center",
                            transform=axes[1, i].transAxes)
            axes[1, i].set_title(name)
            continue
        if name in ("iliopsoas_left", "iliopsoas_right") and l3_slice_index is not None:
            plot_overlay(image, masks[name], slice_index=l3_slice_index, ax=axes[1, i],
                         title=f"{name} (L3 z={l3_slice_index})")
        else:
            plot_overlay(image, masks[name], ax=axes[1, i], title=name)

    # Clinical table at the bottom
    table_ax = fig.add_axes((0.02, -0.04, 0.96, 0.12))
    table_ax.axis("off")
    data_rows = [["target", "n", "vol (ml)", "mean HU", "median HU"]]
    for name, r in results.items():
        vol = "-" if r.volume_ml is None else f"{r.volume_ml:.1f}"
        data_rows.append([
            name, str(r.n_voxels), vol,
            f"{r.mean_hu:.1f}" if r.n_voxels else "-",
            f"{r.median_hu:.1f}" if r.n_voxels else "-",
        ])
    table = table_ax.table(
        cellText=data_rows, loc="upper center", cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    if clinical:
        table_ax2 = fig.add_axes((0.02, -0.14, 0.96, 0.06))
        table_ax2.axis("off")
        clin_rows = [["metric", "value", "suspicion"]]
        for k, v in clinical.items():
            if isinstance(v, dict):
                sus = v.get("fatty_liver_suspicion",
                            v.get("fatty_pancreas_suspicion",
                                  v.get("myosteatosis_flag", "?")))
                val = v.get("ratio", v.get("diff",
                          v.get("imat_fraction", "?")))
                clin_rows.append([k, f"{val}", str(sus)])
            else:
                clin_rows.append([k, str(v), ""])
        table2 = table_ax2.table(
            cellText=clin_rows, loc="upper center", cellLoc="left",
        )
        table2.auto_set_font_size(False)
        table2.set_fontsize(9)
        table2.scale(1, 1.2)

    fig.suptitle("fatAnalyze summary", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.16, 1, 0.96))
    return fig


__all__ = ["plot_summary"]
