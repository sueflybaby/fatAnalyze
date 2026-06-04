"""MR fat-fraction analysis for PDFF / Dixon ROIs.

The input image is expected to be a fat-fraction map where every pixel
represents 0-100 % fat fraction.  The pipeline is:

#. Extract pixel values inside the ROI mask.
#. Compute FF% summary statistics (mean, median, std, percentiles).
#. Bin the FF% values into clinically relevant ranges.
#. Apply clinical grading (hepatic steatosis S0-S3, etc.).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import SimpleITK as sitk

from fatanalyze.roi.extractor import extract_hu, mask_area_cm2, mask_volume_ml
from fatanalyze.interactive.user_roi import UserROI


def _compute_ff_bins(ff: np.ndarray) -> Dict[str, float]:
    """Bin fat-fraction values into clinical ranges.

    Returns fractions (0-1) of voxels falling in each bin.
    """
    if ff.size == 0:
        return {}
    n = float(ff.size)
    bins = [
        ("0-5%",   0.0,   5.0),
        ("5-10%",  5.0,  10.0),
        ("10-20%", 10.0, 20.0),
        ("20-30%", 20.0, 30.0),
        ("30-50%", 30.0, 50.0),
        ("50-100%", 50.0, 100.0),
    ]
    return {label: float(((ff >= lo) & (ff < hi)).sum() / n)
            for label, lo, hi in bins}


def _clinical_flags(ff: np.ndarray, target: str) -> list[str]:
    """Generate clinical flags based on fat-fraction thresholds."""
    flags: list[str] = []
    if ff.size == 0:
        return flags

    mean_ff = float(ff.mean())

    if target.startswith("liver"):
        if mean_ff < 5.0:
            flags.append("Steatosis: S0 (normal)")
        elif mean_ff < 10.0:
            flags.append("Steatosis: S1 (mild)")
        elif mean_ff < 20.0:
            flags.append("Steatosis: S2 (moderate)")
        else:
            flags.append("Steatosis: S3 (severe)")

    if target in ("iliopsoas_left", "iliopsoas_right", "iliopsoas_combined"):
        if mean_ff > 25.0:
            flags.append("Myosteatosis (FF > 25%)")

    if target == "pancreas":
        if mean_ff > 10.0:
            flags.append("Pancreatic steatosis (FF > 10%)")

    if target == "spleen":
        if mean_ff > 10.0:
            flags.append("Splenic steatosis (FF > 10%)")

    return flags


def analyze_mr_roi(
    image: sitk.Image,
    user_roi: UserROI,
) -> Dict[str, Any]:
    """Compute fat-fraction metrics for a user-drawn MR ROI.

    Parameters
    ----------
    image : sitk.Image
        3D volume where pixel values are 0-100 % fat fraction.
    user_roi : UserROI
        User-drawn ROI.

    Returns
    -------
    dict
        Keys: ``target, name, n_voxels, area_cm2, volume_ml, mean_ff,
        median_ff, std_ff, p05_ff, p95_ff, ff_bins, clinical_flags,
        histogram_result``
    """
    ff = extract_hu(image, user_roi.mask)
    spacing = image.GetSpacing()

    n = int(ff.size)
    result: Dict[str, Any] = {
        "target": user_roi.preset,
        "name": user_roi.name,
        "n_voxels": n,
        "area_cm2": mask_area_cm2(user_roi.mask, user_roi.z_index) if n > 0 else 0.0,
        "volume_ml": mask_volume_ml(user_roi.mask) if n > 0 else 0.0,
    }

    if n > 0:
        result["mean_ff"] = float(ff.mean())
        result["median_ff"] = float(np.median(ff))
        result["std_ff"] = float(ff.std())
        result["p05_ff"] = float(np.percentile(ff, 5))
        result["p95_ff"] = float(np.percentile(ff, 95))
        result["ff_bins"] = _compute_ff_bins(ff)
        result["clinical_flags"] = _clinical_flags(ff, user_roi.preset)

        # Build a simple histogram result for the results panel
        result["histogram_result"] = {
            "n_voxels": n,
            "volume_ml": mask_volume_ml(user_roi.mask),
            "mean_ff": float(ff.mean()),
            "median_ff": float(np.median(ff)),
            "std_ff": float(ff.std()),
            "p05_ff": float(np.percentile(ff, 5)),
            "p95_ff": float(np.percentile(ff, 95)),
            "ff_bins": _compute_ff_bins(ff),
            "histogram": _histogram_dict(ff),
        }
    else:
        result["mean_ff"] = float("nan")
        result["median_ff"] = float("nan")
        result["std_ff"] = float("nan")
        result["p05_ff"] = float("nan")
        result["p95_ff"] = float("nan")
        result["ff_bins"] = {}
        result["clinical_flags"] = ["empty_roi"]
        result["histogram_result"] = None

    return result


def _histogram_dict(ff: np.ndarray) -> Dict[str, list[float]]:
    """Build histogram data for the results panel."""
    import math
    if ff.size == 0:
        return {"bin_centers": [], "counts": []}
    bin_width = 2.0
    bins = int(math.ceil(100.0 / bin_width))
    counts, edges = np.histogram(ff, bins=bins, range=(0.0, 100.0))
    bin_centers = ((edges[:-1] + edges[1:]) / 2.0).tolist()
    return {"bin_centers": bin_centers, "counts": counts.tolist()}


__all__ = ["analyze_mr_roi"]
