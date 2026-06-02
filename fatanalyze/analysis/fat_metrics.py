"""Clinical fat indicators derived from HU statistics."""
from __future__ import annotations

from typing import Any, Dict

from fatanalyze.analysis.histogram import HistogramResult


def liver_spleen_ratio(
    liver_result: HistogramResult,
    spleen_result: HistogramResult,
) -> Dict[str, Any]:
    """Liver-to-spleen HU ratio (hepatic steatosis marker).

    Returns
    -------
    dict with:
        ratio : float or None
        fatty_liver_suspicion : bool
        mean_liver_hu, mean_spleen_hu : float
    """
    if liver_result.n_voxels == 0 or spleen_result.n_voxels == 0:
        return {
            "ratio": None,
            "fatty_liver_suspicion": False,
            "mean_liver_hu": liver_result.mean_hu,
            "mean_spleen_hu": spleen_result.mean_hu,
            "note": "missing_organ",
        }
    mean_spleen = spleen_result.mean_hu
    if mean_spleen <= 0:
        ratio = float("inf")
        suspicion = True
    else:
        ratio = liver_result.mean_hu / mean_spleen
        suspicion = ratio < 1.0
    return {
        "ratio": ratio,
        "fatty_liver_suspicion": bool(suspicion),
        "mean_liver_hu": liver_result.mean_hu,
        "mean_spleen_hu": mean_spleen,
    }


def pancreas_spleen_difference(
    pancreas_result: HistogramResult,
    spleen_result: HistogramResult,
) -> Dict[str, Any]:
    """Pancreas - spleen HU difference (pancreatic steatosis marker)."""
    if pancreas_result.n_voxels == 0 or spleen_result.n_voxels == 0:
        return {
            "diff": None,
            "fatty_pancreas_suspicion": False,
            "mean_pancreas_hu": pancreas_result.mean_hu,
            "mean_spleen_hu": spleen_result.mean_hu,
            "note": "missing_organ",
        }
    diff = pancreas_result.mean_hu - spleen_result.mean_hu
    return {
        "diff": diff,
        "fatty_pancreas_suspicion": bool(diff < -5),
        "mean_pancreas_hu": pancreas_result.mean_hu,
        "mean_spleen_hu": spleen_result.mean_hu,
    }


def psoas_imat_fraction(psoas_result: HistogramResult) -> Dict[str, Any]:
    """Myosteatosis indicators from an L3 psoas histogram.

    Reports:
    - imat_fraction : voxels in [-190, -30] HU
    - low_density_fraction : voxels in [-29, 29] HU
    - normal_muscle_fraction : voxels in [30, 150] HU
    - myosteatosis_flag : True if IMAT + LDM > 0.5 OR IMAT > 0.2

    Thresholds: see config/targets.yaml (iliopsoas_*).
    """
    r = psoas_result.ratios
    imat = r.get("imat", 0.0)
    ldm = r.get("low_density", 0.0)
    normal = r.get("normal", 0.0)
    if psoas_result.n_voxels == 0:
        return {
            "imat_fraction": None,
            "low_density_fraction": None,
            "normal_muscle_fraction": None,
            "myosteatosis_flag": False,
            "note": "empty_roi",
        }
    myosteatosis = (imat + ldm) > 0.5 or imat > 0.2
    return {
        "imat_fraction": imat,
        "low_density_fraction": ldm,
        "normal_muscle_fraction": normal,
        "myosteatosis_flag": bool(myosteatosis),
    }


__all__ = [
    "liver_spleen_ratio",
    "pancreas_spleen_difference",
    "psoas_imat_fraction",
]
