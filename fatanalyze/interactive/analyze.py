"""Plug a :class:`UserROI` into the standard histogram + clinical-metric pipeline."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import SimpleITK as sitk

from fatanalyze.analysis.fat_metrics import psoas_imat_fraction
from fatanalyze.analysis.histogram import HistogramResult, compute_ratios
from fatanalyze.config import load_default_config
from fatanalyze.interactive.user_roi import UserROI
from fatanalyze.roi.extractor import extract_hu, mask_area_cm2, mask_volume_ml


def analyze_user_roi(
    image: sitk.Image,
    user_roi: UserROI,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute HU statistics + clinical metrics for a user-drawn ROI.

    The mask is fed through the same :func:`extract_hu` / :func:`compute_ratios`
    / :func:`psoas_imat_fraction` pipeline used for TotalSegmentator ROIs. The
    preset name selects HU ranges and clinical thresholds; for iliopsoas-style
    presets, myosteatosis metrics are populated.

    Parameters
    ----------
    image : sitk.Image
        Source 3D CT volume.
    user_roi : UserROI
        Drawn ROI.
    config : dict, optional
        Full project config; defaults to ``load_default_config()``.

    Returns
    -------
    dict
        Keys:
            target, name, n_voxels, area_cm2, volume_ml,
            mean_hu, median_hu, std_hu, p05_hu, p95_hu,
            ratios, clinical_flags, psoas_metrics, histogram_result
    """
    cfg = config if config is not None else load_default_config()
    hu = extract_hu(image, user_roi.mask)
    spacing = image.GetSpacing()
    hr: HistogramResult = compute_ratios(
        hu, target_name=user_roi.preset, spacing_xyz=spacing, config=cfg,
    )

    psoas_metrics: Optional[Dict[str, Any]] = None
    if user_roi.preset in ("iliopsoas_left", "iliopsoas_right") and hr.n_voxels > 0:
        psoas_metrics = psoas_imat_fraction(hr)

    return {
        "target": user_roi.preset,
        "name": user_roi.name,
        "n_voxels": int(hr.n_voxels),
        "area_cm2": mask_area_cm2(user_roi.mask, user_roi.z_index),
        "volume_ml": mask_volume_ml(user_roi.mask) if hr.n_voxels > 0 else 0.0,
        "mean_hu": float(hr.mean_hu) if hr.n_voxels else float("nan"),
        "median_hu": float(hr.median_hu) if hr.n_voxels else float("nan"),
        "std_hu": float(hr.std_hu) if hr.n_voxels else float("nan"),
        "p05_hu": float(hr.p05_hu) if hr.n_voxels else float("nan"),
        "p95_hu": float(hr.p95_hu) if hr.n_voxels else float("nan"),
        "ratios": dict(hr.ratios),
        "clinical_flags": list(hr.clinical_flags),
        "psoas_metrics": psoas_metrics,
        "histogram_result": hr,
    }


__all__ = ["analyze_user_roi"]
