"""HU histogram + range-ratio computation."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fatanalyze.config import load_default_config


@dataclass
class HistogramResult:
    """Histogram + descriptive statistics for an HU array."""

    target: str
    n_voxels: int
    volume_ml: Optional[float]
    mean_hu: float
    median_hu: float
    std_hu: float
    p05_hu: float
    p95_hu: float
    ratios: Dict[str, float]
    histogram: Dict[str, List[float]]   # bin_centers, counts
    clinical_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def _empty_result(target: str) -> HistogramResult:
    return HistogramResult(
        target=target, n_voxels=0, volume_ml=0.0,
        mean_hu=float("nan"), median_hu=float("nan"), std_hu=float("nan"),
        p05_hu=float("nan"), p95_hu=float("nan"),
        ratios={}, histogram={"bin_centers": [], "counts": []},
        clinical_flags=["empty_roi"],
    )


def _volume_ml(n_voxels: int, spacing_xyz: Tuple[float, float, float] | None) -> float | None:
    if spacing_xyz is None:
        return None
    return float(n_voxels) * float(np.prod(spacing_xyz)) / 1000.0


def compute_ratios(
    hu: np.ndarray,
    target_name: str,
    spacing_xyz: Tuple[float, float, float] | None = None,
    bin_width: int = 5,
    histogram_range: Tuple[int, int] = (-200, 400),
    config: Optional[Dict[str, Any]] = None,
) -> HistogramResult:
    """Compute HU distribution, range ratios, and clinical flags.

    Parameters
    ----------
    hu : np.ndarray
        1D array of HU values (already masked).
    target_name : str
        Target key in ``config["targets"]`` (e.g. ``"liver"``).
    spacing_xyz : tuple, optional
        Voxel spacing in mm. If provided, ``volume_ml`` is reported.
    bin_width : int
        Histogram bin width in HU.
    histogram_range : (low, high)
        Range of the histogram in HU.
    config : dict, optional
        Full project config; loaded on demand.

    Returns
    -------
    HistogramResult
    """
    if hu.size == 0:
        return _empty_result(target_name)

    cfg = config if config is not None else load_default_config()
    target_cfg = cfg.get("targets", {}).get(target_name, {})
    hu_ranges: Dict[str, Tuple[float, float]] = target_cfg.get("hu_ranges", {})

    n = int(hu.size)
    volume_ml = _volume_ml(n, spacing_xyz)

    ratios: Dict[str, float] = {}
    for name, bounds in hu_ranges.items():
        lo, hi = bounds
        mask = (hu >= lo) & (hu <= hi)
        ratios[name] = float(mask.sum()) / n

    # clinical flags
    flags: List[str] = []
    for thr_name, thr_value in target_cfg.get("clinical_thresholds", {}).items():
        if not isinstance(thr_value, (int, float)):
            continue
        if thr_name.startswith("mean_hu") and not np.isnan(hu.mean()):
            if hu.mean() < thr_value:
                flags.append(f"{thr_name}: {hu.mean():.1f} < {thr_value}")
        elif "_fraction_" in thr_name and ratios:
            key = thr_name.split("_fraction_")[0]
            if key in ratios and ratios[key] >= thr_value:
                flags.append(f"{key}_ratio {ratios[key]:.2f} >= {thr_value}")

    bin_edges = np.arange(
        histogram_range[0], histogram_range[1] + bin_width, bin_width,
    )
    counts, _ = np.histogram(hu, bins=bin_edges)
    bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2.0).tolist()

    return HistogramResult(
        target=target_name,
        n_voxels=n,
        volume_ml=volume_ml,
        mean_hu=float(hu.mean()),
        median_hu=float(np.median(hu)),
        std_hu=float(hu.std()),
        p05_hu=float(np.percentile(hu, 5)),
        p95_hu=float(np.percentile(hu, 95)),
        ratios=ratios,
        histogram={"bin_centers": bin_centers, "counts": counts.tolist()},
        clinical_flags=flags,
    )


__all__ = ["HistogramResult", "compute_ratios"]
