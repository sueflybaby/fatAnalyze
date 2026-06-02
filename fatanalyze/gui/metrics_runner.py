"""Wrap :func:`analyze_user_roi` for the multi-ROI GUI workflow.

For psoas presets, L+R ROIs drawn on the same slice are merged via
``sitk.Or`` before :func:`psoas_imat_fraction` is called — that function
expects a single combined muscle mask. Other presets (liver, pancreas,
spleen, custom) are analyzed per-ROI.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import SimpleITK as sitk

from fatanalyze.config import load_default_config
from fatanalyze.gui.roi import ROI
from fatanalyze.interactive.analyze import analyze_user_roi
from fatanalyze.interactive.user_roi import UserROI
from fatanalyze.interactive.polygon_utils import empty_mask_like, rasterize_polygon


PSOAS_PRESETS = ("iliopsoas_left", "iliopsoas_right")


def rasterize(roi: ROI, ref_image: sitk.Image) -> sitk.Image:
    """Populate ``roi.mask`` from its polygon vertices and return it.

    Rasterization is a thin wrapper around
    :func:`fatanalyze.interactive.polygon_utils.rasterize_polygon`; the
    vertices are in pixel coordinates of slice ``roi.z_index``.
    """
    if len(roi.vertices) < 3:
        roi.mask = empty_mask_like(ref_image)
        return roi.mask
    mask = rasterize_polygon(ref_image, roi.z_index, roi.vertices)
    roi.mask = mask
    return mask


def _user_roi_from(roi: ROI) -> UserROI:
    """Convert the GUI :class:`ROI` into the existing :class:`UserROI` dataclass."""
    if roi.mask is None:
        raise ValueError(f"ROI '{roi.name}' has no mask; call rasterize() first")
    return UserROI(
        name=roi.name,
        preset=roi.preset,
        mask=roi.mask,
        z_index=roi.z_index,
        n_points=len(roi.vertices),
    )


def compute_for_rois(
    image: sitk.Image,
    rois: List[ROI],
    config: Optional[dict] = None,
) -> Dict[str, dict]:
    """Compute metrics for a list of ROIs.

    Returns a dict keyed by ROI name. For psoas presets, if both L and R
    are present at the same ``z_index``, the L+R combined result is stored
    under a synthetic key ``"<name>_combined"`` and the per-side ROIs are
    still analyzed (and tagged with ``psoas_metrics=None`` since IMAT
    requires the combined mask).
    """
    cfg = config if config is not None else load_default_config()
    results: Dict[str, dict] = {}

    # Group psoas ROIs by z_index so we can merge L+R
    psoas_by_z: Dict[int, List[ROI]] = {}
    for roi in rois:
        if roi.preset in PSOAS_PRESETS:
            psoas_by_z.setdefault(roi.z_index, []).append(roi)

    for roi in rois:
        # Ensure mask is up to date
        if roi.mask is None:
            rasterize(roi, image)
        # For psoas, we'll fill in psoas_metrics on the combined entry
        uroi = _user_roi_from(roi)
        result = analyze_user_roi(image, uroi, cfg)
        # Psoas sides: clear psoas_metrics (computed on the combined entry)
        if roi.preset in PSOAS_PRESETS:
            same_z = [r for r in psoas_by_z.get(roi.z_index, [])
                      if r.preset in PSOAS_PRESETS]
            if len(same_z) >= 2:
                result["psoas_metrics"] = None
        results[roi.name] = result
        roi.result = result
        roi.status = "analyzed"

    # Psoas combined (L+R) at each z that has both sides
    for z, psoas_rois in psoas_by_z.items():
        sides = [r for r in psoas_rois if r.preset in PSOAS_PRESETS]
        if len(sides) < 2:
            continue
        # Merge masks
        first_mask = sides[0].mask
        if first_mask is None:
            raise RuntimeError(f"ROI '{sides[0].name}' missing mask; call rasterize() first")
        merged_mask = first_mask
        for s in sides[1:]:
            if s.mask is None:
                raise RuntimeError(f"ROI '{s.name}' missing mask; call rasterize() first")
            merged_mask = sitk.Or(merged_mask, s.mask)
        # Use the first side's preset for the metric call (preset name only
        # drives which clinical flags are populated; psoas_* both work)
        combined_uroi = UserROI(
            name=f"{sides[0].name}+{sides[-1].name}",
            preset="iliopsoas_left",  # picks the psoas metrics branch
            mask=merged_mask,
            z_index=z,
            n_points=0,
        )
        combined_result = analyze_user_roi(image, combined_uroi, cfg)
        # Compute combined area / volume from the merged mask
        from fatanalyze.roi.extractor import mask_area_cm2, mask_volume_ml
        if merged_mask is not None:
            combined_result["area_cm2"] = mask_area_cm2(merged_mask, z)
            combined_result["volume_ml"] = mask_volume_ml(merged_mask)
        combined_result["target"] = "iliopsoas_combined"
        combined_result["name"] = f"Combined Psoas @ z={z}"
        results[combined_uroi.name] = combined_result

    return results


__all__ = ["compute_for_rois", "rasterize", "PSOAS_PRESETS"]
