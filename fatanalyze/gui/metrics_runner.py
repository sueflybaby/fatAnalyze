"""Wrap :func:`analyze_user_roi` for the multi-ROI GUI workflow.

For psoas presets, L+R ROIs drawn on the same slice are merged via
``sitk.Or`` before :func:`psoas_imat_fraction` is called — that function
expects a single combined muscle mask. Other presets (liver, pancreas,
spleen, custom) are analyzed per-ROI.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import SimpleITK as sitk

from fatanalyze.config import load_default_config
from fatanalyze.gui.roi import ROI
from fatanalyze.interactive.analyze import analyze_user_roi
from fatanalyze.interactive.user_roi import UserROI
from fatanalyze.interactive.polygon_utils import empty_mask_like, rasterize_polygon
from fatanalyze.io.dicom_loader import OperationCancelled, ProgressCallback


PSOAS_PRESETS = ("iliopsoas_left", "iliopsoas_right")


def rasterize(roi: ROI, ref_image: sitk.Image) -> sitk.Image:
    """Populate ``roi.mask`` (and ``roi.user_roi``) from vertices; return mask.

    Rasterization is a thin wrapper around
    :func:`fatanalyze.interactive.polygon_utils.rasterize_polygon`; the
    vertices are in pixel coordinates of slice ``roi.z_index``.
    """
    if len(roi.vertices) < 3:
        roi.mask = empty_mask_like(ref_image)
    else:
        roi.mask = rasterize_polygon(ref_image, roi.z_index, roi.vertices)
    roi.user_roi = UserROI(
        name=roi.name,
        preset=roi.preset,
        mask=roi.mask,
        z_index=roi.z_index,
        n_points=len(roi.vertices),
    )
    return roi.mask


def compute_for_rois(
    image: sitk.Image,
    rois: List[ROI],
    config: Optional[dict] = None,
    progress: Optional[ProgressCallback] = None,
) -> Dict[str, dict]:
    """Compute metrics for a list of ROIs.

    Parameters
    ----------
    image : sitk.Image
        3D volume the ROIs are drawn on.
    rois : list[ROI]
        ROIs to analyze.
    config : dict, optional
        Pipeline config; defaults to :func:`load_default_config`.
    progress : callable, optional
        ``progress(current, total, message)`` invoked per-ROI and once
        for the psoas-combine step. The callback may raise
        :class:`OperationCancelled` to abort.

    Returns
    -------
    dict[str, dict]
        Keyed by ROI name. For psoas presets, if both L and R are present
        at the same ``z_index``, the L+R combined result is stored under
        a synthetic key ``"<name>_combined"`` and the per-side ROIs are
        still analyzed (and tagged with ``psoas_metrics=None`` since IMAT
        requires the combined mask).
    """
    def report(cur: int, total: int, msg: str) -> None:
        if progress is not None:
            progress(cur, total, msg)

    cfg = config if config is not None else load_default_config()
    results: Dict[str, dict] = {}

    # Group psoas ROIs by z_index so we can merge L+R
    psoas_by_z: Dict[int, List[ROI]] = {}
    for roi in rois:
        if roi.preset in PSOAS_PRESETS:
            psoas_by_z.setdefault(roi.z_index, []).append(roi)

    has_psoas_combine = any(
        len([r for r in sides if r.preset in PSOAS_PRESETS]) >= 2
        for sides in psoas_by_z.values()
    )
    # Total = N (per-ROI) + 1 (psoas combine, if any) + 1 (final "Done.").
    total = len(rois) + (1 if has_psoas_combine else 0) + 1
    step = 0

    report(step, total, f"Starting analysis of {len(rois)} ROI(s)…")

    for roi in rois:
        step += 1
        report(step, total, f"Processing ROI {step}/{len(rois)}: {roi.name}")
        # Ensure mask is up to date
        if roi.user_roi is None:
            rasterize(roi, image)
        # For psoas, we'll fill in psoas_metrics on the combined entry
        uroi = roi.user_roi
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
        step += 1
        report(step, total, f"Combining psoas @ z={z}…")
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

    step += 1
    report(step, total, "Done.")
    return results


def compute_for_rois_mr(
    image: sitk.Image,
    rois: List[ROI],
    progress: Optional[ProgressCallback] = None,
) -> Dict[str, dict]:
    """Compute MR fat-fraction metrics for a list of ROIs.

    Unlike the CT pipeline there is no combined psoas (the MR clinical
    logic is simpler), so each ROI is analyzed independently.
    """
    from fatanalyze.interactive.analyze_mr import analyze_mr_roi

    def report(cur: int, total: int, msg: str) -> None:
        if progress is not None:
            progress(cur, total, msg)

    results: Dict[str, dict] = {}
    total = len(rois) + 1
    step = 0
    report(step, total, f"Starting analysis of {len(rois)} ROI(s)…")

    for roi in rois:
        step += 1
        report(step, total, f"Processing ROI {step}/{len(rois)}: {roi.name}")
        if roi.user_roi is None:
            rasterize(roi, image)
        uroi = roi.user_roi
        result = analyze_mr_roi(image, uroi)
        results[roi.name] = result
        roi.result = result
        roi.status = "analyzed"
    report(total, total, "Done.")
    return results


__all__ = ["compute_for_rois", "compute_for_rois_mr", "rasterize", "PSOAS_PRESETS",
           "OperationCancelled", "ProgressCallback"]
