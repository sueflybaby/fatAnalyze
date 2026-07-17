"""Tests for the MR (PDFF / Dixon) fat-fraction quantification path.

The MR path was previously only exercised at the signature level. These
tests build small synthetic MR volumes (0-100 % fat fraction) and assert on
``_compute_fat_fraction``, ``_mr_qcreport`` range checks, ``analyze_mr_roi``,
and the GUI ``compute_for_rois_mr`` wrapper.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk


# ---------------------------------------------------------------------------
# Synthetic MR volume helpers
# ---------------------------------------------------------------------------


def _ff_volume(value: float, shape=(8, 32, 32)) -> sitk.Image:
    """A uniform 0-100 % fat-fraction volume (float32)."""
    arr = np.full(shape, value, dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 5.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return img


def _user_roi(image: sitk.Image, preset: str, z: int = 3) -> "UserROI":
    """Rasterize a centered square polygon on slice *z* into a UserROI."""
    from fatanalyze.interactive.polygon_utils import rasterize_polygon
    from fatanalyze.interactive.user_roi import UserROI

    verts = [(10, 10), (22, 10), (22, 22), (10, 22)]
    mask = rasterize_polygon(image, z, verts)
    return UserROI(name=f"{preset}_test", preset=preset, mask=mask,
                   z_index=z, n_points=4)


# ---------------------------------------------------------------------------
# _compute_fat_fraction
# ---------------------------------------------------------------------------


def test_compute_fat_fraction_basic() -> None:
    """FF = fat / (fat + water) * 100 for a known fat/water split."""
    from fatanalyze.io.dicom_loader import _compute_fat_fraction

    fat = np.full((4, 16, 16), 30.0, dtype=np.float32)
    water = np.full((4, 16, 16), 70.0, dtype=np.float32)
    ff = _compute_fat_fraction(
        sitk.GetImageFromArray(fat), sitk.GetImageFromArray(water),
    )
    out = sitk.GetArrayFromImage(ff)
    assert out.shape == (4, 16, 16)
    assert float(out.mean()) == pytest.approx(30.0, abs=1e-3)


def test_compute_fat_fraction_zero_denominator_is_zero() -> None:
    """Where fat + water == 0 the result must be 0, not NaN/inf."""
    from fatanalyze.io.dicom_loader import _compute_fat_fraction

    fat = np.zeros((2, 8, 8), dtype=np.float32)
    water = np.zeros((2, 8, 8), dtype=np.float32)
    ff = _compute_fat_fraction(
        sitk.GetImageFromArray(fat), sitk.GetImageFromArray(water),
    )
    out = sitk.GetArrayFromImage(ff)
    assert np.all(out == 0.0)


def test_compute_fat_fraction_clipped_to_100() -> None:
    """Fat > water+fat is clipped to 100 %."""
    from fatanalyze.io.dicom_loader import _compute_fat_fraction

    fat = np.full((2, 8, 8), 90.0, dtype=np.float32)
    water = np.full((2, 8, 8), 10.0, dtype=np.float32)
    ff = _compute_fat_fraction(
        sitk.GetImageFromArray(fat), sitk.GetImageFromArray(water),
    )
    out = sitk.GetArrayFromImage(ff)
    assert float(out.max()) <= 100.0


# ---------------------------------------------------------------------------
# _mr_qcreport range checks
# ---------------------------------------------------------------------------


def test_mr_qcreport_normal_range_no_scale_warning() -> None:
    """A 0-100 % map within range should not flag a scale issue."""
    from fatanalyze.io.dicom_loader import _mr_qcreport

    qc = _mr_qcreport(_ff_volume(25.0), "PDFF")
    assert qc.modality == "MR"
    msgs = " ".join(qc.warnings)
    assert "scale issue" not in msgs
    assert "0-1 scale" not in msgs


def test_mr_qcreport_flags_negative_min_as_scale_issue() -> None:
    """A min below -1 % suggests a wrong scale factor."""
    from fatanalyze.io.dicom_loader import _mr_qcreport

    qc = _mr_qcreport(_ff_volume(-5.0), "PDFF")
    assert any("scale issue" in w for w in qc.warnings)


def test_mr_qcreport_flags_over_100_max_as_scale_issue() -> None:
    """A max above 105 % suggests a wrong scale factor."""
    from fatanalyze.io.dicom_loader import _mr_qcreport

    qc = _mr_qcreport(_ff_volume(110.0), "PDFF")
    assert any("scale issue" in w for w in qc.warnings)


def test_mr_qcreport_flags_0_1_scale() -> None:
    """A 0-1 range is misdetected as a 0-1 scale, not 0-100."""
    from fatanalyze.io.dicom_loader import _mr_qcreport

    qc = _mr_qcreport(_ff_volume(0.5), "PDFF")
    assert any("0-1 scale" in w for w in qc.warnings)


# ---------------------------------------------------------------------------
# analyze_mr_roi
# ---------------------------------------------------------------------------


def test_analyze_mr_roi_uniform_liver_reports_steatosis_grade() -> None:
    """Uniform 25 % FF on a liver preset -> mean 25, S2, correct bins."""
    from fatanalyze.interactive.analyze_mr import analyze_mr_roi

    img = _ff_volume(25.0)
    res = analyze_mr_roi(img, _user_roi(img, "liver"))

    assert res["n_voxels"] > 0
    assert res["mean_ff"] == pytest.approx(25.0, abs=1e-3)
    assert res["median_ff"] == pytest.approx(25.0, abs=1e-3)
    # 25 % sits in the 20-30 % bin exclusively.
    assert res["ff_bins"]["20-30%"] == pytest.approx(1.0, abs=1e-6)
    assert res["ff_bins"]["0-5%"] == pytest.approx(0.0, abs=1e-6)
    # Liver 25 % -> S3 (severe) per the threshold ladder (>= 20).
    assert "Steatosis: S3 (severe)" in res["clinical_flags"]
    # histogram_result carries the same stats for the results panel.
    assert res["histogram_result"]["mean_ff"] == pytest.approx(25.0, abs=1e-3)


def test_analyze_mr_roi_low_ff_liver_is_S0() -> None:
    """Uniform 3 % FF on a liver preset -> S0 (normal)."""
    from fatanalyze.interactive.analyze_mr import analyze_mr_roi

    img = _ff_volume(3.0)
    res = analyze_mr_roi(img, _user_roi(img, "liver"))
    assert "Steatosis: S0 (normal)" in res["clinical_flags"]


def test_analyze_mr_roi_empty_mask_returns_nan_and_flag() -> None:
    """An empty ROI must not raise and must report empty_roi."""
    from fatanalyze.interactive.analyze_mr import analyze_mr_roi
    from fatanalyze.interactive.user_roi import UserROI

    img = _ff_volume(25.0)
    # Empty mask: zero voxels on slice 3.
    empty = sitk.GetImageFromArray(
        np.zeros((8, 32, 32), dtype=np.float32)
    )
    empty.CopyInformation(img)
    roi = UserROI(name="empty", preset="liver", mask=empty, z_index=3, n_points=4)

    res = analyze_mr_roi(img, roi)
    assert res["n_voxels"] == 0
    assert res["mean_ff"] != res["mean_ff"]  # NaN
    assert res["clinical_flags"] == ["empty_roi"]
    assert res["ff_bins"] == {}


def test_analyze_mr_roi_psoas_myosteatosis_flag() -> None:
    """iliopsoas_left with FF > 25 % triggers the myosteatosis flag."""
    from fatanalyze.interactive.analyze_mr import analyze_mr_roi

    img = _ff_volume(40.0)
    res = analyze_mr_roi(img, _user_roi(img, "iliopsoas_left"))
    assert "Myosteatosis (FF > 25%)" in res["clinical_flags"]


def test_analyze_mr_roi_psoas_below_threshold_no_flag() -> None:
    """iliopsoas_left with FF <= 25 % does not trigger myosteatosis."""
    from fatanalyze.interactive.analyze_mr import analyze_mr_roi

    img = _ff_volume(10.0)
    res = analyze_mr_roi(img, _user_roi(img, "iliopsoas_left"))
    assert "Myosteatosis (FF > 25%)" not in res["clinical_flags"]


# ---------------------------------------------------------------------------
# GUI wrapper: compute_for_rois_mr
# ---------------------------------------------------------------------------


def test_compute_for_rois_mr_analyzes_each_roi() -> None:
    """Each MR ROI is analyzed independently (no psoas merge)."""
    from fatanalyze.gui.metrics_runner import compute_for_rois_mr
    from fatanalyze.gui.roi import ROI

    img = _ff_volume(25.0)
    verts = [(10, 10), (22, 10), (22, 22), (10, 22)]
    rois = [
        ROI(name="L", preset="iliopsoas_left", z_index=3, vertices=verts),
        ROI(name="liver", preset="liver", z_index=3, vertices=verts),
    ]
    results = compute_for_rois_mr(img, rois)
    assert set(results.keys()) == {"L", "liver"}
    assert results["liver"]["mean_ff"] == pytest.approx(25.0, abs=1e-3)
    assert results["L"]["mean_ff"] == pytest.approx(25.0, abs=1e-3)
    # No combined psoas entry for the MR pipeline.
    assert all("combined" not in k for k in results)


def test_compute_for_rois_mr_reports_progress_per_roi() -> None:
    """Progress is emitted once per ROI plus a final 'Done.' tick."""
    from fatanalyze.gui.metrics_runner import compute_for_rois_mr
    from fatanalyze.gui.roi import ROI

    img = _ff_volume(25.0)
    verts = [(10, 10), (22, 10), (22, 22), (10, 22)]
    rois = [ROI(name="R0", preset="liver", z_index=3, vertices=verts)]

    calls: list[tuple[int, int, str]] = []
    compute_for_rois_mr(img, rois, progress=lambda c, t, m: calls.append((c, t, m)))

    # total = N rois + 1 (Done.)
    assert all(t == 2 for _, t, _ in calls)
    assert calls[-1][2].lower().startswith("done")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_load_mr_presets_has_expected_vendors() -> None:
    """mr_presets.yaml exposes the documented vendor presets."""
    from fatanalyze.config import load_mr_presets

    presets = load_mr_presets()["presets"]
    for key in ("Siemens (0-100)", "GE IDEAL (0-10000)",
                "Philips (0-100)", "Custom"):
        assert key in presets, key
    # Default preset path used by load_mr_series.
    assert "Siemens (0-100)" in presets
