"""Tests for the interactive user-ROI module (no GUI required)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from fatanalyze.interactive import (
    UserROI,
    analyze_user_roi,
    empty_mask_like,
    load_user_roi,
    rasterize_polygon,
)
from fatanalyze.interactive.polygon_utils import rasterize_polygon as _rasterize
from fatanalyze.interactive.user_roi import _sidecar_path
from fatanalyze.io.dicom_loader import make_synthetic_ct


# ---------------------------------------------------------------------------
# polygon rasterization
# ---------------------------------------------------------------------------


def test_rasterize_polygon_square_fills_expected_voxels() -> None:
    """A 20x20 square on a 64x64 slice should produce ~400 voxels on that slice.

    matplotlib.path.Path.contains_points is exclusive on at least one edge,
    so we accept a small edge-count tolerance (>= 19*19 and <= 21*21).
    """
    img = make_synthetic_ct(size_xyz=(64, 64, 8), spacing_xyz=(1.0, 1.0, 5.0))
    z = 4
    verts = [(20, 20), (40, 20), (40, 40), (20, 40)]
    mask = _rasterize(img, z_index=z, vertices_xy=verts)
    arr = sitk.GetArrayFromImage(mask)
    assert arr.shape == (8, 64, 64)
    n_filled = int(arr[z].sum())
    assert 19 * 19 <= n_filled <= 21 * 21, f"unexpected fill count: {n_filled}"
    # All other slices should be zero.
    for z2 in range(arr.shape[0]):
        if z2 == z:
            continue
        assert arr[z2].sum() == 0


def test_rasterize_polygon_rejects_too_few_vertices() -> None:
    img = make_synthetic_ct()
    with pytest.raises(ValueError):
        _rasterize(img, z_index=0, vertices_xy=[(0, 0), (1, 1)])


def test_rasterize_polygon_rejects_bad_z() -> None:
    img = make_synthetic_ct(size_xyz=(8, 8, 4))
    with pytest.raises(ValueError):
        _rasterize(
            img, z_index=99,
            vertices_xy=[(0, 0), (1, 0), (1, 1), (0, 1)],
        )


def test_empty_mask_like_shares_geometry() -> None:
    img = make_synthetic_ct(size_xyz=(10, 20, 5), spacing_xyz=(0.7, 0.8, 1.25))
    m = empty_mask_like(img)
    assert m.GetSize() == img.GetSize()
    assert m.GetSpacing() == img.GetSpacing()
    assert m.GetOrigin() == img.GetOrigin()
    assert m.GetDirection() == img.GetDirection()
    assert sitk.GetArrayFromImage(m).sum() == 0


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


def test_analyze_user_roi_liver_preset_returns_expected_mean() -> None:
    """A uniform 60-HU ROI should be classified into the 'normal' liver range."""
    # Build a 32x32x8 image filled with 60 HU, except a small (-1000) air
    # region. Polygon fully inside the 60-HU region.
    arr = np.full((8, 32, 32), 60, dtype=np.int16)
    arr[:, 0:4, 0:4] = -1000  # air corner; outside the polygon
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 5.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    verts = [(10, 10), (22, 10), (22, 22), (10, 22)]
    mask = _rasterize(img, z_index=3, vertices_xy=verts)
    roi = UserROI(name="liver_test", preset="liver", mask=mask, z_index=3, n_points=4)

    res = analyze_user_roi(img, roi)
    # Every voxel inside the polygon is 60 HU regardless of MplPath edges.
    assert res["n_voxels"] >= 12 * 12, res["n_voxels"]
    assert res["mean_hu"] == pytest.approx(60.0, abs=1e-6)
    assert res["target"] == "liver"
    # 60 HU is in the 'normal' liver range [40, 80]; other ranges must be 0.
    assert res["ratios"]["normal"] == pytest.approx(1.0, abs=1e-6)
    assert res["ratios"]["fat_mild"] == pytest.approx(0.0, abs=1e-6)
    # Liver preset does not trigger psoas metrics.
    assert res["psoas_metrics"] is None


def test_analyze_user_roi_psoas_preset_populates_myosteatosis() -> None:
    """A psoas preset ROI with HU split across imat/normal should report both."""
    arr = np.full((8, 32, 32), 60, dtype=np.int16)
    # Left half (-150 HU imat) and right half (60 HU normal) on z=2,
    # side-by-side as 8x8 cubes sharing an edge.
    arr[2, 4:12, 4:12] = -150
    arr[2, 4:12, 12:20] = 60
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 5.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    # Polygon footprint exactly matches the union of the two cubes
    # (vertices at the cube boundaries). Imat and normal will be 50/50.
    verts = [(4, 4), (20, 4), (20, 12), (4, 12)]
    mask = _rasterize(img, z_index=2, vertices_xy=verts)
    roi = UserROI(
        name="psoas_test", preset="iliopsoas_left", mask=mask,
        z_index=2, n_points=4,
    )

    res = analyze_user_roi(img, roi)
    pm = res["psoas_metrics"]
    assert pm is not None
    # n_voxels is in [105, 128] depending on MplPath edge handling.
    # In all cases, exactly 64 are imat and ~64 are normal, so ratios
    # stay close to 0.5 each. Tolerate ±10% edge slop.
    assert 0.4 <= pm["imat_fraction"] <= 0.6, pm
    assert 0.4 <= pm["normal_muscle_fraction"] <= 0.6, pm
    assert pm["low_density_fraction"] == pytest.approx(0.0, abs=1e-6)
    # imat (>=0.4) + ldm (0.0) > 0.5 must hold; flag is True.
    assert pm["myosteatosis_flag"] is True
    assert res["n_voxels"] >= 2 * 8 * 8 - 32   # accept at most 32 voxel edge slop


def test_analyze_user_roi_empty_mask_is_safe() -> None:
    img = make_synthetic_ct()
    m = empty_mask_like(img)
    roi = UserROI(
        name="empty", preset="liver", mask=m, z_index=0, n_points=0,
        empty_warning=True,
    )
    res = analyze_user_roi(img, roi)
    assert res["n_voxels"] == 0
    assert res["volume_ml"] == 0.0
    assert res["psoas_metrics"] is None
    # clinical_flags is a list, may contain 'empty_roi' from compute_ratios
    assert "empty_roi" in res["clinical_flags"]


def test_user_roi_preset_validation() -> None:
    img = make_synthetic_ct()
    m = empty_mask_like(img)
    with pytest.raises(ValueError):
        UserROI(name="bad", preset="not_a_real_preset", mask=m, z_index=0)


# ---------------------------------------------------------------------------
# UserROI geometry helpers
# ---------------------------------------------------------------------------


def test_user_roi_area_and_volume() -> None:
    img = make_synthetic_ct(size_xyz=(20, 20, 4), spacing_xyz=(2.0, 2.0, 5.0))
    m = empty_mask_like(img)
    arr = sitk.GetArrayFromImage(m)
    arr[1, 5:15, 5:15] = 1   # 10x10 = 100 px
    m = sitk.GetImageFromArray(arr)
    m.CopyInformation(img)
    roi = UserROI(name="box", preset="liver", mask=m, z_index=1, n_points=4)
    # 100 px * (2*2) mm2 / 100 = 4.0 cm2
    assert roi.area_cm2 == pytest.approx(4.0, abs=1e-6)
    # 100 px * (2*2*5) mm3 / 1000 = 2.0 ml
    assert roi.volume_ml == pytest.approx(2.0, abs=1e-6)
    assert roi.n_voxels == 100


# ---------------------------------------------------------------------------
# save / load roundtrip
# ---------------------------------------------------------------------------


def test_user_roi_save_and_load_roundtrip(tmp_path: Path) -> None:
    # Uniform 70-HU background, no synthetic_ct cube interfering.
    arr = np.full((4, 20, 20), 70, dtype=np.int16)
    arr[:, 0:3, 0:3] = -1000  # air corner
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 5.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    verts = [(5, 5), (12, 5), (12, 12), (5, 12)]
    mask = _rasterize(img, z_index=2, vertices_xy=verts)
    roi = UserROI(
        name="rt_test", preset="liver", mask=mask, z_index=2, n_points=4,
        notes="roundtrip",
    )

    nifti_path = tmp_path / "rt_test.nii.gz"
    roi.save_nifti(nifti_path)
    assert nifti_path.exists()
    sidecar = _sidecar_path(nifti_path)
    assert sidecar.exists()
    assert sidecar.name == "rt_test.nii.gz.json"
    meta = json.loads(sidecar.read_text())
    assert meta["name"] == "rt_test"
    assert meta["preset"] == "liver"
    assert meta["n_voxels"] >= 7 * 7

    loaded = load_user_roi(nifti_path, image=img)
    assert loaded.name == "rt_test"
    assert loaded.preset == "liver"
    assert loaded.n_voxels == roi.n_voxels
    assert loaded.notes == "roundtrip"
    res = analyze_user_roi(img, loaded)
    # All mask voxels are 70 HU (background is uniform 70).
    assert res["mean_hu"] == pytest.approx(70.0, abs=1e-6)
