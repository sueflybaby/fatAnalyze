"""Tests for ROI extractor and L3-approx slice detection."""
from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from fatanalyze.roi.extractor import (
    extract_hu, find_l3_slice, get_3d_mask, get_l3_psoas_mask,
    mask_area_cm2, mask_volume_ml,
)


def test_get_3d_mask_keeps_largest_connected_component() -> None:
    """A 1000-vox mask + a 50-vox noise component -> only the big one survives."""
    arr = np.zeros((8, 32, 32), dtype=np.uint8)
    arr[2:5, 10:30, 10:30] = 1
    arr[6, 28:30, 28:30] = 1  # small noise
    img = sitk.GetImageFromArray(arr)
    cleaned = get_3d_mask(img, keep_largest_cc=True)
    out = sitk.GetArrayFromImage(cleaned)
    assert out.sum() == 3 * 20 * 20


def test_get_3d_mask_erode_shrinks_boundary() -> None:
    arr = np.zeros((8, 32, 32), dtype=np.uint8)
    arr[2:6, 10:30, 10:30] = 1
    img = sitk.GetImageFromArray(arr)
    eroded = get_3d_mask(img, keep_largest_cc=False, erode_voxels=1)
    out = sitk.GetArrayFromImage(eroded)
    # Original is 4x20x20=1600 voxels; after 1-iter 3D erosion it shrinks but >0
    assert out.sum() < 1600
    assert out.sum() > 0


def test_find_l3_slice_picks_largest_psoas_area() -> None:
    """A psoas mask whose area peaks in the middle -> L3-approx z picks the peak."""
    size = (64, 64, 16)
    z, y, x = np.meshgrid(
        np.arange(size[2]), np.arange(size[1]), np.arange(size[0]),
        indexing="ij",
    )
    # Bell-shaped: peak in the middle, smaller at edges
    sigma = size[2] / 3.0
    center = size[2] / 2.0
    radius = 3.0 + 4.0 * np.exp(-0.5 * ((z - center) / sigma) ** 2)
    left = ((x - size[0] * 0.30) ** 2 + (y - size[1] * 0.5) ** 2) <= radius ** 2
    right = ((x - size[0] * 0.70) ** 2 + (y - size[1] * 0.5) ** 2) <= radius ** 2
    left_img = sitk.GetImageFromArray(left.astype(np.uint8))
    right_img = sitk.GetImageFromArray(right.astype(np.uint8))
    z_center, z_min, z_max = find_l3_slice(left_img, right_img, n_buffer=1)
    # Peak should be at center; with buffer of 1 we can verify the range.
    assert z_center == int(round(center))
    assert z_min < z_center < z_max


def test_find_l3_slice_rejects_empty() -> None:
    size = (32, 32, 4)
    z, y, x = np.meshgrid(
        np.arange(size[2]), np.arange(size[1]), np.arange(size[0]),
        indexing="ij",
    )
    empty = np.zeros_like(z, dtype=np.uint8)
    img = sitk.GetImageFromArray(empty)
    with pytest.raises(ValueError):
        find_l3_slice(img, img, n_buffer=1)


def test_get_l3_psoas_mask_returns_buffered_slice() -> None:
    size = (32, 32, 6)
    z, y, x = np.meshgrid(
        np.arange(size[2]), np.arange(size[1]), np.arange(size[0]),
        indexing="ij",
    )
    radius = 4 + (z / size[2]) * 4
    left = ((x - size[0] * 0.30) ** 2 + (y - size[1] * 0.5) ** 2) <= radius ** 2
    right = ((x - size[0] * 0.70) ** 2 + (y - size[1] * 0.5) ** 2) <= radius ** 2
    left_img = sitk.GetImageFromArray(left.astype(np.uint8))
    right_img = sitk.GetImageFromArray(right.astype(np.uint8))
    z_center, _, _ = find_l3_slice(left_img, right_img, n_buffer=1)
    psoas = get_l3_psoas_mask(left_img, right_img, z_center, n_buffer=1)
    out = sitk.GetArrayFromImage(psoas)
    nonzero_slices = np.where(out.sum(axis=(1, 2)) > 0)[0]
    # Buffer of 1 around z_center: at most [z_center-1, z_center, z_center+1]
    assert all(abs(s - z_center) <= 1 for s in nonzero_slices)


def test_extract_hu_returns_masked_values() -> None:
    arr = np.full((4, 8, 8), 40, dtype=np.int16)
    arr[1:3, 2:6, 2:6] = 100
    img = sitk.GetImageFromArray(arr)
    mask = np.zeros_like(arr, dtype=np.uint8)
    mask[1:3, 2:6, 2:6] = 1
    m_img = sitk.GetImageFromArray(mask)
    hu = extract_hu(img, m_img)
    assert hu.size == 2 * 4 * 4
    assert hu.mean() == 100.0


def test_extract_hu_empty_mask() -> None:
    arr = np.full((4, 8, 8), 40, dtype=np.int16)
    img = sitk.GetImageFromArray(arr)
    mask = np.zeros_like(arr, dtype=np.uint8)
    m_img = sitk.GetImageFromArray(mask)
    hu = extract_hu(img, m_img)
    assert hu.size == 0


def test_mask_volume_ml_basic() -> None:
    arr = np.zeros((4, 8, 8), dtype=np.uint8)
    arr[1:3, 2:6, 2:6] = 1
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((2.0, 2.0, 5.0))
    vol = mask_volume_ml(img)
    # 32 voxels * (2*2*5) mm3 / 1000 = 0.64 ml
    assert vol == pytest.approx(0.64, abs=1e-6)


def test_mask_area_cm2_at_slice() -> None:
    arr = np.zeros((4, 8, 8), dtype=np.uint8)
    arr[1, 2:6, 2:6] = 1
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((2.0, 2.0, 5.0))
    a = mask_area_cm2(img, 1)
    # 16 px * (2*2) mm2 / 100 = 0.64 cm2
    assert a == pytest.approx(0.64, abs=1e-6)
    assert mask_area_cm2(img, 0) == 0.0
