"""Shared pytest fixtures and helpers."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when running ``pytest`` from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import SimpleITK as sitk

from fatanalyze.io.dicom_loader import make_synthetic_ct


@pytest.fixture
def small_ct() -> sitk.Image:
    """Tiny synthetic CT: 64x64x16, 40 HU in a centered cube, 0 elsewhere."""
    return make_synthetic_ct(size_xyz=(64, 64, 16), spacing_xyz=(1.0, 1.0, 1.0))


@pytest.fixture
def cube_mask(small_ct) -> sitk.Image:
    """Binary mask with the same centered cube as ``small_ct``."""
    size = small_ct.GetSize()
    arr = np.zeros((size[2], size[1], size[0]), dtype=np.uint8)
    arr[:, size[1] // 4 : 3 * size[1] // 4,
           size[0] // 4 : 3 * size[0] // 4] = 1
    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(small_ct)
    return img


@pytest.fixture
def psoas_masks(small_ct) -> tuple:
    """Two psoas masks (left/right) that grow in area toward slice 8."""
    size = small_ct.GetSize()
    z, y, x = np.meshgrid(
        np.arange(size[2]), np.arange(size[1]), np.arange(size[0]),
        indexing="ij",
    )
    # Two oval blobs that increase radius with z
    radius = 4 + (z / max(1, size[2])) * 4
    left = ((x - size[0] * 0.30) ** 2 + (y - size[1] * 0.5) ** 2) <= radius ** 2
    right = ((x - size[0] * 0.70) ** 2 + (y - size[1] * 0.5) ** 2) <= radius ** 2
    left_img = sitk.GetImageFromArray(left.astype(np.uint8))
    right_img = sitk.GetImageFromArray(right.astype(np.uint8))
    left_img.CopyInformation(small_ct)
    right_img.CopyInformation(small_ct)
    return left_img, right_img
