"""ROI extraction: mask cleanup, L3-approx detection, HU extraction."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi


def get_3d_mask(
    mask: sitk.Image,
    keep_largest_cc: bool = True,
    erode_voxels: int = 0,
) -> sitk.Image:
    """Post-process a binary segmentation.

    Steps:
    1. (Optional) Keep only the largest connected component to drop
       segmentation noise.
    2. (Optional) Erode the boundary by ``erode_voxels`` voxels. Useful
       for organ analysis to avoid partial-volume mixing at the rim.

    Parameters
    ----------
    mask : sitk.Image
        Binary (uint8) segmentation mask.
    keep_largest_cc : bool
        If True, keep only the largest 3D connected component.
    erode_voxels : int
        Number of voxels to shrink the mask inward. 0 = no erosion.

    Returns
    -------
    sitk.Image
        Cleaned binary mask in the same physical space as the input.
    """
    arr = sitk.GetArrayFromImage(mask) > 0
    if keep_largest_cc:
        labels, n = ndi.label(arr)
        if n > 1:
            sizes = ndi.sum(arr, labels, range(1, n + 1))
            keep = (labels == (int(np.argmax(sizes)) + 1))
            arr = keep
    if erode_voxels > 0:
        arr = ndi.binary_erosion(arr, iterations=erode_voxels)
    out = sitk.GetImageFromArray(arr.astype(np.uint8))
    out.CopyInformation(mask)
    return out


def find_l3_slice(
    left_psoas: sitk.Image,
    right_psoas: sitk.Image,
    n_buffer: int = 1,
) -> Tuple[int, int, int]:
    """Find the L3-approximate axial slice index.

    L3-approximation rule: the axial slice where left + right psoas masks
    together have the largest combined area. This matches clinical body
    composition convention (Mourtzakis et al., 2008) closely enough for
    fat quantification at the L3 level.

    Parameters
    ----------
    left_psoas, right_psoas : sitk.Image
        Binary psoas masks. May be 3D.
    n_buffer : int
        Number of slices above and below the argmax slice to include in
        the final mask (default 1). Buffer reduces noise from a single
        argmax spike.

    Returns
    -------
    z_center, z_min, z_max : int
        The argmax slice index and the buffer-inclusive range used.
    """
    if left_psoas.GetSize() != right_psoas.GetSize():
        raise ValueError("Left and right psoas masks must share size")
    left_arr = sitk.GetArrayFromImage(left_psoas) > 0
    right_arr = sitk.GetArrayFromImage(right_psoas) > 0
    n_slices = left_arr.shape[0]
    if n_slices == 0:
        raise ValueError("Empty psoas mask")
    areas = (left_arr | right_arr).sum(axis=(1, 2))
    if areas.max() == 0:
        raise ValueError("Both psoas masks are empty")
    z_center = int(np.argmax(areas))
    z_min = max(0, z_center - n_buffer)
    z_max = min(n_slices - 1, z_center + n_buffer)
    return z_center, z_min, z_max


def get_l3_psoas_mask(
    left_psoas: sitk.Image,
    right_psoas: sitk.Image,
    z_center: int,
    n_buffer: int = 1,
    keep_largest_cc: bool = True,
) -> sitk.Image:
    """Build the L3 psoas analysis mask (left + right merged) over a buffer.

    The returned mask is 3D (same size as inputs) but is non-zero only in
    the ``[z_min, z_max]`` range.
    """
    z_min = max(0, z_center - n_buffer)
    z_max = min(left_psoas.GetSize()[2] - 1, z_center + n_buffer)
    arr_left = sitk.GetArrayFromImage(left_psoas) > 0
    arr_right = sitk.GetArrayFromImage(right_psoas) > 0
    out = np.zeros_like(arr_left, dtype=bool)
    out[z_min : z_max + 1] = arr_left[z_min : z_max + 1] | arr_right[z_min : z_max + 1]
    if keep_largest_cc:
        labels, n = ndi.label(out)
        if n > 1:
            sizes = ndi.sum(out, labels, range(1, n + 1))
            keep = labels == (int(np.argmax(sizes)) + 1)
            out = keep
    img = sitk.GetImageFromArray(out.astype(np.uint8))
    img.CopyInformation(left_psoas)
    return img


def extract_hu(image: sitk.Image, mask: sitk.Image) -> np.ndarray:
    """Return the 1D array of HU values inside ``mask``.

    The mask is treated as a binary selector (any non-zero value counts).
    Returns an empty array if the mask is empty.
    """
    img_arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask) > 0
    if not mask_arr.any():
        return np.empty(0, dtype=img_arr.dtype)
    return img_arr[mask_arr]


def mask_volume_ml(mask: sitk.Image) -> float:
    """Compute the mask volume in millilitres from voxel count and spacing."""
    arr = sitk.GetArrayFromImage(mask) > 0
    voxel_ml = float(np.prod(mask.GetSpacing())) / 1000.0
    return float(arr.sum()) * voxel_ml


def mask_area_cm2(mask: sitk.Image, z_index: int) -> float:
    """Compute the 2D mask area in cm^2 at a single axial slice."""
    arr = sitk.GetArrayFromImage(mask) > 0
    if z_index < 0 or z_index >= arr.shape[0]:
        return 0.0
    voxel_cm2 = float(mask.GetSpacing()[0] * mask.GetSpacing()[1]) / 100.0
    return float(arr[z_index].sum()) * voxel_cm2


__all__ = [
    "get_3d_mask",
    "find_l3_slice",
    "get_l3_psoas_mask",
    "extract_hu",
    "mask_volume_ml",
    "mask_area_cm2",
]
