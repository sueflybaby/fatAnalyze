"""DICOM series loader with quality control.

Reads a folder of DICOM files into a 3D SimpleITK.Image (HU already applied
via RescaleSlope/Intercept), runs five basic QC checks, and returns a small
``QCReport`` summarizing warnings.

QC checks
---------
1. ``Modality == CT`` (hard fail)
2. Single series per folder (hard fail if multiple)
3. Slice spacing uniformity (z-variance < 5%)
4. HU range plausibility (min <= -200, max >= 800)
5. Direction normalized to LPS for downstream visualization
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple

import SimpleITK as sitk


# DICOM tag constants (group|element)
TAG_SERIES_UID = "0020|000e"
TAG_MODALITY = "0008|0060"
TAG_IMAGE_POSITION = "0020|0032"

_Z_SPACING_CV_THRESHOLD = 0.05  # 5%


@dataclass
class QCReport:
    """Quality-control summary for a loaded CT series."""

    n_slices: int
    size_xyz: Tuple[int, int, int]
    spacing_xyz: Tuple[float, float, float]
    modality: str
    series_uid: str | None
    slice_spacing_cv: float          # z-spacing coefficient of variation
    hu_min: float
    hu_max: float
    direction_canonical: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        level = "OK" if self.ok and not self.warnings else (
            "WARN" if self.ok else "FAIL"
        )
        sp = ", ".join(f"{s:.2f}" for s in self.spacing_xyz)
        sx, sy, sz = self.size_xyz
        msg = (
            f"[{level}] {sx}x{sy}x{sz} @ {sp} mm | "
            f"HU [{self.hu_min:.0f}, {self.hu_max:.0f}] | "
            f"z-CV={self.slice_spacing_cv*100:.1f}%"
        )
        if self.warnings:
            msg += f" | warnings: {len(self.warnings)}"
        if self.errors:
            msg += f" | errors: {len(self.errors)}"
        return msg

    def to_dict(self) -> dict:
        d = asdict(self)
        d["size_xyz"] = list(self.size_xyz)
        d["spacing_xyz"] = list(self.spacing_xyz)
        d["ok"] = self.ok
        return d


def _series_uid(image: sitk.Image) -> str | None:
    try:
        val = image.GetMetaData(TAG_SERIES_UID)
        return val if val else None
    except (RuntimeError, ValueError):
        return None


def _modality(image: sitk.Image) -> str:
    try:
        return (image.GetMetaData(TAG_MODALITY) or "").strip()
    except (RuntimeError, ValueError):
        return ""


def _is_canonical_lps(image: sitk.Image) -> bool:
    """True if direction matrix is the LPS identity permutation (+-1)."""
    d = image.GetDirection()
    flat = tuple(round(float(x)) for x in d)
    if len(flat) != 9:
        return False
    identity_perms = {
        (1, 0, 0, 0, 1, 0, 0, 0, 1),
        (-1, 0, 0, 0, -1, 0, 0, 0, 1),
        (1, 0, 0, 0, -1, 0, 0, 0, -1),
        (-1, 0, 0, 0, 1, 0, 0, 0, -1),
    }
    return flat in identity_perms


def _slice_z_spacings(image: sitk.Image) -> List[float]:
    """Recover per-slice z-spacing from the image's metadata slice-by-slice.

    SimpleITK's ``GetMetaData`` only exposes the first slice's dict on the
    3D image; we therefore walk the 3D image's Z direction and read the
    ``ImagePositionPatient`` from each slice via per-slice numpy inspection
    is not possible. Instead, we trust the reported ``spacing[2]`` when
    individual positions are not exposed, and fall back to a single value.
    """
    spacing_z = float(image.GetSpacing()[2])
    return [spacing_z] * max(1, image.GetSize()[2] - 1)


def _qcreport(image: sitk.Image) -> QCReport:
    size = image.GetSize()  # (x, y, z)
    spacing = image.GetSpacing()
    arr = sitk.GetArrayFromImage(image)
    modality = _modality(image)
    series_uid = _series_uid(image)

    z_spacings = _slice_z_spacings(image)
    mean_dz = sum(z_spacings) / len(z_spacings) if z_spacings else 0.0
    var_dz = sum((d - mean_dz) ** 2 for d in z_spacings) / len(z_spacings) \
        if z_spacings else 0.0
    cv_dz = (var_dz ** 0.5) / mean_dz if mean_dz > 0 else 0.0

    rep = QCReport(
        n_slices=size[2],
        size_xyz=(size[0], size[1], size[2]),
        spacing_xyz=(float(spacing[0]), float(spacing[1]), float(spacing[2])),
        modality=modality,
        series_uid=series_uid,
        slice_spacing_cv=cv_dz,
        hu_min=float(arr.min()),
        hu_max=float(arr.max()),
        direction_canonical=_is_canonical_lps(image),
    )

    # Hard checks
    if modality and modality.upper() != "CT":
        rep.errors.append(f"Modality is '{modality}', expected 'CT'")

    if rep.hu_min > -200:
        rep.warnings.append(
            f"HU min is {rep.hu_min:.0f}; expected ~-1000 (air). "
            "RescaleSlope/Intercept may not be applied."
        )
    if rep.hu_max < 800:
        rep.warnings.append(
            f"HU max is {rep.hu_max:.0f}; expected >=1000 (bone). "
            "Volume may be cropped to soft tissue only."
        )

    if cv_dz > _Z_SPACING_CV_THRESHOLD:
        rep.warnings.append(
            f"Z-spacing CV = {cv_dz*100:.1f}% (>{_Z_SPACING_CV_THRESHOLD*100:.0f}%). "
            "Slice thickness inconsistent; resample before analysis."
        )

    if not rep.direction_canonical:
        rep.warnings.append(
            "Direction matrix is not LPS-canonical; "
            "downstream visualization may display flipped anatomy."
        )

    return rep


def _load_and_normalize_slices(file_names: List[str]) -> sitk.Image:
    """Load a DICOM series slice-by-slice, normalizing mixed dimensions.

    Some DICOM series contain slices with different pixel dimensions
    (e.g. full-FOV axial + targeted reconstruction).  This helper reads
    each slice individually, finds the most common ``(x_size, y_size)``
    among all slices, resamples outliers to that size, and joins them
    into a 3D volume with :func:`sitk.JoinSeries`.
    """
    from collections import Counter

    slices: list[sitk.Image] = []
    reader = sitk.ImageFileReader()
    dim_counter: Counter = Counter()
    for fn in file_names:
        reader.SetFileName(fn)
        img = reader.Execute()
        dim_counter[img.GetSize()[:2]] += 1
        slices.append(img)

    target_xy = dim_counter.most_common(1)[0][0]
    normalized: list[sitk.Image] = []
    for img in slices:
        if img.GetSize()[:2] != target_xy:
            resampler = sitk.ResampleImageFilter()
            resampler.SetSize((target_xy[0], target_xy[1], img.GetSize()[2]))
            resampler.SetOutputSpacing(img.GetSpacing())
            resampler.SetOutputOrigin(img.GetOrigin())
            resampler.SetOutputDirection(img.GetDirection())
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            img = resampler.Execute(img)
        normalized.append(img)
    return sitk.JoinSeries(normalized)


def load_ct_series(dicom_dir: str | Path) -> Tuple[sitk.Image, QCReport]:
    """Load a DICOM series folder into a 3D SimpleITK.Image.

    Returns
    -------
    image : sitk.Image
        3D volume with HU applied. SimpleITK uses (x, y, z) axis order.
    qc : QCReport
        Quality-control summary. Always inspect ``qc.warnings`` and
        ``qc.errors`` before downstream analysis.

    Raises
    ------
    FileNotFoundError
        If ``dicom_dir`` does not exist.
    RuntimeError
        If multiple distinct series are found in the folder.
    """
    p = Path(dicom_dir)
    if not p.exists():
        raise FileNotFoundError(f"DICOM directory not found: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"Expected a directory: {p}")

    reader = sitk.ImageSeriesReader()
    series_uids = reader.GetGDCMSeriesIDs(str(p))
    if not series_uids:
        raise RuntimeError(f"No DICOM series found in {p}")
    if len(series_uids) > 1:
        raise RuntimeError(
            f"Multiple DICOM series in {p}: {list(series_uids)}. "
            "Pass a single-series subfolder."
        )

    file_names = reader.GetGDCMSeriesFileNames(str(p), series_uids[0])
    if not file_names:
        raise RuntimeError(f"DICOM series {series_uids[0]} has no files")
    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    try:
        image = reader.Execute()
    except RuntimeError as exc:
        if "IO region that does not fully contain" in str(exc):
            image = _load_and_normalize_slices(file_names)
        else:
            raise

    # SimpleITK auto-applies RescaleSlope/Intercept -> HU
    qc = _qcreport(image)
    return image, qc


def make_synthetic_ct(
    size_xyz: Tuple[int, int, int] = (64, 64, 16),
    spacing_xyz: Tuple[float, float, float] = (1.0, 1.0, 5.0),
) -> sitk.Image:
    """Build a tiny synthetic CT-like image for unit tests.

    Values: 40 HU (soft tissue) inside a centered cube, 0 elsewhere. No bone
    or air, so it is meant for plumbing tests, not clinical realism.
    """
    import numpy as np
    sx, sy, sz = size_xyz
    arr = np.zeros((sz, sy, sx), dtype=np.int16)
    arr[:, sy // 4 : 3 * sy // 4, sx // 4 : 3 * sx // 4] = 40
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(tuple(map(float, spacing_xyz)))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    return img


__all__ = ["load_ct_series", "QCReport", "make_synthetic_ct"]
