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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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
    """Recover per-slice z-spacing from the image metadata.

    For volumes with more than one slice the reported ``spacing[2]`` is
    the gap between consecutive slices (computed by
    ``ImageSeriesReader``).  Single-slice volumes return an empty list.
    """
    n = image.GetSize()[2]
    if n <= 1:
        return []
    spacing_z = float(image.GetSpacing()[2])
    return [spacing_z] * (n - 1)


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

    The returned image carries DICOM metadata copied from the first slice.
    """
    from collections import Counter

    if not file_names:
        raise ValueError("_load_and_normalize_slices: empty file_names list")

    # Sort by InstanceNumber (0020,0013) to ensure correct z-ordering
    reader = sitk.ImageFileReader()
    indexed: List[Tuple[int, str]] = []
    for fn in file_names:
        try:
            reader.SetFileName(fn)
            reader.ReadImageInformation()
            inst = int(reader.GetMetaData("0020|0013"))
        except (RuntimeError, ValueError, TypeError):
            inst = len(indexed)
        indexed.append((inst, fn))
    indexed.sort(key=lambda x: x[0])
    sorted_fnames = [fn for _, fn in indexed]

    slices: list[sitk.Image] = []
    dim_counter: Counter = Counter()
    for fn in sorted_fnames:
        reader.SetFileName(fn)
        img = reader.Execute()
        dim_counter[img.GetSize()[:2]] += 1
        slices.append(img)

    target_xy = dim_counter.most_common(1)[0][0]

    # Reconstruct a reference image from the first slice to carry metadata
    # through the fallback path
    ref_image = slices[0]
    metadata_tags: Dict[str, str] = {}
    if ref_image.HasMetaDataKey(TAG_MODALITY):
        metadata_tags[TAG_MODALITY] = ref_image.GetMetaData(TAG_MODALITY)
    if ref_image.HasMetaDataKey(TAG_SERIES_UID):
        metadata_tags[TAG_SERIES_UID] = ref_image.GetMetaData(TAG_SERIES_UID)

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

    result = sitk.JoinSeries(normalized)

    # Re-attach critical DICOM metadata lost by JoinSeries
    for tag, val in metadata_tags.items():
        result.SetMetaData(tag, val)

    return result


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

    # Proactively check for mixed-dimension slices before Execute().
    # ImageSeriesReader fails with a fragile platform-dependent error when
    # slices have different XY sizes; scanning metadata first avoids it.
    image = None
    if len(file_names) > 1:
        probe = sitk.ImageFileReader()
        first_size = None
        sizes_uniform = True
        for fn in file_names:
            try:
                probe.SetFileName(fn)
                probe.ReadImageInformation()
                sz = probe.GetSize()[:2]
                if first_size is None:
                    first_size = sz
                elif sz != first_size:
                    sizes_uniform = False
                    break
            except RuntimeError:
                sizes_uniform = False
                break
        if not sizes_uniform:
            image = _load_and_normalize_slices(file_names)

    if image is None:
        try:
            image = reader.Execute()
        except RuntimeError:
            image = _load_and_normalize_slices(file_names)

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


# ── MR (PDFF / Dixon) loader ──────────────────────────────────────────────


def _series_description(reader: sitk.ImageFileReader, file_name: str) -> str:
    """Read SeriesDescription from a single DICOM file's metadata."""
    try:
        reader.SetFileName(file_name)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        desc = reader.GetMetaData("0008|103e")
        return (desc or "").strip().upper()
    except RuntimeError:
        return ""


def _load_series_from_files(file_names: List[str]) -> sitk.Image:
    """Load a list of DICOM files as a 3D volume (like ImageSeriesReader).

    If slices have mixed dimensions, falls back to
    :func:`_load_and_normalize_slices` automatically.
    """
    if not file_names:
        raise ValueError("_load_series_from_files: empty file_names list")

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()

    # Proactive mixed-dimension check (avoids fragile error-string parsing)
    image = None
    if len(file_names) > 1:
        probe = sitk.ImageFileReader()
        first_size = None
        sizes_uniform = True
        for fn in file_names:
            try:
                probe.SetFileName(fn)
                probe.ReadImageInformation()
                sz = probe.GetSize()[:2]
                if first_size is None:
                    first_size = sz
                elif sz != first_size:
                    sizes_uniform = False
                    break
            except RuntimeError:
                sizes_uniform = False
                break
        if not sizes_uniform:
            image = _load_and_normalize_slices(file_names)

    if image is None:
        try:
            image = reader.Execute()
        except RuntimeError:
            image = _load_and_normalize_slices(file_names)

    return image


def _kw_match(desc: str, kw_list: List[str]) -> bool:
    """Whole-word match against SeriesDescription.

    The description is split on whitespace, ``/``, ``-``, ``_``, and each
    keyword is checked for exact whole-word equality (case-insensitive).
    This prevents ``FAT`` from matching ``FATIGUE`` or ``NON-FAT``.
    """
    if not kw_list:
        return False
    words = set(desc.upper().replace("/", " ").replace("-", " ").replace("_", " ").split())
    return any(kw.upper() in words for kw in kw_list if kw.strip())


def _find_mr_series(
    dicom_dir: str | Path,
    kw_list: List[str],
    file_reader: sitk.ImageFileReader,
    exclude_uids: set[str] | None = None,
) -> List[str]:
    """Find DICOM series whose SeriesDescription matches one of the keywords.

    Returns the file list of the first matching series, or an empty list.
    ``exclude_uids`` can be used to skip already-matched series.
    """
    if not kw_list:
        return []
    p = Path(dicom_dir)
    series_reader = sitk.ImageSeriesReader()
    for uid in series_reader.GetGDCMSeriesIDs(str(p)):
        if exclude_uids and uid in exclude_uids:
            continue
        fnames = series_reader.GetGDCMSeriesFileNames(str(p), uid)
        if not fnames:
            continue
        desc = _series_description(file_reader, fnames[0])
        if _kw_match(desc, kw_list):
            return fnames
    return []


def _list_all_series(
    dicom_dir: str | Path,
) -> List[Tuple[str, str, List[str]]]:
    """List all DICOM series in the folder.

    Returns list of ``(series_uid, series_description, file_names)``.
    """
    p = Path(dicom_dir)
    reader = sitk.ImageSeriesReader()
    file_reader = sitk.ImageFileReader()
    series: List[Tuple[str, str, List[str]]] = []
    for uid in reader.GetGDCMSeriesIDs(str(p)):
        fnames = reader.GetGDCMSeriesFileNames(str(p), uid)
        desc = _series_description(file_reader, fnames[0]) if fnames else ""
        series.append((uid, desc, fnames))
    return series


def _load_pdff_series_from_files(
    file_names: List[str],
    scale_factor: float = 1.0,
) -> sitk.Image:
    """Load a PDFF series from a pre-scanned file list and apply scaling."""
    image = _load_series_from_files(file_names)
    if abs(scale_factor - 1.0) > 1e-6:
        arr = sitk.GetArrayFromImage(image).astype(np.float32) * scale_factor
        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(image)
        image = out
    return image


def _compute_fat_fraction(
    fat_img: sitk.Image,
    water_img: sitk.Image,
) -> sitk.Image:
    """Compute pixel-wise fat fraction: ``FF = fat / (fat + water) * 100``.

    Regions where ``fat + water == 0`` are set to 0 %.
    Uses float32 throughout to halve peak memory versus float64.
    """
    fat_arr = sitk.GetArrayFromImage(fat_img).astype(np.float32)
    water_arr = sitk.GetArrayFromImage(water_img).astype(np.float32)
    # Compute in float32; back-out float64 only for division zero-guard
    denom = fat_arr + water_arr
    mask = denom > 0.0
    ff_arr = np.zeros_like(denom)
    np.divide(fat_arr, denom, out=ff_arr, where=mask)
    ff_arr *= 100.0
    np.clip(ff_arr, 0.0, 100.0, out=ff_arr)
    out = sitk.GetImageFromArray(ff_arr)
    out.CopyInformation(fat_img)
    return out


def load_mr_series(
    dicom_dir: str | Path,
    preset_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[sitk.Image, QCReport]:
    """Load an MR fat-quantification series.

    Supports two modes (auto-detected from the folder contents):

    * **PDFF** – single series; pixel values are scaled according to the
      vendor preset to produce 0-100 % fat fraction.
    * **Dixon** – two series (fat-only / water-only); the pair is matched
      by SeriesDescription keywords from the preset, then
      :func:`_compute_fat_fraction` is applied.

    Parameters
    ----------
    dicom_dir : str or Path
        Folder containing one DICOM series (PDFF) or two (Dixon).
    preset_cfg : dict, optional
        Vendor-preset dict with keys ``pdff_scale_factor`` and
        ``dixon`` (with sub-keys ``fat_kw`` / ``water_kw``).
        Defaults to ``"Siemens (0-100)"``.

    Returns
    -------
    image : sitk.Image
        3D volume where pixel values are fat fraction in 0-100 %.
    qc : QCReport
        Quality-control summary (MR-aware).
    """
    if preset_cfg is None:
        from fatanalyze.config import load_mr_presets
        presets = load_mr_presets()
        preset_cfg = presets.get("presets", {}).get("Siemens (0-100)", {})

    scale_factor = float(preset_cfg.get("pdff_scale_factor", 1.0))
    dixon_cfg = preset_cfg.get("dixon", {})
    fat_kw: List[str] = dixon_cfg.get("fat_kw", [])
    water_kw: List[str] = dixon_cfg.get("water_kw", [])

    p = Path(dicom_dir)

    # Single scan collects all series info for the whole function
    all_series = _list_all_series(p)

    if not all_series:
        raise RuntimeError(f"No DICOM series found in {p}")

    if len(all_series) == 1:
        # ── PDFF: single series ───────────────────────────────────
        _, _, fnames = all_series[0]
        image = _load_pdff_series_from_files(fnames, scale_factor)
        qc = _mr_qcreport(image, "PDFF")
        return image, qc

    # ── Dixon: two (or more) series matched by keyword ────────────
    if not fat_kw and not water_kw:
        raise RuntimeError(
            "MR preset has empty fat/water keywords. "
            "Select a different vendor preset or configure keywords."
        )

    # Build series-uid -> file-list mapping from the single scan
    file_reader = sitk.ImageFileReader()
    all_uids = {uid for uid, _, _ in all_series}

    fat_files = _find_mr_series(p, fat_kw, file_reader)
    if fat_files:
        # Find the UID of the matched fat series to exclude it
        _reader = sitk.ImageSeriesReader()
        for uid in all_uids:
            _f = _reader.GetGDCMSeriesFileNames(str(p), uid)
            if _f == fat_files:
                exclude_uids = {uid}
                break
        else:
            exclude_uids = set()
    else:
        exclude_uids = set()

    water_files = _find_mr_series(p, water_kw, file_reader, exclude_uids)

    if not fat_files or not water_files:
        choices = "\n".join(
            f"  {i+1}. {uid[:12]}… — {desc or '(no description)'}"
            for i, (uid, desc, _) in enumerate(all_series)
        )
        raise RuntimeError(
            f"Could not auto-match fat/water series in folder.\n"
            f"Available series:\n{choices}\n\n"
            f"Fat keywords: {fat_kw}\nWater keywords: {water_kw}"
        )

    fat_img = _load_series_from_files(fat_files)
    water_img = _load_series_from_files(water_files)

    # ── Structural validation ─────────────────────────────────────
    fat_sz = fat_img.GetSize()
    water_sz = water_img.GetSize()
    errors: List[str] = []
    if fat_sz[:2] != water_sz[:2]:
        errors.append(
            f"XY size mismatch: fat {fat_sz[:2]} vs water {water_sz[:2]}"
        )
    if fat_sz[2] != water_sz[2]:
        errors.append(
            f"Depth mismatch: fat {fat_sz[2]} slices vs water {water_sz[2]}"
        )
    fat_sp = fat_img.GetSpacing()
    water_sp = water_img.GetSpacing()
    if abs(fat_sp[0] - water_sp[0]) > 1e-3 or abs(fat_sp[1] - water_sp[1]) > 1e-3:
        errors.append(
            f"XY spacing mismatch: fat ({fat_sp[0]:.3f}, {fat_sp[1]:.3f}) "
            f"vs water ({water_sp[0]:.3f}, {water_sp[1]:.3f})"
        )
    fat_orig = fat_img.GetOrigin()
    water_orig = water_img.GetOrigin()
    if any(abs(a - b) > 1.0 for a, b in zip(fat_orig, water_orig)):
        errors.append(
            f"Origin mismatch: fat {fat_orig} vs water {water_orig}"
        )
    fat_dir = fat_img.GetDirection()
    water_dir = water_img.GetDirection()
    if any(abs(a - b) > 1e-3 for a, b in zip(fat_dir, water_dir)):
        errors.append("Direction matrix mismatch between fat and water series")
    if errors:
        raise RuntimeError(
            "Dixon fat/water series mismatch:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    image = _compute_fat_fraction(fat_img, water_img)
    qc = _mr_qcreport(image, "Dixon")
    return image, qc


def _mr_qcreport(image: sitk.Image, series_type: str) -> QCReport:
    """Build a QC report for an MR fat-fraction image."""
    size = image.GetSize()
    spacing = image.GetSpacing()
    arr = sitk.GetArrayFromImage(image)

    z_spacings = _slice_z_spacings(image)
    mean_dz = sum(z_spacings) / len(z_spacings) if z_spacings else 0.0
    var_dz = sum((d - mean_dz) ** 2 for d in z_spacings) / len(z_spacings) \
        if z_spacings else 0.0
    cv_dz = (var_dz ** 0.5) / mean_dz if mean_dz > 0 else 0.0

    rep = QCReport(
        n_slices=size[2],
        size_xyz=(size[0], size[1], size[2]),
        spacing_xyz=(float(spacing[0]), float(spacing[1]), float(spacing[2])),
        modality="MR",
        series_uid=f"MR-{series_type}",
        slice_spacing_cv=cv_dz,
        hu_min=float(arr.min()),
        hu_max=float(arr.max()),
        direction_canonical=False,
    )

    rep.warnings.append(f"MR {series_type} fat-fraction map (0-100%)")

    ff = arr.flatten()
    ff_min = float(ff.min())
    ff_max = float(ff.max())
    if ff_min < -1.0:
        rep.warnings.append(
            f"Fat-fraction min = {ff_min:.1f}%; possible scale issue"
        )
    if ff_max > 105.0:
        rep.warnings.append(
            f"Fat-fraction max = {ff_max:.1f}%; possible scale issue"
        )

    if 0 < ff_max < 1.0:
        rep.warnings.append(
            "Fat-fraction range looks like a 0-1 scale (expected 0-100). "
            "Check vendor preset / scale factor."
        )

    if cv_dz > _Z_SPACING_CV_THRESHOLD:
        rep.warnings.append(
            f"Z-spacing CV = {cv_dz*100:.1f}% "
            f"(>{_Z_SPACING_CV_THRESHOLD*100:.0f}%). "
            "Slice thickness inconsistent; resample before analysis."
        )

    return rep


__all__ = ["load_ct_series", "load_mr_series", "QCReport", "make_synthetic_ct"]
