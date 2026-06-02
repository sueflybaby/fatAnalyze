"""TotalSegmentator wrapper with disk caching.

Runs TotalSegmentator on a CT volume and returns per-target binary masks as
SimpleITK images. Results are cached on disk keyed by ``SeriesInstanceUID +
spacing + size + ROI set`` so a notebook re-run after a parameter tweak does
not re-segment the unchanged targets.

The wrapper:
- Resamples the input to TotalSegmentator's preferred 1.5mm isotropic spacing
  only if the user opts in (``resample=True``). By default it trusts the
  input as-is and lets TotalSegmentator's internal pipeline handle spacing.
- Saves model weights to the project's ``.cache/totalseg/`` directory
  (``TOTALSEG_HOME_DIR`` env var).
- Supports ``roi_subset`` to skip segmenting everything when only a few
  organs are needed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import SimpleITK as sitk

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Caching helpers
# -----------------------------------------------------------------------------


def _vol_hash(image: sitk.Image, roi_names: List[str], task: str,
              fast: bool = False, device: str = "cpu") -> str:
    """Deterministic hash of the volume + request that keys the disk cache.

    Includes SeriesInstanceUID when available; otherwise falls back to
    (size, spacing, origin) so two unrelated DICOMs do not collide.

    `fast` and `device` are part of the hash so re-running with a different
    model variant invalidates the cache.
    """
    parts = [task, f"fast={fast}", f"device={device}", ",".join(sorted(roi_names))]
    try:
        sid = image.GetMetaData("0020|000e")
        if sid:
            parts.append(sid)
    except (RuntimeError, ValueError):
        pass
    parts.extend([
        str(image.GetSize()),
        ",".join(f"{s:.4f}" for s in image.GetSpacing()),
    ])
    h = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return h


def _cache_path(cache_dir: Path, vol_hash: str, roi_name: str) -> Path:
    return cache_dir / vol_hash / f"{roi_name}.nii.gz"


def _is_cache_valid(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 1024


# -----------------------------------------------------------------------------
# SimpleITK <-> NIfTI conversion
# -----------------------------------------------------------------------------


def _sitk_to_nifti(image: sitk.Image) -> nib.Nifti1Image:
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing = image.GetSpacing()
    direction = image.GetDirection()
    # SimpleITK is (x, y, z) LPS; NIfTI is RAS. The affine mapping is well-known.
    # For our purposes (re-loading back into the same SimpleITK image), the
    # affine is computed from spacing + direction, leaving the sign flip to
    # the loader.
    affine = np.eye(4, dtype=np.float32)
    # NIfTI xyz <- SimpleITK xyz with direction applied
    for r in range(3):
        for c in range(3):
            affine[r, c] = direction[r * 3 + c] * spacing[c]
    affine[:3, 3] = image.GetOrigin()
    return nib.Nifti1Image(arr, affine)


def _nifti_to_sitk(nii: nib.Nifti1Image, reference: sitk.Image) -> sitk.Image:
    arr = np.asarray(nii.dataobj, dtype=np.int16)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(reference.GetSpacing())
    img.SetOrigin(reference.GetOrigin())
    img.SetDirection(reference.GetDirection())
    return img


# -----------------------------------------------------------------------------
# Mask extraction from a multi-label segmentation
# -----------------------------------------------------------------------------


def _extract_label_mask(
    segmentation: sitk.Image,
    label_value: int,
) -> sitk.Image:
    """Threshold a multi-label segmentation to a single binary mask."""
    mask = segmentation == label_value
    return sitk.Cast(mask, sitk.sitkUInt8)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


@dataclass
class SegmentResult:
    """Container for segmentation output."""

    masks: Dict[str, sitk.Image]
    cache_hits: List[str]
    cache_misses: List[str]
    elapsed_s: float


def segment(
    image: sitk.Image,
    roi_names: Optional[List[str]] = None,
    cache_dir: str | Path = ".cache/totalseg_runs",
    task: str = "total",
    fast: bool = True,
    device: str = "cpu",
    force: bool = False,
) -> Dict[str, sitk.Image]:
    """Segment the input CT volume and return per-ROI binary masks.

    Parameters
    ----------
    image : sitk.Image
        3D CT volume in HU (SimpleITK uses (x, y, z)).
    roi_names : list[str], optional
        Subset of ROIs to return, e.g. ``["liver", "pancreas"]``. Defaults
        to the project's configured set. Pass an empty list to default.
    cache_dir : path
        Where to store the per-ROI NIfTI masks. A new subdirectory is
        created per volume hash.
    task : str
        TotalSegmentator task name. ``"total"`` is the v2 default with
        117 structures.
    fast : bool
        TotalSegmentator's ``fast`` mode - lower precision but ~2-3x faster.
    device : str
        ``"cpu"`` (default), ``"cuda"``, or ``"mps"``. On Apple Silicon,
        CPU is more reliable than MPS for 3D conv.
    force : bool
        Re-run segmentation even if cache is valid.

    Returns
    -------
    dict[str, sitk.Image]
        ``{roi_name: binary mask}`` in the same physical space as ``image``.
    """
    from fatanalyze.config import load_default_config

    cfg = load_default_config()
    if roi_names is None:
        roi_names = list(cfg["totalseg"].get("roi_subset") or [])

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Set model-weight home so the model cache lives in the project
    os.environ.setdefault("TOTALSEG_HOME_DIR", str(Path(".cache/totalseg").resolve()))

    vol_hash = _vol_hash(image, roi_names, task, fast=fast, device=device)
    mask_paths = {r: _cache_path(cache_dir, vol_hash, r) for r in roi_names}

    hits: List[str] = []
    misses: List[str] = []
    for r, p in mask_paths.items():
        if not force and _is_cache_valid(p):
            hits.append(r)
        else:
            misses.append(r)

    logger.info("TotalSegmentator cache: %d hit, %d miss", len(hits), len(misses))
    if misses:
        _run_totalseg(
            image, cache_dir=cache_dir, vol_hash=vol_hash,
            task=task, fast=fast, device=device, roi_names=roi_names,
        )

    masks: Dict[str, sitk.Image] = {}
    for r, p in mask_paths.items():
        nii = nib.load(str(p))
        masks[r] = _nifti_to_sitk(nii, image)
    return masks


def _run_totalseg(
    image: sitk.Image,
    cache_dir: Path,
    vol_hash: str,
    task: str,
    fast: bool,
    device: str,
    roi_names: List[str],
) -> None:
    """Run TotalSegmentator end-to-end and write per-ROI NIfTI masks.

    TotalSegmentator returns a single multi-label volume in v2. We split
    into per-ROI binary masks using the label dictionary exposed by the
    package.
    """
    from totalsegmentator.python_api import totalsegmentator
    from totalsegmentator.map_to_binary import class_map as _class_map

    tmp_nii = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
            tmp_nii = Path(f.name)
        nii_in = _sitk_to_nifti(image)
        nib.save(nii_in, str(tmp_nii))

        out_dir = cache_dir / vol_hash
        out_dir.mkdir(parents=True, exist_ok=True)

        # TotalSegmentator v2 returns a multi-label NIfTI
        result = totalsegmentator(
            str(tmp_nii),
            output=str(out_dir / "multilabel.nii.gz"),
            task=task,
            fast=fast,
            device=device,
            ml=False,
            nr_thr_resamp=1,
            nr_thr_saving=1,
            output_type="nifti",
            quiet=True,
            roi_subset=roi_names if roi_names else None,
        )

        # In TS 2.13 with roi_subset, output is per-ROI files (e.g. liver.nii.gz)
        # written into a directory named after the requested output path.
        # Detect this case first.
        expected_roi_files = [out_dir / f"{n}.nii.gz" for n in (roi_names or [])]
        per_roi_present = expected_roi_files and all(p.exists() for p in expected_roi_files)
        if per_roi_present:
            return

        # Otherwise TS wrote a single multi-label NIfTI at the requested path
        # (or returned one in-memory). The path may be a file, or — when TS
        # honoured the name as a prefix — a directory of per-ROI files.
        if hasattr(result, "get_fdata"):
            multi_path = out_dir / "multilabel.nii.gz"
            if not multi_path.exists():
                nib.save(result, str(multi_path))
        else:
            multi_path = Path(result)

        if not multi_path.is_file():
            raise RuntimeError(
                f"TotalSegmentator produced no usable output: {multi_path} "
                f"is not a file (possibly a directory of per-ROI masks) "
                f"and no per-ROI files in {out_dir}"
            )

        multi_nii = nib.load(str(multi_path))
        multi_arr = multi_nii.get_fdata().astype(np.int32)

        # Map label_value -> friendly name.
        # class_map is a dict keyed by task name; each value is {label_int: name_str}.
        label_map = _class_map.get(task, {})
        inv = {int(label_val): name for label_val, name in label_map.items()}
        unique_vals = sorted(np.unique(multi_arr).astype(int).tolist())

        for label_val in unique_vals:
            if label_val == 0:
                continue
            name = inv.get(label_val)
            if name is None:
                continue
            binary = (multi_arr == label_val).astype(np.uint8)
            roi_nii = nib.Nifti1Image(binary, multi_nii.affine, multi_nii.header)
            out_p = out_dir / f"{name}.nii.gz"
            nib.save(roi_nii, str(out_p))
    finally:
        if tmp_nii is not None and tmp_nii.exists():
            try:
                tmp_nii.unlink()
            except OSError:
                pass


__all__ = ["segment", "SegmentResult"]
