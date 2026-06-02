"""UserROI dataclass: name, preset, mask, geometry, and disk persistence."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import SimpleITK as sitk

from fatanalyze.interactive.polygon_utils import empty_mask_like


def _sidecar_path(mask_path: Path) -> Path:
    """JSON sidecar path for a NIfTI mask (handles .nii.gz suffixes)."""
    name = mask_path.name
    if name.endswith(".nii.gz"):
        stem = name[: -len(".nii.gz")]
        return mask_path.with_name(stem + ".nii.gz.json")
    return mask_path.with_suffix(mask_path.suffix + ".json")


def _valid_presets() -> set[str]:
    """Preset names that map to a target in ``config/targets.yaml``."""
    try:
        from fatanalyze.config import load_default_config
        return set(load_default_config().get("targets", {}).keys())
    except Exception:
        return set()


@dataclass
class UserROI:
    """A user-drawn region of interest.

    Attributes
    ----------
    name : str
        Free-form display name (e.g. ``"psoas_L3_manual"``).
    preset : str
        Key in ``config["targets"]`` (e.g. ``"liver"``, ``"iliopsoas_left"``).
        Drives HU ranges and clinical flags during analysis.
    mask : sitk.Image
        3D binary mask in the same physical space as the source CT volume.
    z_index : int
        Primary axial slice the user drew on.
    n_points : int
        Number of polygon vertices (0 if the user closed without drawing).
    empty_warning : bool
        True if no polygon was finalized.
    notes : str
        Optional free-form notes.
    color : str
        Matplotlib color used for overlay rendering.
    """

    name: str
    preset: str
    mask: sitk.Image
    z_index: int
    n_points: int = 0
    empty_warning: bool = False
    notes: str = ""
    color: str = "#9467bd"

    def __post_init__(self) -> None:
        valid = _valid_presets()
        if valid and self.preset not in valid:
            raise ValueError(
                f"Unknown preset {self.preset!r}. "
                f"Valid presets: {sorted(valid)}"
            )

    @property
    def n_voxels(self) -> int:
        return int((sitk.GetArrayFromImage(self.mask) > 0).sum())

    @property
    def area_cm2(self) -> float:
        arr = sitk.GetArrayFromImage(self.mask) > 0
        if self.z_index < 0 or self.z_index >= arr.shape[0]:
            return 0.0
        sx, sy, _ = self.mask.GetSpacing()
        return float(arr[self.z_index].sum()) * sx * sy / 100.0

    @property
    def volume_ml(self) -> float:
        n = self.n_voxels
        if n == 0:
            return 0.0
        sp = self.mask.GetSpacing()
        return float(n) * float(sp[0] * sp[1] * sp[2]) / 1000.0

    def save_nifti(self, path: str | Path) -> Path:
        """Write the mask to disk as a NIfTI file. Returns the resolved path."""
        import nibabel as nib

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        arr = sitk.GetArrayFromImage(self.mask).astype(np.uint8)
        ref = self.mask
        affine = np.eye(4, dtype=np.float32)
        spacing = ref.GetSpacing()
        direction = ref.GetDirection()
        for r in range(3):
            for c in range(3):
                affine[r, c] = direction[r * 3 + c] * spacing[c]
        affine[:3, 3] = ref.GetOrigin()
        nib.save(nib.Nifti1Image(arr, affine), str(p))
        meta_p = _sidecar_path(p)
        meta_p.write_text(json.dumps(self.meta_dict(), indent=2), encoding="utf-8")
        return p

    def meta_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "preset": self.preset,
            "z_index": int(self.z_index),
            "n_points": int(self.n_points),
            "n_voxels": int(self.n_voxels),
            "area_cm2": float(self.area_cm2),
            "volume_ml": float(self.volume_ml),
            "empty_warning": bool(self.empty_warning),
            "notes": self.notes,
            "color": self.color,
        }


def load_user_roi(
    mask_path: str | Path,
    image: sitk.Image,
    name: Optional[str] = None,
    preset: str = "iliopsoas_left",
) -> UserROI:
    """Re-load a UserROI from a previously saved NIfTI + JSON sidecar."""
    import nibabel as nib

    p = Path(mask_path)
    if not p.exists():
        raise FileNotFoundError(p)
    nii = nib.load(str(p))
    # nibabel returns Nifti1Image (in-memory) or FileBasedImage (on-disk proxy).
    # Both expose ``.dataobj``; ``np.asarray`` materialises to a numpy array.
    arr = np.asarray(nii.dataobj)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {arr.shape}")
    mask = sitk.GetImageFromArray(arr.astype(np.uint8))
    mask.CopyInformation(image)
    meta_p = _sidecar_path(p)
    meta = {}
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
    arr_bool = arr > 0
    z_sums = arr_bool.sum(axis=(1, 2))
    z_index = int(np.argmax(z_sums)) if z_sums.max() > 0 else image.GetSize()[2] // 2
    return UserROI(
        name=name or meta.get("name", p.stem),
        preset=meta.get("preset", preset),
        mask=mask,
        z_index=int(meta.get("z_index", z_index)),
        n_points=int(meta.get("n_points", 0)),
        empty_warning=bool(meta.get("empty_warning", False)),
        notes=meta.get("notes", ""),
        color=meta.get("color", "#9467bd"),
    )


__all__ = ["UserROI", "load_user_roi", "empty_mask_like"]
