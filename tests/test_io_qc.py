"""Tests for the DICOM loader and QC report."""
from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from fatanalyze.io.dicom_loader import (
    QCReport, _is_canonical_lps, _qcreport, make_synthetic_ct,
)


def test_make_synthetic_ct_has_hu_inside_cube() -> None:
    img = make_synthetic_ct(size_xyz=(32, 32, 8), spacing_xyz=(1.0, 1.0, 1.0))
    arr = sitk.GetArrayFromImage(img)
    cube = arr[2:6, 8:24, 8:24]
    assert cube.min() == 40
    assert (cube == 40).all()


def test_qcreport_detects_hu_outliers() -> None:
    """A volume with HU range [0, 100] should trigger the missing-air warning."""
    arr = np.full((8, 32, 32), 50, dtype=np.int16)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    qc = _qcreport(img)
    # Min and max are both narrow; both warnings fire
    assert any("HU min" in w for w in qc.warnings)
    assert any("HU max" in w for w in qc.warnings)
    assert qc.ok is True  # warnings are non-fatal


def test_is_canonical_lps_accepts_identity() -> None:
    img = make_synthetic_ct()
    assert _is_canonical_lps(img) is True


def test_is_canonical_lps_rejects_skewed_direction() -> None:
    img = make_synthetic_ct()
    img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    img.SetDirection((0, 1, 0, 1, 0, 0, 0, 0, -1))  # not LPS
    assert _is_canonical_lps(img) is False


def test_qcreport_rejects_non_ct_modality() -> None:
    arr = np.zeros((4, 16, 16), dtype=np.int16)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((1.0, 1.0, 1.0))
    # Inject a fake Modality tag
    img.SetMetaData("0008|0060", "MR")
    qc = _qcreport(img)
    assert any("Modality" in e for e in qc.errors)
    assert qc.ok is False


def test_load_ct_series_missing_dir() -> None:
    from fatanalyze.io.dicom_loader import load_ct_series
    with pytest.raises(FileNotFoundError):
        load_ct_series("/no/such/dir")


def test_qcreport_summary_contains_key_fields() -> None:
    img = make_synthetic_ct()
    qc = _qcreport(img)
    s = qc.summary()
    for token in ("mm", "HU", "z-CV"):
        assert token in s
    # Status token is one of OK/WARN/FAIL
    assert any(token in s for token in ("OK", "WARN", "FAIL"))


def test_qcreport_to_dict_round_trip() -> None:
    img = make_synthetic_ct()
    qc = _qcreport(img)
    d = qc.to_dict()
    assert d["n_slices"] == img.GetSize()[2]
    assert d["modality"] == ""
    assert isinstance(d["warnings"], list)


# ---------------------------------------------------------------------------
# Modality auto-detection
# ---------------------------------------------------------------------------


def test_detect_dicom_modality_from_fixture() -> None:
    """Real DICOM folder returns 'ct'."""
    from pathlib import Path
    from fatanalyze.io.dicom_loader import detect_dicom_modality
    result = detect_dicom_modality(Path("data/my_case"))
    assert result == "ct"


def test_detect_dicom_modality_empty_dir_raises() -> None:
    """Empty directory raises ValueError."""
    import tempfile
    from pathlib import Path
    from fatanalyze.io.dicom_loader import detect_dicom_modality
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError):
            detect_dicom_modality(Path(tmp))


def test_detect_dicom_modality_nonexistent_raises() -> None:
    """Non-existent directory raises ValueError."""
    from pathlib import Path
    from fatanalyze.io.dicom_loader import detect_dicom_modality
    with pytest.raises(ValueError):
        detect_dicom_modality(Path("/no/such/dir"))
