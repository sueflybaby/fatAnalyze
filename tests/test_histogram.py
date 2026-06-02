"""Tests for HU histogram and range-ratio computation."""
from __future__ import annotations

import numpy as np
import pytest

from fatanalyze.analysis.histogram import compute_ratios


def test_compute_ratios_basic_liver() -> None:
    """Half HU=10, half HU=60 -> liver fat_moderate 50%, normal 50%."""
    hu = np.array([10] * 100 + [60] * 100, dtype=np.int16)
    r = compute_ratios(hu, "liver")
    assert r.n_voxels == 200
    assert r.mean_hu == pytest.approx(35.0, abs=0.1)
    # liver config: fat_severe [-200,0], fat_moderate [0,30], fat_mild [30,40], normal [40,80]
    assert r.ratios["fat_severe"] == pytest.approx(0.0, abs=1e-6)
    assert r.ratios["fat_moderate"] == pytest.approx(0.5, abs=1e-3)
    assert r.ratios["normal"] == pytest.approx(0.5, abs=1e-3)


def test_compute_ratios_psoas_myosteatosis_flags() -> None:
    """Psoas HU dominated by IMAT -> clinical flag raised."""
    rng = np.random.default_rng(0)
    hu = np.concatenate([
        rng.normal(-60, 5, 300),    # IMAT
        rng.normal(10, 5, 100),     # LDM
        rng.normal(50, 15, 600),    # normal
    ]).astype(np.int16)
    r = compute_ratios(hu, "iliopsoas_left")
    assert r.ratios["imat"] > 0.2
    assert any("imat" in f or "low_density" in f for f in r.clinical_flags)


def test_compute_ratios_empty_returns_empty_result() -> None:
    r = compute_ratios(np.empty(0, dtype=np.int16), "liver")
    assert r.n_voxels == 0
    assert "empty_roi" in r.clinical_flags


def test_compute_ratios_volume_ml_from_spacing() -> None:
    hu = np.full(1000, 40, dtype=np.int16)
    r = compute_ratios(hu, "liver", spacing_xyz=(2.0, 2.0, 5.0))
    # 1000 * (2*2*5) / 1000 = 20 ml
    assert r.volume_ml == pytest.approx(20.0, abs=0.01)


def test_compute_ratios_volume_ml_none_when_no_spacing() -> None:
    r = compute_ratios(np.array([40], dtype=np.int16), "liver")
    assert r.volume_ml is None


def test_compute_ratios_percentiles() -> None:
    rng = np.random.default_rng(42)
    hu = rng.normal(50, 10, 1000).astype(np.int16)
    r = compute_ratios(hu, "liver")
    assert r.p05_hu < r.mean_hu < r.p95_hu
