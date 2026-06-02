# Changelog

All notable changes to fatAnalyze are documented here. Versions follow
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-06-02

### Added (v0.2.0)
- **`ipympl>=0.9`** added to the `.[notebook]` optional dependencies. The
  `widget` matplotlib backend (required by `PolygonSelector` in
  JupyterLab/Notebook 7+) ships in `ipympl`. If `ipympl` is missing, the
  notebook setup cells fall back to a native backend or to `inline` with
  a clear install hint.
- Backend-picker logic in cell 1 of `notebooks/single_case.ipynb` and
  cell 1 of `notebooks/interactive_quickstart.ipynb`: tries `widget` →
  `notebook` → `qt5agg`/`qt6agg`/`tkagg` → `inline`.

### Fixed (v0.2.0)
- `notebooks/single_case.ipynb` cell 1 had a stale `matplotlib widget`
  typo (missing `%`) that crashed the kernel on import. Replaced with
  the proper `%matplotlib widget` invocation and the backend fallback.

## [0.2.0] - 2026-06-02

### Added
- **`fatanalyze.interactive` module**: user-drawn ROIs via a
  `matplotlib.widgets.PolygonSelector` 2D polygon on a single axial slice.
  - `draw_roi_2d(image, z_index, name, preset)` opens an interactive figure;
    user clicks polygon vertices and clicks "Finish" to rasterize.
  - `analyze_user_roi(image, user_roi)` reuses the standard histogram and
    clinical-metric pipeline (`compute_ratios`, `psoas_imat_fraction`) with
    the chosen preset's HU ranges and flags.
  - `UserROI` dataclass with `n_voxels`, `area_cm2`, `volume_ml`, plus
    `save_nifti` / `load_user_roi` round-trip for reproducibility.
  - `plot_user_roi` two-panel figure: CT + polygon overlay | HU histogram.
- **Notebook cell block 7** in `notebooks/single_case.ipynb` (headless-safe;
  skipped automatically in `nbconvert` runs).
- **10 new tests** in `tests/test_user_roi.py` covering polygon
  rasterization, empty-mask handling, preset validation, psoas myosteatosis
  metrics, save/load round-trip. Suite is 33/33 green.

### Changed
- `fatanalyze.__version__` bumped `0.1.0` -> `0.2.0`.

### Notes
- No new runtime dependencies; matplotlib (already required) is the only
  dependency.
- The polygon is 2D and 1-slice thick; the resulting `volume_ml` reflects
  a 1-slice volume. Use TS segmentation for whole-organ 3D analysis.
- Polygon edge-inclusion in `matplotlib.path.Path.contains_points` is
  implementation-dependent; tests use range-based assertions to stay
  robust.

## [0.1.0] - 2026-05-30

### Added
- Initial release: DICOM loader with QC, TotalSegmentator wrapper with disk
  cache, L3-approx psoas extractor, HU histogram + clinical metrics
  (liver-spleen ratio, pancreas-spleen difference, psoas IMAT/LDM),
  matplotlib visualizations, single-case notebook, 23 passing tests.
- MONAI fallback noted in README as Plan B.
