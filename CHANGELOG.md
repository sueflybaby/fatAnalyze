# Changelog

All notable changes to fatAnalyze are documented here. Versions follow
[Semantic Versioning](https://semver.org/).

## [0.3.0] - 2026-06-02

### Added
- **`fatanalyze.gui` package** — a native PySide6 GUI replaces Jupyter as
  the primary single-case interface. New `fatanalyze-gui` console script.
  - `app.py`: `QMainWindow` with menu bar, status bar, signal/slot wiring,
    QC dialog after load, and an `export CSV` action.
  - `slice_view.py`: `QGraphicsView` that renders one axial slice as an
    8-bit grayscale `QImage` with live W/L, pan/zoom, and pixel readout in
    the status bar. W/L preset buttons (Soft Tissue, Bone, Lung, Liver,
    Psoas).
  - `polygon_item.py`: editable `QGraphicsPolygonItem` with vertex handles
    (drag to refine, right-click to remove last, double-click to close).
    Signals carried by an embedded `QObject` (PySide6 `QGraphicsItem` is
    not a `QObject`, so direct `Signal` doesn't work).
  - `controls.py`: toolbar with Open DICOM, preset combo, W/L sliders, W/L
    preset combo, Polygon toggle, Clear / Save ROI / Analyze / Export CSV.
  - `roi_list.py`: multi-ROI `QListWidget` with rename + delete.
  - `metrics_runner.py`: thin wrapper around `analyze_user_roi` that
    rasterizes each ROI and, for psoas presets, **merges L+R masks via
    `sitk.Or`** before calling `psoas_imat_fraction` so the combined
    entry drives the myosteatosis breakdown.
  - `results_panel.py`: text table per ROI + embedded `FigureCanvasQTAgg`
    histogram with mean/median overlay lines.
  - `roi.py`: `ROI` dataclass (name, preset, z, vertices, mask, result).
- **9 new tests** in `tests/test_gui.py` covering window construction,
  polygon state machine, ROI rasterization, psoas L+R merge, results-panel
  formatting, and offscreen multi-ROI list manipulation. Full suite
  **42/42 green**.
- **End-to-end smoke test against the real DICOM series**:
  L+R psoas ROIs at z=74 → combined area 9.00 cm²,
  IMAT 12.6 %, LDM 37.3 %, normal 50.1 %, myosteatosis = False.

### Changed
- `pyproject.toml`:
  - Version bumped `0.2.0` → `0.3.0`.
  - New `[gui]` extras (`PySide6>=6.6`), new `[all]` convenience extras.
  - `[notebook]` extras removed (Jupyter workflow deprecated).
  - New `fatanalyze-gui` console script.
- `notebooks/` directory **removed** — replaced by the native GUI.

### Notes
- The GUI delegates all analysis to the existing
  `fatanalyze.interactive.analyze` and `fatanalyze.analysis` modules — no
  new analysis logic, just a UI shell.
- Tested in `QT_QPA_PLATFORM=offscreen` mode; runs the same on a real
  display (no special config needed).
- Per-ROI CSV export schema: one row per ROI; ratio keys are prefixed
  `ratio_`; psoas-only fields (`imat_fraction`, `low_density_fraction`,
  `normal_muscle_fraction`, `myosteatosis_flag`) appear when the
  combined psoas entry is analyzed.

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
