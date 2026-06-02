# fatAnalyze

CT-based ectopic fat analysis for **liver (hepatic steatosis)**, **pancreas (pancreatic steatosis)**, and **psoas (myosteatosis at L3)**. Built around TotalSegmentator v2 with a single-case interactive workflow.

A native **PySide6 GUI** (`fatanalyze-gui`) is the primary interface for
single-case use: open a DICOM folder, navigate the volume slice-by-slice,
draw a polygon ROI on the slice you want, and the same histogram + clinical
metric pipeline runs on the drawn mask. A `fatanalyze` CLI entry point
remains for headless / batch use.

## What it does

1. Loads a DICOM series as a 3D HU volume
2. Runs TotalSegmentator (CPU-first, MPS fallback on Apple Silicon, CUDA on PC)
3. Extracts liver, pancreas, iliopsoas (left/right), and spleen masks
4. Detects the L3-approximate cross-section as the largest axial psoas slice
5. Computes HU histograms, range ratios, and clinical fat metrics
6. Visualizes: per-ROI histograms, segmentation overlays, single-page summary
7. **(v0.3.0)** A native GUI lets you draw freeform ROIs and re-use the same
   metric pipeline on user-drawn regions — no notebook required.

## Installation

```bash
# Clone (or cd into) the project
cd /Users/mac/dev/fatAnalyze

# Create venv (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Editable install of the local package + GUI
pip install -e ".[gui]"
# Or everything: pip install -e ".[all]"
```

> **Required step** — without `pip install -e .`, both the GUI and the CLI
> will fail with `ModuleNotFoundError`. Editable install registers the
> package via a `.pth` file in site-packages.

The TotalSegmentator model weights (~1-2 GB) are downloaded on first run. The
project stores them at `.cache/totalseg/` so the cache travels with the
project (do not commit; `.gitignore` excludes it). The same path is also
exposed via the `TOTALSEG_HOME_DIR` env var, which `fatanalyze.segment` sets
automatically.

## Apple Silicon notes

- PyTorch 2.12+ with `device=cpu` is the **default**. TotalSegmentator's 3D
  conv ops are unstable on the MPS backend (mid-2025) — CPU is more reliable.
- Use `fast=True` for ~2-3× speedup (slightly lower precision, fine for fat
  quantification).
- First segmentation on a real CT: ~5-12 min (CPU, fast). Subsequent runs hit
  the disk cache in <5 s.
- When moving to a PC with NVIDIA GPU: change `device: cpu` to `device: cuda`
  in `fatanalyze/config/targets.yaml`. No code changes required.

## GUI workflow (primary)

```bash
fatanalyze-gui
```

Then in the window:

1. **File → Open DICOM Folder** — pick a folder of `.dcm` files. The QC
   dialog shows slice count, voxel spacing, HU range, and any warnings.
2. **Pick a slice** with the slider (defaults to the middle slice).
3. **Pick a preset** in the toolbar: `iliopsoas_left`, `iliopsoas_right`,
   `liver`, `pancreas`, `spleen`, or `custom`.
4. **Toggle "Polygon: ON"** in the toolbar. **Left-click** on the slice to
   add vertices; **drag** a vertex to refine; **right-click** to remove the
   last vertex; **double-click** (or press the **Save ROI** button) to close
   the polygon. You'll be prompted for a name.
5. **Repeat** for the other psoas side (or any other organ) — each closed
   polygon becomes a row in the **ROI list** on the right.
6. Click **Analyze**. The right pane fills with a metrics table (HU stats,
   range ratios, clinical flags) for each ROI, plus a **psoas-combined**
   entry (L+R mask merged via `sitk.Or`) that drives the
   myosteatosis `IMAT / LDM / Normal` breakdown.
7. **File → Export CSV** writes one row per analyzed ROI to disk.

Mouse bindings outside polygon mode:

- **Middle-mouse drag** → pan
- **Wheel** → zoom
- **Right-mouse drag** → adjust W/L (right = wider, up = higher center)
- **W/L sliders** in the toolbar → numeric W/L
- **W/L preset dropdown** → Soft Tissue / Bone / Lung / Liver / Psoas
- **0** key → reset view to fit window

Supported presets are exactly the keys in `fatanalyze/config/targets.yaml`.
For psoas presets, myosteatosis metrics (IMAT, LDM, normal muscle, flag) are
populated using the L+R merged mask. The polygon is 2D and 1-slice thick, so
`volume_ml` reflects a 1-slice volume; use TotalSegmentator segmentation for
whole-organ 3D analysis.

## CLI workflow (headless / batch)

```bash
# Run the full pipeline on a DICOM series:
fatanalyze path/to/dicom_dir/ --out-dir fatanalyze-out/

# With caching and subset of targets:
fatanalyze path/to/dicom_dir/ \
    --cache-dir .cache/totalseg_runs \
    --targets liver pancreas iliopsoas_left iliopsoas_right
```

The same Python API is also available:

```python
from fatanalyze.io.dicom_loader import load_ct_series
from fatanalyze.segment.totalseg import segment
from fatanalyze.analysis.histogram import compute_ratios
from fatanalyze.analysis.fat_metrics import liver_spleen_ratio, psoas_imat_fraction

image, qc = load_ct_series("./ct_series/")
masks = segment(image, roi_names=["liver", "iliopsoas_left", "iliopsoas_right"])
# ... see notebooks/single_case.ipynb (historical) for the full script
```

## Plan B: MONAI instead of TotalSegmentator

If TotalSegmentator cannot run in your environment (Python version mismatch,
PyTorch incompatibility, model download blocked), replace the
`fatanalyze.segment.totalseg` module with a MONAI-based wrapper:

```python
# fatanalyze/segment/monai.py
from monai.bundle import download, load
from monai.inferers import SlidingWindowInferer
```

Pre-trained MONAI spleen / liver / pancreas models can be downloaded via
`monai.bundle.download(name="spleen_ct_segmentation", ...)`. The rest of the
pipeline (`roi/`, `analysis/`, `viz/`, `gui/`) is segmentation-engine agnostic
— it only needs `dict[name -> SimpleITK.Image]`.

## Configuration

All HU thresholds, label mappings, and segmenter settings live in
`fatanalyze/config/targets.yaml`. Edit and reload (or call
`fatanalyze.config.load_default_config.cache_clear()`).

## Project layout

```
fatanalyze/
  config/
    targets.yaml             HU thresholds, label maps
  io/
    dicom_loader.py          DICOM -> SimpleITK + QC
  segment/
    totalseg.py              TotalSegmentator wrapper + disk cache
  roi/
    extractor.py             mask post-processing + L3-approx
  analysis/
    histogram.py             HU histogram + range ratios
    fat_metrics.py           clinical fat indicators
  viz/
    histogram_plot.py
    overlay.py
    summary.py
  interactive/                # user-drawn ROIs (v0.2.0)
    polygon_utils.py
    polygon_drawer.py
    user_roi.py
    analyze.py
    viz.py
  gui/                        # native PySide6 GUI (v0.3.0)
    app.py                   # QMainWindow, signal/slot wiring, entry point
    slice_view.py            # QGraphicsView: CT slice + W/L + pan/zoom
    polygon_item.py          # editable QGraphicsPolygonItem
    controls.py              # toolbar (preset, W/L, draw, save, analyze)
    roi_list.py              # multi-ROI list with rename/delete
    metrics_runner.py        # wraps analyze_user_roi + psoas L+R merge
    results_panel.py         # text table + embedded histogram
    roi.py                   # ROI dataclass
tests/
  test_io_qc.py
  test_histogram.py
  test_l3_detection.py
  test_user_roi.py
  test_gui.py                # v0.3.0 — offscreen Qt smoke tests
```

## License

MIT

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
