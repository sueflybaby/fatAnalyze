# fatAnalyze

CT-based ectopic fat analysis for **liver (hepatic steatosis)**, **pancreas (pancreatic steatosis)**, and **psoas (myosteatosis at L3)**. Built around TotalSegmentator v2 with a single-case interactive workflow.

## What it does

1. Loads a DICOM series as a 3D HU volume
2. Runs TotalSegmentator (CPU-first, MPS fallback on Apple Silicon, CUDA on PC)
3. Extracts liver, pancreas, iliopsoas (left/right), and spleen masks
4. Detects the L3-approximate cross-section as the largest axial psoas slice
5. Computes HU histograms, range ratios, and clinical fat metrics
6. Visualizes: per-ROI histograms, segmentation overlays, single-page summary

## Installation

```bash
# Clone (or cd into) the project
cd /Users/mac/dev/fatAnalyze

# Create venv (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Editable install of the local package + notebook extras
pip install -e ".[notebook]"
```

> **Required step** — without `pip install -e .`, `import fatanalyze` will
> raise `ModuleNotFoundError` in Jupyter. Editable install registers the
> package via a `.pth` file in site-packages.

The TotalSegmentator model weights (~1-2 GB) are downloaded on first run. The project
stores them at `.cache/totalseg/` so the cache travels with the project (do not
commit; `.gitignore` excludes it). The same path is also exposed via the
`TOTALSEG_HOME_DIR` env var, which `fatanalyze.segment` sets automatically.

## Apple Silicon notes

- PyTorch 2.12+ with `device=cpu` is the **default**. TotalSegmentator's 3D conv
  ops are unstable on the MPS backend (mid-2025) — CPU is more reliable.
- Use `fast=True` for ~2-3× speedup (slightly lower precision, fine for fat
  quantification).
- First segmentation on a real CT: ~5-12 min (CPU, fast). Subsequent runs hit
  the disk cache in <5 s.
- When moving to a PC with NVIDIA GPU: change `device: cpu` to `device: cuda`
  in `fatanalyze/config/targets.yaml`. No code changes required.

## Quick start

```python
from fatanalyze.io.dicom_loader import load_ct_series
from fatanalyze.segment.totalseg import segment
from fatanalyze.roi.extractor import (
    get_3d_mask, find_l3_slice, get_l3_psoas_mask, extract_hu,
)
from fatanalyze.analysis.histogram import compute_ratios
from fatanalyze.analysis.fat_metrics import (
    liver_spleen_ratio, pancreas_spleen_difference, psoas_imat_fraction,
)
from fatanalyze.viz.histogram_plot import plot_histogram
from fatanalyze.viz.overlay import plot_overlay

# 1. Load
image, qc = load_ct_series("./ct_series/")
print("QC:", qc.summary())

# 2. Segment
masks = segment(
    image,
    roi_names=["liver", "pancreas", "spleen", "iliopsoas_left", "iliopsoas_right"],
    cache_dir="./.cache/totalseg_runs",
)

# 3. 3D masks for liver, pancreas, spleen
liver = get_3d_mask(masks["liver"])
pancreas = get_3d_mask(masks["pancreas"])
spleen = get_3d_mask(masks["spleen"])

# 4. L3-approx slice for psoas
z = find_l3_slice(masks["iliopsoas_left"], masks["iliopsoas_right"])
psoas_l3 = get_l3_psoas_mask(
    masks["iliopsoas_left"], masks["iliopsoas_right"], z, n_buffer=1,
)

# 5. HU extraction
liver_hu = extract_hu(image, liver)
pancreas_hu = extract_hu(image, pancreas)
spleen_hu = extract_hu(image, spleen)
psoas_hu = extract_hu(image, psoas_l3)

# 6. Histogram + clinical metrics
liver_stats = compute_ratios(liver_hu, target_name="liver", spacing=image.GetSpacing())
lsr = liver_spleen_ratio(liver_stats.mean_hu, compute_ratios(spleen_hu, "spleen").mean_hu)
print(f"Liver-spleen ratio: {lsr:.2f}")

# 7. Visualize
plot_histogram(liver_hu, target_name="liver", title="Liver")
plot_overlay(image, liver, slice_index=z, title="Liver (L3-approx slice)")
```

The notebook `notebooks/single_case.ipynb` walks through the same flow cell by
cell with all visualizations inline.

## User-drawn ROIs

Beyond TotalSegmentator's auto-segmentation, you can draw a freehand polygon
on any axial slice and run the same histogram + clinical-metric pipeline on
it. Useful for:

- Spot-checking TS results (does the auto-mask capture the right region?)
- Analysing structures TS missed or got wrong
- Custom anatomical regions not in the standard preset list

The drawer uses `matplotlib.widgets.PolygonSelector` (no extra deps):

```python
from fatanalyze.interactive import draw_roi_2d, analyze_user_roi, plot_user_roi

# 1. Open the interactive drawer on a chosen axial slice
roi = draw_roi_2d(
    image,
    z_index=74,                  # axial slice to draw on
    name="psoas_L3_manual",
    preset="iliopsoas_left",     # reuse existing HU ranges + clinical flags
)
# Click to add polygon vertices, drag to move, click "Finish" to close.

# 2. Same pipeline as TS: histogram, range ratios, clinical flags
result = analyze_user_roi(image, roi)
print(f"area={result['area_cm2']:.1f} cm^2  mean_hu={result['mean_hu']:.1f}")
print(f"ratios={result['ratios']}")
if result['psoas_metrics']:
    print(f"myosteatosis={result['psoas_metrics']['myosteatosis_flag']}")

# 3. Two-panel visualization
fig = plot_user_roi(image, roi, analysis=result)
fig.savefig("user_roi_psoas_L3_manual.png", dpi=120, bbox_inches="tight")

# 4. Persist the mask + metadata for later re-use / batch reprocess
roi.save_nifti("user_roi_psoas_L3_manual.nii.gz")
```

Supported presets are exactly the keys in `fatanalyze/config/targets.yaml`
(`liver`, `pancreas`, `spleen`, `iliopsoas_left`, `iliopsoas_right`).
Psoas presets additionally populate `psoas_metrics` (IMAT/LDM fractions +
myosteatosis flag). The polygon is 2D and 1-slice thick, so `volume_ml`
reflects a 1-slice volume; use TS segmentation for whole-organ 3D analysis.

The interactive cell block is also embedded in
`notebooks/single_case.ipynb` (block 7). It is wrapped in `try/except` so
headless / `nbconvert` runs skip it automatically.

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
pipeline (`roi/`, `analysis/`, `viz/`) is segmentation-engine agnostic — it
only needs `dict[name -> SimpleITK.Image]`.

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
notebooks/
  single_case.ipynb
tests/
  test_io_qc.py
  test_histogram.py
  test_l3_detection.py
  test_user_roi.py           # v0.2.0
```

## License

MIT

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
