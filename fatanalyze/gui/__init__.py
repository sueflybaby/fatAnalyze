"""Native PySide6 GUI for fatAnalyze — single-case CT, draw-and-analyze workflow.

The GUI is a thin layer on top of the existing analysis modules:

- :mod:`fatanalyze.io.dicom_loader` for DICOM ingestion
- :mod:`fatanalyze.interactive` for polygon rasterization + NIfTI sidecars
- :mod:`fatanalyze.interactive.analyze` for HU stats + clinical metrics
- :mod:`fatanalyze.viz.histogram_plot` for the embedded histogram

The GUI never re-implements analysis; it just wires UI events to those
already-tested functions.
"""
from __future__ import annotations

from fatanalyze.gui.app import FatAnalyzeWindow, main
from fatanalyze.gui.roi import ROI


__all__ = ["FatAnalyzeWindow", "ROI", "main"]
