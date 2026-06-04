"""Results panel: text table for HU stats + embedded matplotlib histogram."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from fatanalyze.modality import Modality

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


class ResultsPanel(QWidget):
    """Two-pane display: a text table of metrics + a histogram canvas.

    Modality-aware: shows HU stats for CT, FF% stats for MR.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._modality: Modality = Modality.CT
        self._fig = Figure(figsize=(5, 3), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax = self._fig.add_subplot(111)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        font = self.text.font()
        font.setFamily("Menlo")
        self.text.setFont(font)
        self.text.setPlaceholderText(self.tr("No ROIs analyzed yet.") + "\n\n"
                                      "1. " + self.tr("Open a DICOM folder") + "\n"
                                      "2. " + self.tr("Pick a slice with the slider") + "\n"
                                      "3. " + self.tr("Click 'ROI Draw: OFF' to enable drawing") + "\n"
                                      "4. " + self.tr("Left-click to add vertices, double-click to close") + "\n"
                                      "5. " + self.tr("Click 'Analyze' to compute metrics"))

        self._header = QLabel(self.tr("Results"))
        self._header.setStyleSheet("font-weight: bold; font-size: 14px; padding: 4px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._header)
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.text)
        splitter.addWidget(self._canvas)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        self._current_result: Optional[Dict[str, Any]] = None
        self.clear()

    # -- public API ----------------------------------------------------

    def clear(self) -> None:
        self._current_result = None
        self._modality = Modality.CT
        self.text.clear()
        self._ax.clear()
        self._ax.set_title("Histogram (no data)")
        self._ax.set_xlabel("HU")
        self._ax.set_ylabel("Voxel count")
        self._canvas.draw_idle()

    def show_result(self, name: str, result: Dict[str, Any]) -> None:
        self._current_result = result
        self.text.setPlainText(self._format_metrics(name, result))
        self._draw_histogram(result)

    def show_all(
        self,
        results: Dict[str, Dict[str, Any]],
        modality: Modality = Modality.CT,
    ) -> None:
        self._modality = modality
        if not results:
            self.clear()
            return
        is_mr = modality == Modality.MR
        if is_mr:
            lines = [f"{'name':<22} {'preset':<18} {'area':>8} {'vol':>7} "
                     f"{'meanFF':>7} {'medFF':>7}  flags"]
            lines.append("-" * 88)
            for name, r in results.items():
                flags = ",".join(r.get("clinical_flags") or []) or "-"
                lines.append(
                    f"{name[:22]:<22} {r.get('target','?'):<18} "
                    f"{r.get('area_cm2', 0):>7.2f}c {r.get('volume_ml', 0):>6.2f}m "
                    f"{r.get('mean_ff', float('nan')):>7.1f} "
                    f"{r.get('median_ff', float('nan')):>7.1f}  {flags[:14]}"
                )
        else:
            lines = [f"{'name':<22} {'preset':<18} {'area':>8} {'vol':>7} "
                     f"{'mean':>7} {'median':>7}  flags"]
            lines.append("-" * 88)
            for name, r in results.items():
                flags = ",".join(r.get("clinical_flags") or []) or "-"
                lines.append(
                    f"{name[:22]:<22} {r.get('target','?'):<18} "
                    f"{r.get('area_cm2', 0):>7.2f}c {r.get('volume_ml', 0):>6.2f}m "
                    f"{r.get('mean_hu', float('nan')):>7.1f} "
                    f"{r.get('median_hu', float('nan')):>7.1f}  {flags[:14]}"
                )
                if r.get("psoas_metrics"):
                    pm = r["psoas_metrics"]
                    lines.append(
                        f"  └─ IMAT {pm.get('imat_fraction', 0)*100:5.1f}%   "
                        f"LDM {pm.get('low_density_fraction', 0)*100:5.1f}%   "
                        f"Normal {pm.get('normal_muscle_fraction', 0)*100:5.1f}%   "
                        f"myosteatosis={pm.get('myosteatosis_flag', False)}"
                    )
        self.text.setPlainText("\n".join(lines))
        first = next(iter(results.values()))
        self._draw_histogram(first)

    # -- internals -----------------------------------------------------

    def _draw_histogram(self, result: Dict[str, Any]) -> None:
        self._ax.clear()
        is_mr = self._modality == Modality.MR
        hr = result.get("histogram_result")

        if hr is None:
            self._ax.set_title("Histogram (empty)")
            self._ax.set_xlabel("Fat Fraction (%)" if is_mr else "HU")
            self._ax.set_ylabel("Voxel count")
            self._canvas.draw_idle()
            return

        if is_mr:
            n_voxels = hr.get("n_voxels", 0) if isinstance(hr, dict) else getattr(hr, "n_voxels", 0)
            if n_voxels == 0:
                self._ax.set_title("Histogram (empty)")
                self._ax.set_xlabel("Fat Fraction (%)")
                self._ax.set_ylabel("Voxel count")
                self._canvas.draw_idle()
                return
            hist_data = hr.get("histogram", {}) if isinstance(hr, dict) else {}
            bin_centers = np.asarray(hist_data.get("bin_centers", []), dtype=float)
            counts = np.asarray(hist_data.get("counts", []), dtype=float)
            if len(bin_centers) >= 2 and len(counts) == len(bin_centers):
                width = float(bin_centers[1] - bin_centers[0])
            else:
                width = 1.0
            self._ax.bar(bin_centers, counts, width=width,
                         color="#888", edgecolor="black", linewidth=0.3)
            mean_ff = result.get("mean_ff", float("nan"))
            median_ff = result.get("median_ff", float("nan"))
            if not np.isnan(mean_ff):
                self._ax.axvline(mean_ff, color="blue", linestyle="-", linewidth=1.0,
                                 label=f"mean={mean_ff:.1f}%")
            if not np.isnan(median_ff):
                self._ax.axvline(median_ff, color="green", linestyle="-", linewidth=1.0,
                                 label=f"median={median_ff:.1f}%")
            title = f"Fat Fraction — {result.get('name', '?')} ({n_voxels} voxels)"
            self._ax.set_title(title)
            self._ax.set_xlabel("Fat Fraction (%)")
            self._ax.set_ylabel("Voxel count")
            self._ax.legend(loc="upper right", fontsize=8)
        else:
            from fatanalyze.analysis.histogram import HistogramResult
            if isinstance(hr, HistogramResult):
                if hr.n_voxels == 0:
                    self._ax.set_title("Histogram (empty)")
                    self._ax.set_xlabel("HU")
                    self._ax.set_ylabel("Voxel count")
                    self._canvas.draw_idle()
                    return
                bin_centers = np.asarray(hr.histogram.get("bin_centers", []), dtype=float)
                counts = np.asarray(hr.histogram.get("counts", []), dtype=float)
                if len(bin_centers) >= 2 and len(counts) == len(bin_centers):
                    width = float(bin_centers[1] - bin_centers[0])
                else:
                    width = 1.0
                self._ax.bar(bin_centers, counts, width=width,
                             color="#888", edgecolor="black", linewidth=0.3)
                thresholds = result.get("ratios_thresholds") or {}
                for label, hu in thresholds.items():
                    self._ax.axvline(hu, color="red", linestyle="--", linewidth=0.8)
                mean_hu = result.get("mean_hu", float("nan"))
                median_hu = result.get("median_hu", float("nan"))
                if not np.isnan(mean_hu):
                    self._ax.axvline(mean_hu, color="blue", linestyle="-", linewidth=1.0,
                                     label=f"mean={mean_hu:.1f}")
                if not np.isnan(median_hu):
                    self._ax.axvline(median_hu, color="green", linestyle="-", linewidth=1.0,
                                     label=f"median={median_hu:.1f}")
                title = f"Histogram — {result.get('name', '?')} ({hr.n_voxels} voxels)"
                self._ax.set_title(title)
                self._ax.set_xlabel("HU")
                self._ax.set_ylabel("Voxel count")
                self._ax.legend(loc="upper right", fontsize=8)
        self._canvas.draw_idle()

    def _format_metrics(self, name: str, r: Dict[str, Any]) -> str:
        is_mr = self._modality == Modality.MR
        lines = [
            f"ROI: {name}    (preset: {r.get('target', '?')})",
            "-" * 56,
            f"  n_voxels : {r.get('n_voxels', 0):>10}",
            f"  area     : {r.get('area_cm2', 0):>9.2f} cm²",
            f"  volume   : {r.get('volume_ml', 0):>9.2f} ml",
        ]
        if is_mr:
            lines += [
                f"  mean FF% : {r.get('mean_ff', float('nan')):>10.1f}",
                f"  median FF: {r.get('median_ff', float('nan')):>10.1f}",
                f"  std FF%  : {r.get('std_ff', float('nan')):>10.1f}",
                f"  p05 / p95: {r.get('p05_ff', float('nan')):>6.1f}% / "
                f"{r.get('p95_ff', float('nan')):>6.1f}%",
                "",
                "  " + self.tr("Fat Fraction Bins:") + ":",
            ]
            bins = r.get("ff_bins") or {}
            if not bins:
                lines.append("    " + self.tr("(none)"))
            for k, v in bins.items():
                try:
                    lines.append(f"    {k:<24} {v*100:6.2f}%")
                except (TypeError, ValueError):
                    lines.append(f"    {k:<24} {v}")
        else:
            lines += [
                f"  mean HU  : {r.get('mean_hu', float('nan')):>10.1f}",
                f"  median HU: {r.get('median_hu', float('nan')):>10.1f}",
                f"  std HU   : {r.get('std_hu', float('nan')):>10.1f}",
                f"  p05 / p95: {r.get('p05_hu', float('nan')):>6.1f} / "
                f"{r.get('p95_hu', float('nan')):>6.1f}",
                "",
                "  " + self.tr("Ratios:") + ":",
            ]
            ratios = r.get("ratios") or {}
            if not ratios:
                lines.append("    " + self.tr("(none)"))
            for k, v in ratios.items():
                try:
                    lines.append(f"    {k:<24} {v*100:6.2f}%")
                except (TypeError, ValueError):
                    lines.append(f"    {k:<24} {v}")
            pm = r.get("psoas_metrics")
            if pm:
                lines.append("")
                lines.append("  " + self.tr("Psoas / myosteatosis:") + ":")
                lines.append(f"    IMAT fraction     : {pm.get('imat_fraction', 0)*100:6.2f}%")
                lines.append(f"    Low-density (LDM) : {pm.get('low_density_fraction', 0)*100:6.2f}%")
                lines.append(f"    Normal muscle     : {pm.get('normal_muscle_fraction', 0)*100:6.2f}%")
                lines.append(f"    {self.tr('Myosteatosis flag')} : {pm.get('myosteatosis_flag', False)}")
        flags = r.get("clinical_flags") or []
        if flags:
            lines.append("")
            lines.append("  " + self.tr("Clinical flags:") + ": " + ', '.join(flags))
        return "\n".join(lines)

    def retranslate(self) -> None:
        """Re-apply tr() to every visible widget after the locale changes."""
        self._header.setText(self.tr("Results"))
        self.text.setPlaceholderText(self.tr("No ROIs analyzed yet.") + "\n\n"
                                      "1. " + self.tr("Open a DICOM folder") + "\n"
                                      "2. " + self.tr("Pick a slice with the slider") + "\n"
                                      "3. " + self.tr("Click 'ROI Draw: OFF' to enable drawing") + "\n"
                                      "4. " + self.tr("Left-click to add vertices, double-click to close") + "\n"
                                      "5. " + self.tr("Click 'Analyze' to compute metrics"))
        if self._current_result:
            pass
        else:
            is_mr = self._modality == Modality.MR
            self._ax.clear()
            self._ax.set_title("Histogram (no data)")
            self._ax.set_xlabel("Fat Fraction (%)" if is_mr else "HU")
            self._ax.set_ylabel("Voxel count")
            self._canvas.draw_idle()


__all__ = ["ResultsPanel"]
