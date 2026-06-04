"""Main window: wires slice view, controls, ROI list, and results panel."""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import SimpleITK as sitk

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from fatanalyze.config import load_mr_presets
from fatanalyze.modality import Modality
from fatanalyze.gui.controls import ControlsBar
from fatanalyze.gui.control_panel import ControlPanel
from fatanalyze.gui.i18n import install_locale, current_locale, SUPPORTED_LOCALES, reset_for_test
from fatanalyze.gui.metrics_runner import compute_for_rois, compute_for_rois_mr, rasterize
from fatanalyze.gui.polygon_item import PolygonItem
from fatanalyze.gui.results_panel import ResultsPanel
from fatanalyze.gui.roi import ROI
from fatanalyze.gui.roi_list import ROIListWidget
from fatanalyze.gui.slice_view import SliceView
from fatanalyze.io.dicom_loader import load_ct_series, load_mr_series
from fatanalyze.interactive.user_roi import UserROI


PRESET_COLORS: Dict[str, QColor] = {
    "iliopsoas_left":  QColor(255,  80,  80),
    "iliopsoas_right": QColor( 80,  80, 255),
    "liver":           QColor( 80, 200,  80),
    "pancreas":        QColor(220, 180,  60),
    "spleen":          QColor(200,  80, 200),
    "custom":          QColor(180, 180, 180),
}


class FatAnalyzeWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(self.tr("fatAnalyze"))
        self.resize(1280, 800)

        self._image: Optional[sitk.Image] = None
        self._qc = None
        self._modality: Modality = Modality.CT
        self._active_polygon: Optional[PolygonItem] = None
        self._polygons_by_name: Dict[str, PolygonItem] = {}
        self._results: Dict[str, dict] = {}
        self._menu_actions: Dict[str, QAction] = {}

        self._build_ui()
        self._wire_signals()

    # -- UI scaffolding -----------------------------------------------

    def _build_ui(self) -> None:
        self.controls = ControlsBar(self)
        self.addToolBar(Qt.TopToolBarArea, self.controls)

        self._build_menu()

        central = QWidget()
        self.setCentralWidget(central)
        hsplit = QSplitter(Qt.Horizontal)

        # --- Left side: control panel (Display + ROI) ---
        self.panel = ControlPanel(self)
        hsplit.addWidget(self.panel)

        # --- Center: slice view + slider ---
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        self.slice_view = SliceView(self)
        center_layout.addWidget(self.slice_view, 1)

        slider_row = QHBoxLayout()
        self._slice_label_label = QLabel(self.tr("Slice:"))
        slider_row.addWidget(self._slice_label_label)
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setRange(0, 0)
        self.slice_slider.setEnabled(False)
        slider_row.addWidget(self.slice_slider, 1)
        self.slice_label = QLabel("— / —")
        slider_row.addWidget(self.slice_label)
        center_layout.addLayout(slider_row)

        hsplit.addWidget(center)

        # --- Right: ROI list + results ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.roi_list = ROIListWidget(self)
        right_layout.addWidget(self.roi_list, 1)

        self.results = ResultsPanel(self)
        right_layout.addWidget(self.results, 2)

        hsplit.addWidget(right)
        hsplit.setStretchFactor(0, 0)   # left panel: fixed-ish
        hsplit.setStretchFactor(1, 2)   # slice view: dominant
        hsplit.setStretchFactor(2, 1)   # right: secondary
        hsplit.setSizes([230, 720, 400])
        hsplit.setCollapsible(0, False)
        hsplit.setCollapsible(2, False)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(hsplit)

        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage(self.tr("Open a DICOM folder to begin."))
        self.slice_view.pixel_hovered.connect(self._on_pixel_hovered)

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        self._file_menu = menubar.addMenu(self.tr("&File"))

        act = QAction(self.tr("Open DICOM Folder…"), self)
        act.setShortcut(QKeySequence.Open)
        act.triggered.connect(self._on_open_folder)
        self._file_menu.addAction(act)
        self._menu_actions["open"] = act

        act = QAction(self.tr("Export CSV…"), self)
        act.triggered.connect(self._on_export_csv)
        self._file_menu.addAction(act)
        self._menu_actions["export"] = act

        self._file_menu.addSeparator()
        act = QAction(self.tr("Quit"), self)
        act.setShortcut(QKeySequence.Quit)
        act.triggered.connect(self.close)
        self._file_menu.addAction(act)
        self._menu_actions["quit"] = act

        self._analysis_menu = menubar.addMenu(self.tr("&Analysis"))
        act = QAction(self.tr("Run Analyze"), self)
        act.setShortcut("Ctrl+R")
        act.triggered.connect(self._on_analyze)
        self._analysis_menu.addAction(act)
        self._menu_actions["run_analyze"] = act

        self._help_menu = menubar.addMenu(self.tr("&Help"))
        act = QAction(self.tr("About fatAnalyze"), self)
        act.triggered.connect(self._on_about)
        self._help_menu.addAction(act)
        self._menu_actions["about"] = act

    def _wire_signals(self) -> None:
        # --- Top toolbar (high-level actions) ---
        self.controls.open_folder_requested.connect(self._on_open_folder)
        self.controls.analyze_requested.connect(self._on_analyze)
        self.controls.export_csv_requested.connect(self._on_export_csv)
        self.controls.language_changed.connect(self._on_language_changed)
        self.controls.modality_changed.connect(self._on_modality_changed)
        # --- Side panel (Display + ROI) ---
        self.panel.preset_changed.connect(self._on_preset_changed)
        self.panel.window_level_changed.connect(self.slice_view.set_window_level)
        self.panel.wl_preset_changed.connect(self.slice_view.apply_wl_preset)
        self.panel.draw_toggle_requested.connect(self._on_draw_toggled)
        self.panel.clear_roi_requested.connect(self._on_clear_polygon)
        self.panel.save_roi_requested.connect(self._on_save_polygon)
        self.panel.mr_preset_changed.connect(self._on_mr_preset_changed)
        # --- View <-> state ---
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        self.slice_view.slice_changed.connect(self._on_view_slice_changed)
        self.roi_list.roi_selected.connect(self._on_roi_selected)
        self.slice_view.polygon_closed.connect(self._on_save_polygon)

    # -- Language switching --------------------------------------------

    def _on_modality_changed(self, mod: str) -> None:
        self._modality = Modality(mod)
        # Tell the side panel to swap W/L sliders and show/hide the
        # appropriate preset combo (CT W/L Preset vs MR vendor Preset).
        self.panel.set_modality(mod)
        # Clear image when switching modality
        self._image = None
        self._qc = None
        self._results.clear()
        self.results.clear()
        self.roi_list.clear()
        self._polygons_by_name.clear()
        self._active_polygon = None
        self.slice_slider.setRange(0, 0)
        self.slice_slider.setEnabled(False)
        self.slice_label.setText(self.tr("— / —"))
        self.statusBar().showMessage(
            self.tr("Switched to {mode} mode. Open a folder to begin.").format(
                mode="MR" if mod == "mr" else "CT"
            )
        )

    def _on_language_changed(self, locale: str) -> None:
        install_locale(QApplication.instance(), locale)
        self.retranslate()

    def _on_mr_preset_changed(self, preset_name: str) -> None:
        """The MR vendor preset affects the next MR folder load."""
        if self._modality == Modality.MR and self._image is not None:
            self.statusBar().showMessage(
                self.tr("MR preset set to '{p}'. Reopen folder to apply.").format(
                    p=preset_name
                ), 4000,
            )
        else:
            self.statusBar().showMessage(
                self.tr("MR preset set to '{p}'.").format(p=preset_name), 2000,
            )

    def retranslate(self) -> None:
        self.setWindowTitle(self.tr("fatAnalyze"))
        self._file_menu.setTitle(self.tr("&File"))
        self._analysis_menu.setTitle(self.tr("&Analysis"))
        self._help_menu.setTitle(self.tr("&Help"))
        self._menu_actions["open"].setText(self.tr("Open DICOM Folder…"))
        self._menu_actions["export"].setText(self.tr("Export CSV…"))
        self._menu_actions["quit"].setText(self.tr("Quit"))
        self._menu_actions["run_analyze"].setText(self.tr("Run Analyze"))
        self._menu_actions["about"].setText(self.tr("About fatAnalyze"))
        self._slice_label_label.setText(self.tr("Slice:"))
        self.statusBar().showMessage(self.tr("Open a DICOM folder to begin."))
        self.controls.retranslate()
        self.panel.retranslate()
        self.roi_list.retranslate()
        self.results.retranslate()

    # -- slots ---------------------------------------------------------

    def _on_open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, self.tr("Open DICOM Folder"), str(Path.cwd()),
        )
        if not folder:
            return
        try:
            if self._modality == Modality.MR:
                preset_cfg = None
                try:
                    from fatanalyze.config import load_mr_presets
                    mr_presets = load_mr_presets()
                    mr_preset_name = self.panel.current_mr_preset()
                    preset_cfg = mr_presets.get("presets", {}).get(mr_preset_name, {})
                except Exception:
                    pass
                image, qc = load_mr_series(Path(folder), preset_cfg)
            else:
                image, qc = load_ct_series(Path(folder))
        except Exception as exc:
            QMessageBox.critical(self, self.tr("Load failed"), str(exc))
            return
        self._image = image
        self._qc = qc
        self.slice_view.set_image(image)
        self.slice_slider.setRange(0, image.GetDepth() - 1)
        self.slice_slider.setValue(image.GetDepth() // 2)
        self.slice_slider.setEnabled(True)
        self.slice_label.setText(f"{self.slice_slider.value()+1} / {image.GetDepth()}")
        if self._modality == Modality.MR:
            self.slice_view.set_window_level(100.0, 50.0)
            self.panel.set_wl_sliders(100, 50)
        else:
            w, l = self.slice_view.get_window_level()
            self.panel.set_wl_sliders(w, l)
        self._active_polygon = None
        self._polygons_by_name.clear()
        self.roi_list.clear()
        self._results.clear()
        self.results.clear()
        QMessageBox.information(
            self, self.tr("DICOM QC"),
            qc.summary() if hasattr(qc, "summary") else str(qc),
        )
        self.statusBar().showMessage(
            self.tr("Loaded {n} slices from {folder}").format(
                n=image.GetDepth(), folder=folder
            ), 5000,
        )

    def _on_preset_changed(self, preset: str) -> None:
        if self._active_polygon is not None and self._active_polygon.vertex_count() == 0:
            self._active_polygon.set_color(PRESET_COLORS.get(preset, QColor(180, 180, 180)))

    def _on_slice_changed(self, z: int) -> None:
        if self._image is None:
            return
        self.slice_view.set_slice(z)
        depth = self._image.GetDepth()
        self.slice_label.setText(f"{z+1} / {depth}")

    def _on_view_slice_changed(self, z: int) -> None:
        """Sync the slider when the view scrolls via mouse wheel."""
        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(z)
        self.slice_slider.blockSignals(False)
        depth = self._image.GetDepth()
        self.slice_label.setText(f"{z+1} / {depth}")

    def _on_draw_toggled(self, on: bool) -> None:
        if not on:
            if self._active_polygon is not None and self._active_polygon.vertex_count() < 3:
                self._active_polygon.clear()
                self.slice_view._scene.removeItem(self._active_polygon)
                self._active_polygon = None
            self.slice_view.polygon_mode = False
            self.slice_view.active_polygon = None
            return
        if self._image is None:
            QMessageBox.warning(self, self.tr("No image"),
                                self.tr("Open a DICOM folder first."))
            self.panel.draw_btn.setChecked(False)
            return
        preset = self.panel.current_preset()
        color = PRESET_COLORS.get(preset, QColor(180, 180, 180))
        self._active_polygon = PolygonItem(color=color)
        self.slice_view._scene.addItem(self._active_polygon)
        self.slice_view.active_polygon = self._active_polygon
        self.slice_view.polygon_mode = True
        self.statusBar().showMessage(
            self.tr("ROI drawing ON (preset: {preset}). "
                    "Left-click to add vertices, double-click to close.").format(
                preset=preset
            ),
        )

    def _on_clear_polygon(self) -> None:
        if self._active_polygon is None:
            return
        self._active_polygon.clear()
        self.statusBar().showMessage(self.tr("ROI cleared."), 2000)

    def _on_save_polygon(self) -> None:
        if self._active_polygon is None or self._active_polygon.vertex_count() < 3:
            QMessageBox.information(self, self.tr("Save ROI"),
                                    self.tr("Draw at least 3 vertices first."))
            return
        self._active_polygon.close()
        preset = self.panel.current_preset()
        default_name = f"{preset}"
        name, ok = QInputDialog.getText(self, self.tr("Save ROI"),
                                        self.tr("ROI name:"), text=default_name)
        if not ok or not name.strip():
            return
        name = name.strip()
        z = self.slice_view.z_index
        roi = ROI(name=name, preset=preset, z_index=z,
                  vertices=self._active_polygon.get_vertices())
        self.roi_list.add_roi(roi)
        self._polygons_by_name[name] = self._active_polygon
        self._active_polygon = None
        self.panel.draw_btn.setChecked(False)
        self.statusBar().showMessage(
            self.tr("ROI '{name}' added ({n} vertices).").format(
                name=name, n=len(roi.vertices)
            ), 4000,
        )

    def _on_roi_selected(self, roi: ROI) -> None:
        if roi.name in self._results:
            self.results.show_result(roi.name, self._results[roi.name])

    def _on_analyze(self) -> None:
        if self._image is None:
            QMessageBox.warning(self, self.tr("No image"),
                                self.tr("Open a DICOM folder first."))
            return
        rois = self.roi_list.get_rois()
        if not rois:
            QMessageBox.information(self, self.tr("No ROIs"),
                                    self.tr("Draw at least one ROI first."))
            return
        try:
            if self._modality == Modality.MR:
                self._results = compute_for_rois_mr(self._image, rois)
            else:
                self._results = compute_for_rois(self._image, rois)
        except Exception as exc:
            QMessageBox.critical(self, self.tr("Analyze failed"), str(exc))
            return
        for name in self._results:
            self.roi_list.mark_analyzed(name)
        self.results.show_all(self._results, self._modality)
        self.statusBar().showMessage(
            self.tr("Analyzed {n} ROI(s).").format(n=len(self._results)), 4000,
        )

    def _on_export_csv(self) -> None:
        if not self._results:
            QMessageBox.information(self, self.tr("No results"),
                                    self.tr("Click 'Analyze' first."))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, self.tr("Export metrics to CSV"), "fatAnalyze-metrics.csv",
            "CSV files (*.csv)",
        )
        if not path:
            return
        is_mr = self._modality == Modality.MR
        rows = []
        for name, r in self._results.items():
            if is_mr:
                row = {
                    "name": name,
                    "target": r.get("target", ""),
                    "n_voxels": r.get("n_voxels", 0),
                    "area_cm2": f"{r.get('area_cm2', 0):.4f}",
                    "volume_ml": f"{r.get('volume_ml', 0):.4f}",
                    "mean_ff": f"{r.get('mean_ff', float('nan')):.2f}",
                    "median_ff": f"{r.get('median_ff', float('nan')):.2f}",
                    "std_ff": f"{r.get('std_ff', float('nan')):.2f}",
                    "p05_ff": f"{r.get('p05_ff', float('nan')):.2f}",
                    "p95_ff": f"{r.get('p95_ff', float('nan')):.2f}",
                    "clinical_flags": ";".join(r.get("clinical_flags") or []),
                }
                for k, v in (r.get("ff_bins") or {}).items():
                    row[f"ffbin_{k}"] = f"{v*100:.2f}%" if isinstance(v, (int, float)) else v
            else:
                row = {
                    "name": name,
                    "target": r.get("target", ""),
                    "n_voxels": r.get("n_voxels", 0),
                    "area_cm2": f"{r.get('area_cm2', 0):.4f}",
                    "volume_ml": f"{r.get('volume_ml', 0):.4f}",
                    "mean_hu": f"{r.get('mean_hu', float('nan')):.2f}",
                    "median_hu": f"{r.get('median_hu', float('nan')):.2f}",
                    "std_hu": f"{r.get('std_hu', float('nan')):.2f}",
                    "p05_hu": f"{r.get('p05_hu', float('nan')):.2f}",
                    "p95_hu": f"{r.get('p95_hu', float('nan')):.2f}",
                    "clinical_flags": ";".join(r.get("clinical_flags") or []),
                }
                for k, v in (r.get("ratios") or {}).items():
                    row[f"ratio_{k}"] = f"{v:.4f}" if isinstance(v, (int, float)) else v
                pm = r.get("psoas_metrics")
                if pm:
                    row["imat_fraction"] = f"{pm.get('imat_fraction', 0):.4f}"
                    row["low_density_fraction"] = f"{pm.get('low_density_fraction', 0):.4f}"
                    row["normal_muscle_fraction"] = f"{pm.get('normal_muscle_fraction', 0):.4f}"
                    row["myosteatosis_flag"] = pm.get("myosteatosis_flag", False)
            rows.append(row)
        if rows:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        for name in self._results:
            self.roi_list.mark_exported(name)
        self.statusBar().showMessage(
            self.tr("Exported {n} rows to {path}").format(n=len(rows), path=path), 5000,
        )

    def _on_about(self) -> None:
        QMessageBox.about(
            self, self.tr("About fatAnalyze"),
            "<b>fatAnalyze v0.4.0</b><br>"
            + self.tr("CT ectopic-fat analysis (liver, pancreas, psoas at L3).") + "<br>"
            + self.tr("Native PySide6 GUI; the analysis pipeline is unchanged.") + "<br><br>"
            + self.tr("DICOM → polygon ROI → HU stats + clinical metrics.")
            + "<br><br>"
            + self.tr("MR PDFF/Dixon fat fraction support (FF% stats + steatosis grading)."),
        )

    def _on_pixel_hovered(self, x: int, y: int, val: float) -> None:
        if self._image is None:
            return
        if val != val:
            return
        if self._modality == Modality.MR:
            self.statusBar().showMessage(
                f"x={x} y={y}  FF%={val:.1f}  "
                f"z={self.slice_view.z_index+1}/{self._image.GetDepth()}",
            )
        else:
            self.statusBar().showMessage(
                f"x={x} y={y}  HU={val:.1f}  "
                f"z={self.slice_view.z_index+1}/{self._image.GetDepth()}",
            )


# -- entry point ---------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Console-script entry point: ``fatanalyze-gui`` / ``python -m fatanalyze.gui``."""
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("fatAnalyze")
    install_locale(app, "zh_CN")
    win = FatAnalyzeWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FatAnalyzeWindow", "main"]
