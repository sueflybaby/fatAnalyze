"""Main window: wires slice view, controls, ROI list, and results panel."""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import SimpleITK as sitk

from PySide6.QtCore import Qt, QPointF
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

from fatanalyze.gui.controls import ControlsBar
from fatanalyze.gui.metrics_runner import compute_for_rois, rasterize
from fatanalyze.gui.polygon_item import PolygonItem
from fatanalyze.gui.results_panel import ResultsPanel
from fatanalyze.gui.roi import ROI
from fatanalyze.gui.roi_list import ROIListWidget
from fatanalyze.gui.slice_view import SliceView
from fatanalyze.io.dicom_loader import load_ct_series
from fatanalyze.interactive.user_roi import UserROI


# Color per preset (used to draw the polygon in the slice view)
PRESET_COLORS: Dict[str, QColor] = {
    "iliopsoas_left":  QColor(255,  80,  80),
    "iliopsoas_right": QColor( 80,  80, 255),
    "liver":           QColor( 80, 200,  80),
    "pancreas":        QColor(220, 180,  60),
    "spleen":          QColor(200,  80, 200),
    "custom":          QColor(180, 180, 180),
}


class FatAnalyzeWindow(QMainWindow):
    """The single-window fatAnalyze GUI.

    Layout (horizontal splitter):
        ┌──────────────────────────────┬─────────────────────┐
        │ Toolbar                      │                     │
        │ Slice view (with polygons)   │ ROI list            │
        │ Slice slider + status        │ Results panel       │
        └──────────────────────────────┴─────────────────────┘
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("fatAnalyze")
        self.resize(1280, 800)

        self._image: Optional[sitk.Image] = None
        self._qc = None
        self._active_polygon: Optional[PolygonItem] = None
        self._polygons_by_name: Dict[str, PolygonItem] = {}
        self._results: Dict[str, dict] = {}

        self._build_ui()
        self._wire_signals()

    # -- UI scaffolding -----------------------------------------------

    def _build_ui(self) -> None:
        # Top toolbar
        self.controls = ControlsBar(self)
        self.addToolBar(Qt.TopToolBarArea, self.controls)

        # Menu bar
        self._build_menu()

        # Central layout: horizontal splitter
        central = QWidget()
        self.setCentralWidget(central)
        hsplit = QSplitter(Qt.Horizontal)

        # Left side: slice view + slider
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.slice_view = SliceView(self)
        left_layout.addWidget(self.slice_view, 1)

        # Slice slider row
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Slice:"))
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setRange(0, 0)
        self.slice_slider.setEnabled(False)
        slider_row.addWidget(self.slice_slider, 1)
        self.slice_label = QLabel("— / —")
        slider_row.addWidget(self.slice_label)
        left_layout.addLayout(slider_row)

        hsplit.addWidget(left)

        # Right side: ROI list (top) + results (bottom)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.roi_list = ROIListWidget(self)
        right_layout.addWidget(self.roi_list, 1)

        self.results = ResultsPanel(self)
        right_layout.addWidget(self.results, 2)

        hsplit.addWidget(right)
        hsplit.setStretchFactor(0, 3)
        hsplit.setStretchFactor(1, 2)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(hsplit)

        # Status bar
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Open a DICOM folder to begin.")
        self.slice_view.pixel_hovered.connect(self._on_pixel_hovered)

    def _build_menu(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        open_action = QAction("Open DICOM Folder…", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._on_open_folder)
        file_menu.addAction(open_action)

        export_action = QAction("Export CSV…", self)
        export_action.triggered.connect(self._on_export_csv)
        file_menu.addAction(export_action)

        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        analysis_menu = menubar.addMenu("&Analysis")
        run_action = QAction("Run Analyze", self)
        run_action.setShortcut("Ctrl+R")
        run_action.triggered.connect(self._on_analyze)
        analysis_menu.addAction(run_action)

        help_menu = menubar.addMenu("&Help")
        about_action = QAction("About fatAnalyze", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _wire_signals(self) -> None:
        # Toolbar → main window
        self.controls.open_folder_requested.connect(self._on_open_folder)
        self.controls.preset_changed.connect(self._on_preset_changed)
        self.controls.window_level_changed.connect(self.slice_view.set_window_level)
        self.controls.wl_preset_changed.connect(self.slice_view.apply_wl_preset)
        self.controls.draw_toggle_requested.connect(self._on_draw_toggled)
        self.controls.clear_roi_requested.connect(self._on_clear_polygon)
        self.controls.save_roi_requested.connect(self._on_save_polygon)
        self.controls.analyze_requested.connect(self._on_analyze)
        self.controls.export_csv_requested.connect(self._on_export_csv)
        # Slice slider
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        # ROI list
        self.roi_list.roi_selected.connect(self._on_roi_selected)

    # -- slots ---------------------------------------------------------

    def _on_open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Open DICOM Folder", str(Path.cwd()),
        )
        if not folder:
            return
        try:
            image, qc = load_ct_series(Path(folder))
        except Exception as exc:
            QMessageBox.critical(self, "DICOM load failed", str(exc))
            return
        self._image = image
        self._qc = qc
        self.slice_view.set_image(image)
        # Wire slider
        self.slice_slider.setRange(0, image.GetDepth() - 1)
        self.slice_slider.setValue(image.GetDepth() // 2)
        self.slice_slider.setEnabled(True)
        self.slice_label.setText(f"{self.slice_slider.value()+1} / {image.GetDepth()}")
        # Reflect current W/L
        w, l = self.slice_view.get_window_level()
        self.controls.set_wl_sliders(w, l)
        # Clear stale state
        self._active_polygon = None
        self._polygons_by_name.clear()
        self.roi_list.clear()
        self._results.clear()
        self.results.clear()
        # Show QC report
        QMessageBox.information(
            self, "DICOM QC",
            qc.summary() if hasattr(qc, "summary") else str(qc),
        )
        self.statusBar().showMessage(
            f"Loaded {image.GetDepth()} slices from {folder}", 5000,
        )

    def _on_preset_changed(self, preset: str) -> None:
        # Color of the *next* polygon will use this preset
        if self._active_polygon is not None and self._active_polygon.vertex_count() == 0:
            self._active_polygon.set_color(PRESET_COLORS.get(preset, QColor(180, 180, 180)))

    def _on_slice_changed(self, z: int) -> None:
        if self._image is None:
            return
        self.slice_view.set_slice(z)
        depth = self._image.GetDepth()
        self.slice_label.setText(f"{z+1} / {depth}")

    def _on_draw_toggled(self, on: bool) -> None:
        if not on:
            if self._active_polygon is not None and self._active_polygon.vertex_count() < 3:
                # Discard empty polygon and any leaked handles
                self._active_polygon.clear()
                self.slice_view._scene.removeItem(self._active_polygon)
                self._active_polygon = None
            return
        if self._image is None:
            QMessageBox.warning(self, "No image", "Open a DICOM folder first.")
            self.controls.draw_btn.setChecked(False)
            return
        # Start a new polygon
        preset = self.controls.current_preset()
        color = PRESET_COLORS.get(preset, QColor(180, 180, 180))
        self._active_polygon = PolygonItem(color=color)
        self.slice_view._scene.addItem(self._active_polygon)
        self.statusBar().showMessage(
            f"Polygon drawing ON (preset: {preset}). "
            f"Left-click to add vertices, double-click to close.",
        )

    def _on_clear_polygon(self) -> None:
        if self._active_polygon is None:
            return
        self._active_polygon.clear()
        self.statusBar().showMessage("Polygon cleared.", 2000)

    def _on_save_polygon(self) -> None:
        if self._active_polygon is None or self._active_polygon.vertex_count() < 3:
            QMessageBox.information(self, "Save ROI",
                                    "Draw at least 3 vertices first.")
            return
        self._active_polygon.close()
        # Build a ROI object and add to list
        preset = self.controls.current_preset()
        default_name = f"{preset}"
        name, ok = QInputDialog.getText(self, "Save ROI", "ROI name:", text=default_name)
        if not ok or not name.strip():
            return
        name = name.strip()
        z = self.slice_view.z_index
        roi = ROI(
            name=name,
            preset=preset,
            z_index=z,
            vertices=self._active_polygon.get_vertices(),
        )
        self.roi_list.add_roi(roi)
        self._polygons_by_name[name] = self._active_polygon
        self._active_polygon = None
        self.controls.draw_btn.setChecked(False)
        self.statusBar().showMessage(f"ROI '{name}' added ({len(roi.vertices)} vertices).",
                                      4000)

    def _on_roi_selected(self, roi: ROI) -> None:
        # When user clicks a row, show its metrics
        if roi.name in self._results:
            self.results.show_result(roi.name, self._results[roi.name])

    def _on_analyze(self) -> None:
        if self._image is None:
            QMessageBox.warning(self, "No image", "Open a DICOM folder first.")
            return
        rois = self.roi_list.get_rois()
        if not rois:
            QMessageBox.information(self, "No ROIs", "Draw at least one ROI first.")
            return
        try:
            self._results = compute_for_rois(self._image, rois)
        except Exception as exc:
            QMessageBox.critical(self, "Analyze failed", str(exc))
            return
        # Mark all analyzed
        for name in self._results:
            self.roi_list.mark_analyzed(name)
        self.results.show_all(self._results)
        self.statusBar().showMessage(
            f"Analyzed {len(self._results)} ROI(s).", 4000,
        )

    def _on_export_csv(self) -> None:
        if not self._results:
            QMessageBox.information(self, "No results", "Click 'Analyze' first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export metrics to CSV", "fatAnalyze-metrics.csv",
            "CSV files (*.csv)",
        )
        if not path:
            return
        rows = []
        for name, r in self._results.items():
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
        self.statusBar().showMessage(f"Exported {len(rows)} rows to {path}", 5000)

    def _on_about(self) -> None:
        QMessageBox.about(
            self, "About fatAnalyze",
            "<b>fatAnalyze v0.3.0</b><br>"
            "CT ectopic-fat analysis (liver, pancreas, psoas at L3).<br>"
            "Native PySide6 GUI; the analysis pipeline is unchanged.<br><br>"
            "DICOM → polygon ROI → HU stats + clinical metrics.",
        )

    def _on_pixel_hovered(self, x: int, y: int, hu: float) -> None:
        if self._image is None:
            return
        if hu != hu:  # NaN
            return
        self.statusBar().showMessage(
            f"x={x} y={y}  HU={hu:.1f}  z={self.slice_view.z_index+1}/{self._image.GetDepth()}",
        )

    # -- mouse forwarding (when polygon mode is on) --------------------

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if (self.controls.draw_btn.isChecked()
                and self._active_polygon is not None
                and event.button() == Qt.LeftButton):
            # Translate to scene coords
            scene_pt = self.slice_view.mapToScene(
                self.slice_view.mapFrom(self, event.position().toPoint())
            )
            self._active_polygon.add_vertex(scene_pt.x(), scene_pt.y())
            return
        if (self.controls.draw_btn.isChecked()
                and self._active_polygon is not None
                and event.button() == Qt.RightButton):
            self._active_polygon.remove_last_vertex()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[override]
        if (self.controls.draw_btn.isChecked()
                and self._active_polygon is not None
                and self._active_polygon.vertex_count() >= 3
                and event.button() == Qt.LeftButton):
            self._on_save_polygon()
            return
        super().mouseDoubleClickEvent(event)


# -- entry point ---------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Console-script entry point: ``fatanalyze-gui`` / ``python -m fatanalyze.gui``."""
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("fatAnalyze")
    win = FatAnalyzeWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
