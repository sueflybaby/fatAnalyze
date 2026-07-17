"""Main window: wires slice view, controls, ROI list, and results panel."""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import SimpleITK as sitk

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
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
from fatanalyze import __version__
from fatanalyze.gui.controls import ControlsBar
from fatanalyze.gui.control_panel import ControlPanel
from fatanalyze.gui.i18n import install_locale, current_locale, SUPPORTED_LOCALES, reset_for_test
from fatanalyze.gui.metrics_runner import compute_for_rois, compute_for_rois_mr, rasterize
from fatanalyze.gui.polygon_item import PolygonItem
from fatanalyze.gui.results_panel import ResultsPanel
from fatanalyze.gui.roi import ROI
from fatanalyze.gui.roi_list import ROIListWidget
from fatanalyze.gui.slice_view import SliceView
from fatanalyze.io.dicom_loader import (
    OperationCancelled,
    ProgressCallback,
    detect_dicom_modality,
    load_ct_series,
    load_mr_series,
)
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
        self.setWindowTitle(self.tr("BodyFatAnalyzer"))
        self.resize(1280, 800)

        self._image: Optional[sitk.Image] = None
        self._qc = None
        self._modality: Modality = Modality.CT
        self._active_polygon: Optional[PolygonItem] = None
        self._polygons_by_name: Dict[str, PolygonItem] = {}
        self._polygon_z: Dict[str, int] = {}       # name → z_index for saved ROIs
        self._active_polygon_z: Optional[int] = None  # z_index of the polygon being drawn
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
        act = QAction(self.tr("About BodyFatAnalyzer"), self)
        act.triggered.connect(self._on_about)
        self._help_menu.addAction(act)
        self._menu_actions["about"] = act

    def _wire_signals(self) -> None:
        # --- Top toolbar (high-level actions) ---
        self.controls.open_folder_requested.connect(self._on_open_folder)
        self.controls.export_csv_requested.connect(self._on_export_csv)
        self.controls.language_changed.connect(self._on_language_changed)
        self.controls.modality_changed.connect(self._on_modality_changed)
        # --- Side panel (Display + ROI) ---
        self.panel.preset_changed.connect(self._on_preset_changed)
        self.panel.window_level_changed.connect(self.slice_view.set_window_level)
        self.panel.wl_preset_changed.connect(self.slice_view.apply_wl_preset)
        self.panel.draw_toggle_requested.connect(self._on_draw_toggled)
        self.panel.clear_all_rois_requested.connect(self._on_clear_all_rois)
        self.panel.mr_preset_changed.connect(self._on_mr_preset_changed)
        # --- ROI list (Analyze moved to its header) ---
        self.roi_list.analyze_requested.connect(self._on_analyze)
        # --- View <-> state ---
        self.slice_slider.valueChanged.connect(self._on_slice_changed)
        self.slice_view.slice_changed.connect(self._on_view_slice_changed)
        self.roi_list.roi_selected.connect(self._on_roi_selected)
        self.roi_list.roi_removed.connect(self._on_roi_removed)
        self.slice_view.polygon_closed.connect(self._on_save_polygon)

    # -- Language switching --------------------------------------------

    def _on_modality_changed(self, mod: str) -> None:
        self._modality = Modality(mod)
        # Tell the side panel to swap W/L sliders and show/hide the
        # appropriate preset combo (CT W/L Preset vs MR vendor Preset).
        self.panel.set_modality(mod)
        # Clear image when switching modality. Must destroy polygons from
        # the scene as well as drop the Python references — otherwise the
        # old ROIs would stay rendered on top of the new image.
        self._reset_rois_state()
        self._image = None
        self._qc = None
        self.slice_slider.setRange(0, 0)
        self.slice_slider.setEnabled(False)
        self.slice_label.setText(self.tr("— / —"))
        self.statusBar().showMessage(
            self.tr("Switched to {mode} mode. Open a folder to begin.").format(
                mode="MR" if mod == "mr" else "CT"
            )
        )

    def _run_with_progress(
        self,
        title: str,
        label: str,
        total: int,
        fn,
        *args,
        **kwargs,
    ):
        """Run ``fn(*args, progress=cb, **kwargs)`` under a modal QProgressDialog.

        ``fn`` is expected to accept a ``progress`` keyword argument
        matching :data:`ProgressCallback` and to call it at safe
        checkpoints. If the user clicks Cancel, :class:`OperationCancelled`
        is raised by the callback and surfaces here.

        Returns ``fn``'s return value, or ``None`` if the user cancelled.
        Non-cancellation exceptions (e.g. load failures) are caught, shown
        in a :class:`QMessageBox`, and converted to ``None``.
        """
        dlg = QProgressDialog(label, self.tr("Cancel"), 0, total, self)
        dlg.setWindowTitle(title)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)  # show immediately
        dlg.setAutoClose(False)     # we close it ourselves on cancel/error
        dlg.setAutoReset(False)

        def progress_cb(current: int, total_steps: int, message: str = "") -> None:
            if total_steps > 0 and dlg.maximum() != total_steps:
                dlg.setMaximum(total_steps)
            dlg.setValue(current)
            if message:
                dlg.setLabelText(message)
            QApplication.processEvents()
            if dlg.wasCanceled():
                raise OperationCancelled()

        try:
            try:
                progress_cb(0, total, label)
                return fn(*args, progress=progress_cb, **kwargs)
            except OperationCancelled:
                return None
            except Exception as exc:
                QMessageBox.critical(
                    self, self.tr("Operation failed"), str(exc),
                )
                return None
        finally:
            dlg.close()

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
        self.setWindowTitle(self.tr("BodyFatAnalyzer"))
        self._file_menu.setTitle(self.tr("&File"))
        self._analysis_menu.setTitle(self.tr("&Analysis"))
        self._help_menu.setTitle(self.tr("&Help"))
        self._menu_actions["open"].setText(self.tr("Open DICOM Folder…"))
        self._menu_actions["export"].setText(self.tr("Export CSV…"))
        self._menu_actions["quit"].setText(self.tr("Quit"))
        self._menu_actions["run_analyze"].setText(self.tr("Run Analyze"))
        self._menu_actions["about"].setText(self.tr("About BodyFatAnalyzer"))
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
        # Auto-detect modality from the first DICOM file in the folder.
        try:
            detected = detect_dicom_modality(Path(folder))
            if detected != self._modality.value.lower():
                self.controls.set_modality(detected)
        except Exception:
            pass  # fall through to existing logic

        is_mr = self._modality == Modality.MR
        total_stages = 4 if is_mr else 3

        def do_load(progress: ProgressCallback):
            if is_mr:
                preset_cfg = None
                try:
                    from fatanalyze.config import load_mr_presets
                    mr_presets = load_mr_presets()
                    mr_preset_name = self.panel.current_mr_preset()
                    preset_cfg = mr_presets.get("presets", {}).get(mr_preset_name, {})
                except Exception:
                    pass
                return load_mr_series(Path(folder), preset_cfg, progress=progress)
            return load_ct_series(Path(folder), progress=progress)

        try:
            image, qc = self._run_with_progress(
                title=self.tr("Loading DICOM"),
                label=self.tr("Loading DICOM folder…"),
                total=total_stages,
                fn=do_load,
            )
        except OperationCancelled:
            self.statusBar().showMessage(self.tr("Loading cancelled."), 3000)
            return
        if image is None:
            return  # _run_with_progress already showed the error dialog

        # Wipe all ROI / draw state from the previous exam BEFORE the new
        # image renders, so the scene is clean when the new slice appears.
        self._reset_rois_state()
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
        QMessageBox.information(
            self, self.tr("DICOM QC"),
            self._format_qc_summary(qc) if hasattr(qc, "summary") else str(qc),
        )
        self.statusBar().showMessage(
            self.tr("Loaded {n} slices from {folder}").format(
                n=image.GetDepth(), folder=folder
            ), 5000,
        )

    def _reset_rois_state(self) -> None:
        """Destroy all polygons from the scene and clear every ROI-related slot.

        Called when the user opens a new exam or switches modality, so the
        next image starts from a clean slate (no leftover polygons, no
        half-drawn active polygon, no stale results, no stuck draw mode).

        Does not show a confirmation prompt — callers guard if needed.
        """
        if self._active_polygon is not None:
            self._active_polygon.destroy()
            self._active_polygon = None
        for polygon in list(self._polygons_by_name.values()):
            polygon.destroy()
        self._active_polygon_z = None
        self._polygons_by_name.clear()
        self._polygon_z.clear()
        self.roi_list.clear()
        self._results.clear()
        self.results.clear()
        # If draw mode was left on, the user would click on a fresh image
        # and create vertices in a polygon that's about to be replaced.
        # Reset both the view's mode and the panel button state.
        self.slice_view.polygon_mode = False
        self.slice_view.active_polygon = None
        if self.panel.draw_btn.isChecked():
            self.panel.draw_btn.setChecked(False)
        self._update_clear_button()

    def _format_qc_summary(self, qc) -> str:
        if not qc.warnings and not qc.errors:
            level = self.tr("OK")
        elif not qc.errors:
            level = self.tr("WARN")
        else:
            level = self.tr("FAIL")
        sp = ", ".join(f"{s:.2f}" for s in qc.spacing_xyz)
        sx, sy, sz = qc.size_xyz
        parts = [
            f"[{level}] {sx}×{sy}×{sz} @ {sp} {self.tr('mm')}",
            f"{self.tr('HU')} [{qc.hu_min:.0f}, {qc.hu_max:.0f}]",
            f"{self.tr('z-CV')}={qc.slice_spacing_cv*100:.1f}%",
        ]
        msg = " | ".join(parts)
        if qc.warnings:
            msg += f" | {self.tr('warnings')}: {len(qc.warnings)}"
        if qc.errors:
            msg += f" | {self.tr('errors')}: {len(qc.errors)}"
        return msg

    def _on_preset_changed(self, preset: str) -> None:
        if self._active_polygon is not None and self._active_polygon.vertex_count() == 0:
            self._active_polygon.set_color(PRESET_COLORS.get(preset, QColor(180, 180, 180)))

    def _refresh_roi_visibility(self, z: int) -> None:
        """Show only ROIs belonging to slice *z*, hide all others."""
        for name, polygon in self._polygons_by_name.items():
            polygon.setVisible(self._polygon_z.get(name) == z)
        if self._active_polygon is not None:
            self._active_polygon.setVisible(self._active_polygon_z == z)

    def _on_slice_changed(self, z: int) -> None:
        if self._image is None:
            return
        self.slice_view.set_slice(z)
        self._refresh_roi_visibility(z)
        depth = self._image.GetDepth()
        self.slice_label.setText(f"{z+1} / {depth}")

    def _on_view_slice_changed(self, z: int) -> None:
        """Sync the slider when the view scrolls via mouse wheel."""
        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(z)
        self.slice_slider.blockSignals(False)
        self._refresh_roi_visibility(z)
        depth = self._image.GetDepth()
        self.slice_label.setText(f"{z+1} / {depth}")

    def _update_clear_button(self) -> None:
        has_rois = len(self.roi_list.get_rois()) > 0
        has_active = (self._active_polygon is not None
                      and self._active_polygon.vertex_count() > 0)
        self.panel.set_clear_all_enabled(has_rois or has_active)

    def _on_draw_toggled(self, on: bool) -> None:
        if not on:
            self._active_polygon_z = None
            if self._active_polygon is not None and self._active_polygon.vertex_count() < 3:
                self._active_polygon.destroy()
                self._active_polygon = None
            self.slice_view.polygon_mode = False
            self.slice_view.active_polygon = None
            self._update_clear_button()
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
        self._active_polygon_z = self.slice_view.z_index
        self._active_polygon.signals.vertices_changed.connect(self._update_clear_button)
        self._update_clear_button()
        self.statusBar().showMessage(
            self.tr("ROI drawing ON (preset: {preset}). "
                    "Left-click to add vertices, double-click to close.").format(
                preset=preset
            ),
        )

    def _on_clear_all_rois(self) -> None:
        has_rois = len(self.roi_list.get_rois()) > 0
        has_active = (self._active_polygon is not None
                      and self._active_polygon.vertex_count() > 0)
        if not has_rois and not has_active:
            return
        confirm = QMessageBox.question(
            self, self.tr("Clear All ROIs"),
            self.tr("Are you sure you want to clear all ROIs? This cannot be undone."),
        )
        if confirm != QMessageBox.Yes:
            return
        if self._active_polygon is not None:
            self._active_polygon.destroy()
            self._active_polygon = None
        for polygon in list(self._polygons_by_name.values()):
            polygon.destroy()
        self._active_polygon_z = None
        self._polygons_by_name.clear()
        self._polygon_z.clear()
        self.roi_list.clear()
        self._update_clear_button()
        self.statusBar().showMessage(self.tr("All ROIs cleared."), 3000)

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
        name = roi.name
        self._polygons_by_name[name] = self._active_polygon
        self._polygon_z[name] = z
        self._active_polygon = None
        self._update_clear_button()
        self.panel.draw_btn.setChecked(False)
        self.statusBar().showMessage(
            self.tr("ROI '{name}' added ({n} vertices).").format(
                name=name, n=len(roi.vertices)
            ), 4000,
        )

    def _on_roi_removed(self, name: str) -> None:
        self._polygon_z.pop(name, None)
        polygon = self._polygons_by_name.pop(name, None)
        if polygon is not None:
            polygon.destroy()
        self._update_clear_button()

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

        # Decide the total number of progress steps up front so the dialog
        # can show a determinate progress bar with a real percentage.
        has_psoas_combine = any(
            r.preset in ("iliopsoas_left", "iliopsoas_right") for r in rois
        )
        total = len(rois) + (1 if has_psoas_combine else 0) + 1

        is_mr = self._modality == Modality.MR

        def do_analyze(progress: ProgressCallback):
            if is_mr:
                return compute_for_rois_mr(self._image, rois, progress=progress)
            return compute_for_rois(self._image, rois, progress=progress)

        try:
            self._results = self._run_with_progress(
                title=self.tr("Analyzing ROIs"),
                label=self.tr("Analyzing ROIs…"),
                total=total,
                fn=do_analyze,
            )
        except OperationCancelled:
            self.statusBar().showMessage(self.tr("Analysis cancelled."), 3000)
            return
        if self._results is None:
            return  # cancelled or failed (dialog already shown)
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
            self, self.tr("Export metrics to CSV"), "BodyFatAnalyzer-metrics.csv",
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
            self, self.tr("About BodyFatAnalyzer"),
            f"<b>BodyFatAnalyzer v{__version__}</b><br>"
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
    app.setApplicationName("BodyFatAnalyzer")
    icon_path = Path(__file__).parent / "resources" / "app_icon.svg"
    if icon_path.exists():
        icon = QIcon(str(icon_path))
        app.setWindowIcon(icon)
    install_locale(app, "zh_CN")
    win = FatAnalyzeWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FatAnalyzeWindow", "main"]
