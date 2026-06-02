"""Toolbar: preset combo, W/L sliders, polygon actions."""
from __future__ import annotations

from typing import List

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QToolBar,
    QWidget,
)

from fatanalyze.gui.slice_view import WL_PRESETS


# Preset options exposed to the user. Order matters — first option is default.
PRESET_CHOICES: List[str] = [
    "iliopsoas_left",
    "iliopsoas_right",
    "liver",
    "pancreas",
    "spleen",
    "custom",
]


class ControlsBar(QToolBar):
    """Single-row toolbar with preset, W/L, and drawing actions.

    Signals
    -------
    open_folder_requested
    preset_changed(str)
    window_level_changed(float, float)
    wl_preset_changed(str)
    draw_toggle_requested(bool)
    clear_roi_requested
    save_roi_requested
    analyze_requested
    export_csv_requested
    """

    open_folder_requested = Signal()
    preset_changed = Signal(str)
    window_level_changed = Signal(float, float)
    wl_preset_changed = Signal(str)
    draw_toggle_requested = Signal(bool)
    clear_roi_requested = Signal()
    save_roi_requested = Signal()
    analyze_requested = Signal()
    export_csv_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__("ROI Tools", parent)
        self.setMovable(False)
        self._build()

    def _build(self) -> None:
        # --- Open folder ---
        self.addAction("Open DICOM…")
        self.widgetForAction(self.actions()[-1]).setObjectName("open_folder_btn")
        self.actions()[-1].triggered.connect(self.open_folder_requested.emit)
        self.addSeparator()

        # --- Preset combo ---
        self.addWidget(QLabel(" Preset: "))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(PRESET_CHOICES)
        self.preset_combo.setCurrentText("iliopsoas_left")
        self.preset_combo.currentTextChanged.connect(self.preset_changed.emit)
        self.addWidget(self.preset_combo)
        self.addSeparator()

        # --- W/L sliders ---
        self.addWidget(QLabel(" W "))
        self.w_slider = QSlider(Qt.Horizontal)
        self.w_slider.setRange(1, 4000)
        self.w_slider.setValue(400)
        self.w_slider.setMaximumWidth(140)
        self.addWidget(self.w_slider)
        self.addWidget(QLabel(" L "))
        self.l_slider = QSlider(Qt.Horizontal)
        self.l_slider.setRange(-1000, 1000)
        self.l_slider.setValue(40)
        self.l_slider.setMaximumWidth(140)
        self.addWidget(self.l_slider)
        self.w_slider.valueChanged.connect(self._emit_wl)
        self.l_slider.valueChanged.connect(self._emit_wl)
        self.addSeparator()

        # --- W/L preset combo ---
        self.addWidget(QLabel(" W/L Preset: "))
        self.wl_preset_combo = QComboBox()
        self.wl_preset_combo.addItems(list(WL_PRESETS.keys()))
        self.wl_preset_combo.setCurrentText("Soft Tissue")
        self.wl_preset_combo.currentTextChanged.connect(self.wl_preset_changed.emit)
        self.addWidget(self.wl_preset_combo)
        self.addSeparator()

        # --- Drawing actions ---
        self.draw_btn = QPushButton("Polygon: OFF")
        self.draw_btn.setCheckable(True)
        self.draw_btn.toggled.connect(self._on_draw_toggled)
        self.addWidget(self.draw_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_roi_requested.emit)
        self.addWidget(clear_btn)
        save_btn = QPushButton("Save ROI")
        save_btn.clicked.connect(self.save_roi_requested.emit)
        self.addWidget(save_btn)
        self.addSeparator()
        analyze_btn = QPushButton("Analyze")
        analyze_btn.clicked.connect(self.analyze_requested.emit)
        self.addWidget(analyze_btn)
        export_btn = QPushButton("Export CSV")
        export_btn.clicked.connect(self.export_csv_requested.emit)
        self.addWidget(export_btn)

    def _emit_wl(self, *_) -> None:
        self.window_level_changed.emit(float(self.w_slider.value()),
                                        float(self.l_slider.value()))

    def _on_draw_toggled(self, checked: bool) -> None:
        self.draw_btn.setText(f"Polygon: {'ON' if checked else 'OFF'}")
        self.draw_toggle_requested.emit(checked)

    def set_wl_sliders(self, w: float, l: float) -> None:
        """Set W/L sliders without emitting signals (used to reflect state)."""
        self.w_slider.blockSignals(True)
        self.l_slider.blockSignals(True)
        self.w_slider.setValue(int(w))
        self.l_slider.setValue(int(l))
        self.w_slider.blockSignals(False)
        self.l_slider.blockSignals(False)

    def current_preset(self) -> str:
        return self.preset_combo.currentText()


__all__ = ["ControlsBar", "PRESET_CHOICES"]
