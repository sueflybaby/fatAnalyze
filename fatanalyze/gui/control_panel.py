"""Side dock panel: Display (preset, W/L) + ROI controls.

Houses the controls that used to clutter the top toolbar. Emits the
same signals as :class:`ControlsBar` so :class:`FatAnalyzeWindow` can
treat it as a drop-in replacement for the lower-frequency controls.
"""
from __future__ import annotations

from typing import List, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from fatanalyze.config import load_mr_presets
from fatanalyze.gui.slice_view import WL_PRESETS


# Anatomical-target preset choices. Internal key drives lookup;
# English label is translated at display time.
PRESET_CHOICES: List[Tuple[str, str]] = [
    ("iliopsoas_left",  "Left Psoas"),
    ("iliopsoas_right", "Right Psoas"),
    ("liver",           "Liver"),
    ("pancreas",        "Pancreas"),
    ("spleen",          "Spleen"),
    ("custom",          "Custom"),
]


_MR_PRESET_KEYS: list[str] = []


def _load_mr_preset_keys() -> list[str]:
    global _MR_PRESET_KEYS
    if not _MR_PRESET_KEYS:
        _MR_PRESET_KEYS = list(load_mr_presets().get("presets", {}).keys())
    return _MR_PRESET_KEYS


class ControlPanel(QWidget):
    """Vertical side panel with ``Display`` and ``ROI`` group boxes.

    Signals mirror the subset of :class:`ControlsBar` we still need:
    ``preset_changed``, ``window_level_changed``, ``wl_preset_changed``,
    ``draw_toggle_requested``, ``clear_roi_requested``, ``save_roi_requested``,
    ``mr_preset_changed``.
    """

    preset_changed = Signal(str)
    window_level_changed = Signal(float, float)
    wl_preset_changed = Signal(str)
    draw_toggle_requested = Signal(bool)
    clear_roi_requested = Signal()
    save_roi_requested = Signal()
    mr_preset_changed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._modality: str = "ct"
        self.setMinimumWidth(220)
        self.setMaximumWidth(280)
        self._build()

    # -- build --------------------------------------------------------

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(8)

        root.addWidget(self._build_display_group())
        root.addWidget(self._build_roi_group())
        root.addStretch(1)

    def _build_display_group(self) -> QGroupBox:
        box = QGroupBox(self.tr("Display"))
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 10, 8, 8)
        v.setSpacing(6)

        # --- Anatomical preset ---
        self.preset_combo = QComboBox()
        for key, label_en in PRESET_CHOICES:
            self.preset_combo.addItem(self.tr(label_en), key)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.currentIndexChanged.connect(self._emit_preset)
        self.preset_label = QLabel(self.tr("Preset:"))
        self.preset_label.setMinimumWidth(72)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(self.preset_label, 0)
        row.addWidget(self.preset_combo, 1)
        container = QWidget()
        container.setLayout(row)
        v.addWidget(container)

        # --- W/L sliders with value readouts ---
        self._w_label, self.w_slider = self._slider_with_label(
            " W ", minimum=1, maximum=4000, value=400,
        )
        v.addLayout(self._labeled_row(" W ", self._w_label,
                                      self.w_slider,
                                      spacing=4))
        self.w_slider.valueChanged.connect(self._on_w_changed)

        self._l_label, self.l_slider = self._slider_with_label(
            " L ", minimum=-1000, maximum=1000, value=40,
        )
        v.addLayout(self._labeled_row(" L ", self._l_label,
                                      self.l_slider,
                                      spacing=4))
        self.l_slider.valueChanged.connect(self._on_l_changed)

        # --- W/L preset (CT) ---
        self.wl_preset_combo = QComboBox()
        for name in WL_PRESETS.keys():
            self.wl_preset_combo.addItem(self.tr(name))
        self.wl_preset_combo.setCurrentIndex(0)
        self.wl_preset_combo.currentIndexChanged.connect(self._emit_wl_preset)
        v.addWidget(self._labeled("W/L Preset:", self.wl_preset_combo))

        # --- MR vendor preset (MR) ---
        self.mr_preset_combo = QComboBox()
        for key in _load_mr_preset_keys():
            self.mr_preset_combo.addItem(key)
        self.mr_preset_combo.setCurrentIndex(0)
        self.mr_preset_combo.currentIndexChanged.connect(self._emit_mr_preset)
        v.addWidget(self._labeled("MR Preset:", self.mr_preset_combo))

        self._apply_modality_visibility()
        return box

    def _build_roi_group(self) -> QGroupBox:
        box = QGroupBox(self.tr("ROI"))
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 10, 8, 8)
        v.setSpacing(6)

        self.draw_btn = QPushButton(self.tr("ROI Draw: OFF"))
        self.draw_btn.setCheckable(True)
        self.draw_btn.toggled.connect(self._on_draw_toggled)
        v.addWidget(self.draw_btn)

        row = QHBoxLayout()
        row.setSpacing(6)
        self._clear_btn = QPushButton(self.tr("Clear"))
        self._clear_btn.clicked.connect(self.clear_roi_requested.emit)
        row.addWidget(self._clear_btn)
        self._save_btn = QPushButton(self.tr("Save ROI"))
        self._save_btn.clicked.connect(self.save_roi_requested.emit)
        row.addWidget(self._save_btn)
        v.addLayout(row)

        return box

    # -- helpers ------------------------------------------------------

    def _labeled(self, label_text: str, widget: QWidget) -> QWidget:
        """Wrap a widget in a horizontal row with a left label."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        lbl = QLabel(self.tr(label_text))
        lbl.setMinimumWidth(72)
        row.addWidget(lbl, 0)
        row.addWidget(widget, 1)
        container = QWidget()
        container.setLayout(row)
        return container

    def _labeled_row(self, label_text: str, value_label: QLabel,
                     slider: QSlider, spacing: int = 4) -> QHBoxLayout:
        """Build ``[label] [slider  ] [value]`` row layout."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(spacing)
        lbl = QLabel(self.tr(label_text))
        lbl.setMinimumWidth(20)
        row.addWidget(lbl, 0)
        row.addWidget(slider, 1)
        value_label.setMinimumWidth(36)
        value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(value_label, 0)
        return row

    def _slider_with_label(self, default_text: str,
                           minimum: int, maximum: int, value: int
                           ) -> tuple[QLabel, QSlider]:
        lbl = QLabel(str(value))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        lbl.setText(str(value))
        return lbl, slider

    # -- modality-aware visibility -----------------------------------

    def set_modality(self, modality: str) -> None:
        if modality == self._modality:
            return
        self._modality = modality
        self._apply_modality_visibility()
        self._apply_modality_labels()
        self._emit_wl()

    def _apply_modality_visibility(self) -> None:
        is_mr = self._modality == "mr"
        # In CT: show W/L Preset, hide MR Preset
        # In MR: show MR Preset, hide W/L Preset
        self.wl_preset_combo.parent().setVisible(not is_mr)
        self.mr_preset_combo.parent().setVisible(is_mr)

    def _apply_modality_labels(self) -> None:
        is_mr = self._modality == "mr"
        if is_mr:
            # Update slider ranges to FF%
            self.w_slider.setRange(1, 100)
            self.l_slider.setRange(0, 100)
            self.w_slider.setValue(100)
            self.l_slider.setValue(50)
        else:
            self.w_slider.setRange(1, 4000)
            self.l_slider.setRange(-1000, 1000)
            self.w_slider.setValue(400)
            self.l_slider.setValue(40)

    # -- signal emitters ----------------------------------------------

    def _emit_preset(self, _idx: int) -> None:
        self.preset_changed.emit(self.current_preset())

    def _emit_mr_preset(self, _idx: int) -> None:
        self.mr_preset_changed.emit(self.current_mr_preset())

    def _emit_wl_preset(self, _idx: int) -> None:
        keys = list(WL_PRESETS.keys())
        if 0 <= self.wl_preset_combo.currentIndex() < len(keys):
            self.wl_preset_changed.emit(keys[self.wl_preset_combo.currentIndex()])

    def _emit_wl(self) -> None:
        self.window_level_changed.emit(float(self.w_slider.value()),
                                        float(self.l_slider.value()))

    def _on_w_changed(self, val: int) -> None:
        self._w_label.setText(str(val))
        self._emit_wl()

    def _on_l_changed(self, val: int) -> None:
        self._l_label.setText(str(val))
        self._emit_wl()

    def _on_draw_toggled(self, checked: bool) -> None:
        self.draw_btn.setText(self.tr("ROI Draw: ON") if checked
                              else self.tr("ROI Draw: OFF"))
        self.draw_toggle_requested.emit(checked)

    # -- public API ---------------------------------------------------

    def current_preset(self) -> str:
        data = self.preset_combo.currentData()
        return data if data else PRESET_CHOICES[0][0]

    def current_mr_preset(self) -> str:
        data = self.mr_preset_combo.currentText()
        return data if data else _load_mr_preset_keys()[0]

    def set_wl_sliders(self, w: float, l: float) -> None:
        self.w_slider.blockSignals(True)
        self.l_slider.blockSignals(True)
        self.w_slider.setValue(int(w))
        self.l_slider.setValue(int(l))
        self.w_slider.blockSignals(False)
        self.l_slider.blockSignals(False)
        self._w_label.setText(str(int(w)))
        self._l_label.setText(str(int(l)))

    def set_preset(self, key: str) -> None:
        for i, (k, _) in enumerate(PRESET_CHOICES):
            if k == key:
                self.preset_combo.setCurrentIndex(i)
                return

    # -- retranslation ------------------------------------------------

    def retranslate(self) -> None:
        # Display group title
        self.findChildren(QGroupBox)[0].setTitle(self.tr("Display"))
        if len(self.findChildren(QGroupBox)) > 1:
            self.findChildren(QGroupBox)[1].setTitle(self.tr("ROI"))

        # Preset label
        self.preset_label.setText(self.tr("Preset:"))

        # Anatomical preset combo
        current_key = self.current_preset()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        for key, label_en in PRESET_CHOICES:
            self.preset_combo.addItem(self.tr(label_en), key)
        idx = self.preset_combo.findData(current_key)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)
        self.preset_combo.blockSignals(False)

        # W/L preset combo
        keys = list(WL_PRESETS.keys())
        current_wl_key = (
            keys[self.wl_preset_combo.currentIndex()]
            if 0 <= self.wl_preset_combo.currentIndex() < len(keys)
            else keys[0]
        )
        self.wl_preset_combo.blockSignals(True)
        self.wl_preset_combo.clear()
        for k in keys:
            self.wl_preset_combo.addItem(self.tr(k))
        idx = self.wl_preset_combo.findText(self.tr(current_wl_key))
        if idx >= 0:
            self.wl_preset_combo.setCurrentIndex(idx)
        self.wl_preset_combo.blockSignals(False)

        # Buttons
        is_on = self.draw_btn.isChecked()
        self.draw_btn.setText(self.tr("ROI Draw: ON") if is_on
                              else self.tr("ROI Draw: OFF"))
        self._clear_btn.setText(self.tr("Clear"))
        self._save_btn.setText(self.tr("Save ROI"))


__all__ = ["ControlPanel", "PRESET_CHOICES"]
