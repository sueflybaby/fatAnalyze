"""Toolbar: preset combo, W/L sliders, polygon actions, language combo.

Strings are wrapped with ``self.tr(...)`` so they pick up the active
locale. ``retranslate()`` re-applies all tr() calls after the user
switches language.
"""
from __future__ import annotations

from typing import List, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QPushButton,
    QSlider,
    QToolBar,
    QWidget,
)

from fatanalyze.config import load_mr_presets
from fatanalyze.gui.i18n import SUPPORTED_LOCALES, current_locale
from fatanalyze.gui.slice_view import WL_PRESETS


# Preset options exposed to the user. (internal_key, default_english_label).
# The internal key drives preset lookup; the label is what the user sees
# and is run through self.tr() at display time.
PRESET_CHOICES: List[Tuple[str, str]] = [
    ("iliopsoas_left",  "Left Psoas"),
    ("iliopsoas_right", "Right Psoas"),
    ("liver",           "Liver"),
    ("pancreas",        "Pancreas"),
    ("spleen",          "Spleen"),
    ("custom",          "Custom"),
]


# MR vendor presets keys (from mr_presets.yaml)
_MR_PRESET_KEYS: list[str] = []


def _load_mr_preset_keys() -> list[str]:
    global _MR_PRESET_KEYS
    if not _MR_PRESET_KEYS:
        presets = load_mr_presets()
        _MR_PRESET_KEYS = list(presets.get("presets", {}).keys())
    return _MR_PRESET_KEYS


class ControlsBar(QToolBar):
    """Single-row toolbar with modality toggle, preset, W/L, drawing, language.

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
    language_changed(str)
        Emitted with the new locale code (e.g. ``"zh_CN"``).
    modality_changed(str)
        Emitted with the new modality (``"ct"`` or ``"mr"``).
    mr_preset_changed(str)
        Emitted when the MR vendor preset changes.
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
    language_changed = Signal(str)
    modality_changed = Signal(str)   # "ct" | "mr"
    mr_preset_changed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__("ROI Tools", parent)
        self.setMovable(False)
        self._modality: str = "ct"
        # Track translatable widgets so retranslate() can re-apply tr()
        self._w_label: QLabel
        self._l_label: QLabel
        self._preset_label: QLabel
        self._wl_preset_label: QLabel
        self._mr_preset_label: QLabel
        self._open_action = None
        self._clear_btn: QPushButton
        self._save_btn: QPushButton
        self._analyze_btn: QPushButton
        self._export_btn: QPushButton
        self._build()

    def _build(self) -> None:
        # --- Modality toggle button ---
        self.modality_btn = QPushButton("CT")
        self.modality_btn.setCheckable(True)
        self.modality_btn.setFixedWidth(48)
        self.modality_btn.setStyleSheet(
            "QPushButton { font-weight: bold; }"
            "QPushButton:checked { background-color: #4472C4; color: white; }"
        )
        self.modality_btn.toggled.connect(self._on_modality_toggled)
        self.addWidget(self.modality_btn)
        self.addSeparator()

        # --- Open folder ---
        self._open_action = self.addAction(self.tr("Open DICOM…"))
        self._open_action.triggered.connect(self.open_folder_requested.emit)
        self.addSeparator()

        # --- Preset combo ---
        self._preset_label = QLabel(" " + self.tr("Preset:") + " ")
        self.addWidget(self._preset_label)
        self.preset_combo = QComboBox()
        for key, label_en in PRESET_CHOICES:
            self.preset_combo.addItem(self.tr(label_en), key)
        self.preset_combo.setCurrentIndex(0)
        self.preset_combo.currentIndexChanged.connect(self._emit_preset)
        self.addWidget(self.preset_combo)
        self.addSeparator()

        # --- W/L sliders (also used as FF% Range sliders in MR mode) ---
        self._w_label = QLabel(self.tr(" W "))
        self.addWidget(self._w_label)
        self.w_slider = QSlider(Qt.Horizontal)
        self.w_slider.setRange(1, 4000)
        self.w_slider.setValue(400)
        self.w_slider.setMaximumWidth(140)
        self.addWidget(self.w_slider)
        self._l_label = QLabel(self.tr(" L "))
        self.addWidget(self._l_label)
        self.l_slider = QSlider(Qt.Horizontal)
        self.l_slider.setRange(-1000, 1000)
        self.l_slider.setValue(40)
        self.l_slider.setMaximumWidth(140)
        self.addWidget(self.l_slider)
        self.w_slider.valueChanged.connect(self._emit_wl)
        self.l_slider.valueChanged.connect(self._emit_wl)
        self.addSeparator()

        # --- W/L preset combo (CT) ---
        self._wl_preset_label = QLabel(" " + self.tr("W/L Preset:") + " ")
        self.addWidget(self._wl_preset_label)
        self.wl_preset_combo = QComboBox()
        for name in WL_PRESETS.keys():
            self.wl_preset_combo.addItem(self.tr(name))
        self.wl_preset_combo.setCurrentIndex(0)
        self.wl_preset_combo.currentIndexChanged.connect(self._emit_wl_preset)
        self.addWidget(self.wl_preset_combo)

        # --- MR vendor preset combo (hidden in CT mode) ---
        self._mr_preset_label = QLabel(" " + self.tr("MR Preset:") + " ")
        self.addWidget(self._mr_preset_label)
        self.mr_preset_combo = QComboBox()
        for key in _load_mr_preset_keys():
            self.mr_preset_combo.addItem(key)
        self.mr_preset_combo.setCurrentIndex(0)
        self.mr_preset_combo.currentIndexChanged.connect(self._emit_mr_preset)
        self.addWidget(self.mr_preset_combo)

        for w in (self._mr_preset_label, self.mr_preset_combo):
            w.hide()

        self.addSeparator()

        # --- Drawing actions ---
        self.draw_btn = QPushButton(self.tr("ROI Draw: OFF"))
        self.draw_btn.setCheckable(True)
        self.draw_btn.toggled.connect(self._on_draw_toggled)
        self.addWidget(self.draw_btn)
        self._clear_btn = QPushButton(self.tr("Clear"))
        self._clear_btn.clicked.connect(self.clear_roi_requested.emit)
        self.addWidget(self._clear_btn)
        self._save_btn = QPushButton(self.tr("Save ROI"))
        self._save_btn.clicked.connect(self.save_roi_requested.emit)
        self.addWidget(self._save_btn)
        self.addSeparator()
        self._analyze_btn = QPushButton(self.tr("Analyze"))
        self._analyze_btn.clicked.connect(self.analyze_requested.emit)
        self.addWidget(self._analyze_btn)
        self._export_btn = QPushButton(self.tr("Export CSV"))
        self._export_btn.clicked.connect(self.export_csv_requested.emit)
        self.addWidget(self._export_btn)
        self.addSeparator()

        # --- Language toggle button ---
        self._lang_btn = QPushButton()
        self._update_lang_btn_text()
        self._lang_btn.clicked.connect(self._toggle_language)
        self.addWidget(self._lang_btn)

    # -- modality ----------------------------------------------------

    @property
    def current_modality(self) -> str:
        return self._modality

    def _on_modality_toggled(self, checked: bool) -> None:
        self._modality = "mr" if checked else "ct"
        self.modality_btn.setText("MR" if checked else "CT")
        self._reconfigure_for_modality()
        self.modality_changed.emit(self._modality)

    def _reconfigure_for_modality(self) -> None:
        is_mr = self._modality == "mr"
        if is_mr:
            self._w_label.setText(self.tr("FF Range"))
            self._l_label.setText(self.tr("Center"))
            self.w_slider.setRange(1, 100)
            self.l_slider.setRange(0, 100)
            self.w_slider.setValue(100)
            self.l_slider.setValue(50)
        else:
            self._w_label.setText(self.tr(" W "))
            self._l_label.setText(self.tr(" L "))
            self.w_slider.setRange(1, 4000)
            self.l_slider.setRange(-1000, 1000)
            self.w_slider.setValue(400)
            self.l_slider.setValue(40)

        self._wl_preset_label.setVisible(not is_mr)
        self.wl_preset_combo.setVisible(not is_mr)
        self._mr_preset_label.setVisible(is_mr)
        self.mr_preset_combo.setVisible(is_mr)

        self._emit_wl()

    def set_modality(self, modality: str) -> None:
        """Programmatically set the modality (``"ct"`` or ``"mr"``)."""
        if modality == self._modality:
            return
        self._modality = modality
        is_mr = modality == "mr"
        self.modality_btn.blockSignals(True)
        self.modality_btn.setChecked(is_mr)
        self.modality_btn.setText("MR" if is_mr else "CT")
        self.modality_btn.blockSignals(False)
        self._reconfigure_for_modality()
        self.modality_changed.emit(modality)

    # -- signal emitters ----------------------------------------------

    def _emit_mr_preset(self, _idx: int) -> None:
        self.mr_preset_changed.emit(self.current_mr_preset())

    def _emit_preset(self, _idx: int) -> None:
        self.preset_changed.emit(self.current_preset())

    def _emit_wl(self, *_) -> None:
        self.window_level_changed.emit(float(self.w_slider.value()),
                                        float(self.l_slider.value()))

    def _emit_wl_preset(self, _idx: int) -> None:
        # Map display index back to the underlying WL_PRESETS key
        keys = list(WL_PRESETS.keys())
        if 0 <= self.wl_preset_combo.currentIndex() < len(keys):
            self.wl_preset_changed.emit(keys[self.wl_preset_combo.currentIndex()])

    def _on_draw_toggled(self, checked: bool) -> None:
        self.draw_btn.setText(self.tr("ROI Draw: ON") if checked else self.tr("ROI Draw: OFF"))
        self.draw_toggle_requested.emit(checked)

    def _toggle_language(self) -> None:
        current = current_locale()
        target = "zh_CN" if current == "en" else "en"
        self.language_changed.emit(target)

    # -- public API ----------------------------------------------------

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

    def current_preset(self) -> str:
        data = self.preset_combo.currentData()
        return data if data else PRESET_CHOICES[0][0]

    def set_preset(self, key: str) -> None:
        """Programmatically select a preset by its internal key."""
        for i, (k, _) in enumerate(PRESET_CHOICES):
            if k == key:
                self.preset_combo.setCurrentIndex(i)
                return

    def set_locale(self, locale: str) -> None:
        """Update the language button text to match current locale."""
        if locale not in SUPPORTED_LOCALES:
            return
        self._update_lang_btn_text()

    def _update_lang_btn_text(self) -> None:
        self._lang_btn.setText("中文" if current_locale() == "en" else "English")

    # -- retranslation -------------------------------------------------

    def retranslate(self) -> None:
        """Re-apply tr() to every visible widget after the locale changes."""
        self.setWindowTitle(self.tr("ROI Tools"))
        if self._open_action is not None:
            self._open_action.setText(self.tr("Open DICOM…"))

        self.modality_btn.setText("MR" if self._modality == "mr" else "CT")
        self._preset_label.setText(" " + self.tr("Preset:") + " ")
        is_mr = self._modality == "mr"
        self._w_label.setText(self.tr("FF Range") if is_mr else self.tr(" W "))
        self._l_label.setText(self.tr("Center") if is_mr else self.tr(" L "))
        self._wl_preset_label.setText(" " + self.tr("W/L Preset:") + " ")
        self._mr_preset_label.setText(" " + self.tr("MR Preset:") + " ")

        # Preset combo: re-add items with new translations
        current_key = self.current_preset()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        for key, label_en in PRESET_CHOICES:
            self.preset_combo.addItem(self.tr(label_en), key)
        idx = self.preset_combo.findData(current_key)
        if idx >= 0:
            self.preset_combo.setCurrentIndex(idx)
        self.preset_combo.blockSignals(False)

        # W/L preset combo: re-add with new translations
        keys = list(WL_PRESETS.keys())
        current_wl_key = keys[self.wl_preset_combo.currentIndex()] if 0 <= self.wl_preset_combo.currentIndex() < len(keys) else keys[0]
        self.wl_preset_combo.blockSignals(True)
        self.wl_preset_combo.clear()
        for k in keys:
            self.wl_preset_combo.addItem(self.tr(k))
        idx = self.wl_preset_combo.findText(self.tr(current_wl_key))
        if idx >= 0:
            self.wl_preset_combo.setCurrentIndex(idx)
        self.wl_preset_combo.blockSignals(False)

        # Buttons
        self._update_lang_btn_text()
        self._clear_btn.setText(self.tr("Clear"))
        self._save_btn.setText(self.tr("Save ROI"))
        self._analyze_btn.setText(self.tr("Analyze"))
        self._export_btn.setText(self.tr("Export CSV"))
        is_on = self.draw_btn.isChecked()
        self.draw_btn.setText(self.tr("ROI Draw: ON") if is_on else self.tr("ROI Draw: OFF"))


__all__ = ["ControlsBar", "PRESET_CHOICES"]
