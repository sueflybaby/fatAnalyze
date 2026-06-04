"""Top toolbar: Modality toggle, Open, [spacer], Export CSV, Language.

Lower-frequency controls (preset, W/L, MR preset, ROI draw/clear/save)
now live in :class:`fatanalyze.gui.control_panel.ControlPanel` on the
left side of the main window. The "Analyze" button now sits in the
ROI list panel header. This toolbar keeps only the high-level actions
the user reaches for most often.

Layout (left → right):
    Modality: [CT] | Open DICOM…   ←expandable→   Export CSV | 中文

Strings are wrapped with ``self.tr(...)`` so they pick up the active
locale. ``retranslate()`` re-applies all tr() calls after the user
switches language.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolBar,
    QWidget,
)

from fatanalyze.gui.i18n import SUPPORTED_LOCALES, current_locale


class ControlsBar(QToolBar):
    """Top toolbar: modality, open, [spacer], export, language.

    Signals
    -------
    open_folder_requested
    export_csv_requested
    language_changed(str)
        Emitted with the new locale code (e.g. ``"zh_CN"``).
    modality_changed(str)
        Emitted with the new modality (``"ct"`` or ``"mr"``).
    """

    open_folder_requested = Signal()
    export_csv_requested = Signal()
    language_changed = Signal(str)
    modality_changed = Signal(str)   # "ct" | "mr"

    def __init__(self, parent=None) -> None:
        super().__init__("ROI Tools", parent)
        self.setMovable(False)
        self._modality: str = "ct"
        self._open_action = None
        self._modality_label: QLabel
        self._export_btn: QPushButton
        self._build()

    def _build(self) -> None:
        # --- Left: Modality label + toggle button ---
        self._modality_label = QLabel(self.tr("Modality:"))
        self._modality_label.setMinimumWidth(60)
        self.addWidget(self._modality_label)

        self.modality_btn = QPushButton("CT")
        self.modality_btn.setCheckable(True)
        self.modality_btn.setFixedWidth(56)
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

        # --- Spacer pushes the rest to the right ---
        # QSizePolicy.Policy values: Expanding=7, Preferred=5
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Preferred)
        self.addWidget(spacer)

        # --- Right: Export CSV ---
        self._export_btn = QPushButton(self.tr("Export CSV"))
        self._export_btn.clicked.connect(self.export_csv_requested.emit)
        self.addWidget(self._export_btn)
        self.addSeparator()

        # --- Right: Language toggle button ---
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
        self.modality_changed.emit(self._modality)

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
        self.modality_changed.emit(modality)

    # -- language ----------------------------------------------------

    def _toggle_language(self) -> None:
        current = current_locale()
        target = "zh_CN" if current == "en" else "en"
        self.language_changed.emit(target)

    # -- public API ---------------------------------------------------

    def set_locale(self, locale: str) -> None:
        """Update the language button text to match current locale."""
        if locale not in SUPPORTED_LOCALES:
            return
        self._update_lang_btn_text()

    def _update_lang_btn_text(self) -> None:
        self._lang_btn.setText("中文" if current_locale() == "en" else "English")

    # -- retranslation ------------------------------------------------

    def retranslate(self) -> None:
        """Re-apply tr() to every visible widget after the locale changes."""
        self.setWindowTitle(self.tr("ROI Tools"))
        if self._open_action is not None:
            self._open_action.setText(self.tr("Open DICOM…"))
        self._modality_label.setText(self.tr("Modality:"))
        self.modality_btn.setText("MR" if self._modality == "mr" else "CT")
        self._export_btn.setText(self.tr("Export CSV"))
        self._update_lang_btn_text()


__all__ = ["ControlsBar"]

