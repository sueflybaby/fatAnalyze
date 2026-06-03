"""Tests for the i18n module and locale switching in the GUI."""
from __future__ import annotations

import sys
from typing import Generator

import pytest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QMainWindow

from fatanalyze.gui.i18n import (
    LOCALE_LABELS,
    SUPPORTED_LOCALES,
    TRANSLATIONS,
    JsonTranslator,
    current_locale,
    install_locale,
    reset_for_test,
)
from fatanalyze.gui.app import FatAnalyzeWindow
from fatanalyze.gui.controls import ControlsBar
from fatanalyze.gui.results_panel import ResultsPanel
from fatanalyze.gui.roi_list import ROIListWidget


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(autouse=True)
def reset_locale(app: QApplication) -> Generator:
    """Reset to English before each test and install a clean translator."""
    install_locale(app, "en")
    yield
    reset_for_test()


# ---------------------------------------------------------------------------
# JsonTranslator
# ---------------------------------------------------------------------------

class TestJsonTranslator:
    def test_returns_translation_when_found(self) -> None:
        t = JsonTranslator("zh_CN")
        assert t.translate("", "Left Psoas") == "左腰大肌"

    def test_returns_source_when_not_found(self) -> None:
        t = JsonTranslator("zh_CN")
        assert t.translate("", "nonexistent_key_xyz") == "nonexistent_key_xyz"

    def test_returns_source_for_english(self) -> None:
        t = JsonTranslator("en")
        assert t.translate("", "Left Psoas") == "Left Psoas"

    def test_empty_source_returns_empty(self) -> None:
        t = JsonTranslator("zh_CN")
        assert t.translate("", "") == ""

    def test_plural_form_returns_source(self) -> None:
        t = JsonTranslator("zh_CN")
        assert t.translate("", "voxels", n=2) == "voxels"

    def test_all_translation_keys_map_to_strings(self) -> None:
        for locale, table in TRANSLATIONS.items():
            assert isinstance(locale, str)
            for src, dst in table.items():
                assert isinstance(src, str)
                assert isinstance(dst, str)
                assert dst != "", f"Empty translation for '{src}' in '{locale}'"


# ---------------------------------------------------------------------------
# install_locale / current_locale
# ---------------------------------------------------------------------------

class TestLocaleManagement:
    def test_install_and_query(self, app: QApplication) -> None:
        install_locale(app, "zh_CN")
        assert current_locale() == "zh_CN"

    def test_switch_back_to_english(self, app: QApplication) -> None:
        install_locale(app, "zh_CN")
        install_locale(app, "en")
        assert current_locale() == "en"

    def test_initial_locale_is_english(self) -> None:
        assert current_locale() == "en"

    def test_supported_locales_list(self) -> None:
        assert "en" in SUPPORTED_LOCALES
        assert "zh_CN" in SUPPORTED_LOCALES

    def test_locale_labels(self) -> None:
        assert LOCALE_LABELS["en"] == "English"
        assert LOCALE_LABELS["zh_CN"] == "中文"


# ---------------------------------------------------------------------------
# Smoke: FatAnalyzeWindow builds and responds to LanguageChange
# ---------------------------------------------------------------------------

class TestWindowLocale:
    def test_window_title_in_english(self, app: QApplication) -> None:
        win = FatAnalyzeWindow()
        assert win.windowTitle() == "fatAnalyze"

    def _switch(self, app, win, locale):
        """Switch locale via the public path used by production code."""
        win._on_language_changed(locale)

    def test_window_title_in_chinese(self, app: QApplication) -> None:
        win = FatAnalyzeWindow()
        self._switch(app, win, "zh_CN")
        file_menu = win._file_menu
        assert file_menu.title() == "文件(&F)"

    def test_file_menu_items_translate(self, app: QApplication) -> None:
        win = FatAnalyzeWindow()
        self._switch(app, win, "zh_CN")
        expected = {
            "open": "打开 DICOM 文件夹…",
            "export": "导出 CSV…",
            "quit": "退出",
        }
        for key, expected_text in expected.items():
            actual = win._menu_actions[key].text()
            assert actual.replace("&", "").replace("…", "…") == expected_text.replace("…", "…")

    def test_controls_bar_retranslate(self, app: QApplication) -> None:
        """Verify toolbar buttons change language after a switch."""
        win = FatAnalyzeWindow()
        assert "Preset:" in win.controls._preset_label.text()
        self._switch(app, win, "zh_CN")
        assert "预设:" in win.controls._preset_label.text()

    def test_slice_label_translate(self, app: QApplication) -> None:
        win = FatAnalyzeWindow()
        self._switch(app, win, "zh_CN")
        assert "切片:" in win._slice_label_label.text()

    def test_results_panel_placeholder(self, app: QApplication) -> None:
        win = FatAnalyzeWindow()
        self._switch(app, win, "zh_CN")
        placeholder = win.results.text.placeholderText()
        assert "尚未分析任何 ROI" in placeholder

    def test_about_dialog_uses_tr(self, app: QApplication) -> None:
        win = FatAnalyzeWindow()
        self._switch(app, win, "zh_CN")
        assert win.tr("CT ectopic-fat analysis (liver, pancreas, psoas at L3).") == \
               "CT 异位脂肪分析（肝、胰、L3 水平腰大肌）。"

    def test_double_switch(self, app: QApplication) -> None:
        """en → zh_CN → en should leave everything in English."""
        win = FatAnalyzeWindow()
        self._switch(app, win, "zh_CN")
        self._switch(app, win, "en")
        assert win._file_menu.title() == "&File"
        assert "Preset:" in win.controls._preset_label.text()

    def test_status_bar_updates(self, app: QApplication) -> None:
        win = FatAnalyzeWindow()
        assert "Open a DICOM folder" in win.statusBar().currentMessage()
        self._switch(app, win, "zh_CN")
        assert "打开 DICOM 文件夹" in win.statusBar().currentMessage()


# ---------------------------------------------------------------------------
# Signal: language_changed triggers install_locale
# ---------------------------------------------------------------------------

class TestLanguageSignal:
    def test_button_toggles_to_chinese(self, app: QApplication) -> None:
        win = FatAnalyzeWindow()
        assert current_locale() == "en"
        # Button shows target language: "中文" when in English mode
        assert win.controls._lang_btn.text() == "中文"
        win.controls._toggle_language()
        app.processEvents()
        assert current_locale() == "zh_CN"
        assert win.controls._lang_btn.text() == "English"
        # Verify translation took effect
        assert win.controls._preset_label.text() == " 预设: "
        assert "预设:" in win.controls._preset_label.text()

    def test_button_toggles_back_to_english(self, app: QApplication) -> None:
        win = FatAnalyzeWindow()
        # Switch to zh
        win.controls._toggle_language()
        app.processEvents()
        assert current_locale() == "zh_CN"
        # Switch back to en
        win.controls._toggle_language()
        app.processEvents()
        assert current_locale() == "en"
        assert win.controls._lang_btn.text() == "中文"
        assert "Preset:" in win.controls._preset_label.text()
