"""Lightweight i18n for the fatAnalyze GUI.

Strings are wrapped with ``self.tr("...")`` (or ``QObject.tr`` for module
constants). A custom :class:`QTranslator` subclass reads from a Python
dict maintained in this file, so there is no ``.ts`` / ``.qm`` build
step. Add new translations to ``TRANSLATIONS`` and they become
available in the language combo automatically.

Usage
-----
::

    from fatanalyze.gui.i18n import install_locale, current_locale
    install_locale(QApplication.instance(), "zh_CN")

When the translator is replaced, Qt emits ``QEvent::LanguageChange`` to
every top-level widget. Each widget re-translates its own visible text
in its ``changeEvent`` / ``retranslateUi`` override.
"""
from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import QObject, QTranslator


# ---------------------------------------------------------------------------
# Translation table
# ---------------------------------------------------------------------------
# Format: TRANSLATIONS[locale][source_text] = translated_text
# English strings are the source; the tr() function returns the source
# as a fallback if no translation is found.
TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "zh_CN": {
        # --- Menu ---
        "&File": "文件(&F)",
        "&Analysis": "分析(&A)",
        "&Help": "帮助(&H)",
        "Open DICOM Folder…": "打开 DICOM 文件夹…",
        "Export CSV…": "导出 CSV…",
        "Quit": "退出",
        "Run Analyze": "运行分析",
        "About fatAnalyze": "关于 fatAnalyze",

        # --- Toolbar ---
        "ROI Tools": "ROI 工具",
        "Preset:": "预设:",
        "W/L Preset:": "窗宽窗位:",
        "Open DICOM…": "打开 DICOM…",
        "ROI Draw: OFF": "ROI勾画: 关",
        "ROI Draw: ON": "ROI勾画: 开",
        "Clear": "清空",
        "Save ROI": "保存 ROI",
        "Analyze": "分析",
        "Export CSV": "导出 CSV",
        " W ": " 窗宽 ",
        " L ": " 窗位 ",

        # --- Side panel group titles ---
        "Display": "显示",
        "ROI": "ROI",

        # --- Slice / labels ---
        "Slice:": "切片:",
        "— / —": "— / —",

        # --- Preset choices (combo display) ---
        "Left Psoas": "左腰大肌",
        "Right Psoas": "右腰大肌",
        "Liver": "肝脏",
        "Pancreas": "胰腺",
        "Spleen": "脾脏",
        "Custom": "自定义",

        # --- W/L preset names ---
        "Soft Tissue": "软组织",
        "Bone": "骨",
        "Lung": "肺",

        # --- Results panel ---
        "Results": "结果",
        "No ROIs analyzed yet.": "尚未分析任何 ROI。",
        "flags": "标志",
        "Open a DICOM folder": "打开 DICOM 文件夹",
        "Pick a slice with the slider": "用滑块选切片",
        "Click 'ROI Draw: OFF' to enable drawing": "点击「ROI勾画: 关」开始绘制",
        "Left-click to add vertices, double-click to close": "左键加点，双击闭合",
        "Click 'Analyze' to compute metrics": "点击「分析」计算指标",
        # --- Status / hover ---
        "Open a DICOM folder to begin.": "打开 DICOM 文件夹开始。",
        "ROI cleared.": "已清空 ROI 勾画。",
        "Load failed": "加载失败",
        "DICOM QC": "DICOM 质控",
        "No image": "未加载图像",
        "Open a DICOM folder first.": "请先打开 DICOM 文件夹。",
        "No ROIs": "无 ROI",
        "Draw at least one ROI first.": "请先画至少一个 ROI。",
        "Analyze failed": "分析失败",
        "Save ROI": "保存 ROI",
        "Draw at least 3 vertices first.": "请先画至少 3 个顶点。",
        "ROI name:": "ROI 名称:",

        # --- ROI list ---
        "Rename": "重命名",
        "Delete": "删除",
        "Rename ROI": "重命名 ROI",
        "New name:": "新名称:",
        "Duplicate": "重名",
        "ROI '{name}' already exists": "ROI「{name}」已存在",
        "Delete ROI": "删除 ROI",
        "Delete ROI '{name}'?": "删除 ROI「{name}」？",

        # --- About ---
        "CT ectopic-fat analysis (liver, pancreas, psoas at L3).": "CT 异位脂肪分析（肝、胰、L3 水平腰大肌）。",
        "Native PySide6 GUI; the analysis pipeline is unchanged.": "原生 PySide6 GUI；分析管线保持不变。",
        "DICOM → polygon ROI → HU stats + clinical metrics.": "DICOM → 多边形 ROI → HU 统计 + 临床指标。",
        "MR PDFF/Dixon fat fraction support (FF% stats + steatosis grading).": "MR PDFF/Dixon 脂肪分数支持（FF% 统计 + 脂肪变性分级）。",

        # --- MR keys ---
        "MR": "MR",
        "FF Range": "FF 范围",
        "Center": "中心",
        "MR Preset:": "MR 预设:",
        "Switched to {mode} mode. Open a folder to begin.": "已切换到 {mode} 模式，请打开文件夹开始。",
        "Loaded {n} slices from {folder}": "已从 {folder} 加载 {n} 层切片",
        "MR preset set to '{p}'.": "MR 预设已设为「{p}」。",
        "MR preset set to '{p}'. Reopen folder to apply.": "MR 预设已设为「{p}」。重新打开文件夹以应用。",

        # --- MR Fat Fraction ---
        "Fat Fraction Bins:": "脂肪分数区间:",
        "Fat Fraction (%)": "脂肪分数 (%)",
        "mean_ff": "平均 FF%",
        "Steatosis: S0 (normal)": "脂肪变性: S0（正常）",
        "Steatosis: S1 (mild)": "脂肪变性: S1（轻度）",
        "Steatosis: S2 (moderate)": "脂肪变性: S2（中度）",
        "Steatosis: S3 (severe)": "脂肪变性: S3（重度）",
        "Myosteatosis (FF > 25%)": "肌少性脂肪变性（FF > 25%）",
        "Pancreatic steatosis (FF > 10%)": "胰腺脂肪变性（FF > 10%）",
        "Splenic steatosis (FF > 10%)": "脾脏脂肪变性（FF > 10%）",

        # --- Metric formatting keys ---
        "(none)": "(无)",
        "Ratios:": "比值:",
        "Clinical flags:": "临床标志:",
        "Psoas / myosteatosis:": "腰大肌 / 肌少性脂肪变性:",
        "Myosteatosis flag : ": "肌少性脂肪变性 : ",

        # --- Language combo ---
        "Language:": "语言:",
    },
}


SUPPORTED_LOCALES: list[str] = ["en", "zh_CN"]
LOCALE_LABELS: Dict[str, str] = {
    "en": "English",
    "zh_CN": "中文",
}


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------

class JsonTranslator(QTranslator):
    """A :class:`QTranslator` backed by the in-memory ``TRANSLATIONS`` dict.

    Qt calls :meth:`translate` for every ``tr()`` evaluation; we look the
    source string up in our pre-built flat table and return the
    translation (or the source as fallback). No .ts / .qm build needed.
    """

    def __init__(self, locale: str, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._locale = locale
        self._table: Dict[str, str] = TRANSLATIONS.get(locale, {})

    @property
    def locale(self) -> str:
        return self._locale

    def translate(  # type: ignore[override]
        self,
        context: str,
        source_text: str,
        disambiguation: Optional[str] = None,
        n: int = -1,
    ) -> str:
        if not source_text:
            return ""
        if n >= 0 and n != 1:
            # Plural forms not handled; return source.
            return source_text
        return self._table.get(source_text, source_text)


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_current_translator: Optional[JsonTranslator] = None


def install_locale(app, locale: str) -> JsonTranslator:
    """Install (or replace) the active translator on ``app``.

    Replacing the translator causes Qt to emit
    ``QEvent::LanguageChange`` to every top-level widget, which in
    turn triggers their ``retranslateUi`` overrides.
    """
    global _current_translator
    if _current_translator is not None:
        app.removeTranslator(_current_translator)
        _current_translator.deleteLater()
    translator = JsonTranslator(locale)
    app.installTranslator(translator)
    _current_translator = translator
    return translator


def current_locale() -> str:
    """Return the currently installed locale code (e.g. ``"en"`` or ``"zh_CN"``)."""
    if _current_translator is None:
        return "en"
    return _current_translator.locale


def reset_for_test() -> None:
    """Drop the current translator reference. Tests-only; production code uses install_locale."""
    global _current_translator
    _current_translator = None


__all__ = [
    "JsonTranslator",
    "TRANSLATIONS",
    "SUPPORTED_LOCALES",
    "LOCALE_LABELS",
    "install_locale",
    "current_locale",
    "reset_for_test",
]
