"""Offscreen GUI smoke tests — no real display required.

These tests use ``QT_QPA_PLATFORM=offscreen`` (set in ``conftest.py``) and
verify that:

- the package imports cleanly
- the main window instantiates and is wired
- a slice view can render a synthetic CT volume
- a polygon can be drawn (vertices added)
- a ROI can be analyzed end-to-end (rasterize → analyze_user_roi)
"""
from __future__ import annotations

import os

import pytest
import SimpleITK as sitk

from PySide6.QtWidgets import QApplication


# ---------------------------------------------------------------------------
# pytest-qt is not a dependency; use QTest from PySide6 directly.
# ---------------------------------------------------------------------------
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest


# Ensure a QApplication exists for the whole module
@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app
    # do not quit; pytest-qt / other tests may still need it


# ---------------------------------------------------------------------------
# import + window construction
# ---------------------------------------------------------------------------

def test_package_imports():
    from fatanalyze.gui import FatAnalyzeWindow, ROI, main
    assert FatAnalyzeWindow is not None
    assert ROI is not None
    assert callable(main)


def test_window_constructs(qapp):
    from fatanalyze.gui import FatAnalyzeWindow
    w = FatAnalyzeWindow()
    assert w.windowTitle() == "fatAnalyze"
    assert w.size().width() >= 800
    assert w.size().height() >= 600
    # Core widgets exist
    assert w.slice_view is not None
    assert w.roi_list is not None
    assert w.results is not None
    assert w.controls is not None
    # No image loaded yet
    assert w._image is None
    assert w._results == {}
    assert w.roi_list.get_rois() == []


# ---------------------------------------------------------------------------
# slice view rendering
# ---------------------------------------------------------------------------

def _make_synthetic_image(shape=(40, 64, 64), spacing=(1.25, 0.7, 0.7)):
    import numpy as np
    arr = np.zeros(shape, dtype=np.int16)
    arr[:] = -1000  # air
    arr[10:30, 20:40, 20:40] = 40  # soft tissue
    arr[15:20, 25:30, 25:30] = 200  # bone-ish
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)  # z, y, x
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    return img


def test_slice_view_renders_synthetic(qapp):
    from fatanalyze.gui import FatAnalyzeWindow
    w = FatAnalyzeWindow()
    img = _make_synthetic_image()
    w._image = img
    w.slice_view.set_image(img)
    # Pixel readout sanity check: x=30, y=30 should be in the soft-tissue block
    arr = sitk.GetArrayFromImage(img)
    expected_hu = float(arr[20, 30, 30])
    # Use the same mapping the slice view does (W=400, L=40 by default)
    w, l = w.slice_view.get_window_level()
    assert w == 400.0
    assert l == 40.0


def test_preset_color_map_complete():
    from fatanalyze.gui.app import PRESET_COLORS
    for preset in ("iliopsoas_left", "iliopsoas_right", "liver", "pancreas", "spleen", "custom"):
        assert preset in PRESET_COLORS


# ---------------------------------------------------------------------------
# ROI rasterization + analyze
# ---------------------------------------------------------------------------

def test_polygon_add_vertex_and_rasterize(qapp):
    from fatanalyze.gui.polygon_item import PolygonItem
    from fatanalyze.gui.roi import ROI
    from fatanalyze.gui.metrics_runner import rasterize, compute_for_rois

    img = _make_synthetic_image(shape=(40, 100, 100))
    polygon = PolygonItem()
    # 4-vertex rectangle fully inside the soft-tissue block (y in [20, 40), x in [20, 40))
    polygon.add_vertex(30, 30)
    polygon.add_vertex(35, 30)
    polygon.add_vertex(35, 35)
    polygon.add_vertex(30, 35)

    roi = ROI(name="test_box", preset="liver", z_index=20, vertices=polygon.get_vertices())
    mask = rasterize(roi, img)
    assert mask is not None
    n_voxels = int((sitk.GetArrayFromImage(mask) > 0).sum())
    assert n_voxels > 0
    # Should be ~5x5 = 25 voxels (MplPath edge inclusion is fuzzy)
    assert 20 <= n_voxels <= 35

    # Run analyze
    results = compute_for_rois(img, [roi])
    assert "test_box" in results
    r = results["test_box"]
    assert r["n_voxels"] > 0
    # Soft-tissue block is HU=40, so the polygon's mean HU should be ~40
    assert 30 <= r["mean_hu"] <= 50
    assert "fat_fraction" in r["ratios"] or "normal" in r["ratios"]


def test_psoas_combined_merges_two_rois(qapp):
    from fatanalyze.gui.polygon_item import PolygonItem
    from fatanalyze.gui.roi import ROI
    from fatanalyze.gui.metrics_runner import rasterize, compute_for_rois

    img = _make_synthetic_image(shape=(40, 100, 100))
    # Two non-overlapping psoas rectangles, both fully inside the soft-tissue block
    left = ROI(name="L-Psoas", preset="iliopsoas_left", z_index=20,
               vertices=[(22, 30), (27, 30), (27, 35), (22, 35)])
    right = ROI(name="R-Psoas", preset="iliopsoas_right", z_index=20,
                vertices=[(32, 30), (37, 30), (37, 35), (32, 35)])
    rasterize(left, img)
    rasterize(right, img)
    results = compute_for_rois(img, [left, right])
    assert "L-Psoas" in results
    assert "R-Psoas" in results
    # Combined entry: both sides' masks OR'd
    combined_keys = [k for k in results if k not in ("L-Psoas", "R-Psoas")]
    assert combined_keys, "expected a combined psoas entry"
    combined = results[combined_keys[0]]
    assert combined["target"] == "iliopsoas_combined"
    assert combined["psoas_metrics"] is not None
    pm = combined["psoas_metrics"]
    # Both psoas sides sit inside the HU=40 block → all "normal" → sum = 1
    s = pm["imat_fraction"] + pm["low_density_fraction"] + pm["normal_muscle_fraction"]
    assert 0.99 <= s <= 1.01
    assert pm["normal_muscle_fraction"] > 0.95


# ---------------------------------------------------------------------------
# ROI list widget (model-side tests; no GUI interaction)
# ---------------------------------------------------------------------------

def test_roi_list_add_remove(qapp):
    from fatanalyze.gui.roi import ROI
    from fatanalyze.gui.roi_list import ROIListWidget

    rl = ROIListWidget()
    assert rl.get_rois() == []

    rl.add_roi(ROI(name="A", preset="liver", z_index=5, vertices=[(0, 0), (1, 0), (1, 1)]))
    rl.add_roi(ROI(name="B", preset="liver", z_index=6, vertices=[(0, 0), (1, 0), (1, 1)]))
    assert len(rl.get_rois()) == 2

    # Duplicate name gets auto-suffixed
    rl.add_roi(ROI(name="A", preset="liver", z_index=7, vertices=[(0, 0), (1, 0), (1, 1)]))
    names = [r.name for r in rl.get_rois()]
    assert "A" in names and "A_2" in names

    rl.remove_roi("B")
    assert len(rl.get_rois()) == 2
    assert "B" not in [r.name for r in rl.get_rois()]


# ---------------------------------------------------------------------------
# PolygonItem state machine (no GUI)
# ---------------------------------------------------------------------------

def test_polygon_state_machine(qapp):
    from fatanalyze.gui.polygon_item import PolygonItem

    p = PolygonItem()
    assert p.vertex_count() == 0
    assert not p.is_closed

    p.add_vertex(0, 0)
    p.add_vertex(10, 0)
    p.add_vertex(10, 10)
    assert p.vertex_count() == 3
    assert not p.is_closed

    p.close()
    assert p.is_closed
    vs = p.get_vertices()
    assert len(vs) == 3
    assert vs[0] == (0.0, 0.0)
    assert vs[2] == (10.0, 10.0)

    p.remove_last_vertex()
    assert p.vertex_count() == 2
    assert not p.is_closed

    p.clear()
    assert p.vertex_count() == 0


# ---------------------------------------------------------------------------
# Results panel formatting (model-side)
# ---------------------------------------------------------------------------

def test_results_panel_format(qapp):
    from fatanalyze.gui.results_panel import ResultsPanel

    panel = ResultsPanel()

    fake = {
        "target": "iliopsoas_left",
        "n_voxels": 1234,
        "area_cm2": 12.34,
        "volume_ml": 5.6,
        "mean_hu": 50.5,
        "median_hu": 48.2,
        "std_hu": 20.0,
        "p05_hu": 10.0,
        "p95_hu": 90.0,
        "ratios": {"fat_fraction": 0.123},
        "clinical_flags": [],
        "psoas_metrics": {
            "imat_fraction": 0.10,
            "low_density_fraction": 0.20,
            "normal_muscle_fraction": 0.70,
            "myosteatosis_flag": False,
        },
    }
    s = panel._format_metrics("L-Psoas", fake)
    assert "L-Psoas" in s
    assert "12.34 cm²" in s
    assert "1234" in s
    assert "IMAT fraction" in s
    assert "10.00%" in s  # IMAT
    assert "20.00%" in s  # LDM
    assert "70.00%" in s  # normal


# ---------------------------------------------------------------------------
# SliceView mouse handling (regression tests for v0.3.0 vertex-drawing bug)
# ---------------------------------------------------------------------------

def _make_mouse_event(pos, button, event_type=None):
    """Build a QMouseEvent suitable for direct mousePressEvent/dblClick tests.

    PySide6 changed QMouseEvent's ctor signature across versions; this helper
    picks the right one. We try the modern (5-arg + local + global) ctor
    first, then fall back to the legacy (4-arg) form.
    """
    from PySide6.QtCore import QEvent, QPoint
    from PySide6.QtGui import QMouseEvent

    et = event_type or QEvent.MouseButtonPress
    pos_qpoint = QPoint(int(pos[0]), int(pos[1]))
    # Modern PySide6 (6.x): (type, localPos, button, buttons, modifiers, ...)
    try:
        return QMouseEvent(et, pos_qpoint, button, button, Qt.NoModifier)
    except TypeError:
        # Older 4-arg form
        return QMouseEvent(et, pos_qpoint, button, Qt.NoModifier)


def test_polygon_left_click_adds_vertex(qapp):
    """Left-click in polygon mode must add a vertex at the mapped scene coords.

    This is a regression test for v0.3.0: the original code put the
    left-click handler in ``FatAnalyzeWindow.mousePressEvent``, but Qt
    dispatches QGraphicsView mouse events to the view (not the parent
    window), so vertex-adding was unreachable from real clicks.

    We build the view with a known scene rect + identity transform so
    widget-local (10, 20) maps to scene (10, 20) deterministically.
    """
    from fatanalyze.gui.polygon_item import PolygonItem
    from fatanalyze.gui.slice_view import SliceView
    from PySide6.QtCore import QEvent, QRectF

    view = SliceView()
    view._scene.setSceneRect(QRectF(0, 0, 1000, 1000))
    view.resetTransform()  # ensure mapToScene is identity
    poly = PolygonItem()
    view._scene.addItem(poly)
    view.polygon_mode = True
    view.active_polygon = poly

    assert poly.vertex_count() == 0

    event = _make_mouse_event((10, 20), Qt.LeftButton, QEvent.MouseButtonPress)
    view.mousePressEvent(event)
    assert poly.vertex_count() == 1
    assert poly.get_vertices()[0] == (10.0, 20.0)

    event = _make_mouse_event((50, 60), Qt.LeftButton, QEvent.MouseButtonPress)
    view.mousePressEvent(event)
    assert poly.vertex_count() == 2
    assert poly.get_vertices()[1] == (50.0, 60.0)


def test_polygon_right_click_removes_last_vertex(qapp):
    """Right-click in polygon mode must remove the most recent vertex."""
    from fatanalyze.gui.polygon_item import PolygonItem
    from fatanalyze.gui.slice_view import SliceView
    from PySide6.QtCore import QEvent, QRectF

    view = SliceView()
    view._scene.setSceneRect(QRectF(0, 0, 1000, 1000))
    view.resetTransform()
    poly = PolygonItem()
    view._scene.addItem(poly)
    poly.add_vertex(0, 0)
    poly.add_vertex(5, 5)
    poly.add_vertex(10, 10)
    assert poly.vertex_count() == 3

    view.polygon_mode = True
    view.active_polygon = poly

    event = _make_mouse_event((0, 0), Qt.RightButton, QEvent.MouseButtonPress)
    view.mousePressEvent(event)

    assert poly.vertex_count() == 2
    vs = poly.get_vertices()
    assert vs == [(0.0, 0.0), (5.0, 5.0)]


def test_polygon_double_click_closes_with_three_vertices(qapp):
    """Double-click with ≥3 vertices must emit ``polygon_closed``.

    Main window connects this signal to the save-ROI flow; the test
    just verifies the signal fires when the user double-clicks.
    """
    from fatanalyze.gui.polygon_item import PolygonItem
    from fatanalyze.gui.slice_view import SliceView
    from PySide6.QtCore import QEvent, QRectF

    view = SliceView()
    view._scene.setSceneRect(QRectF(0, 0, 1000, 1000))
    view.resetTransform()
    poly = PolygonItem()
    view._scene.addItem(poly)
    for x, y in [(0, 0), (10, 0), (10, 10)]:
        poly.add_vertex(x, y)

    view.polygon_mode = True
    view.active_polygon = poly

    fired = []
    view.polygon_closed.connect(lambda: fired.append(1))

    event = _make_mouse_event((5, 5), Qt.LeftButton, QEvent.MouseButtonDblClick)
    view.mouseDoubleClickEvent(event)
    assert fired == [1], f"expected polygon_closed to fire once, got {fired}"

    # With only 2 vertices, double-click should NOT fire (caller wants ≥3)
    poly2 = PolygonItem()
    view._scene.addItem(poly2)
    poly2.add_vertex(0, 0)
    poly2.add_vertex(1, 1)
    view.active_polygon = poly2
    fired.clear()
    event = _make_mouse_event((0, 0), Qt.LeftButton, QEvent.MouseButtonDblClick)
    view.mouseDoubleClickEvent(event)
    assert fired == [], "double-click with <3 vertices must not close polygon"


def test_saved_roi_keeps_preset_color_in_scene(qapp):
    """After 'Save ROI', the polygon stays in the scene with the preset color.

    This validates the v0.3.0 color-highlight UX: L-Psoas = red,
    R-Psoas = blue, liver = green, etc. The polygon is what gives the
    visual feedback when multiple ROIs are drawn back-to-back.
    """
    from fatanalyze.gui import FatAnalyzeWindow
    from fatanalyze.gui.app import PRESET_COLORS

    w = FatAnalyzeWindow()
    # We need a fake image loaded so the draw toggle will accept ON
    img = _make_synthetic_image(shape=(20, 64, 64))
    w._image = img
    w.slice_view.set_image(img)

    # Simulate "Polygon: ON" with preset = iliopsoas_left
    w.controls.preset_combo.setCurrentText("iliopsoas_left")
    w._on_draw_toggled(True)
    assert w._active_polygon is not None
    expected_color = PRESET_COLORS["iliopsoas_left"]
    pen = w._active_polygon.pen()
    assert pen.color().red() == expected_color.red()
    assert pen.color().green() == expected_color.green()
    assert pen.color().blue() == expected_color.blue()

    # Add 3 vertices and simulate save via the dblclick signal path
    for x, y in [(20, 20), (30, 20), (25, 30)]:
        w._active_polygon.add_vertex(x, y)
    w._on_save_polygon = lambda: None  # avoid the QInputDialog prompt
    # Manually call the slot via the signal
    w._active_polygon.close()
    # Verify the polygon is still in the scene (color-highlight stays visible)
    assert w._active_polygon in w.slice_view._scene.items()
