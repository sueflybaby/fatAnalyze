"""Multi-ROI list widget with save/load/delete."""
from __future__ import annotations

from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fatanalyze.gui.roi import ROI


class ROIListWidget(QWidget):
    """List of user-drawn ROIs; supports select/edit/delete/save/load.

    Signals
    -------
    roi_selected(ROI)
        Emitted when the user clicks a row (or selects via keyboard).
    rois_changed()
        Emitted whenever the list is mutated.
    """

    roi_selected = Signal(object)  # ROI
    rois_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rois: Dict[str, ROI] = {}  # name -> ROI (names are unique)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.list_widget = QListWidget(self)
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.list_widget, 1)

        # Action buttons
        btn_row = QHBoxLayout()
        self.rename_btn = QPushButton("Rename")
        self.rename_btn.clicked.connect(self._on_rename)
        btn_row.addWidget(self.rename_btn)

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._on_delete)
        btn_row.addWidget(self.delete_btn)

        layout.addLayout(btn_row)

    # -- public API ----------------------------------------------------

    def add_roi(self, roi: ROI) -> None:
        """Add a ROI; auto-suffix the name if it already exists."""
        base = roi.name
        candidate = base
        n = 2
        while candidate in self._rois:
            candidate = f"{base}_{n}"
            n += 1
        roi.name = candidate
        self._rois[candidate] = roi
        self._refresh_row(candidate)
        self.rois_changed.emit()

    def remove_roi(self, name: str) -> None:
        if name in self._rois:
            del self._rois[name]
            for i in range(self.list_widget.count()):
                it = self.list_widget.item(i)
                if it.data(Qt.UserRole) == name:
                    self.list_widget.takeItem(i)
                    break
            self.rois_changed.emit()

    def get_rois(self) -> List[ROI]:
        return list(self._rois.values())

    def get_roi(self, name: str) -> Optional[ROI]:
        return self._rois.get(name)

    def selected_roi(self) -> Optional[ROI]:
        items = self.list_widget.selectedItems()
        if not items:
            return None
        name = items[0].data(Qt.UserRole)
        return self._rois.get(name)

    def mark_analyzed(self, name: str) -> None:
        roi = self._rois.get(name)
        if roi is None:
            return
        roi.status = "analyzed"
        self._refresh_row(name)

    def mark_exported(self, name: str) -> None:
        roi = self._rois.get(name)
        if roi is None:
            return
        roi.status = "exported"
        self._refresh_row(name)

    def clear(self) -> None:
        self._rois.clear()
        self.list_widget.clear()
        self.rois_changed.emit()

    # -- internals -----------------------------------------------------

    def _refresh_row(self, name: str) -> None:
        roi = self._rois[name]
        label = f"{roi.name:<16} {roi.preset:<16} z={roi.z_index:>3}  v={roi.n_vertices:>2}  [{roi.status}]"
        # Find or create the item
        for i in range(self.list_widget.count()):
            it = self.list_widget.item(i)
            if it.data(Qt.UserRole) == name:
                it.setText(label)
                return
        item = QListWidgetItem(label)
        item.setData(Qt.UserRole, name)
        self.list_widget.addItem(item)

    def _on_selection_changed(self) -> None:
        roi = self.selected_roi()
        if roi is not None:
            self.roi_selected.emit(roi)

    def _on_rename(self) -> None:
        roi = self.selected_roi()
        if roi is None:
            return
        new_name, ok = QInputDialog.getText(self, "Rename ROI", "New name:", text=roi.name)
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()
        if new_name == roi.name:
            return
        if new_name in self._rois:
            QMessageBox.warning(self, "Duplicate", f"ROI '{new_name}' already exists")
            return
        del self._rois[roi.name]
        roi.name = new_name
        self._rois[new_name] = roi
        self._refresh_row(new_name)
        self.rois_changed.emit()

    def _on_delete(self) -> None:
        roi = self.selected_roi()
        if roi is None:
            return
        confirm = QMessageBox.question(
            self, "Delete ROI",
            f"Delete ROI '{roi.name}'?",
        )
        if confirm == QMessageBox.Yes:
            self.remove_roi(roi.name)


__all__ = ["ROIListWidget"]
