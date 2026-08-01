"""Dedicated nonblocking Qt window for simulated manual fault review.

This module is imported only by the desktop UI so headless simulation keeps
PySide6 optional.  A custom dialog is used instead of ``QMessageBox`` because
the macOS native message-box backend may synthesize an ``OK`` button before the
message and button state are updated.
"""

from __future__ import annotations

from typing import Any, Mapping

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget


class ManualReviewDialog(QDialog):
    """Persistent status window that cannot dismiss an active review by accident."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("人工审核")
        self.setWindowModality(Qt.NonModal)
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        self.setMinimumWidth(520)
        self._recovery_id = ""
        self._waiting = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(16)

        self.title_label = QLabel("⚠️ 产线故障人工审核")
        self.title_label.setStyleSheet("font-size:18px;font-weight:700;color:#d29922")
        layout.addWidget(self.title_label)

        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        self.message_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.message_label.setStyleSheet("font-size:15px;line-height:1.5")
        layout.addWidget(self.message_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        self.confirm_button = QPushButton("确定")
        self.confirm_button.setDefault(True)
        self.confirm_button.clicked.connect(self._acknowledge)
        layout.addWidget(self.confirm_button, 0, Qt.AlignRight)

    @property
    def recovery_id(self) -> str:
        return self._recovery_id

    def apply_popup(self, popup: Mapping[str, Any]) -> None:
        """Render one state snapshot without entering a nested Qt event loop."""

        recovery_id = str(popup.get("recovery_id", ""))
        status = str(popup.get("status", ""))
        waiting = status == "MANUAL_REVIEW"
        first_display = recovery_id != self._recovery_id or not self.isVisible()
        self._recovery_id = recovery_id
        self._waiting = waiting
        self.message_label.setText(str(popup.get("message", "")))

        if waiting:
            self.title_label.setText("⚠️ 产线故障人工审核")
            self.title_label.setStyleSheet("font-size:18px;font-weight:700;color:#d29922")
            self.progress_bar.show()
            self.confirm_button.hide()
        else:
            self.title_label.setText("✅ 人工审核完成")
            self.title_label.setStyleSheet("font-size:18px;font-weight:700;color:#238636")
            self.progress_bar.hide()
            self.confirm_button.show()

        self.adjustSize()
        if first_display:
            self.show()
            self.raise_()
            self.activateWindow()

    def dismiss(self) -> None:
        """Hide stale/reset state without destroying the reusable widget."""

        self._waiting = False
        self._recovery_id = ""
        self.hide()

    def reject(self) -> None:  # type: ignore[override]
        # Escape must not dismiss an in-progress manual review.  Once complete,
        # treat it exactly like the explicit confirmation button.
        if self._waiting:
            return
        self._acknowledge()

    def _acknowledge(self) -> None:
        if self._waiting:
            return
        self.hide()
        parent = self.parentWidget()
        if parent is not None:
            parent.show()
            parent.raise_()
            parent.activateWindow()


__all__ = ["ManualReviewDialog"]
