from __future__ import annotations

from datetime import datetime

from pyqtgraph.Qt import QtWidgets


class EventTimelineWidget(QtWidgets.QGroupBox):
    def __init__(self, parent=None, max_rows: int = 200) -> None:
        super().__init__("Event Timeline", parent)
        self._max_rows = max_rows

        layout = QtWidgets.QVBoxLayout(self)
        self._table = QtWidgets.QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Time", "Event", "Confidence", "Severity"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)

        layout.addWidget(self._table)

    def add_event(self, event: dict) -> None:
        timestamp = float(event.get("timestamp", 0.0))
        ts_text = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S") if timestamp > 0 else "--:--:--"

        label = str(event.get("label", "Unknown"))
        confidence = float(event.get("confidence", 0.0))
        severity = str(event.get("severity", "low"))

        self._table.insertRow(0)
        self._table.setItem(0, 0, QtWidgets.QTableWidgetItem(ts_text))
        self._table.setItem(0, 1, QtWidgets.QTableWidgetItem(label))
        self._table.setItem(0, 2, QtWidgets.QTableWidgetItem(f"{confidence * 100.0:5.1f}%"))
        self._table.setItem(0, 3, QtWidgets.QTableWidgetItem(severity.upper()))

        while self._table.rowCount() > self._max_rows:
            self._table.removeRow(self._table.rowCount() - 1)

    def clear_events(self) -> None:
        self._table.setRowCount(0)
