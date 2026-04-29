from __future__ import annotations

from typing import Sequence

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets


class WaveformPanel(QtWidgets.QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._canvas = pg.GraphicsLayoutWidget()
        layout.addWidget(self._canvas)

        self._plot_v = self._canvas.addPlot(title="Voltage")
        self._plot_v.showGrid(x=True, y=True, alpha=0.25)
        self._plot_v.setLabel("left", "V")
        self._plot_v.setLabel("bottom", "Time", units="ms")
        self._curve_v = self._plot_v.plot(pen=pg.mkPen("#f5c242", width=2))

        self._canvas.nextRow()

        self._plot_i = self._canvas.addPlot(title="Current")
        self._plot_i.showGrid(x=True, y=True, alpha=0.25)
        self._plot_i.setLabel("left", "A")
        self._plot_i.setLabel("bottom", "Time", units="ms")
        self._curve_i = self._plot_i.plot(pen=pg.mkPen("#51d6ff", width=2))

    def update_waveforms(self, v_wave: Sequence[float], i_wave: Sequence[float], fs_hz: float) -> None:
        if not v_wave or not i_wave:
            return

        v = np.asarray(v_wave, dtype=np.float32)
        i = np.asarray(i_wave, dtype=np.float32)
        n = len(v)
        if n == 0:
            return

        t_ms = np.arange(n, dtype=np.float32) * (1000.0 / float(fs_hz))
        self._curve_v.setData(t_ms, v)
        self._curve_i.setData(t_ms, i)


class HarmonicSpectrumPanel(QtWidgets.QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plot = pg.PlotWidget(title="Harmonic Spectrum (1..13)")
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.setLabel("left", "Magnitude")
        self._plot.setLabel("bottom", "Harmonic Order")
        layout.addWidget(self._plot)

        self._orders = np.arange(1, 14, dtype=np.float32)
        self._bars_v = pg.BarGraphItem(x=self._orders - 0.18, height=np.zeros(13), width=0.3, brush="#f5c242")
        self._bars_i = pg.BarGraphItem(x=self._orders + 0.18, height=np.zeros(13), width=0.3, brush="#51d6ff")

        self._plot.addItem(self._bars_v)
        self._plot.addItem(self._bars_i)
        self._plot.setXRange(0, 14)

    def update_harmonics(self, mags_v: Sequence[float], mags_i: Sequence[float]) -> None:
        if len(mags_v) < 13 or len(mags_i) < 13:
            return

        v = np.asarray(mags_v[:13], dtype=np.float32)
        i = np.asarray(mags_i[:13], dtype=np.float32)

        self._plot.removeItem(self._bars_v)
        self._plot.removeItem(self._bars_i)

        self._bars_v = pg.BarGraphItem(x=self._orders - 0.18, height=v, width=0.3, brush="#f5c242")
        self._bars_i = pg.BarGraphItem(x=self._orders + 0.18, height=i, width=0.3, brush="#51d6ff")

        self._plot.addItem(self._bars_v)
        self._plot.addItem(self._bars_i)


class ProbabilityPanel(QtWidgets.QGroupBox):
    def __init__(self, class_names: Sequence[str], parent=None) -> None:
        super().__init__("Class Probabilities", parent)
        self._class_names = list(class_names)

        layout = QtWidgets.QGridLayout(self)
        self._name_labels: list[QtWidgets.QLabel] = []
        self._bars: list[QtWidgets.QProgressBar] = []
        self._value_labels: list[QtWidgets.QLabel] = []

        for row, name in enumerate(self._class_names):
            name_label = QtWidgets.QLabel(name)
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(0)
            bar.setTextVisible(False)
            value_label = QtWidgets.QLabel("0.0%")

            layout.addWidget(name_label, row, 0)
            layout.addWidget(bar, row, 1)
            layout.addWidget(value_label, row, 2)

            self._name_labels.append(name_label)
            self._bars.append(bar)
            self._value_labels.append(value_label)

    def update_probabilities(
        self,
        class_names: Sequence[str],
        probabilities: Sequence[float],
        top1_label: str,
        active_labels: Sequence[str] | None = None,
    ) -> None:
        if len(class_names) != len(self._class_names):
            return
        if len(probabilities) != len(self._class_names):
            return

        active_set = set(active_labels) if active_labels else {top1_label}

        for idx, prob in enumerate(probabilities):
            value = max(0.0, min(1.0, float(prob)))
            self._bars[idx].setValue(int(round(value * 1000.0)))
            self._value_labels[idx].setText(f"{value * 100.0:5.1f}%")

            is_active = self._class_names[idx] in active_set
            font = self._name_labels[idx].font()
            font.setBold(is_active)
            self._name_labels[idx].setFont(font)

            if is_active:
                self._bars[idx].setStyleSheet(
                    "QProgressBar::chunk { background-color: #4CAF50; }"
                )
            else:
                self._bars[idx].setStyleSheet("")
