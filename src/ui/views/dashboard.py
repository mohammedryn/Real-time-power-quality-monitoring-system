from __future__ import annotations

from datetime import datetime

from pyqtgraph.Qt import QtWidgets

from src.runtime.pipeline import InferenceSnapshot
from src.ui.views.events import EventTimelineWidget
from src.ui.widgets.plots import HarmonicSpectrumPanel, ProbabilityPanel, WaveformPanel


class DashboardView(QtWidgets.QWidget):
    def __init__(self, class_names: list[str], fs_hz: float, parent=None) -> None:
        super().__init__(parent)
        self._fs_hz = fs_hz
        self._last_event_signature: tuple[str, int] | None = None

        root = QtWidgets.QVBoxLayout(self)

        top_row = QtWidgets.QHBoxLayout()
        root.addLayout(top_row, stretch=0)

        self._top1_card = QtWidgets.QLabel("Top-1: --")
        self._top1_card.setStyleSheet("font-size: 20px; font-weight: 700; padding: 8px;")

        self._timestamp = QtWidgets.QLabel("Updated: --")
        self._timestamp.setStyleSheet("font-size: 14px; padding: 8px;")

        top_row.addWidget(self._top1_card, stretch=1)
        top_row.addWidget(self._timestamp, stretch=0)

        middle = QtWidgets.QSplitter()
        middle.setOrientation(1)  # Vertical
        root.addWidget(middle, stretch=1)

        top_panels = QtWidgets.QWidget()
        top_layout = QtWidgets.QHBoxLayout(top_panels)
        top_layout.setContentsMargins(0, 0, 0, 0)

        self._waveforms = WaveformPanel()
        self._harmonics = HarmonicSpectrumPanel()

        top_layout.addWidget(self._waveforms, stretch=3)
        top_layout.addWidget(self._harmonics, stretch=2)

        bottom_panels = QtWidgets.QWidget()
        bottom_layout = QtWidgets.QHBoxLayout(bottom_panels)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        left_col = QtWidgets.QVBoxLayout()
        right_col = QtWidgets.QVBoxLayout()
        bottom_layout.addLayout(left_col, stretch=3)
        bottom_layout.addLayout(right_col, stretch=2)

        self._probabilities = ProbabilityPanel(class_names)
        left_col.addWidget(self._probabilities)

        self._metrics_group = QtWidgets.QGroupBox("Live Metrics")
        metrics_grid = QtWidgets.QGridLayout(self._metrics_group)
        self._metric_labels: dict[str, QtWidgets.QLabel] = {}
        metric_order = [
            ("rms_v", "RMS-V"),
            ("rms_i", "RMS-I"),
            ("thd_v", "THD-V"),
            ("thd_i", "THD-I"),
            ("dpf", "DPF"),
            ("pf", "PF"),
            ("frequency_hz", "Freq (Hz)"),
        ]

        for idx, (key, label) in enumerate(metric_order):
            metrics_grid.addWidget(QtWidgets.QLabel(label), idx, 0)
            value = QtWidgets.QLabel("--")
            metrics_grid.addWidget(value, idx, 1)
            self._metric_labels[key] = value

        left_col.addWidget(self._metrics_group)

        self._events = EventTimelineWidget()
        right_col.addWidget(self._events, stretch=3)

        self._health = QtWidgets.QPlainTextEdit()
        self._health.setReadOnly(True)
        self._health.setMaximumBlockCount(500)
        right_col.addWidget(self._health, stretch=2)

        middle.addWidget(top_panels)
        middle.addWidget(bottom_panels)
        middle.setSizes([550, 450])

    def update_snapshot(self, snapshot: InferenceSnapshot) -> None:
        self._top1_card.setText(
            f"Top-1: {snapshot.top1_label} ({snapshot.top1_confidence * 100.0:5.1f}%)"
        )
        ts_text = datetime.fromtimestamp(snapshot.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        self._timestamp.setText(f"Updated: {ts_text}")

        if snapshot.v_phys is not None and snapshot.i_phys is not None:
            self._waveforms.update_waveforms(snapshot.v_phys, snapshot.i_phys, self._fs_hz)

        self._harmonics.update_harmonics(snapshot.harmonics_v, snapshot.harmonics_i)
        self._probabilities.update_probabilities(
            class_names=snapshot.class_names,
            probabilities=snapshot.probabilities,
            top1_label=snapshot.top1_label,
        )

        metrics = snapshot.metrics
        self._metric_labels["rms_v"].setText(f"{metrics['rms_v']:.3f} V")
        self._metric_labels["rms_i"].setText(f"{metrics['rms_i']:.3f} A")
        self._metric_labels["thd_v"].setText(f"{metrics['thd_v'] * 100.0:.2f}%")
        self._metric_labels["thd_i"].setText(f"{metrics['thd_i'] * 100.0:.2f}%")
        self._metric_labels["dpf"].setText(f"{metrics['dpf']:.3f}")
        self._metric_labels["pf"].setText(f"{metrics['pf']:.3f}")
        self._metric_labels["frequency_hz"].setText(f"{metrics['frequency_hz']:.2f}")

        if snapshot.event is not None:
            signature = (snapshot.event["label"], int(snapshot.timestamp))
            if signature != self._last_event_signature:
                self._events.add_event(snapshot.event)
                self._last_event_signature = signature

        self._health.setPlainText(self._format_health(snapshot.health))

    def _format_health(self, health: dict) -> str:
        runtime = health.get("runtime", {})
        counters = runtime.get("counters", {})
        stages = runtime.get("stages", {})

        lines = [
            "System Health",
            "------------",
            f"Uptime: {runtime.get('uptime_sec', 0.0):.1f}s",
            f"Frames ingested: {counters.get('frames_ingested', 0)}",
            f"Frames scored: {counters.get('frames_scored', 0)}",
            f"Frames dropped (acq): {counters.get('frames_dropped_acq', 0)}",
            f"Results dropped: {counters.get('results_dropped', 0)}",
        ]

        if "serial_status" in health:
            lines.append(f"Serial status: {health['serial_status']}")

        receiver = health.get("receiver")
        if receiver is not None:
            lines.extend(
                [
                    f"Receiver accepted: {receiver.get('accepted_frames', 0)}",
                    f"CRC failures: {receiver.get('crc_failures', 0)}",
                    f"Parse failures: {receiver.get('parse_failures', 0)}",
                    f"Timeouts: {receiver.get('timeouts', 0)}",
                    f"Reconnects: {receiver.get('reconnects', 0)}",
                ]
            )

        acq = stages.get("acquisition_ms", {})
        model = stages.get("model_ms", {})
        total = stages.get("inference_total_ms", {})

        lines.extend(
            [
                "",
                "Latency",
                "-------",
                f"Acquisition mean/p95: {acq.get('mean_ms', 0.0):.2f} / {acq.get('p95_ms', 0.0):.2f} ms",
                f"Model mean/p95: {model.get('mean_ms', 0.0):.2f} / {model.get('p95_ms', 0.0):.2f} ms",
                f"End-to-end mean/p95: {total.get('mean_ms', 0.0):.2f} / {total.get('p95_ms', 0.0):.2f} ms",
            ]
        )

        if "inference_latency_ms_mean" in health:
            lines.append(
                f"Inference latency mean/p95: {health.get('inference_latency_ms_mean', 0.0):.2f} / "
                f"{health.get('inference_latency_ms_p95', 0.0):.2f} ms"
            )

        if "device_temp_c" in health:
            lines.append(f"Device temp: {health['device_temp_c']:.1f} C")

        return "\n".join(lines)
