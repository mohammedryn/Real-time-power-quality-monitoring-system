from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

from pyqtgraph.Qt import QtCore, QtWidgets

from src.dsp.preprocess import load_config
from src.runtime.pipeline import ArtifactPredictor, RuntimePipeline
from src.ui.views.dashboard import DashboardView


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg: dict, pipeline: RuntimePipeline) -> None:
        super().__init__()
        self._cfg = cfg
        self._pipeline = pipeline
        self._last_seq: int | None = None

        self.setWindowTitle("PQ Monitor - Handheld Dashboard")
        self.resize(1280, 800)

        fs_hz = float(cfg["signal"]["fs_hz"])
        class_names = list(cfg["classes"]["names"])
        self._dashboard = DashboardView(class_names=class_names, fs_hz=fs_hz)
        self.setCentralWidget(self._dashboard)

        ui_fps = int(cfg.get("runtime", {}).get("ui_target_fps", 30))
        interval_ms = max(16, int(1000 / max(1, ui_fps)))

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(interval_ms)

    def _refresh(self) -> None:
        snapshot = self._pipeline.get_latest_snapshot()
        if snapshot is None:
            return
        if self._last_seq == snapshot.seq:
            return

        self._dashboard.update_snapshot(snapshot)
        self._last_seq = snapshot.seq

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API naming
        self._timer.stop()
        self._pipeline.stop()
        event.accept()


def _default_session_log(cfg: dict) -> str:
    root = Path(cfg["paths"]["live_sessions"])
    root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(root / f"ui_session_{ts}.jsonl")


def _load_theme_if_available(app: QtWidgets.QApplication, cfg: dict) -> None:
    theme_dir = Path("assets") / "ui_theme"
    qss_path = theme_dir / "style.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))
        return

    app.setStyle("Fusion")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pi touch UI for live power-quality monitoring")
    parser.add_argument("--port", required=True, help="Serial device path, e.g. /dev/ttyACM0")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument(
        "--receiver-mode",
        choices=["feature", "raw"],
        default="feature",
        help="feature: use MCU 282-feature frames, raw: host DSP fallback",
    )
    parser.add_argument("--model", default=None, help="Path to model artifact")
    parser.add_argument("--scaler", default=None, help="Optional scaler artifact")
    parser.add_argument("--session-log", default=None, help="Output JSONL session log path")
    parser.add_argument("--windowed", action="store_true", help="Run in windowed mode (no fullscreen)")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    cfg = load_config(args.config)

    predictor = ArtifactPredictor(
        class_names=list(cfg["classes"]["names"]),
        model_path=args.model,
        scaler_path=args.scaler,
    )

    session_log = args.session_log or _default_session_log(cfg)

    pipeline = RuntimePipeline(
        cfg,
        predictor,
        port=args.port,
        receiver_mode=args.receiver_mode,
        session_log_path=session_log,
    )
    pipeline.start()

    app = QtWidgets.QApplication(sys.argv)
    _load_theme_if_available(app, cfg)

    window = MainWindow(cfg, pipeline)
    if args.windowed:
        window.show()
    else:
        window.showFullScreen()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
