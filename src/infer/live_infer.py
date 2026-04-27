from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

from src.dsp.preprocess import load_config
from src.runtime.pipeline import ArtifactPredictor, RuntimePipeline


def _default_session_log(cfg: dict) -> str:
    root = Path(cfg["paths"]["live_sessions"])
    root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(root / f"session_{ts}.jsonl")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live power-quality inference runner")
    parser.add_argument("--port", required=True, help="Serial device path, e.g. /dev/ttyACM0")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument(
        "--receiver-mode",
        choices=["feature", "raw"],
        default="feature",
        help="feature: use MCU 282-feature frames, raw: host DSP fallback",
    )
    parser.add_argument("--model", default=None, help="Path to model artifact (.joblib/.pkl/.keras/.h5)")
    parser.add_argument("--scaler", default=None, help="Optional scaler artifact path")
    parser.add_argument("--session-log", default=None, help="Output JSONL session log path")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N scored frames (0 = infinite)")
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

    print(f"[live] receiver_mode={args.receiver_mode} port={args.port}")
    print(f"[live] session_log={session_log}")

    scored = 0
    pipeline.start()
    try:
        while True:
            snapshot = pipeline.get_result(timeout=1.0)
            if snapshot is None:
                continue

            scored += 1
            print(
                f"seq={snapshot.seq:>6} "
                f"top1={snapshot.top1_label:<20} "
                f"conf={snapshot.top1_confidence:0.3f} "
                f"RMS_V={snapshot.metrics['rms_v']:8.3f} "
                f"THD_V={snapshot.metrics['thd_v']*100:6.2f}%"
            )

            if args.max_frames > 0 and scored >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()

    print(f"[live] scored_frames={scored}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
