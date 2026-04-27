from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import struct
import sys
from typing import Iterable, Iterator, Union

import numpy as np

from src.dsp.preprocess import load_config
from src.io.frame_protocol import (
    N_FEATURES,
    N_SAMPLES,
    FeatureFrame,
    ParsedFrame,
    iter_frames_from_bytes,
    parse_feature_frame,
    parse_frame,
)
from src.runtime.pipeline import ArtifactPredictor, RuntimePipeline


def _default_session_log(cfg: dict) -> str:
    root = Path(cfg["paths"]["live_sessions"])
    root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(root / f"replay_{ts}.jsonl")


def _replay_from_npy(path: Path) -> Iterator[dict]:
    array = np.load(path)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[1] != N_FEATURES:
        raise ValueError(f"Expected npy shape (*, {N_FEATURES}), got {array.shape}")

    for idx, row in enumerate(array):
        yield {"seq": idx, "features": row.astype(np.float32)}


def _replay_from_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fp:
        for idx, line in enumerate(fp):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "seq" not in payload:
                payload["seq"] = idx
            yield payload


def _replay_from_binary(path: Path) -> Iterator[Union[FeatureFrame, ParsedFrame]]:
    blob = path.read_bytes()
    for frame_bytes in iter_frames_from_bytes(blob):
        _, n = struct.unpack_from("<HH", frame_bytes, 4)
        if n == N_FEATURES:
            yield parse_feature_frame(frame_bytes)
        elif n == N_SAMPLES:
            yield parse_frame(frame_bytes)


def load_replay_source(path: str) -> Iterable[Union[FeatureFrame, ParsedFrame, dict, np.ndarray]]:
    replay_path = Path(path)
    if not replay_path.exists():
        raise FileNotFoundError(f"Replay input not found: {replay_path}")

    suffix = replay_path.suffix.lower()
    if suffix == ".npy":
        return _replay_from_npy(replay_path)
    if suffix in {".jsonl", ".json"}:
        return _replay_from_jsonl(replay_path)
    if suffix in {".bin", ".dat"}:
        return _replay_from_binary(replay_path)

    raise ValueError("Unsupported replay input format. Use .npy, .jsonl, or .bin")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline replay inference")
    parser.add_argument("--input", required=True, help="Replay input (.npy/.jsonl/.bin)")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--model", default=None, help="Path to model artifact")
    parser.add_argument("--scaler", default=None, help="Optional scaler artifact path")
    parser.add_argument("--session-log", default=None, help="Output JSONL session log path")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N scored frames (0 = all)")
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
    replay_source = load_replay_source(args.input)

    pipeline = RuntimePipeline(
        cfg,
        predictor,
        replay_source=replay_source,
        session_log_path=session_log,
    )

    print(f"[replay] input={args.input}")
    print(f"[replay] session_log={session_log}")

    scored = 0
    pipeline.start()
    try:
        while True:
            snapshot = pipeline.get_result(timeout=0.5)
            if snapshot is not None:
                scored += 1
                print(
                    f"seq={snapshot.seq:>6} "
                    f"top1={snapshot.top1_label:<20} "
                    f"conf={snapshot.top1_confidence:0.3f}"
                )
                if args.max_frames > 0 and scored >= args.max_frames:
                    break

            if pipeline.source_exhausted and pipeline.pending_results() == 0:
                break
    finally:
        pipeline.stop()

    print(f"[replay] scored_frames={scored}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
