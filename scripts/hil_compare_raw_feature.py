from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dsp.feature_index import FEATURE_INDEX, PARITY_TOLERANCES, TOTAL_FEATURES
from src.dsp.features import extract_features
from src.dsp.preprocess import load_config, preprocess_frame
from src.io.serial_receiver import SerialFrameReceiver


# Anchor features used for optional cross-run matching.
# These are stable physics metrics and help pair similar frames when captures
# are not contemporaneous (e.g., after a firmware reflash).
ANCHOR_INDEX = {
    "v_rms": 2,
    "i_rms": 14,
    "thd_v": 50,
    "thd_i": 51,
}


def _capture_raw_features(
    port: str,
    cfg: dict,
    target_frames: int,
    timeout: float,
) -> tuple[np.ndarray, list[int]]:
    receiver = SerialFrameReceiver(port=port, timeout=timeout, mode="raw")
    expected_n = int(cfg["signal"]["samples_per_frame"])

    vectors: list[np.ndarray] = []
    seqs: list[int] = []

    receiver.open()
    try:
        while len(vectors) < target_frames:
            frame = receiver.read_frame(frame_timeout=timeout)
            if frame is None:
                continue

            processed = preprocess_frame(frame.v_raw, frame.i_raw, cfg, expected_n=expected_n)
            vector = extract_features(processed["v_phys"], processed["i_phys"]).astype(np.float32)
            if vector.shape != (TOTAL_FEATURES,):
                continue

            vectors.append(vector)
            seqs.append(int(frame.seq))
    finally:
        receiver.close()

    if not vectors:
        raise RuntimeError("No raw frames captured")

    return np.stack(vectors, axis=0), seqs


def _capture_feature_vectors(
    port: str,
    target_frames: int,
    timeout: float,
) -> tuple[np.ndarray, list[int]]:
    receiver = SerialFrameReceiver(port=port, timeout=timeout, mode="feature")

    vectors: list[np.ndarray] = []
    seqs: list[int] = []

    receiver.open()
    try:
        while len(vectors) < target_frames:
            frame = receiver.read_frame(frame_timeout=timeout)
            if frame is None:
                continue

            vector = np.asarray(frame.features, dtype=np.float32)
            if vector.shape != (TOTAL_FEATURES,):
                continue

            vectors.append(vector)
            seqs.append(int(frame.seq))
    finally:
        receiver.close()

    if not vectors:
        raise RuntimeError("No feature frames captured")

    return np.stack(vectors, axis=0), seqs


def _extract_anchor_matrix(vectors: np.ndarray) -> np.ndarray:
    cols = [ANCHOR_INDEX[name] for name in ("v_rms", "i_rms", "thd_v", "thd_i")]
    return vectors[:, cols]


def _pair_by_index(raw_vectors: np.ndarray, mcu_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair_count = min(len(raw_vectors), len(mcu_vectors))
    idx = np.arange(pair_count, dtype=np.int64)
    return idx, idx, np.zeros(pair_count, dtype=np.float64)


def _pair_by_anchor(raw_vectors: np.ndarray, mcu_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_anchor = _extract_anchor_matrix(raw_vectors)
    mcu_anchor = _extract_anchor_matrix(mcu_vectors)

    # Robustly normalize anchor dimensions using combined median/MAD.
    combined = np.vstack([raw_anchor, mcu_anchor])
    med = np.median(combined, axis=0)
    mad = np.median(np.abs(combined - med), axis=0)
    std = np.std(combined, axis=0)
    scale = np.where(mad > 1e-9, mad, std)
    scale = np.where(scale > 1e-9, scale, 1.0)

    raw_n = (raw_anchor - med) / scale
    mcu_n = (mcu_anchor - med) / scale

    # Pair greedily by smallest anchor distance, one-to-one.
    dmat = np.linalg.norm(raw_n[:, None, :] - mcu_n[None, :, :], axis=2)
    candidates: list[tuple[float, int, int]] = []
    for i in range(dmat.shape[0]):
        for j in range(dmat.shape[1]):
            candidates.append((float(dmat[i, j]), i, j))
    candidates.sort(key=lambda x: x[0])

    used_raw: set[int] = set()
    used_mcu: set[int] = set()
    pair_count = min(len(raw_vectors), len(mcu_vectors))

    raw_idx: list[int] = []
    mcu_idx: list[int] = []
    distances: list[float] = []

    for dist, i, j in candidates:
        if i in used_raw or j in used_mcu:
            continue
        used_raw.add(i)
        used_mcu.add(j)
        raw_idx.append(i)
        mcu_idx.append(j)
        distances.append(dist)
        if len(raw_idx) >= pair_count:
            break

    return (
        np.asarray(raw_idx, dtype=np.int64),
        np.asarray(mcu_idx, dtype=np.int64),
        np.asarray(distances, dtype=np.float64),
    )


def _build_pairs(
    raw_vectors: np.ndarray,
    mcu_vectors: np.ndarray,
    pairing: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pairing == "index":
        return _pair_by_index(raw_vectors, mcu_vectors)
    if pairing == "anchor":
        return _pair_by_anchor(raw_vectors, mcu_vectors)
    raise ValueError(f"Unknown pairing mode: {pairing}")


def _pairwise_metrics(
    raw_vectors: np.ndarray,
    mcu_vectors: np.ndarray,
    pairing: str,
) -> dict[str, Any]:
    raw_idx, mcu_idx, pair_dist = _build_pairs(raw_vectors, mcu_vectors, pairing)
    pair_count = len(raw_idx)

    raw = raw_vectors[raw_idx]
    mcu = mcu_vectors[mcu_idx]

    abs_diff = np.abs(raw - mcu)
    frame_max_abs = abs_diff.max(axis=1)
    frame_mean_abs = abs_diff.mean(axis=1)

    slice_metrics: list[dict] = []
    for name, (start, stop) in FEATURE_INDEX.items():
        section = abs_diff[:, start:stop]
        section_frame_max = section.max(axis=1)
        tol = float(PARITY_TOLERANCES.get(name, 1e-3))

        slice_metrics.append(
            {
                "slice": name,
                "start": start,
                "stop": stop,
                "tolerance": tol,
                "max_abs": float(section.max()),
                "mean_abs": float(section.mean()),
                "p95_abs": float(np.quantile(section, 0.95)),
                "pass_rate": float(np.mean(section_frame_max <= tol)),
            }
        )

    summary: dict[str, Any] = {
        "pair_count": int(pair_count),
        "global_max_abs": float(abs_diff.max()),
        "global_mean_abs": float(abs_diff.mean()),
        "frame_max_abs_mean": float(frame_max_abs.mean()),
        "frame_max_abs_p95": float(np.quantile(frame_max_abs, 0.95)),
        "frame_max_abs_max": float(frame_max_abs.max()),
        "frame_mean_abs_mean": float(frame_mean_abs.mean()),
        "pairing": {
            "method": pairing,
            "distance_mean": float(pair_dist.mean()) if len(pair_dist) else 0.0,
            "distance_p95": float(np.quantile(pair_dist, 0.95)) if len(pair_dist) else 0.0,
            "distance_max": float(pair_dist.max()) if len(pair_dist) else 0.0,
            "raw_indices": raw_idx.tolist(),
            "mcu_indices": mcu_idx.tolist(),
            "distances": pair_dist.tolist(),
        },
        "slice_metrics": slice_metrics,
    }

    return summary


def _distribution_metrics(raw_vectors: np.ndarray, mcu_vectors: np.ndarray) -> dict[str, Any]:
    # Distribution metrics do not require pairing. They compare aggregate
    # feature populations captured in each mode.
    per_feature_mean_delta = np.abs(raw_vectors.mean(axis=0) - mcu_vectors.mean(axis=0))

    slice_metrics: list[dict[str, Any]] = []
    for name, (start, stop) in FEATURE_INDEX.items():
        raw_slice = raw_vectors[:, start:stop].reshape(-1)
        mcu_slice = mcu_vectors[:, start:stop].reshape(-1)

        row = {
            "slice": name,
            "start": start,
            "stop": stop,
            "raw_mean": float(np.mean(raw_slice)),
            "mcu_mean": float(np.mean(mcu_slice)),
            "abs_mean_delta": float(abs(np.mean(raw_slice) - np.mean(mcu_slice))),
            "raw_std": float(np.std(raw_slice)),
            "mcu_std": float(np.std(mcu_slice)),
            "abs_std_delta": float(abs(np.std(raw_slice) - np.std(mcu_slice))),
            "raw_p95": float(np.quantile(raw_slice, 0.95)),
            "mcu_p95": float(np.quantile(mcu_slice, 0.95)),
            "abs_p95_delta": float(abs(np.quantile(raw_slice, 0.95) - np.quantile(mcu_slice, 0.95))),
        }
        slice_metrics.append(row)

    return {
        "raw_frame_count": int(len(raw_vectors)),
        "feature_frame_count": int(len(mcu_vectors)),
        "feature_mean_abs_delta_mean": float(np.mean(per_feature_mean_delta)),
        "feature_mean_abs_delta_p95": float(np.quantile(per_feature_mean_delta, 0.95)),
        "feature_mean_abs_delta_max": float(np.max(per_feature_mean_delta)),
        "slice_metrics": slice_metrics,
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_reports(
    output_dir: Path,
    raw_vectors: np.ndarray,
    mcu_vectors: np.ndarray,
    raw_seqs: list[int],
    mcu_seqs: list[int],
    pairwise_summary: dict[str, Any],
    distribution_summary: dict[str, Any],
    baseline_summary: dict[str, Any] | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "raw_mode_features.npy", raw_vectors)
    np.save(output_dir / "feature_mode_vectors.npy", mcu_vectors)

    metadata = {
        "raw_frame_count": int(len(raw_vectors)),
        "feature_frame_count": int(len(mcu_vectors)),
        "raw_seq_first": raw_seqs[0] if raw_seqs else None,
        "raw_seq_last": raw_seqs[-1] if raw_seqs else None,
        "feature_seq_first": mcu_seqs[0] if mcu_seqs else None,
        "feature_seq_last": mcu_seqs[-1] if mcu_seqs else None,
        "pairwise_comparison": pairwise_summary,
        "distribution_comparison": distribution_summary,
        "baseline_raw_drift": baseline_summary,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    _write_csv(
        output_dir / "pairwise_slice_metrics.csv",
        ["slice", "start", "stop", "tolerance", "max_abs", "mean_abs", "p95_abs", "pass_rate"],
        pairwise_summary["slice_metrics"],
    )
    # Backward-compatible name for existing workflow references.
    _write_csv(
        output_dir / "slice_metrics.csv",
        ["slice", "start", "stop", "tolerance", "max_abs", "mean_abs", "p95_abs", "pass_rate"],
        pairwise_summary["slice_metrics"],
    )

    _write_csv(
        output_dir / "distribution_slice_metrics.csv",
        [
            "slice", "start", "stop",
            "raw_mean", "mcu_mean", "abs_mean_delta",
            "raw_std", "mcu_std", "abs_std_delta",
            "raw_p95", "mcu_p95", "abs_p95_delta",
        ],
        distribution_summary["slice_metrics"],
    )

    if baseline_summary is not None:
        _write_csv(
            output_dir / "baseline_raw_drift_slice_metrics.csv",
            ["slice", "start", "stop", "tolerance", "max_abs", "mean_abs", "p95_abs", "pass_rate"],
            baseline_summary["slice_metrics"],
        )

    pair_info = pairwise_summary.get("pairing", {})
    pairs_rows = []
    raw_indices = pair_info.get("raw_indices", [])
    mcu_indices = pair_info.get("mcu_indices", [])
    distances = pair_info.get("distances", [])
    for i in range(min(len(raw_indices), len(mcu_indices), len(distances))):
        pairs_rows.append(
            {
                "pair_id": i,
                "raw_index": raw_indices[i],
                "feature_index": mcu_indices[i],
                "anchor_distance": distances[i],
            }
        )
    if pairs_rows:
        _write_csv(
            output_dir / "pairing_matches.csv",
            ["pair_id", "raw_index", "feature_index", "anchor_distance"],
            pairs_rows,
        )


def _wait_for_user(message: str, skip_prompts: bool) -> None:
    print(message)
    if skip_prompts:
        return
    input("Press Enter when ready...")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hardware-in-loop parity: compare raw-mode host features vs feature-mode Teensy vectors",
    )
    parser.add_argument("--port", required=True, help="Serial device path, e.g. /dev/ttyACM0")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to configuration file")
    parser.add_argument("--frames", type=int, default=50, help="Frames to capture in each mode")
    parser.add_argument("--timeout", type=float, default=1.0, help="Per-frame timeout in seconds")
    parser.add_argument(
        "--pairing",
        choices=["anchor", "index"],
        default="anchor",
        help="Pairing method for cross-run frame comparison",
    )
    parser.add_argument(
        "--baseline-raw-drift",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture a second raw-mode batch to estimate natural source drift floor",
    )
    parser.add_argument(
        "--baseline-frames",
        type=int,
        default=None,
        help="Optional frame count for baseline raw-drift run (defaults to --frames)",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/hil_parity",
        help="Directory for output logs and comparison artifacts",
    )
    parser.add_argument(
        "--skip-prompts",
        action="store_true",
        help="Do not pause for flashing prompts",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)

    _wait_for_user(
        "Step 1: Flash Teensy with PQ_RAW_MODE=1. Keep a stable source/load connected.",
        args.skip_prompts,
    )
    t0 = time.time()
    raw_vectors, raw_seqs = _capture_raw_features(args.port, cfg, args.frames, args.timeout)
    print(f"Captured {len(raw_vectors)} raw-mode frames in {time.time() - t0:.2f}s")

    baseline_summary: dict[str, Any] | None = None
    if args.baseline_raw_drift:
        baseline_frames = int(args.baseline_frames or args.frames)
        _wait_for_user(
            "Step 1B: Keep PQ_RAW_MODE=1 and capture a second raw batch for baseline drift.",
            args.skip_prompts,
        )
        t0b = time.time()
        raw_vectors_b, _raw_seqs_b = _capture_raw_features(args.port, cfg, baseline_frames, args.timeout)
        print(f"Captured {len(raw_vectors_b)} baseline raw-drift frames in {time.time() - t0b:.2f}s")
        baseline_summary = _pairwise_metrics(raw_vectors, raw_vectors_b, pairing=args.pairing)

    _wait_for_user(
        "Step 2: Flash Teensy with feature mode (PQ_RAW_MODE=0) using the same source/load.",
        args.skip_prompts,
    )
    t1 = time.time()
    mcu_vectors, mcu_seqs = _capture_feature_vectors(args.port, args.frames, args.timeout)
    print(f"Captured {len(mcu_vectors)} feature-mode frames in {time.time() - t1:.2f}s")

    pairwise_summary = _pairwise_metrics(raw_vectors, mcu_vectors, pairing=args.pairing)
    distribution_summary = _distribution_metrics(raw_vectors, mcu_vectors)
    _write_reports(
        output_dir,
        raw_vectors,
        mcu_vectors,
        raw_seqs,
        mcu_seqs,
        pairwise_summary,
        distribution_summary,
        baseline_summary,
    )

    print("\nComparison summary")
    print(f"pairing method:     {pairwise_summary['pairing']['method']}")
    print(f"paired frames:      {pairwise_summary['pair_count']}")
    print(f"pairwise max abs:   {pairwise_summary['global_max_abs']:.6f}")
    print(f"pairwise mean abs:  {pairwise_summary['global_mean_abs']:.6f}")
    print(f"pairwise p95 frame: {pairwise_summary['frame_max_abs_p95']:.6f}")
    print(f"distribution mean(|delta(feature means)|): {distribution_summary['feature_mean_abs_delta_mean']:.6f}")
    print(f"distribution p95(|delta(feature means)|):  {distribution_summary['feature_mean_abs_delta_p95']:.6f}")
    if baseline_summary is not None:
        print(f"baseline raw-drift p95 frame max abs:      {baseline_summary['frame_max_abs_p95']:.6f}")
    print("note: pairwise metrics compare non-contemporaneous capture batches; use distribution and baseline drift metrics for robust parity interpretation")
    print(f"artifacts written to {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
