from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import statistics
import time

import serial

TIMING_RE = re.compile(r"#TIMING\s+dsp_us=(?P<dsp>\d+)\s+total_us=(?P<total>\d+)")


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(round((p / 100.0) * (len(sorted_values) - 1)))
    return float(sorted_values[idx])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture PQ_DEBUG_TIMING logs from Teensy serial output")
    parser.add_argument("--port", required=True, help="Serial device path, e.g. /dev/ttyACM0")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud")
    parser.add_argument("--seconds", type=float, default=30.0, help="Capture duration")
    parser.add_argument(
        "--output-dir",
        default="artifacts/hardware_timing",
        help="Output directory root",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_log_path = out_dir / "timing_raw.log"
    csv_path = out_dir / "timing_samples.csv"
    summary_path = out_dir / "summary.json"

    samples: list[tuple[int, int]] = []
    start = time.time()

    with serial.Serial(args.port, args.baud, timeout=1.0) as ser, raw_log_path.open("w", encoding="utf-8") as raw_fp:
        while time.time() - start < args.seconds:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            raw_fp.write(line + "\n")

            match = TIMING_RE.search(line)
            if match is None:
                continue

            dsp_us = int(match.group("dsp"))
            total_us = int(match.group("total"))
            samples.append((dsp_us, total_us))

    with csv_path.open("w", encoding="utf-8") as csv_fp:
        csv_fp.write("dsp_us,total_us\n")
        for dsp_us, total_us in samples:
            csv_fp.write(f"{dsp_us},{total_us}\n")

    dsp_values = [float(s[0]) for s in samples]
    total_values = [float(s[1]) for s in samples]

    summary = {
        "run_id": run_id,
        "port": args.port,
        "sample_count": len(samples),
        "dsp_us": {
            "mean": statistics.fmean(dsp_values) if dsp_values else 0.0,
            "p95": _percentile(dsp_values, 95),
            "max": max(dsp_values) if dsp_values else 0.0,
        },
        "total_us": {
            "mean": statistics.fmean(total_values) if total_values else 0.0,
            "p95": _percentile(total_values, 95),
            "max": max(total_values) if total_values else 0.0,
        },
        "raw_log": str(raw_log_path),
        "csv": str(csv_path),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
