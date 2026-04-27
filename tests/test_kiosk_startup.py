from __future__ import annotations

from pathlib import Path
import shlex
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_TEMPLATE = REPO_ROOT / "src" / "system" / "service" / "pq-monitor.service"
KIOSK_SETUP = REPO_ROOT / "src" / "system" / "kiosk_setup.sh"


def _build_exec_start(model_path: str = "", scaler_path: str = "") -> str:
    cmd = (
        f"source {shlex.quote(str(KIOSK_SETUP))}; "
        f"build_exec_start /opt/pq-monitor {shlex.quote(model_path)} {shlex.quote(scaler_path)}"
    )
    result = subprocess.run(
        ["bash", "-c", cmd],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def test_service_template_omits_optional_artifact_flags() -> None:
    exec_start = next(
        line for line in SERVICE_TEMPLATE.read_text(encoding="utf-8").splitlines()
        if line.startswith("ExecStart=")
    )

    assert " --port ${PQ_SERIAL_PORT}" in exec_start
    assert " --config ${PQ_CONFIG_PATH}" in exec_start
    assert " --receiver-mode ${PQ_RECEIVER_MODE}" in exec_start
    assert " --model " not in exec_start
    assert " --scaler " not in exec_start


def test_kiosk_exec_start_omits_empty_artifact_flags() -> None:
    exec_start = _build_exec_start()

    assert exec_start == (
        "/opt/pq-monitor/.venv/bin/python -m src.ui.app "
        "--port ${PQ_SERIAL_PORT} "
        "--config ${PQ_CONFIG_PATH} "
        "--receiver-mode ${PQ_RECEIVER_MODE}"
    )


def test_kiosk_exec_start_includes_non_empty_artifact_flags() -> None:
    exec_start = _build_exec_start(
        model_path="artifacts/models/pq_model.joblib",
        scaler_path="artifacts/scalers/pq_scaler.joblib",
    )

    assert " --model ${PQ_MODEL_PATH}" in exec_start
    assert " --scaler ${PQ_SCALER_PATH}" in exec_start
