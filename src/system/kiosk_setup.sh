#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sudo ./src/system/kiosk_setup.sh --repo /opt/pq-monitor [options]

Options:
  --repo PATH            Repository root on Pi (required)
  --user USER            Service user (default: pi)
  --port DEVICE          Serial device path (default: /dev/ttyACM0)
  --config PATH          Config path relative to repo (default: configs/default.yaml)
  --receiver-mode MODE   feature or raw (default: feature)
  --model PATH           Model artifact path relative to repo
  --scaler PATH          Scaler artifact path relative to repo
EOF
}

build_exec_start() {
  local repo_root="$1"
  local model_path="${2:-}"
  local scaler_path="${3:-}"
  local exec_start

  exec_start="$repo_root/.venv/bin/python -m src.ui.app --port \${PQ_SERIAL_PORT} --config \${PQ_CONFIG_PATH} --receiver-mode \${PQ_RECEIVER_MODE}"
  if [[ -n "$model_path" ]]; then
    exec_start+=" --model \${PQ_MODEL_PATH}"
  fi
  if [[ -n "$scaler_path" ]]; then
    exec_start+=" --scaler \${PQ_SCALER_PATH}"
  fi

  printf '%s\n' "$exec_start"
}

main() {
  local REPO_ROOT=""
  local RUN_USER="pi"
  local SERIAL_PORT="/dev/ttyACM0"
  local CONFIG_PATH="configs/default.yaml"
  local RECEIVER_MODE="feature"
  local MODEL_PATH=""
  local SCALER_PATH=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --repo)
        REPO_ROOT="$2"; shift 2 ;;
      --user)
        RUN_USER="$2"; shift 2 ;;
      --port)
        SERIAL_PORT="$2"; shift 2 ;;
      --config)
        CONFIG_PATH="$2"; shift 2 ;;
      --receiver-mode)
        RECEIVER_MODE="$2"; shift 2 ;;
      --model)
        MODEL_PATH="$2"; shift 2 ;;
      --scaler)
        SCALER_PATH="$2"; shift 2 ;;
      -h|--help)
        usage; exit 0 ;;
      *)
        echo "Unknown option: $1"
        usage
        exit 1 ;;
    esac
  done

  if [[ -z "$REPO_ROOT" ]]; then
    echo "--repo is required"
    usage
    exit 1
  fi

  if [[ ! -d "$REPO_ROOT" ]]; then
    echo "Repository path does not exist: $REPO_ROOT"
    exit 1
  fi

  if [[ "$EUID" -ne 0 ]]; then
    echo "Run as root (use sudo)"
    exit 1
  fi

  local SERVICE_TEMPLATE="$REPO_ROOT/src/system/service/pq-monitor.service"
  local LOGROTATE_TEMPLATE="$REPO_ROOT/src/system/service/pq-monitor.logrotate"

  if [[ ! -f "$SERVICE_TEMPLATE" ]]; then
    echo "Missing service template: $SERVICE_TEMPLATE"
    exit 1
  fi

  install -d -m 0755 /var/log/pq-monitor
  chown "$RUN_USER":"$RUN_USER" /var/log/pq-monitor

  cat >/etc/default/pq-monitor <<EOF
PQ_SERIAL_PORT=$SERIAL_PORT
PQ_CONFIG_PATH=$CONFIG_PATH
PQ_RECEIVER_MODE=$RECEIVER_MODE
PQ_MODEL_PATH=$MODEL_PATH
PQ_SCALER_PATH=$SCALER_PATH
EOF

  local EXEC_START
  EXEC_START="$(build_exec_start "$REPO_ROOT" "$MODEL_PATH" "$SCALER_PATH")"

  sed \
    -e "s|^User=.*$|User=$RUN_USER|" \
    -e "s|^Group=.*$|Group=$RUN_USER|" \
    -e "s|^WorkingDirectory=.*$|WorkingDirectory=$REPO_ROOT|" \
    -e "s|^ExecStart=.*$|ExecStart=$EXEC_START|" \
    "$SERVICE_TEMPLATE" >/etc/systemd/system/pq-monitor.service

  if [[ -f "$LOGROTATE_TEMPLATE" ]]; then
    install -m 0644 "$LOGROTATE_TEMPLATE" /etc/logrotate.d/pq-monitor
  fi

  systemctl daemon-reload
  systemctl enable pq-monitor.service
  systemctl restart pq-monitor.service

  echo "kiosk setup completed"
  systemctl status pq-monitor.service --no-pager -n 20
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
