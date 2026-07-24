#!/usr/bin/env bash
set -euo pipefail

CALIBRATION_PATTERN='[p]rometheus-judge-calibration-v0.8.3'
PROJECT_DIR='/workspace/ingen_eval/phase_a_design'
PYTHON_BIN='/tmp/prometheus-venv/bin/python'
MODEL_DIR='/tmp/prometheus_7b_v2_0'
LOG_PATH='/workspace/experiments/prometheus_full_v0.8.3.log'

while pgrep -f "${CALIBRATION_PATTERN}" >/dev/null; do
  sleep 20
done

cd "${PROJECT_DIR}"
exec "${PYTHON_BIN}" W02_Prometheus_Full_Run.py \
  --run-id w02-two-model-prometheus-diagnostic-v0.8.3 \
  --model-dir "${MODEL_DIR}" \
  --allow-failed-calibration \
  >"${LOG_PATH}" 2>&1
