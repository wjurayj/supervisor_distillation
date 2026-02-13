#!/usr/bin/env bash
set -euo pipefail

SUPERVISOR="Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
WORKER="Qwen/Qwen2.5-7B-Instruct-Turbo"
SEED=42
N=20

echo "=== Supervisor: $SUPERVISOR | Worker: $WORKER ==="
python examples/run_experiment.py \
  --supervisor-model "$SUPERVISOR" \
  --worker-model "$WORKER" \
  --seed "$SEED" \
  -n "$N" \
  --num-distractors 19 \
  --log-dir "logs/qasper_distractors"
