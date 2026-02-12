#!/usr/bin/env bash
set -euo pipefail

SUPERVISORS=(
  "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
  "meta-llama/Llama-3.3-70B-Instruct-Turbo"
)

WORKERS=(
  "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
  "Qwen/Qwen2.5-7B-Instruct-Turbo"
)

SEED=42
N=20

for sup in "${SUPERVISORS[@]}"; do
  for wrk in "${WORKERS[@]}"; do
    echo "=== Supervisor: $sup | Worker: $wrk ==="
    python examples/run_experiment.py \
      --supervisor-model "$sup" \
      --worker-model "$wrk" \
      --seed "$SEED" \
      -n "$N"
    echo ""
  done
done
