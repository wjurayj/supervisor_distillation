#!/usr/bin/env bash
set -euo pipefail

SUPERVISOR="Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
WORKER="Qwen/Qwen2.5-7B-Instruct-Turbo"
SEED=42
N=20

FLAGS=(
  "structured-jobs"
  "structured-output"
  "builtin-chunking"
  "explicit-convergence"
  "synthesis-cot"
)

run_config() {
  local label="$1"
  local flags="$2"
  local distractors="$3"
  local dname="d${distractors}"

  echo "=== [${dname}] ${label} ==="
  # shellcheck disable=SC2086
  python examples/run_experiment.py \
    --supervisor-model "$SUPERVISOR" \
    --worker-model "$WORKER" \
    --seed "$SEED" \
    -n "$N" \
    --num-distractors "$distractors" \
    --log-dir "logs/ablation_${dname}/${label}" \
    $flags
  echo ""
}

for d in 0 9; do
  # 1. none (baseline)
  run_config "none" "" "$d"

  # 2. all flags on
  run_config "all" "--all-minions" "$d"

  # 3-7. all-but-one
  for f in "${FLAGS[@]}"; do
    all_but=""
    for g in "${FLAGS[@]}"; do
      if [ "$g" != "$f" ]; then
        all_but="${all_but} --${g}"
      fi
    done
    run_config "all-but-${f}" "$all_but" "$d"
  done

  # 8-12. only-one
  for f in "${FLAGS[@]}"; do
    run_config "only-${f}" "--${f}" "$d"
  done
done

echo "=== Ablation complete ==="
