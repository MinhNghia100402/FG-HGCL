#!/usr/bin/env bash
# Final test (docs/final_lock.md): B0f, B2, B3, B7 x 3 model seeds x {l2d5, D1, D2} in CDELDA's harness.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/sda1/nghiavm/research/.venv/bin/python
IDS="B0f B2 B3 B7"
$PY -u scripts/run_ablation.py harness --variant l2d5 --ids $IDS 2>&1 | grep --line-buffered -E "\+ FG-HGCL|ERROR|Traceback"
for v in D1 D2; do
  $PY -u scripts/run_ablation.py harness --variant $v --ids $IDS --suffix _fine --override '{"eval_every": 2}' 2>&1 | grep --line-buffered -E "\+ FG-HGCL|ERROR|Traceback"
done
echo QUEUE_DONE
