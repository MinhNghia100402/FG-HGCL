#!/usr/bin/env bash
# D1, D2 re-run with a fine epoch grid (eval every 2 epochs, 150 epochs): on the small datasets every model peaked at
# the first point of the 10-epoch grid, so the common-epoch comparison was not reliable.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/sda1/nghiavm/research/.venv/bin/python
IDS="B2 B3 B4 B5 B7 B8 B9 B9u B3_bilin C2"
OV='{"epochs": 150, "eval_every": 2}'
for v in D1 D2; do
  $PY scripts/run_ablation.py inner --variant $v --ids B0f $IDS --override "$OV" --suffix _fine 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
done
for v in D1 D2; do echo "#### report $v (fine)"; $PY scripts/run_ablation.py report --variant $v --ref B0f --ids $IDS --suffix _fine; done
echo QUEUE_DONE
