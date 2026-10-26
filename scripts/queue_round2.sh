#!/usr/bin/env bash
# Round 2: v22-style hypergraph tower (inner placement, z-score, k 20) under strict Protocol A; small datasets at lr 3e-4.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/sda1/nghiavm/research/.venv/bin/python
NEW="V22_hg V22 B3i B7i B7i_C2"
OLD="B3 B7"
SMALL='{"lr": 0.0003, "epochs": 300, "eval_every": 4}'
$PY scripts/run_ablation.py inner --variant l2d5 --ids $NEW 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
for v in D1 D2; do
  $PY scripts/run_ablation.py inner --variant $v --ids B0f $OLD $NEW --override "$SMALL" --suffix _lr3 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
done
echo "#### report l2d5"; $PY scripts/run_ablation.py report --variant l2d5 --ref B0f --ids B3 B7 $NEW
for v in D1 D2; do echo "#### report $v (lr 3e-4)"; $PY scripts/run_ablation.py report --variant $v --ref B0f --ids $OLD $NEW --suffix _lr3; done
echo QUEUE_DONE
