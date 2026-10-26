#!/usr/bin/env bash
# Selected config (docs/hgcmlda_selection.md): ep1000, 3 model seeds, D1 and D2, HGCMLDA original (leaky) protocol.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/sda1/nghiavm/research/.venv/bin/python
for ds in 1 2; do
  $PY -u scripts/hgcmlda_protocol.py --dataset $ds --variant leaky --seeds 2026 2027 2028 --epochs 1000 --tag final_ep1000 \
      --cfg '{"epochs": 1000}' 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
done
echo QUEUE_DONE
