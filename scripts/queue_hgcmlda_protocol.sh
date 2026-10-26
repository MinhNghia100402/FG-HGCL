#!/usr/bin/env bash
# FG-HGCL B3 under HGCMLDA's original 5-CV protocol (warm, 1:1), clean and leaky, D1 and D2, 3 model seeds.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/sda1/nghiavm/research/.venv/bin/python
for ds in 1 2; do for v in clean leaky; do
  $PY -u scripts/hgcmlda_protocol.py --dataset $ds --variant $v 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
done; done
echo QUEUE_DONE
