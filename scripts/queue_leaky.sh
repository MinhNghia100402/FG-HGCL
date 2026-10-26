#!/usr/bin/env bash
# LEAKY comparison (HGCMLDA original protocol): FG-HGCL B3 (locked) and HGCMLDA, cold protocol of CDELDA's harness.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/sda1/nghiavm/research/.venv/bin/python
for v in D1 D2 l2d5; do $PY -u scripts/run_leaky.py --model fghgcl --variant $v 2>&1 | grep --line-buffered -E "\+ FG-HGCL|ERROR|Traceback"; done
echo FGHGCL_LEAKY_DONE
for v in D1 D2 l2d5; do $PY -u scripts/run_leaky.py --model hgcmlda --variant $v 2>&1 | grep --line-buffered -E "\+ HGCMLDA|ERROR|Traceback|done in"; done
echo QUEUE_DONE
