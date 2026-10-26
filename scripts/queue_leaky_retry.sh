#!/usr/bin/env bash
# Re-run the FG-HGCL leaky seeds that failed on GMM (checkpointed names are skipped by the harness).
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/sda1/nghiavm/research/.venv/bin/python
for v in D2 l2d5; do $PY -u scripts/run_leaky.py --model fghgcl --variant $v 2>&1 | grep --line-buffered -E "\+ FG-HGCL|ERROR|Traceback"; done
echo RETRY_DONE
