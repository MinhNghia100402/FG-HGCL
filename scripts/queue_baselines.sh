#!/usr/bin/env bash
# B0 (CDELDA TwoTower content) and B1 (HGCMLDA-clean) in the CDELDA harness, cold protocol, all datasets.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/sda1/nghiavm/research/.venv/bin/python
for v in l2d5 D1 D2; do $PY -u scripts/run_baseline.py --model B0 --variant $v; done
for v in D1 D2 l2d5; do $PY -u scripts/run_baseline.py --model B1 --variant $v; done
echo QUEUE_DONE
