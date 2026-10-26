#!/usr/bin/env bash
# Reproduce HGCMLDA (released code, original leaky protocol) on Dataset2, seeds 0-2, with the HGCMLDA repo's own runner.
# Results are written by that runner to /sda1/nghiavm/research/HGCMLDA/results/<run_id>/ (metrics.json, split.npz).
set -uo pipefail
cd /sda1/nghiavm/research/HGCMLDA
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for s in 0 1 2; do
  /sda1/nghiavm/research/.venv/bin/python -u scripts/run_hgcmlda.py --dataset 2 --variant original --seed $s 2>&1 | grep --line-buffered -E "\[runner\]|Error|Traceback"
done
echo QUEUE_DONE
