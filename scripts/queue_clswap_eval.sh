#!/usr/bin/env bash
# Evaluation stage of docs/hgcmlda_cl_swap.md. usage: bash scripts/queue_clswap_eval.sh <gpu> "<dataset:cl:tau:seed:splitseed> ..."
set -u
# CPU limit: shared server, at most 16 cores for all my jobs together (override with CPUS=...)
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for job in $2; do
  IFS=: read -r ds cl tau seed sps <<< "$job"
  echo "### D$ds $cl tau $tau seed $seed split $sps"
  taskset -c "${CPUS:-0-15}" /sda1/nghiavm/research/.venv/bin/python -u scripts/hgcmlda_cl_swap.py --dataset "$ds" --cl "$cl" --tau "$tau" --seed "$seed" --split-seed "$sps" 2>&1 \
    | grep --line-buffered -E "\[runner\] CV|\[cl_swap\]|Error|Traceback"
done
echo QUEUE_DONE
