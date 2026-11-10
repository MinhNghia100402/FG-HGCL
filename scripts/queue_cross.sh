#!/usr/bin/env bash
# Development stage of docs/hgcmlda_cross_cl.md (split seed 1, model seed 0).
# usage: bash scripts/queue_cross.sh <gpu> "<dataset:cl:cross> ..."   (cross 0 = off)
set -u
# CPU limit: shared server, at most 16 cores for all my jobs together (override with CPUS=...)
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
for job in $2; do
  IFS=: read -r ds cl cross <<< "$job"
  echo "### D$ds $cl cross $cross"
  taskset -c "${CPUS:-0-15}" /sda1/nghiavm/research/.venv/bin/python -u scripts/hgcmlda_cl_swap.py --dataset "$ds" --cl "$cl" --cross "$cross" --split-seed 1 2>&1 \
    | grep --line-buffered -E "\[runner\] CV|\[cl_swap\] CL|Error|Traceback"
done
echo QUEUE_DONE
