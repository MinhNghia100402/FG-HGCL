#!/usr/bin/env bash
# Stage 2 of docs/hypergraph_views_d12.md: selected candidate (A0.1) + references R0, F0, seeds 2026-2028.
# usage: CUDA_VISIBLE_DEVICES=<gpu> bash scripts/queue_assoc2.sh <dataset 1|2> <variant leaky|clean>
set -u
# CPU limit: shared server, at most 16 cores for all my jobs together (override with CPUS=...)
cd "$(dirname "$0")/.."
PY=/sda1/nghiavm/research/.venv/bin/python
DS=$1; VAR=$2
if [ "$DS" = 1 ]; then R0=B0f; LL='"link_loss": "infonce"'; else R0=B0fv; LL='"link_loss": "infonce+bce"'; fi
run() {
  taskset -c "${CPUS:-0-15}" $PY scripts/hgcmlda_protocol.py --dataset "$DS" --variant "$VAR" --model "$R0" --seeds 2026 2027 2028 --epochs 1000 \
      --tag "as2_$1" --cfg "{\"epochs\": 1000, $LL$2}"
}
run A0.1 ', "lam_assoc": 0.1'
run R0 ""
run F0 ', "assoc_fuse": true'
