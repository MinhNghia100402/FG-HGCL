#!/usr/bin/env bash
# Ablation: link loss of the no-hypergraph FG-HGCL model (R0 of docs/hypergraph_views_d12.md) on D1/D2, leaky.
# R0 itself (as2_R0) uses infonce (D1) / infonce+bce (D2); this runs bce only, seeds 2026-2028.
# usage: CUDA_VISIBLE_DEVICES=<gpu> bash scripts/queue_linkloss.sh <dataset 1|2>
set -u
# CPU limit: shared server, at most 16 cores for all my jobs together (override with CPUS=...)
cd "$(dirname "$0")/.."
DS=$1
if [ "$DS" = 1 ]; then R0=B0f; else R0=B0fv; fi
taskset -c "${CPUS:-0-15}" /sda1/nghiavm/research/.venv/bin/python scripts/hgcmlda_protocol.py --dataset "$DS" --variant leaky --model "$R0" \
  --seeds 2026 2027 2028 --epochs 1000 --tag ll_bce --cfg '{"epochs": 1000, "link_loss": "bce"}'
