#!/usr/bin/env bash
# 3-seed confirmation of InfoNCE link loss under HGCMLDA's original protocol (selection: inner validation, seed 2026):
# D1 = B3 + infonce, D2 = v2 + infonce+bce. Switch the locked config only if the 3-seed inner-val mean beats the current one.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-4} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/sda1/nghiavm/research/.venv/bin/python
V='"dis_views": ["name", "defn", "syn"], "dis_view_feats": true'
$PY -u scripts/hgcmlda_protocol.py --dataset 1 --variant leaky --seeds 2026 2027 2028 --epochs 1000 --tag final_nce \
    --cfg '{"epochs": 1000, "link_loss": "infonce"}' 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
$PY -u scripts/hgcmlda_protocol.py --dataset 2 --variant leaky --seeds 2026 2027 2028 --epochs 1000 --tag final_nceb \
    --cfg "{\"epochs\": 1000, $V, \"link_loss\": \"infonce+bce\"}" 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
echo QUEUE_DONE
