#!/usr/bin/env bash
# Stage 1 of docs/hypergraph_views_d12.md: association-hypergraph view on D1 / D2, seed 2026.
# usage: CUDA_VISIBLE_DEVICES=<gpu> bash scripts/queue_assoc.sh <dataset 1|2> <variant leaky|clean>
set -u
cd "$(dirname "$0")/.."
PY=/sda1/nghiavm/research/.venv/bin/python
DS=$1; VAR=$2
if [ "$DS" = 1 ]; then R0=B0f; R1=B3; LL='"link_loss": "infonce"'; else R0=B0fv; R1=B3v; LL='"link_loss": "infonce+bce"'; fi
run() {  # model tag cfg-extra
  $PY scripts/hgcmlda_protocol.py --dataset "$DS" --variant "$VAR" --model "$1" --seeds 2026 --epochs 1000 \
      --tag "as1_$2" --cfg "{\"epochs\": 1000, $LL$3}"
}
run $R0 R0 ""
run $R1 R1 ""
run $R0 F0 ', "assoc_fuse": true'
for l in 0.1 0.5 1.0; do run $R0 A$l ", \"lam_assoc\": $l"; done
run $R0 A0.5_nobce ', "lam_assoc": 0.5, "assoc_bce": false'
for l in 0.1 0.5 1.0; do run $R0 F$l ", \"lam_assoc\": $l, \"assoc_fuse\": true"; done
