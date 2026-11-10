#!/usr/bin/env bash
# docs/collab_hypergraph_d12.md: content hyperedges carrying train link profiles (hg_signal links).
# usage: CUDA_VISIBLE_DEVICES=<gpu> bash scripts/queue_collab.sh <stage 1|2> [tags...]
# stage 1: CS0, CS0.1, CS0.5 at seed 2026 on D1/D2 x leaky/clean; stage 2: given tags at seeds 2026-2028.
set -u
# CPU limit: shared server, at most 16 cores for all my jobs together (override with CPUS=...)
cd "$(dirname "$0")/.."
PY=/sda1/nghiavm/research/.venv/bin/python
STAGE=$1; shift
TAGS=${*:-CS0 CS0.1 CS0.5}
SEEDS="2026"; [ "$STAGE" = 2 ] && SEEDS="2026 2027 2028"
for DS in 1 2; do
  if [ "$DS" = 1 ]; then BASE='"link_loss": "infonce"'; NEG=3; else BASE='"dis_views": ["name", "defn", "syn"], "dis_view_feats": true, "link_loss": "infonce+bce"'; NEG=4; fi
  for VAR in leaky clean; do
    for T in $TAGS; do
      LAM=${T#CS}
      taskset -c "${CPUS:-0-15}" $PY scripts/hgcmlda_protocol.py --dataset "$DS" --variant "$VAR" --model B3 --seeds $SEEDS --epochs 1000 \
          --neg_seed $NEG --tag "cs${STAGE}_$T" --cfg "{\"epochs\": 1000, $BASE, \"hg_signal\": \"links\", \"lam_cl\": $LAM}"
    done
  done
done
echo QUEUE_DONE
