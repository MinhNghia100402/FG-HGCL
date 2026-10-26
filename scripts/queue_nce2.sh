#!/usr/bin/env bash
# InfoNCE as main link loss, evaluated where each paper reports: D3 = CDELDA harness (inner validation here),
# D1 / D2 = HGCMLDA original protocol (selection on inner validation inside each fold's train part, seed 2026).
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-3} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/sda1/nghiavm/research/.venv/bin/python
$PY scripts/run_ablation.py inner --variant l2d5 --ids B3v B3_nce B3_nce_t05 B3_nceb B3v_nce B3v_nceb 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
echo "#### report l2d5 (vs B3)"; $PY scripts/paired.py B3 B0f B3v B3_nce B3_nce_t05 B3_nceb B3v_nce B3v_nceb
echo "#### report l2d5 (vs B3v = both-cold of FG-HGCL v2)"; $PY scripts/paired.py "both|B3_vfeat" B3v B3v_nce B3v_nceb
V='"dis_views": ["name", "defn", "syn"], "dis_view_feats": true'
for ds in 1 2; do
  base='"epochs": 1000'; [ $ds = 2 ] && base="\"epochs\": 1000, $V"      # current locked config: D1 = B3, D2 = v2
  for t in nce nceb; do
    ll=infonce; [ $t = nceb ] && ll=infonce+bce
    $PY -u scripts/hgcmlda_protocol.py --dataset $ds --variant leaky --seeds 2026 --epochs 1000 --tag "sel_$t" \
        --cfg "{$base, \"link_loss\": \"$ll\"}" 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
  done
done
echo QUEUE_DONE
