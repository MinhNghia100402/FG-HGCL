#!/usr/bin/env bash
# FG-HGCL v2 (docs/final_lock_v2.md): D3 harness test round 2 (v2 + ablation B0fv), then D1/D2 HGCMLDA-protocol check.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/sda1/nghiavm/research/.venv/bin/python
export CCDIFF_SWEEP_RESULTS=$PWD/results/harness/fg_hgcl_v2
cfg(){ $PY -c "import sys,json;sys.argv=['x'];sys.path.insert(0,'scripts');import run_ablation as r;print(json.dumps({**r.config('$1'),**json.loads('$2')}))"; }
SCN='{"disease": {"m": 128, "epochs": 490}, "lncRNA": {"dis_views": [], "dis_view_feats": false, "epochs": 400}}'
SCN0='{"disease": {"epochs": 460}, "lncRNA": {"epochs": 210}}'
for s in 2026 2027 2028; do
  $PY -u scripts/run_harness.py --name "FG-HGCL v2 s$s" --variant l2d5 --fixed --cfg "$(cfg B3v "{\"epochs\": 230, \"seed\": $s}")" --scn-cfg "$SCN" 2>&1 | grep --line-buffered -E "\+ FG|ERROR|Traceback"
  $PY -u scripts/run_harness.py --name "FG-HGCL B0fv s$s" --variant l2d5 --fixed --cfg "$(cfg B0fv "{\"epochs\": 30, \"seed\": $s}")" --scn-cfg "$SCN0" 2>&1 | grep --line-buffered -E "\+ FG|ERROR|Traceback"
done
echo D4_DONE
for ds in 1 2; do
  $PY -u scripts/hgcmlda_protocol.py --dataset $ds --variant leaky --seeds 2026 2027 2028 --epochs 1000 --tag v2_ep1000 \
      --cfg '{"epochs": 1000, "dis_views": ["name", "defn", "syn"], "dis_view_feats": true}' 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
done
echo QUEUE_DONE
