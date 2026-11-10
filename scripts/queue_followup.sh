#!/usr/bin/env bash
# Follow-up tests of docs/followup_lock.md. LANE=a: F1 (C-normal, shared B3 config) + D1 on splits 1, 2 + D2 on split 2;
# LANE=b: D2 on split 1. Each lane runs on its own GPU (GPU=...).
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-2} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/sda1/nghiavm/research/.venv/bin/python
HG=/sda1/nghiavm/research/HGCMLDA/results
cfg(){ $PY -c "import sys,json;sys.argv=['x'];sys.path.insert(0,'scripts');import run_ablation as r;print(json.dumps({**r.config('$1'),**json.loads('$2')}))"; }

fg_on_split(){   # dataset, split seed: HGCMLDA on the new split, then the locked FG-HGCL configs on its split.npz
  local d=$1 k=$2
  $PY -u scripts/hgcmlda_newsplit.py --dataset "$d" --split-seed "$k" 2>&1 | grep --line-buffered -E "newsplit|\[runner\] CV|Traceback|Error"
  local sp; sp=$(ls -d $HG/*_D${d}_original_s0_split${k} | tail -1)/split.npz
  local V='"dis_views": ["name", "defn", "syn"], "dis_view_feats": true'
  if [ "$d" = 1 ]; then
    locked='{"epochs": 1000, "link_loss": "infonce"}'; abl=B0f
  else
    locked="{\"epochs\": 1000, $V, \"link_loss\": \"infonce+bce\"}"; abl=B0fv
  fi
  ll=$($PY -c "import json;print(json.loads('$locked')['link_loss'])")
  $PY -u scripts/hgcmlda_protocol.py --dataset "$d" --variant leaky --model B3 --epochs 1000 --split "$sp" \
      --tag "split${k}_locked" --cfg "$locked" 2>&1 | grep --line-buffered -E "^==|Traceback|Error"
  $PY -u scripts/hgcmlda_protocol.py --dataset "$d" --variant leaky --model "$abl" --epochs 1000 --split "$sp" \
      --tag "split${k}_nohg" --cfg "{\"epochs\": 1000, \"link_loss\": \"$ll\"}" 2>&1 | grep --line-buffered -E "^==|Traceback|Error"
  $PY -u scripts/hgcmlda_protocol.py --dataset "$d" --variant leaky --model B3 --epochs 1000 --split "$sp" \
      --tag "split${k}_shared" --cfg '{"epochs": 1000}' 2>&1 | grep --line-buffered -E "^==|Traceback|Error"
}

if [ "${LANE:-a}" = a ]; then
  export CCDIFF_SWEEP_RESULTS=$PWD/results/harness/followup
  for s in 2026 2027 2028; do
    $PY -u scripts/run_harness.py --name "FG-HGCL B3 warm s$s" --variant l2d5 --protocol warm --fixed \
        --cfg "$(cfg B3 "{\"epochs\": 1000, \"seed\": $s}")" 2>&1 | grep --line-buffered -E "\+ FG|ERROR|Traceback"
  done
  echo F1_DONE
  fg_on_split 1 1; fg_on_split 1 2; echo D1_DONE
  fg_on_split 2 2; echo D2S2_DONE
else
  fg_on_split 2 1; echo D2S1_DONE
fi
echo QUEUE_DONE
