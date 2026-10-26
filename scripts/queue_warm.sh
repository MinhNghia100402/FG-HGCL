#!/usr/bin/env bash
# C-normal (warm) on D3: candidate selection on inner validation only (harness warm folds, 12.5 % of train positives).
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-2} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/sda1/nghiavm/research/.venv/bin/python
cfg(){ $PY -c "import sys,json;sys.argv=['x'];sys.path.insert(0,'scripts');import run_ablation as r;print(json.dumps({**r.config('$1'),**json.loads('$2')}))"; }
for i in B0f B3 B3v B3_nce B3_nceb B3v_nce B3v_nceb; do
  $PY scripts/inner_val.py --tag "warm|$i" --scenario warm --inner-seeds 0 1 2 --cfg "$(cfg $i '{"epochs": 1000, "eval_every": 20}')" 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
done
echo "#### report warm"; $PY scripts/paired.py "warm|B0f" "warm|B3" "warm|B3v" "warm|B3_nce" "warm|B3_nceb" "warm|B3v_nce" "warm|B3v_nceb"
echo QUEUE_DONE
