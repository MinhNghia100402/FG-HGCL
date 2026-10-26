#!/usr/bin/env bash
# C-normal (warm) test on D3 (docs/final_lock_v2.md): locked B3_nceb 600 epochs + ablation B0f 1000 epochs, 3 seeds.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-2} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CCDIFF_SWEEP_RESULTS=$PWD/results/harness/fg_hgcl_v2
PY=/sda1/nghiavm/research/.venv/bin/python
cfg(){ $PY -c "import sys,json;sys.argv=['x'];sys.path.insert(0,'scripts');import run_ablation as r;print(json.dumps({**r.config('$1'),**json.loads('$2')}))"; }
for s in 2026 2027 2028; do
  $PY -u scripts/run_harness.py --name "FG-HGCL warm s$s" --variant l2d5 --protocol warm --fixed --cfg "$(cfg B3_nceb "{\"epochs\": 600, \"seed\": $s}")" 2>&1 | grep --line-buffered -E "\+ FG|ERROR|Traceback"
  $PY -u scripts/run_harness.py --name "FG-HGCL B0f warm s$s" --variant l2d5 --protocol warm --fixed --cfg "$(cfg B0f "{\"epochs\": 1000, \"seed\": $s}")" 2>&1 | grep --line-buffered -E "\+ FG|ERROR|Traceback"
done
echo QUEUE_DONE
