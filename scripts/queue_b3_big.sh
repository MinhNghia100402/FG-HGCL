#!/usr/bin/env bash
# Larger B3 (same architecture, wider): embedding m and hidden width; D3 inner validation only. Runs after queue_b3_tune.
set -uo pipefail
cd "$(dirname "$0")/.."
until grep -q QUEUE_DONE results/logs/b3_tune.log 2>/dev/null; do sleep 30; done
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/sda1/nghiavm/research/.venv/bin/python
B3=$($PY -c "import sys;sys.argv=['x'];sys.path.insert(0,'scripts');import run_ablation as r,json;print(json.dumps(r.config('B3')))")
run(){ cfg=$($PY -c "import json,sys;print(json.dumps({**json.loads(sys.argv[1]),**json.loads(sys.argv[2])}))" "$B3" "$2")
       $PY scripts/inner_val.py --tag "B3+$1" --inner-seeds 0 1 2 --cfg "$cfg" 2>&1 | grep --line-buffered -E "^==|Error|Traceback"; }
run m256          '{"m": 256}'
run h1024         '{"hidden": 1024}'
run h2048         '{"hidden": 2048}'
run m128_h1024    '{"m": 128, "hidden": 1024}'
run m256_h2048    '{"m": 256, "hidden": 2048}'
run m128_h1024_d4 '{"m": 128, "hidden": 1024, "drop": 0.4}'
echo "#### report"; $PY scripts/paired.py B3 B0f B3+m256 B3+h1024 B3+h2048 B3+m128_h1024 B3+m256_h2048 B3+m128_h1024_d4
echo QUEUE_DONE
