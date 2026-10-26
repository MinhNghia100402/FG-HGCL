#!/usr/bin/env bash
# B3 kept as is (architecture fixed); one-factor-at-a-time hyperparameter / seed-ensemble sweep on D3 (l2d5) inner
# validation only. Tags "B3+<change>"; compare with scripts/paired.py B3 ...
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/sda1/nghiavm/research/.venv/bin/python
B3=$($PY -c "import sys;sys.argv=['x'];sys.path.insert(0,'scripts');import run_ablation as r,json;print(json.dumps(r.config('B3')))")
run(){ cfg=$($PY -c "import json,sys;print(json.dumps({**json.loads(sys.argv[1]),**json.loads(sys.argv[2])}))" "$B3" "$2")
       $PY scripts/inner_val.py --tag "B3+$1" --inner-seeds 0 1 2 --cfg "$cfg" 2>&1 | grep --line-buffered -E "^==|Error|Traceback"; }
run m64      '{"m": 64}'
run m128     '{"m": 128}'
run h512     '{"hidden": 512}'
run drop0    '{"drop": 0.0}'
run drop0.4  '{"drop": 0.4}'
run wd1e-4   '{"wd": 0.0001}'
run wd1e-2   '{"wd": 0.01}'
run k5       '{"k_lnc": 5, "k_dis": 5}'
run k20      '{"k_lnc": 20, "k_dis": 20}'
run g8       '{"g_lnc": 8, "g_dis": 5}'
run g32      '{"g_lnc": 32, "g_dis": 15}'
run lr5e-4   '{"lr": 0.0005, "epochs": 800}'
run L2       '{"hg_layers": 2}'
run ens3     '{"ens": 3}'
run ens5     '{"ens": 5}'
echo "#### report"; $PY scripts/paired.py B3 B0f $(for t in m64 m128 h512 drop0 drop0.4 wd1e-4 wd1e-2 k5 k20 g8 g32 lr5e-4 L2 ens3 ens5; do echo -n "B3+$t "; done)
echo QUEUE_DONE
