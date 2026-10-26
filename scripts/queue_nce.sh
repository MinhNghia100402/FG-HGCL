#!/usr/bin/env bash
# InfoNCE as main link loss vs BCE, inner validation (harness both-cold): l2d5 (10-epoch grid), D1 / D2 (fine 2-epoch grid).
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/sda1/nghiavm/research/.venv/bin/python
IDS="B3_nce B3_nce_t05 B3_nceb B3v_nce B3v_nceb"
$PY scripts/run_ablation.py inner --variant l2d5 --ids B3v $IDS 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
rm -f results/skip_variants.txt.bak; mv results/skip_variants.txt results/skip_variants.txt.bak 2>/dev/null || true
for v in D1 D2; do
  $PY scripts/run_ablation.py inner --variant $v --ids B3v $IDS --override '{"epochs": 150, "eval_every": 2}' --suffix _fine 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
done
mv results/skip_variants.txt.bak results/skip_variants.txt 2>/dev/null || true
echo "#### report l2d5"; $PY scripts/paired.py B3 B0f B3v $IDS
for v in D1 D2; do echo "#### report $v"; $PY scripts/paired.py $v:B3_fine $v:B0f_fine $(for i in B3v $IDS; do echo -n "$v:${i}_fine "; done); done
echo QUEUE_DONE
