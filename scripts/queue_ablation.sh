#!/usr/bin/env bash
# Inner-validation ablation matrix (configs/fg_hgcl_both_cold.yaml) on l2d5, D1, D2, then paired reports.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/sda1/nghiavm/research/.venv/bin/python
IDS="B2 B3 B4 B5 B7 B8 B9 B9u B3_bilin C2"
$PY scripts/run_ablation.py inner --variant l2d5 --ids $IDS 2>&1 | grep -E "^==|Error|Traceback"
for v in D1 D2; do
  $PY scripts/run_ablation.py inner --variant $v --ids B0f $IDS 2>&1 | grep -E "^==|Error|Traceback"
done
for v in l2d5 D1 D2; do echo "#### report $v"; $PY scripts/run_ablation.py report --variant $v --ref B0f --ids $IDS; done
echo QUEUE_DONE
