#!/usr/bin/env bash
# HGCMLDA -> FG-HGCL module-replacement chain (configs/chain.yaml) on inner validation; D1, D2 first, then D3 (l2d5).
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/sda1/nghiavm/research/.venv/bin/python
STEPS="S0 S1 S2 S3 S4 S5 S5b S6m S6g L0 L5"
for v in D1 D2 l2d5; do
  $PY scripts/run_ablation.py inner --variant $v --ids $STEPS 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
  echo "#### report $v"; $PY scripts/run_ablation.py report --variant $v --ref S0 --ids S1 S2 S3 S4 S5 S5b S6m S6g
  echo "#### leaky $v"; $PY scripts/run_ablation.py report --variant $v --ref L0 --ids L5
done
echo QUEUE_DONE
