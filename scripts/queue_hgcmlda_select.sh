#!/usr/bin/env bash
# Candidate selection for HGCMLDA's original (leaky) protocol on inner validation (docs/hgcmlda_selection.md).
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/sda1/nghiavm/research/.venv/bin/python
declare -A C=( [base]='{}' [ens3]='{"ens": 3}' [m64]='{"m": 64}' [m128]='{"m": 128}' [m128h512]='{"m": 128, "hidden": 512}'
               [lam1]='{"lam_cl": 1.0}' [gated]='{"fusion": "gated"}' [k11g9]='{"k_lnc": 11, "k_dis": 11, "g_lnc": 9, "g_dis": 9}'
               [drop04]='{"drop": 0.4}' [ep1000]='{"epochs": 1000}' )
for ds in 1 2; do for t in base ens3 m64 m128 m128h512 lam1 gated k11g9 drop04 ep1000; do
  extra=(); [ "$t" = ep1000 ] && extra=(--epochs 1000)
  $PY -u scripts/hgcmlda_protocol.py --dataset $ds --variant leaky --seeds 2026 --tag "sel_$t" --cfg "${C[$t]}" "${extra[@]}" 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
done; done
echo QUEUE_DONE
