#!/usr/bin/env bash
# C2 (association-hypergraph contrastive view) then C1 (lam_cl, tau sensitivity of B3) on inner validation.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/sda1/nghiavm/research/.venv/bin/python
BASE='"m": 32, "drop": 0.2, "wd": 0.001, "epochs": 500, "eval_every": 10'
run(){ $PY scripts/inner_val.py --tag "$1" --inner-seeds 0 1 2 --cfg "{$BASE, $2}" 2>&1 | grep -E "^==|Error|Traceback"; }
for la in 0.1 0.5 1.0; do run "C2_feat_assoc$la" "\"views\": [], \"lam_assoc\": $la"; done
run C2_knngmm_cl0.5_assoc0.5 '"views": ["knn","gmm"], "lam_cl": 0.5, "lam_assoc": 0.5'
for lam in 0.01 0.05 0.1 0.5 1.0; do for tau in 0.1 0.2 0.5 0.8; do
  run "C1_knngmm_l${lam}_t${tau}" "\"views\": [\"knn\",\"gmm\"], \"lam_cl\": $lam, \"tau\": $tau"
done; done
echo QUEUE_DONE
