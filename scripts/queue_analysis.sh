#!/usr/bin/env bash
# Report analyses on the locked configs (post-lock, descriptive only; nothing here changes the lock):
#   1. dump runs (scores + inner-val curves) for per-degree / per-group / learning-curve analysis, incl. no-hypergraph R0
#   2. sensitivity of the clean locked config to one hyperparameter at a time (3 seeds)
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-1} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY="taskset -c ${CPUS:-0-15} /sda1/nghiavm/research/.venv/bin/python -u"
V='"dis_views": ["name", "defn", "syn"], "dis_view_feats": true'
B1='"epochs": 1000, "link_loss": "infonce"'
B2="\"epochs\": 1000, $V, \"link_loss\": \"infonce+bce\""
run() {  # dataset variant model tag cfg [extra args]
    $PY scripts/hgcmlda_protocol.py --dataset $1 --variant $2 --model $3 --seeds 2026 2027 2028 --epochs 1000 \
        --tag $4 --cfg "$5" ${6:-} 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
}
for v in clean leaky; do
    run 1 $v B3 dump_final "{$B1}" --dump
    run 2 $v B3 dump_final "{$B2}" --dump
    run 1 $v B0f dump_nohg "{$B1}" --dump
    run 2 $v B0fv dump_nohg "{$B2}" --dump
done
for s in '"lam_cl": 0.1' '"lam_cl": 1.0' '"tau": 0.1' '"tau": 0.5' '"link_k": 16' '"link_k": 256' \
         '"k_lnc": 5, "k_dis": 5' '"k_lnc": 20, "k_dis": 20' '"m": 64' '"hg_layers": 2'; do
    t=$(echo "$s" | tr -d '" :,' )
    run 1 clean B3 sens_$t "{$B1, $s}"
    run 2 clean B3 sens_$t "{$B2, $s}"
done
echo QUEUE_DONE
