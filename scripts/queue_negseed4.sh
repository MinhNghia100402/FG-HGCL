#!/usr/bin/env bash
# D2 with the original Dataset2 negative-sampling seed (4): same training (deterministic, epochs from inner-val), only
# the 1:1 test negatives change, so FG-HGCL is scored on exactly HGCMLDA's D2 test pairs.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-1} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY="taskset -c ${CPUS:-0-15} /sda1/nghiavm/research/.venv/bin/python -u"
B='"epochs": 1000, "dis_views": ["name", "defn", "syn"], "dis_view_feats": true, "link_loss": "infonce+bce"'
for v in leaky clean; do
    for m in B3 B0fv; do
        $PY scripts/hgcmlda_protocol.py --dataset 2 --variant $v --model $m --seeds 2026 2027 2028 --epochs 1000 \
            --neg_seed 4 --dump --tag neg4 --cfg "{$B}" 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
    done
done
echo QUEUE_DONE
