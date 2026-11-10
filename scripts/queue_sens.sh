#!/usr/bin/env bash
# Sensitivity of FG-HGCL (clean setting, 3 seeds), one hyperparameter at a time; DS=1|2 (split of queue_analysis.sh part 2).
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-1} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY="taskset -c ${CPUS:-0-15} /sda1/nghiavm/research/.venv/bin/python -u"
V='"dis_views": ["name", "defn", "syn"], "dis_view_feats": true'
if [ "$DS" = 1 ]; then B='"epochs": 1000, "link_loss": "infonce"'; else B="\"epochs\": 1000, $V, \"link_loss\": \"infonce+bce\""; fi
for s in '"lam_cl": 0.1' '"lam_cl": 1.0' '"tau": 0.1' '"tau": 0.5' '"link_k": 16' '"link_k": 256' \
         '"k_lnc": 5, "k_dis": 5' '"k_lnc": 20, "k_dis": 20' '"m": 64' '"hg_layers": 2'; do
    t=$(echo "$s" | tr -d '" :,' )
    $PY scripts/hgcmlda_protocol.py --dataset $DS --variant clean --model B3 --seeds 2026 2027 2028 --epochs 1000 \
        --tag sens_$t --cfg "{$B, $s}" 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
done
echo QUEUE_DONE
