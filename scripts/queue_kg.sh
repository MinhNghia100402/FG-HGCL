#!/usr/bin/env bash
# Knowledge-hyperedge views (scripts/build_kg_views.py) on top of the locked configs, HGCMLDA protocol, 3 seeds.
# Reference = D*_clean_B3_final_nce(b).json (same configs without kg). Paired decision on inner-val.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-1} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY="taskset -c ${CPUS:-0-15} /sda1/nghiavm/research/.venv/bin/python -u"
V='"dis_views": ["name", "defn", "syn"], "dis_view_feats": true'
VAR=${VARIANT:-clean}
run() {  # dataset base-json-body tag kg-json
    $PY scripts/hgcmlda_protocol.py --dataset $1 --variant $VAR --seeds 2026 2027 2028 --epochs 1000 \
        --tag $3 --cfg "{\"epochs\": 1000, $2, $4}" 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
}
for kg in mirna locus mirna+locus onto; do
    case $kg in
        mirna) K='"kg_lnc": ["mirna"]';; locus) K='"kg_lnc": ["locus"]';;
        mirna+locus) K='"kg_lnc": ["mirna", "locus"]';; onto) K='"kg_dis": ["onto"]';;
    esac
    run 1 '"link_loss": "infonce"' final_nce_kg_$kg "$K"
    run 2 "$V, \"link_loss\": \"infonce+bce\"" final_nceb_kg_$kg "$K"
done
echo QUEUE_DONE
