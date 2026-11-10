#!/usr/bin/env bash
# Side experiment: lncRNA RNA-FM block -> RIBOSPAN-10K-15 (scripts/build_lnc_ribospan.py), locked final configs,
# HGCMLDA protocol, 3 seeds. Leaky RNA-FM reference = D*_leaky_B3_final_nce(b).json; clean RNA-FM run here as reference.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-3} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY="taskset -c ${CPUS:-0-15} /sda1/nghiavm/research/.venv/bin/python -u"
V='"dis_views": ["name", "defn", "syn"], "dis_view_feats": true'
C1='{"epochs": 1000, "link_loss": "infonce"}'
C2="{\"epochs\": 1000, $V, \"link_loss\": \"infonce+bce\"}"
run() {  # dataset variant cfg tag lnc_file
    CCDIFF_LNC_FILE=$5 $PY scripts/hgcmlda_protocol.py --dataset $1 --variant $2 --seeds 2026 2027 2028 --epochs 1000 \
        --tag $4 --cfg "$3" 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
}
run 1 leaky "$C1" final_nce_rs10k lnc_ribospan10k.npy
run 2 leaky "$C2" final_nceb_rs10k lnc_ribospan10k.npy
run 1 clean "$C1" final_nce lnc_ortho.npy
run 1 clean "$C1" final_nce_rs10k lnc_ribospan10k.npy
run 2 clean "$C2" final_nceb lnc_ortho.npy
run 2 clean "$C2" final_nceb_rs10k lnc_ribospan10k.npy
echo QUEUE_DONE
