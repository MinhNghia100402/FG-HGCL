#!/usr/bin/env bash
# D2 new-split runs again with Dataset2's own negative seed (4), the seed HGCMLDA's code uses on these splits.
# Same args as the *_split{1,2}_{locked,nohg} runs; training is unchanged, only the 1:1 CV negatives differ.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-1} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY="taskset -c ${CPUS:-0-15} /sda1/nghiavm/research/.venv/bin/python -u"
$PY scripts/hgcmlda_protocol.py --dataset 2 --variant leaky --model B3 --seeds 2026 2027 2028 --epochs 1000 --split /sda1/nghiavm/research/HGCMLDA/results/20261027-075932_D2_original_s0_split1/split.npz --neg_seed 4 --tag split1_locked_neg4 --cfg '{"epochs": 1000, "dis_views": ["name", "defn", "syn"], "dis_view_feats": true, "link_loss": "infonce+bce"}' 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
$PY scripts/hgcmlda_protocol.py --dataset 2 --variant leaky --model B3 --seeds 2026 2027 2028 --epochs 1000 --split /sda1/nghiavm/research/HGCMLDA/results/20261027-083128_D2_original_s0_split2/split.npz --neg_seed 4 --tag split2_locked_neg4 --cfg '{"epochs": 1000, "dis_views": ["name", "defn", "syn"], "dis_view_feats": true, "link_loss": "infonce+bce"}' 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
$PY scripts/hgcmlda_protocol.py --dataset 2 --variant leaky --model B0fv --seeds 2026 2027 2028 --epochs 1000 --split /sda1/nghiavm/research/HGCMLDA/results/20261027-075932_D2_original_s0_split1/split.npz --neg_seed 4 --tag split1_nohg_neg4 --cfg '{"epochs": 1000, "link_loss": "infonce+bce"}' 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
$PY scripts/hgcmlda_protocol.py --dataset 2 --variant leaky --model B0fv --seeds 2026 2027 2028 --epochs 1000 --split /sda1/nghiavm/research/HGCMLDA/results/20261027-083128_D2_original_s0_split2/split.npz --neg_seed 4 --tag split2_nohg_neg4 --cfg '{"epochs": 1000, "link_loss": "infonce+bce"}' 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
echo QUEUE_DONE
