#!/usr/bin/env bash
# Same as queue_split_neg4.sh for the FG-HGCL-base 1000-epoch column (*_split{1,2}_shared) of tab:newsplit.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-1} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY="taskset -c ${CPUS:-0-15} /sda1/nghiavm/research/.venv/bin/python -u"
while pgrep -u "$(id -u)" -f queue_split_neg4.sh >/dev/null; do sleep 30; done
$PY scripts/hgcmlda_protocol.py --dataset 2 --variant leaky --model B3 --seeds 2026 2027 2028 --epochs 1000 --split /sda1/nghiavm/research/HGCMLDA/results/20261027-075932_D2_original_s0_split1/split.npz --neg_seed 4 --tag split1_shared_neg4 --cfg '{"epochs": 1000}' 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
$PY scripts/hgcmlda_protocol.py --dataset 2 --variant leaky --model B3 --seeds 2026 2027 2028 --epochs 1000 --split /sda1/nghiavm/research/HGCMLDA/results/20261027-083128_D2_original_s0_split2/split.npz --neg_seed 4 --tag split2_shared_neg4 --cfg '{"epochs": 1000}' 2>&1 | grep --line-buffered -E "^==|Error|Traceback"
echo QUEUE_DONE
