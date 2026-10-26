#!/usr/bin/env bash
# CDELDA full model (TwoTower-PEFT-disease, LoRA S-BioBERT) as a BASELINE on D1 / D2 in CDELDA's harness, cold protocol.
# CDELDA's code unchanged, launched through IHC-LDA scripts/cdelda_launch.py (local weights), PEFT_EPOCHS=100 as in
# IHC-LDA scripts/cdelda_repro.sh (which reproduced CDELDA's Table 2 within 0.007).
set -uo pipefail
cd /sda1/nghiavm/research/IHC-LDA/data/raw/CDELDA
export CDELDA_DATA_ROOT=/sda1/nghiavm/research/FG-HGCL/data/harness
export CCDIFF_SWEEP_RESULTS=/sda1/nghiavm/research/FG-HGCL/results/harness/baselines
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${GPU:-0} PEFT_EPOCHS=100
for v in D1 D2; do
  /sda1/nghiavm/research/.venv/bin/python -u ../../../scripts/cdelda_launch.py bench.sweep_run --variants $v --protocol cold \
      --only "TwoTower-PEFT-disease (LoRA S-BioBERT)" 2>&1 | grep --line-buffered -E "\+ TwoTower|ERROR|Traceback|done in|PEFT|peft"
done
echo QUEUE_DONE
