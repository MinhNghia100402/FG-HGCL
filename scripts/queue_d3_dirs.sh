#!/usr/bin/env bash
# D3 (l2d5) inner validation: direction 1 (multi-view disease-text hypergraph), extra D1/D2 labels, direction 2
# (per-scenario tuning for lncRNA-cold and disease-cold). Tags "<scenario>|<id>". Test blocks are never scored.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/sda1/nghiavm/research/.venv/bin/python
cfg(){ $PY -c "import sys,json;sys.argv=['x'];sys.path.insert(0,'scripts');import run_ablation as r;print(json.dumps({**r.config(sys.argv[1] if False else '$1'),**json.loads('$2')}))"; }
run(){ $PY scripts/inner_val.py --tag "$1|$2" --scenario $1 --inner-seeds 0 1 2 --cfg "$(cfg $3 "$4")" 2>&1 | grep --line-buffered -E "^==|Error|Traceback"; }
V='"dis_views": ["name", "defn", "syn"]'
for scn in both disease lncRNA; do
  run $scn B0f        B0f '{}'
  run $scn B3         B3  '{}'
  run $scn B3_views   B3  "{$V}"
  run $scn B3_vfeat   B3  "{$V, \"dis_view_feats\": true}"
  run $scn B0f_vfeat  B0f "{$V, \"dis_view_feats\": true}"
  run $scn B3_m128    B3  '{"m": 128}'
  run $scn B3_m128_vfeat B3 "{$V, \"dis_view_feats\": true, \"m\": 128}"
done
run both B3_extra B3 '{"extra_src": ["D1", "D2"]}'
for scn in both disease lncRNA; do echo "#### report $scn"; $PY scripts/paired.py "$scn|B0f" "$scn|B3" "$scn|B3_views" "$scn|B3_vfeat" "$scn|B0f_vfeat" "$scn|B3_m128" "$scn|B3_m128_vfeat" $([ $scn = both ] && echo "both|B3_extra"); done
echo QUEUE_DONE
