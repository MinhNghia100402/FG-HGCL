"""LEAKY comparison runs (HGCMLDA's original protocol) in CDELDA's harness, cold protocol. Results go to
results/harness/leaky/ and must be reported as "leaky (HGCMLDA protocol)", never as both-cold results.

  --model hgcmlda : HGCMLDA released code (IHC-LDA HGCMLDA-clean adapter) with IL / ID computed from the FULL matrix,
                    exactly as the original prepareData.py; training edges and loss sets stay train-only as in the original.
  --model fghgcl  : FG-HGCL B3 (locked config, locked epochs) with leak=True (fghgcl/leaky.py), 3 model seeds.

Usage: CUDA_VISIBLE_DEVICES=0 /sda1/nghiavm/research/.venv/bin/python scripts/run_leaky.py --model hgcmlda --variant D1
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IHC = Path("/sda1/nghiavm/research/IHC-LDA")
CDELDA = IHC / "data/raw/CDELDA"
OUT = ROOT / "results" / "harness" / "leaky"
PY = "/sda1/nghiavm/research/.venv/bin/python"
FULL = {}


def hgcmlda(variant):
    for p in (str(IHC / "scripts"), str(IHC), str(CDELDA / "snapshot_src"), str(CDELDA)):
        sys.path.insert(0, p)
    os.chdir(CDELDA)
    import bench.runner as br
    import src.baselines.hgcmlda_clean as hc
    from hgcmlda_harness import HGCMLDAHarness
    from src.baselines.hgcmlda_similarity import clean_similarities

    orig_inputs = hc.training_inputs

    def leaky_inputs(train_edges, dss, lnc_train_mask, dis_train_mask):
        inp = orig_inputs(train_edges, dss, lnc_train_mask, dis_train_mask)
        inp["IL"], inp["ID"] = clean_similarities(FULL["M"], dss)        # FULL matrix = original prepareData.py
        return inp
    hc.training_inputs = leaky_inputs

    class Leaky(HGCMLDAHarness):
        def fit(self, M, Clnc, Cdis, train_lnc, train_dis):
            FULL["M"] = (np.asarray(M) > 0).astype(np.float64)
            return super().fit(M, Clnc, Cdis, train_lnc, train_dis)

    name = "HGCMLDA-LEAKY (original protocol, paper HP, Dataset1)"
    orig = br.build_registry
    br.build_registry = lambda device: orig(device) + [(name, lambda: Leaky())]
    sys.argv = ["bench.sweep_run", "--variants", variant, "--protocol", "cold", "--only", name]
    runpy.run_module("bench.sweep_run", run_name="__main__", alter_sys=True)


def fghgcl(variant, seeds):
    sys.path.insert(0, str(ROOT / "scripts"))
    from paired import common_epoch, load_runs
    from run_ablation import config
    runs = load_runs()
    t = "B3" if variant == "l2d5" else f"{variant}:B3_fine"           # the locked epochs of docs/final_lock.md
    ep, _ = common_epoch(runs[t])
    for s in seeds:
        cfg = {**config("B3"), "epochs": int(ep), "seed": s, "leak": True}
        subprocess.run([PY, str(ROOT / "scripts/run_harness.py"), "--name", f"FG-HGCL B3 LEAKY s{s}", "--variant", variant,
                        "--fixed", "--cfg", json.dumps(cfg)], check=True, env=os.environ)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["hgcmlda", "fghgcl"], required=True)
    ap.add_argument("--variant", default="l2d5")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    a = ap.parse_args()
    os.environ.setdefault("CDELDA_DATA_ROOT", str(ROOT / "data" / "harness"))
    os.environ["CCDIFF_SWEEP_RESULTS"] = str(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    hgcmlda(a.variant) if a.model == "hgcmlda" else fghgcl(a.variant, a.seeds)
