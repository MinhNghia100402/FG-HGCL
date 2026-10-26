"""Baselines B0 / B1 in CDELDA's unmodified harness, cold protocol (plan §6, §7 Phase 2, item 1).

  B0  "TwoTower (content)"                  CDELDA's own content two-tower (snapshot_src/twoside_models.TwoTowerContent),
                                            taken from CDELDA's registry unchanged (no LoRA, per the project's scope).
  B1  "HGCMLDA-clean (paper HP, Dataset1)"  HGCMLDA released code with similarities recomputed from TRAIN edges only,
                                            IHC-LDA scripts/hgcmlda_harness.py (read-only reuse).
CDELDA's LoRA model is not run; its published / rerun numbers are cited from results/reference (scripts/collect_reference.py).

Usage: CUDA_VISIBLE_DEVICES=0 /sda1/nghiavm/research/.venv/bin/python scripts/run_baseline.py --model B0 --variant D1
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IHC = Path("/sda1/nghiavm/research/IHC-LDA")
CDELDA = IHC / "data/raw/CDELDA"
NAMES = {"B0": "TwoTower (content)", "B1": "HGCMLDA-clean (paper HP, Dataset1)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(NAMES), required=True)
    ap.add_argument("--variant", default="l2d5")
    a = ap.parse_args()
    os.environ.setdefault("CDELDA_DATA_ROOT", str(ROOT / "data" / "harness"))
    os.environ.setdefault("CCDIFF_SWEEP_RESULTS", str(ROOT / "results" / "harness" / "baselines"))
    Path(os.environ["CCDIFF_SWEEP_RESULTS"]).mkdir(parents=True, exist_ok=True)
    for p in (str(IHC / "scripts"), str(IHC), str(CDELDA / "snapshot_src"), str(CDELDA)):
        sys.path.insert(0, p)
    os.chdir(CDELDA)
    if a.model == "B1":
        import bench.runner as br
        from hgcmlda_harness import HGCMLDAHarness
        orig = br.build_registry
        br.build_registry = lambda device: orig(device) + [(NAMES["B1"], lambda: HGCMLDAHarness())]
    sys.argv = ["bench.sweep_run", "--variants", a.variant, "--protocol", "cold", "--only", NAMES[a.model]]
    runpy.run_module("bench.sweep_run", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
