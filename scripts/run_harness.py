"""Run an FG-HGCL configuration through CDELDA's unmodified harness (cold protocol: disease / lncRNA / both-cold).

The model is appended to bench.runner.build_registry at run time; results go to results/harness/bench_rdl2d5_cold.json
(checkpointed per model name, same file format as CDELDA's own results).

Usage (any cwd):
  CUDA_VISIBLE_DEVICES=0 /sda1/nghiavm/research/.venv/bin/python scripts/run_harness.py --name "FG-HGCL B2 gmm" \
      --cfg '{"views": ["gmm"], "lam_cl": 0}'
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CDELDA = Path(os.environ.get("CDELDA_REPO", "/sda1/nghiavm/research/IHC-LDA/data/raw/CDELDA"))   # clone of CDELDA's repo
os.environ.setdefault("CDELDA_DATA_ROOT", str(ROOT / "data" / "harness"))   # l2d5 symlinked to IHC-LDA, D1/D2 built here
os.environ.setdefault("CCDIFF_SWEEP_RESULTS", str(ROOT / "results" / "harness"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cfg", default="{}", help="JSON overrides of fghgcl.model.Config")
    ap.add_argument("--variant", default="l2d5")
    ap.add_argument("--protocol", default="cold", choices=["cold", "warm"], help="harness protocol (warm = C-normal)")
    ap.add_argument("--no-refit", action="store_true")
    ap.add_argument("--fixed", action="store_true", help="train once for cfg.epochs (no per-fold epoch selection)")
    ap.add_argument("--scn-cfg", default="{}", help='per-scenario overrides, e.g. {"disease": {"m": 128, "epochs": 400}}')
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(CDELDA))
    sys.path.insert(0, str(CDELDA / "snapshot_src"))
    os.chdir(CDELDA)
    from fghgcl.harness import FGHGCLHarness, build
    cfg, Model = build(json.loads(a.cfg))
    out = Path(os.environ["CCDIFF_SWEEP_RESULTS"])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "configs.jsonl", "a") as f:
        f.write(json.dumps({"name": a.name, "cfg": cfg.to_dict(), "refit": not a.no_refit, "fixed": a.fixed, "scn_cfg": json.loads(a.scn_cfg), "variant": a.variant}) + "\n")

    import bench.runner as br
    orig = br.build_registry
    br.build_registry = lambda device: orig(device) + [(a.name, lambda: FGHGCLHarness(cfg, refit=not a.no_refit, fixed=a.fixed,
                                                                                     scn_over=json.loads(a.scn_cfg)))]
    sys.argv = ["bench.sweep_run", "--variants", a.variant, "--protocol", a.protocol, "--only", a.name]
    runpy.run_module("bench.sweep_run", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
