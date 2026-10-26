"""Run the ablation matrix of configs/fg_hgcl_both_cold.yaml (plan §6).

  inner   : inner-validation selection runs (scripts/inner_val.py) for the given ids and variant
  report  : paired comparison against a reference id (scripts/paired.py), common epoch per config
  harness : final test in CDELDA's harness for LOCKED ids: --fixed, epochs = the common inner-val epoch of that id,
            one run per model seed; names "FG-HGCL <id> s<seed>"

Usage:
  /sda1/nghiavm/research/.venv/bin/python scripts/run_ablation.py inner   --variant D1 --ids B0f B2 B3
  /sda1/nghiavm/research/.venv/bin/python scripts/run_ablation.py report  --variant D1 --ref B0f --ids B2 B3
  /sda1/nghiavm/research/.venv/bin/python scripts/run_ablation.py harness --variant l2d5 --ids B0f B3 --seeds 2026 2027 2028
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable                                     # the interpreter running this script
CFG = ROOT / "configs" / "fg_hgcl_both_cold.yaml"


def config(i: str) -> dict:
    y = yaml.safe_load(CFG.read_text())
    if i in y["experiments"]:
        return {**y["base"], **y["experiments"][i]}
    ch = yaml.safe_load((ROOT / "configs" / "chain.yaml").read_text())   # HGCMLDA -> FG-HGCL chain steps
    if i in ch["steps"]:
        return {**ch["base"], **ch["steps"][i]}
    raise SystemExit(f"unknown id {i}; known: {list(y['experiments']) + list(ch['steps'])}")


SUFFIX = ""


def tag(i, v):
    return (i if v == "l2d5" else f"{v}:{i}") + SUFFIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["inner", "report", "harness"])
    ap.add_argument("--variant", default="l2d5")
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--ref", default="B0f")
    ap.add_argument("--inner-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--override", default="{}", help="JSON overrides applied on top of every id (e.g. a finer epoch grid)")
    ap.add_argument("--suffix", default="", help="tag suffix for runs made with --override")
    a = ap.parse_args()
    global SUFFIX
    SUFFIX = a.suffix
    over = json.loads(a.override)
    skip = ROOT / "results" / "skip_variants.txt"            # variants listed here are skipped by queued runs
    ch = yaml.safe_load((ROOT / "configs" / "chain.yaml").read_text())
    chain_skip = a.variant in ch.get("skip_variants", []) and all(i in ch["steps"] for i in a.ids)
    if a.stage in ("inner", "report") and (chain_skip or (skip.exists() and a.variant in skip.read_text().split())):
        print(f"[run_ablation] {a.variant} listed in {skip.name}: skipped", flush=True)
        return
    if a.stage == "inner":
        for i in a.ids:
            subprocess.run([PY, str(ROOT / "scripts/inner_val.py"), "--tag", tag(i, a.variant), "--variant", a.variant,
                            "--inner-seeds", *map(str, a.inner_seeds), "--cfg", json.dumps({**config(i), **over})], check=True)
    elif a.stage == "report":
        subprocess.run([PY, str(ROOT / "scripts/paired.py"), tag(a.ref, a.variant), *[tag(i, a.variant) for i in a.ids]],
                       check=True)
    else:
        sys.path.insert(0, str(ROOT / "scripts"))
        from paired import common_epoch, load_runs
        runs = load_runs()
        for i in a.ids:
            t = tag(i, a.variant)
            if t not in runs:
                raise SystemExit(f"{t}: no inner-validation run; run the 'inner' stage first")
            ep, _ = common_epoch(runs[t])
            for s in a.seeds:
                cfg = {**config(i), **over, "epochs": int(ep), "seed": s}
                env = {**os.environ, "CCDIFF_SWEEP_RESULTS": str(ROOT / "results" / "harness" / "fg_hgcl")}
                subprocess.run([PY, str(ROOT / "scripts/run_harness.py"), "--name", f"FG-HGCL {i} s{s}",
                                "--variant", a.variant, "--fixed", "--cfg", json.dumps(cfg)], check=True, env=env)


if __name__ == "__main__":
    main()
