"""Run the released HGCMLDA code on a NEW 5-CV split (same data, same independent set, same paper HP).

The only change is the shuffle seed of the CV pairs: original/Dataset*/prepareData.py calls np.random.seed(0) before
shuffling the ones and zeros; here that one call uses --split-seed instead. Every other seed (model seed, the
evaluation's np.random.seed(3)) is untouched. The run itself is HGCMLDA's scripts/run_hgcmlda.py, so the result lands
in HGCMLDA/results/<run_id>/ with its split.npz, which scripts/hgcmlda_protocol.py --split then reuses for FG-HGCL.

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/hgcmlda_newsplit.py --dataset 1 --split-seed 1 [--seed 0]
"""
import argparse
import runpy
import sys

import numpy as np

HG_RUNNER = "/sda1/nghiavm/research/HGCMLDA/scripts/run_hgcmlda.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=int, choices=[1, 2], required=True)
    ap.add_argument("--split-seed", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0, help="HGCMLDA model seed (as the reproduction runs)")
    a = ap.parse_args()
    orig_seed = np.random.seed

    def seed(s=None):                                 # only prepareData's np.random.seed(0) is redirected
        if s == 0 and sys._getframe(1).f_code.co_filename.endswith("prepareData.py"):
            print(f"[newsplit] prepareData shuffle seed 0 -> {a.split_seed}", flush=True)
            s = a.split_seed
        return orig_seed(s)

    np.random.seed = seed
    sys.argv = [HG_RUNNER, "--dataset", str(a.dataset), "--variant", "original", "--seed", str(a.seed),
                "--tag", f"split{a.split_seed}"]
    runpy.run_path(HG_RUNNER, run_name="__main__")


if __name__ == "__main__":
    main()
