"""Paired comparison of inner-val runs (same fold x inner seed) against a reference tag (latest entry per tag).

mode "curve" (default): every config is scored at ONE common epoch = argmax of its inner-val curve averaged over all
(fold, inner seed); "per-run": each run's own best epoch (optimistic, kept for reference).
Tags of non-l2d5 variants are "<variant>:<id>" (scripts/run_ablation.py).
Usage: python scripts/paired.py [--per-run] REF_TAG [TAG ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def load_runs(path=ROOT / "results" / "inner_val.jsonl"):
    runs = {}
    for line in open(path):
        r = json.loads(line)
        runs[r["tag"]] = r
    return runs


def common_epoch(r):
    """Epoch maximising the inner-val curve averaged over every (fold, inner seed); ties -> fewer epochs."""
    eps = [e for e, _ in r["rows"][0]["hist"]]
    curve = np.mean([[v for _, v in x["hist"]] for x in r["rows"]], 0)
    i = max(range(len(eps)), key=lambda j: (curve[j], -eps[j]))
    return eps[i], i


def scores(r, mode="curve"):
    if mode == "per-run":
        return {(x["fold"], x["inner_seed"]): x["val"] for x in r["rows"]}, int(np.median(r["epochs"]))
    e, i = common_epoch(r)
    return {(x["fold"], x["inner_seed"]): x["hist"][i][1] for x in r["rows"]}, e


def main(argv):
    mode = "curve"
    if argv and argv[0] in ("--per-run", "--curve"):
        mode = argv.pop(0)[2:]
    runs = load_runs()
    ref, tags = argv[0], argv[1:] or [t for t in runs if t != argv[0]]
    R, re_ = scores(runs[ref], mode)
    print(f"mode={mode}  reference {ref}: {np.mean(list(R.values())):.4f} @ {re_}")
    print(f"{'tag':<26}{'mean':>8}{'Δ':>9}{'±sd':>8}{'win/n':>8}{'p(t)':>8}  epoch")
    for t in tags:
        T, e = scores(runs[t], mode)
        ks = sorted(set(R) & set(T))
        d = np.array([T[k] - R[k] for k in ks])
        p = stats.ttest_1samp(d, 0).pvalue if len(d) > 1 else float("nan")
        print(f"{t:<26}{np.mean([T[k] for k in ks]):>8.4f}{d.mean():>+9.4f}{d.std(ddof=1):>8.4f}"
              f"{f'{(d > 0).sum()}/{len(d)}':>8}{p:>8.3f}  {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
