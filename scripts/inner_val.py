"""Model selection on INNER VALIDATION ONLY (harness both-cold folds of any harness variant). The harness test block of every fold is
never scored: inside each fold's train part, 20 % of train lncRNAs x 20 % of train diseases form the validation block.

Each config is trained on the inner-train block; the reported value is the best inner-val AUPR 1:1 over the epoch grid
(and the epoch), mean +- std over folds and inner seeds. Results are appended to results/inner_val.jsonl.

Usage:
  CUDA_VISIBLE_DEVICES=0 /sda1/nghiavm/research/.venv/bin/python scripts/inner_val.py --tag B2_gmm \
      --cfg '{"views": ["gmm"]}' [--folds 0 1 2 3 4] [--inner-seeds 0]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fghgcl.harness import SEED, aupr_1to1, build, sigmoid, harness_folds, inner_split, load_semsim, load_variant  # noqa: E402
from fghgcl.model import FGHGCL, Config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--variant", default="l2d5", help="l2d5 (= D3) | D1 | D2")
    ap.add_argument("--scenario", default="both", choices=["both", "lncRNA", "disease", "warm"], help="cold scenario of the harness")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--inner-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "inner_val.jsonl")
    a = ap.parse_args()
    cfg, Model = build(json.loads(a.cfg))
    M, Xl, Xd = load_variant(a.variant)
    dss = load_semsim()
    L, D = M.shape
    lf, df = harness_folds(L), harness_folds(D)
    leak = None
    if getattr(cfg, "leak", False):                     # LEAKY comparison runs only (HGCMLDA protocol): full-matrix IL / ID
        from fghgcl.leaky import leaky_similarities
        leak = leaky_similarities(M, dss)
    rows, t0 = [], time.time()
    if a.scenario == "warm":                            # C-normal: harness warm folds over positives (bench/runner.py)
        pos = np.argwhere(M)
        wf = np.array_split(np.random.default_rng(SEED).permutation(len(pos)), 5)
        allL, allD = np.ones(L, bool), np.ones(D, bool)
        for f in a.folds:
            Mtr = M.copy()
            Mtr[pos[wf[f], 0], pos[wf[f], 1]] = False   # the harness test positives are never seen
            for s in a.inner_seeds:                     # inner validation: 12.5 % of the train positives (IHC-LDA adapter)
                rng = np.random.default_rng(SEED + s)
                tp = np.argwhere(Mtr)
                vp = tp[rng.permutation(len(tp))[:int(round(0.125 * len(tp)))]]
                Mi = Mtr.copy()
                Mi[vp[:, 0], vp[:, 1]] = False
                z = np.argwhere(~Mtr)                   # candidate negatives: pairs unlabeled in the train matrix
                vn = z[rng.choice(len(z), len(vp), replace=False)]
                y = np.r_[np.ones(len(vp)), np.zeros(len(vn))].astype(bool)

                def val(S):
                    return aupr_1to1(np.r_[S[vp[:, 0], vp[:, 1]], S[vn[:, 0], vn[:, 1]]], y)
                m = Model(cfg)
                hist = m.fit(Mi, Xl, Xd, dss, allL, allD, val=val, leak_sims=leak)
                ep, v = max(hist, key=lambda r: (r[1], -r[0]))
                rows.append({"fold": f, "inner_seed": s, "best_epoch": ep, "val": v, "hist": hist, "loss": m.loss_log})
                print(f"[{a.tag}] fold {f} seed {s}: val {v:.4f} @ {ep}", flush=True)
    for f in (a.folds if a.scenario != "warm" else []):
        tl, td = np.ones(L, bool), np.ones(D, bool)
        if a.scenario in ("both", "lncRNA"):
            tl[lf[f]] = False
        if a.scenario in ("both", "disease"):
            td[df[f]] = False
        Mtr = M & tl[:, None] & td[None, :]
        for s in a.inner_seeds:
            ml, md, vr, vc = inner_split(tl, td, a.scenario, SEED + s)
            Yv = Mtr[np.ix_(vr, vc)]
            if cfg.ens == 1:
                m = Model(cfg)
                hist = m.fit(Mtr & ml[:, None] & md[None, :], Xl, Xd, dss, ml, md,
                             val=lambda S: aupr_1to1(S[np.ix_(vr, vc)], Yv), leak_sims=leak)
            else:                                       # seed ensemble: mean sigmoid score at every evaluated epoch
                blocks = []
                for j in range(cfg.ens):
                    m, store = Model(replace(cfg, seed=cfg.seed + j)), []
                    eps = [e for e, _ in m.fit(Mtr & ml[:, None] & md[None, :], Xl, Xd, dss, ml, md,
                                               val=lambda S: store.append(sigmoid(S[np.ix_(vr, vc)])) or 0.0)]
                    blocks.append(store)
                hist = [(e, aupr_1to1(np.mean([b[t] for b in blocks], 0), Yv)) for t, e in enumerate(eps)]
            ep, v = max(hist, key=lambda r: (r[1], -r[0]))
            rows.append({"fold": f, "inner_seed": s, "best_epoch": ep, "val": v, "hist": hist, "loss": m.loss_log})
            print(f"[{a.tag}] fold {f} seed {s}: val {v:.4f} @ {ep}", flush=True)
    vals = np.array([r["val"] for r in rows])
    res = {"tag": a.tag, "variant": a.variant, "scenario": a.scenario, "cfg": cfg.to_dict(), "mean": float(vals.mean()), "std": float(vals.std()),
           "epochs": [r["best_epoch"] for r in rows], "rows": rows, "seconds": round(time.time() - t0)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as fh:
        fh.write(json.dumps(res) + "\n")
    print(f"== {a.tag}: inner-val AUPR 1:1 {res['mean']:.4f} ± {res['std']:.4f}  ({res['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
