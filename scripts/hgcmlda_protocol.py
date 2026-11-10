"""FG-HGCL under HGCMLDA's ORIGINAL evaluation protocol (D1 = Dataset1, D2 = Dataset2), for a like-for-like comparison
with the HGCMLDA paper and its reproduction in /sda1/nghiavm/research/HGCMLDA (reports/phase0_table_D*.md).

Protocol (original/Dataset*/prepareData.py + main_experiments.py, splits read from the reproduction's split.npz):
  - an independent test set (positives + negatives) is held out; the remaining positives and zeros are cut into 5 folds;
  - fold k trains on the ones / zeros of the other folds (independent pairs never used), warm start (every node seen);
  - evaluation 1:1: fold-k test positives vs the first n_pos test zeros after np.random.seed(s) + shuffle, then the
    independent positives vs shuffled independent negatives (same RNG stream, as in the original). The original uses
    s = 3 in Dataset1 and s = 4 in Dataset2 (original/Dataset2/main_experiments.py:78); this script used s = 3 for both
    until 2026-11-05 (found by the score-agreement check of scripts/analysis_report.py), so D2 results with the default
    are on a different 1:1 negative sample of the same pool; --neg_seed 4 reproduces Dataset2's exact negatives.
  - metrics: paper-style (mean interpolated ROC AUC, pooled trapezoidal AUPR; HGCMLDA src/eval/metrics.py) + fold means.
Variants:
  clean : FG-HGCL on content features (no association-derived input)
  leaky : + IL / ID from the FULL matrix as in prepareData.py (the original HGCMLDA input; fghgcl/leaky.py)
Epochs: chosen per fold on an inner validation (12.5 % of that fold's train ones and zeros), then refit on all train pairs.

Usage: CUDA_VISIBLE_DEVICES=0 /sda1/nghiavm/research/.venv/bin/python scripts/hgcmlda_protocol.py --dataset 1 --variant clean
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fghgcl.harness import build, load_variant, load_semsim  # noqa: E402
from fghgcl.leaky import leaky_similarities  # noqa: E402

HG = Path("/sda1/nghiavm/research/HGCMLDA/results")
SPLITS = {1: HG / "20261005-035834_D1_original_s0_p0/split.npz", 2: HG / "20261005-042559_D2_clean_s0_p0/split.npz"}


def paper_style(ys, ps):
    """HGCMLDA src/eval/metrics.py: mean of interpolated ROC curves; pooled trapezoidal AUPR."""
    mean_fpr = np.linspace(0, 1, 1000)
    tprs = []
    for y, p in zip(ys, ps):
        fpr, tpr, _ = roc_curve(y, p)
        t = np.interp(mean_fpr, fpr, tpr)
        t[0] = 0.0
        tprs.append(t)
    mt = np.mean(tprs, 0)
    mt[-1] = 1.0
    prec, rec, _ = precision_recall_curve(np.concatenate(ys), np.concatenate(ps))
    return {"auc": float(auc(mean_fpr, mt)), "aupr": float(auc(rec, prec))}


def mask_of(pairs, shape):
    m = np.zeros(shape, bool)
    m[pairs[:, 0], pairs[:, 1]] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=int, choices=[1, 2], required=True)
    ap.add_argument("--variant", choices=["clean", "leaky"], required=True)
    ap.add_argument("--model", default="B3", help="config id of configs/*.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--cfg", default="{}", help="JSON overrides on top of --model (selection candidates)")
    ap.add_argument("--tag", default="", help="suffix of the result file")
    ap.add_argument("--split", default=None, help="split.npz of another HGCMLDA run (default: the reproduction split)")
    ap.add_argument("--neg_seed", type=int, default=3, help="np.random seed of the 1:1 test negatives (original: 3 for D1, 4 for D2)")
    ap.add_argument("--dump", action="store_true", help="also save test pairs, scores and inner-val curves (results/hgcmlda_protocol/dump/)")
    a = ap.parse_args()
    from run_ablation import config
    M, Xl, Xd = load_variant(f"D{a.dataset}")
    dss = load_semsim()
    split = Path(a.split) if a.split else SPLITS[a.dataset]
    sp = np.load(split)
    L, D = M.shape
    ind_pos, ind_neg = mask_of(sp["ind_pos"], M.shape), mask_of(sp["ind_neg"], M.shape)
    zeros_cv = ~M & ~ind_neg
    leak = leaky_similarities(M, dss) if a.variant == "leaky" else None   # prepareData.py: IL / ID from ld_true
    allL, allD = np.ones(L, bool), np.ones(D, bool)
    out = {"neg_seed": a.neg_seed, "dataset": a.dataset, "variant": a.variant, "model": a.model, "cfg_over": json.loads(a.cfg), "tag": a.tag, "split": str(split), "seeds": {}}
    for seed in a.seeds:
        cfg, Model = build({**config(a.model), "epochs": a.epochs, "eval_every": 10, "seed": seed, **json.loads(a.cfg)})
        ys, ps, yi, pi, folds, t0 = [], [], [], [], [], time.time()
        dump = {}
        np.random.seed(a.neg_seed)                        # the original seeds once per fold inside evaluate_validation
        for k in range(5):
            tpos, tneg = sp[f"fold{k}_test_pos"], sp[f"fold{k}_test_neg"]
            train_ones = M & ~mask_of(tpos, M.shape) & ~ind_pos
            train_zeros = zeros_cv & ~mask_of(tneg, M.shape)
            # inner validation for the epoch count: 12.5 % of the train ones and of the train zeros
            rng = np.random.default_rng(seed + k)
            o, z = np.argwhere(train_ones), np.argwhere(train_zeros)
            vo, vz = o[rng.permutation(len(o))[:len(o) // 8]], z[rng.permutation(len(z))[:len(z) // 8]]
            vmask = mask_of(vo, M.shape) | mask_of(vz, M.shape)
            Mi = train_ones & ~vmask
            m = Model(cfg)

            def val(S):
                pp, nn_ = S[vo[:, 0], vo[:, 1]], S[vz[:, 0], vz[:, 1]][:len(vo)]
                return average_precision_score(np.r_[np.ones(len(pp)), np.zeros(len(nn_))], np.r_[pp, nn_])
            hist = m.fit(Mi, Xl, Xd, dss, allL, allD, val=val, leak_sims=leak,
                         train_mask=(train_ones | train_zeros) & ~vmask)
            ep, vbest = max(hist, key=lambda r: (r[1], -r[0]))
            S = 0.0
            for j in range(getattr(cfg, "ens", 1)):        # seed ensemble: same epoch count, mean sigmoid score
                m = Model(replace(cfg, seed=cfg.seed + j))
                m.fit(train_ones, Xl, Xd, dss, allL, allD, epochs=ep, leak_sims=leak, train_mask=train_ones | train_zeros)
                S = S + 1.0 / (1.0 + np.exp(-np.clip(m.scores(), -50, 50))) / getattr(cfg, "ens", 1)
            # evaluation exactly as main_experiments.evaluate_validation / evaluate_test
            np.random.seed(a.neg_seed)
            idx = np.arange(len(tneg)).reshape(1, -1)
            np.random.shuffle(idx.T)
            neg = tneg[idx[0, :len(tpos)]]
            y = np.r_[np.ones(len(tpos)), np.zeros(len(neg))]
            p = np.r_[S[tpos[:, 0], tpos[:, 1]], S[neg[:, 0], neg[:, 1]]]
            idx2 = np.arange(len(sp["ind_neg"])).reshape(1, -1)
            np.random.shuffle(idx2.T)
            ineg = sp["ind_neg"][idx2[0, :len(sp["ind_pos"])]]
            y2 = np.r_[np.ones(len(sp["ind_pos"])), np.zeros(len(ineg))]
            p2 = np.r_[S[sp["ind_pos"][:, 0], sp["ind_pos"][:, 1]], S[ineg[:, 0], ineg[:, 1]]]
            ys.append(y), ps.append(p), yi.append(y2), pi.append(p2)
            if a.dump:
                dump.update({f"f{k}_cv_pairs": np.r_[tpos, neg], f"f{k}_cv_y": y, f"f{k}_cv_p": p,
                             f"f{k}_ind_pairs": np.r_[sp["ind_pos"], ineg], f"f{k}_ind_y": y2, f"f{k}_ind_p": p2,
                             f"f{k}_train_ones": np.argwhere(train_ones), f"f{k}_hist": np.array(hist, float)})
            folds.append({"epoch": int(ep), "inner_val_ap": float(vbest), "cv_auc": float(roc_auc_score(y, p)), "cv_ap": float(average_precision_score(y, p)),
                          "ind_auc": float(roc_auc_score(y2, p2)), "ind_ap": float(average_precision_score(y2, p2))})
            print(f"[D{a.dataset} {a.variant} {a.model} s{seed}] fold {k}: ep {ep} CV AUC {folds[-1]['cv_auc']:.4f} "
                  f"AP {folds[-1]['cv_ap']:.4f} | ind AUC {folds[-1]['ind_auc']:.4f}", flush=True)
        r = {"cv": paper_style(ys, ps), "independent": paper_style(yi, pi), "folds": folds, "seconds": round(time.time() - t0),
             "inner_val_ap": float(np.mean([f["inner_val_ap"] for f in folds]))}
        out["seeds"][str(seed)] = r
        if a.dump:
            d = ROOT / "results" / "hgcmlda_protocol" / "dump"
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / f"D{a.dataset}_{a.variant}_{a.model}{('_' + a.tag) if a.tag else ''}_s{seed}.npz", **dump)
        print(f"== D{a.dataset} {a.variant} {a.model}{a.tag} s{seed}: innerval {r['inner_val_ap']:.4f} | CV AUC {r['cv']['auc']:.4f} AUPR {r['cv']['aupr']:.4f} | "
              f"independent AUC {r['independent']['auc']:.4f} AUPR {r['independent']['aupr']:.4f}", flush=True)
    for part in ("cv", "independent"):
        for mt in ("auc", "aupr"):
            v = [out["seeds"][s][part][mt] for s in out["seeds"]]
            out[f"{part}_{mt}"] = [float(np.mean(v)), float(np.std(v))]
    out["inner_val_ap"] = [float(np.mean([out["seeds"][s]["inner_val_ap"] for s in out["seeds"]])),
                           float(np.std([out["seeds"][s]["inner_val_ap"] for s in out["seeds"]]))]
    p = ROOT / "results" / "hgcmlda_protocol" / f"D{a.dataset}_{a.variant}_{a.model}{('_' + a.tag) if a.tag else ''}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k.endswith(("auc", "aupr"))}))


if __name__ == "__main__":
    main()
