"""Descriptive analyses for the report (post-lock; nothing here is used to choose a configuration).

Inputs: results/hgcmlda_protocol/D*_{clean,leaky}_*.json and dump/*.npz (scripts/queue_analysis.sh, --dump),
HGCMLDA reproduction runs in /sda1/nghiavm/research/HGCMLDA/results (same split.npz, same 1:1 test negatives: the
evaluation code draws them with np.random.seed(3) exactly as main_experiments.py; checked below by label order and by
the agreement of the two models' scores on the negatives).
Outputs results/analysis_report.json:
  paired      fold-level FG-HGCL vs HGCMLDA (CV AUC, CV AP per fold; FG-HGCL = mean of 3 seeds, HGCMLDA = mean of its
              seeds), mean delta, folds won, Nadeau-Bengio corrected t (n = 5, n_te / n_tr = 1/4) and two-sided p
  strata      CV AUC pooled over the 5 folds by train degree of the lncRNA / disease of each test pair, and by whether
              the lncRNA has content features (FG-HGCL, FG-HGCL-HG, HGCMLDA, popularity)
  popularity  per-fold AUC / AP of (deg_l + 1) * deg_d and of deg_d alone (train degrees of the fold)
  curves      inner-val AP vs epoch (mean over folds and seeds) for every dump
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "hgcmlda_protocol"
HG = Path("/sda1/nghiavm/research/HGCMLDA/results")
HG_RUNS = {("D1", "leaky"): "*_D1_original_s[0-4]_p0", ("D1", "clean"): "*_D1_clean_s[0-4]_p0",
           ("D2", "leaky"): "*_D2_original_s[0-2]", ("D2", "clean"): "*_D2_clean_s[0-4]_p0"}
FINAL = {"D1": "B3_final_nce", "D2": "B3_neg4"}          # D2: --neg_seed 4 = Dataset2's own 1:1 negatives
DUMP = {"D1": ("B3_dump_final", "B0f_dump_nohg"), "D2": ("B3_neg4", "B0fv_neg4")}
BINS = [(0, 0, "0"), (1, 2, "1–2"), (3, 10, "3–10"), (11, 10 ** 9, "> 10")]


def corrected_t(d, ratio=0.25):
    d = np.asarray(d, float)
    n, m, s2 = len(d), d.mean(), d.var(ddof=1)
    if s2 == 0:
        return float("inf") if m else 0.0, 0.0 if m else 1.0
    t = m / np.sqrt((1 / n + ratio) * s2)
    return float(t), float(2 * stats.t.sf(abs(t), n - 1))


def hg_runs(ds, var):
    return sorted(r for r in glob.glob(str(HG / HG_RUNS[(ds, var)])) if "split" not in Path(r).name)


def hg_scores(ds, var):
    """HGCMLDA test scores per fold, mean over its seeds: list of 5 arrays aligned with our cv pairs."""
    out = []
    for k in range(5):
        P = [np.loadtxt(f"{r}/{k + 1}predict_valid.txt") for r in hg_runs(ds, var)]
        out.append(np.mean(P, 0))
    return out


def dumps(ds, var, tag):
    return [np.load(f) for f in sorted(glob.glob(str(R / "dump" / f"{ds}_{var}_{tag}_s*.npz")))]


def auc_safe(y, s):
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def main():
    out = {"paired": {}, "strata": {}, "popularity": {}, "curves": {}, "checks": {}}
    for ds in ("D1", "D2"):
        M = np.load(ROOT / "data" / "harness" / f"data_rd_{ds}" / "M.npy") > 0
        Xl = np.load(ROOT / "data" / "harness" / f"data_rd_{ds}" / "lnc_ortho.npy")
        has = np.any(Xl != 0, 1)
        for var in ("leaky", "clean"):
            key = f"{ds}_{var}"
            # ---- fold-level paired comparison
            js = json.loads((R / f"{ds}_{var}_{FINAL[ds]}.json").read_text())
            fg = {m: np.mean([[f[m] for f in s["folds"]] for s in js["seeds"].values()], 0) for m in ("cv_auc", "cv_ap")}
            hgm = [json.loads(Path(r, "metrics.json").read_text())["cv"]["folds"] for r in hg_runs(ds, var)]
            hg = {"cv_auc": np.mean([[f["auc"] for f in m] for m in hgm], 0), "cv_ap": np.mean([[f["ap"] for f in m] for m in hgm], 0)}
            pr = {"n_hg_seeds": len(hgm)}
            for m in ("cv_auc", "cv_ap"):
                d = fg[m] - hg[m]
                t, p = corrected_t(d)
                pr[m] = {"fg": fg[m].round(4).tolist(), "hg": hg[m].round(4).tolist(), "delta_mean": float(d.mean()),
                         "delta_min": float(d.min()), "wins": int((d > 0).sum()), "t_corr": t, "p_corr": p}
            out["paired"][key] = pr
            # ---- per-pair strata
            fgd, nhd = dumps(ds, var, DUMP[ds][0]), dumps(ds, var, DUMP[ds][1])
            hgs = hg_scores(ds, var)
            rows = {"lnc_deg": [], "dis_deg": [], "has": [], "y": [], "FG-HGCL": [], "FG-HGCL-HG": [], "HGCMLDA": [], "Popularity": []}
            pop = {"pop_ld": [], "pop_d": []}
            agree = []
            for k in range(5):
                pairs, y = fgd[0][f"f{k}_cv_pairs"], fgd[0][f"f{k}_cv_y"]
                tr = np.zeros(M.shape, bool)
                to = fgd[0][f"f{k}_train_ones"]
                tr[to[:, 0], to[:, 1]] = True
                dl, dd = tr.sum(1), tr.sum(0)
                s_fg = np.mean([z[f"f{k}_cv_p"] for z in fgd], 0)
                s_nh = np.mean([z[f"f{k}_cv_p"] for z in nhd], 0)
                s_pop = (dl[pairs[:, 0]] + 1.0) * dd[pairs[:, 1]]
                assert len(hgs[k]) == len(y)
                neg = y == 0
                agree.append(stats.spearmanr(s_fg[neg], hgs[k][neg]).correlation)
                rows["lnc_deg"].append(dl[pairs[:, 0]]), rows["dis_deg"].append(dd[pairs[:, 1]]), rows["has"].append(has[pairs[:, 0]])
                rows["y"].append(y), rows["FG-HGCL"].append(s_fg), rows["FG-HGCL-HG"].append(s_nh)
                rows["HGCMLDA"].append(hgs[k]), rows["Popularity"].append(s_pop)
                pop["pop_ld"].append([auc_safe(y, s_pop), float(average_precision_score(y, s_pop))])
                pop["pop_d"].append([auc_safe(y, dd[pairs[:, 1]] + 1e-9 * np.arange(len(y))), float(average_precision_score(y, dd[pairs[:, 1]]))])
            out["checks"][key] = {"spearman_neg_fg_vs_hg": [round(float(a), 3) for a in agree]}
            # popularity with the paper-style metrics (same pairs as FG-HGCL; independent set scored per fold, averaged)
            sys.path.insert(0, str(ROOT / "scripts"))
            from hgcmlda_protocol import paper_style
            ys, ps, yi, pi_ = [], [], [], []
            for k in range(5):
                tr = np.zeros(M.shape, bool)
                to = fgd[0][f"f{k}_train_ones"]
                tr[to[:, 0], to[:, 1]] = True
                dl, dd = tr.sum(1), tr.sum(0)
                for pr_, y_, Y, P in ((f"f{k}_cv_pairs", f"f{k}_cv_y", ys, ps), (f"f{k}_ind_pairs", f"f{k}_ind_y", yi, pi_)):
                    pp = fgd[0][pr_]
                    Y.append(fgd[0][y_]), P.append((dl[pp[:, 0]] + 1.0) * dd[pp[:, 1]])
            out["popularity"][key + "_paper"] = {"cv": paper_style(ys, ps), "independent": paper_style(yi, pi_)}
            out["popularity"][key] = {k_: np.mean(v, 0).round(4).tolist() for k_, v in pop.items()}
            cat = {k_: np.concatenate(v) for k_, v in rows.items()}
            st = {}
            for side in ("lnc_deg", "dis_deg"):
                for lo, hi, name in BINS:
                    sel = (cat[side] >= lo) & (cat[side] <= hi)
                    st[f"{side}:{name}"] = {"n_pos": int(cat["y"][sel].sum()), "n_neg": int((1 - cat["y"][sel]).sum()),
                                            **{mth: auc_safe(cat["y"][sel], cat[mth][sel]) for mth in ("FG-HGCL", "FG-HGCL-HG", "HGCMLDA", "Popularity")}}
            for name, sel in (("has_content", cat["has"]), ("no_content", ~cat["has"])):
                st[f"lnc:{name}"] = {"n_pos": int(cat["y"][sel].sum()), "n_neg": int((1 - cat["y"][sel]).sum()),
                                     **{mth: auc_safe(cat["y"][sel], cat[mth][sel]) for mth in ("FG-HGCL", "FG-HGCL-HG", "HGCMLDA", "Popularity")}}
            st["all"] = {mth: auc_safe(cat["y"], cat[mth]) for mth in ("FG-HGCL", "FG-HGCL-HG", "HGCMLDA", "Popularity")}
            out["strata"][key] = st
            # ---- inner-val curves
            for tag, zs in (("FG-HGCL", fgd), ("FG-HGCL-HG", nhd)):
                H = np.array([z[f"f{k}_hist"] for z in zs for k in range(5)])        # (runs, steps, 2)
                out["curves"][f"{key}:{tag}"] = {"epoch": H[0, :, 0].tolist(), "mean": H[:, :, 1].mean(0).round(5).tolist(),
                                                 "sd": H[:, :, 1].std(0).round(5).tolist()}
    (ROOT / "results" / "analysis_report.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for k, v in out["paired"].items():
        print(k, {m: (round(v[m]["delta_mean"], 4), v[m]["wins"], round(v[m]["p_corr"], 4)) for m in ("cv_auc", "cv_ap")})
    for k, v in out["strata"].items():
        print(k, {s: {m: (round(x, 3) if isinstance(x, float) else x) for m, x in r.items()} for s, r in v.items()})
    print(json.dumps(out["popularity"]), json.dumps(out["checks"]))


if __name__ == "__main__":
    main()
