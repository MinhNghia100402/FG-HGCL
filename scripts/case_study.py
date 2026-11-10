"""Case study (descriptive, post-lock): train the final clean configuration on ALL known D1 / D2 associations and check
the top-ranked UNKNOWN pairs against LncRNADisease v3.0 (IHC-LDA data/raw/external/lncrnadisease_v3, downloaded
2026-10-09; Homo sapiens, ncRNA Category = LncRNA). D1 was built from LncRNADisease 2.0 + Lnc2Cancer, D2 from MNDR v3.0,
so a v3.0 record that is 0 in the matrix is independent evidence the model never saw.
Matching: exact lncRNA symbol (upper case) and exact disease name (lower case); only pairs whose lncRNA and disease
both occur in the matched v3.0 records are candidates ("evaluable universe").
Epochs: median of the per-fold epochs chosen on inner-val in the 5-CV run of the same config (3 seeds).
Baselines: popularity (deg_l * deg_d, train degrees), and the same model without the hypergraph branch.
Within-disease metrics: mean over diseases with >= 1 v3.0-only pair of the AUC and P@10 among that disease's candidates.
Output: results/case_study_D<k>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from fghgcl.harness import build, load_semsim, load_variant  # noqa: E402
from run_ablation import config  # noqa: E402

RAW = Path("/sda1/nghiavm/research/IHC-LDA/data/raw")
V = {"dis_views": ["name", "defn", "syn"], "dis_view_feats": True}
CFG = {1: ("B3", "B0f", {"epochs": 1000, "link_loss": "infonce"}, "D1_clean_B3_final_nce.json"),
       2: ("B3", "B0fv", {"epochs": 1000, **V, "link_loss": "infonce+bce"}, "D2_clean_B3_final_nceb.json")}


def evidence(ds, M):
    v3 = pd.read_csv(RAW / "external/lncrnadisease_v3/website_alldata.tsv", sep="\t", usecols=[0, 1, 2, 3, 6, 11])
    v3 = v3[(v3.Species == "Homo sapiens") & (v3["ncRNA Category"] == "LncRNA")]
    ln = [x.strip() for x in open(RAW / ds / "lnc_names.txt") if x.strip()]
    dn = [x.strip() for x in open(RAW / ds / "dis_names.txt") if x.strip()]
    li, di = {n.upper(): i for i, n in enumerate(ln)}, {n.lower(): j for j, n in enumerate(dn)}
    E, pmid = np.zeros(M.shape, bool), {}
    for a, b, p in zip(v3["ncRNA Symbol"].astype(str).str.upper(), v3["Disease Name"].astype(str).str.lower(), v3["PubMed ID"].astype(str)):
        if a in li and b in di:
            E[li[a], di[b]] = True
            pmid.setdefault((li[a], di[b]), set()).add(p)
    return E, pmid, ln, dn


def train_scores(model, over, ep, M, Xl, Xd, dss, seeds):
    S = 0.0
    allL, allD = np.ones(M.shape[0], bool), np.ones(M.shape[1], bool)
    for s in seeds:
        cfg, Model = build({**config(model), **over, "epochs": ep, "eval_every": 10, "seed": s})
        m = Model(cfg)
        m.fit(M, Xl, Xd, dss, allL, allD, epochs=ep, train_mask=np.ones(M.shape, bool))
        S = S + 1 / (1 + np.exp(-np.clip(m.scores(), -50, 50))) / len(seeds)
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=int, choices=[1, 2], required=True)
    a = ap.parse_args()
    ds = f"D{a.dataset}"
    model, nohg, over, ref = CFG[a.dataset]
    M, Xl, Xd = load_variant(ds)
    dss = load_semsim()
    js = json.loads((ROOT / "results/hgcmlda_protocol" / ref).read_text())
    ep = int(np.median([f["epoch"] for s in js["seeds"].values() for f in s["folds"]]))
    E, pmid, ln, dn = evidence(ds, M)
    cand = ~M & E.any(1)[:, None] & E.any(0)[None, :]
    y = E[cand]
    scores = {"FG-HGCL": train_scores(model, over, ep, M, Xl, Xd, dss, [2026, 2027, 2028]),
              "FG-HGCL-HG": train_scores(nohg, over, ep, M, Xl, Xd, dss, [2026, 2027, 2028]),
              "Popularity": np.outer(M.sum(1), M.sum(0)).astype(float)}
    out = {"dataset": ds, "epochs": ep, "n_candidates": int(cand.sum()), "n_evidence": int(y.sum()),
           "base_rate": float(y.mean()), "methods": {}}
    rng = np.random.default_rng(0)
    for name, S in scores.items():
        s = S[cand] + rng.uniform(0, 1e-9, cand.sum())            # deterministic tie break
        order = np.argsort(-s)
        r = {"auc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s))}
        for k in (20, 50, 100, 200):
            r[f"p@{k}"] = float(y[order[:k]].mean())
        # within-disease ranking (removes disease popularity): mean AUC and hit rate of v3.0-only pairs in the top 10
        aucs, hits = [], []
        for j in np.nonzero(cand.any(0))[0]:
            c = cand[:, j]
            yj = E[c, j]
            if 0 < yj.sum() < c.sum():
                aucs.append(roc_auc_score(yj, S[c, j]))
                top = np.argsort(-S[c, j])[:10]
                hits.append(yj[top].mean())
        r["macro_auc_dis"], r["macro_p@10_dis"], r["n_dis"] = float(np.mean(aucs)), float(np.mean(hits)), len(aucs)
        out["methods"][name] = r
    # per-disease top-10 among unknown pairs (diseases with the most v3.0-only evidence)
    S = scores["FG-HGCL"]
    newd = (E & ~M).sum(0)
    out["cases"] = []
    for j in np.argsort(-newd)[:3]:
        rows = [i for i in np.argsort(-S[:, j]) if not M[i, j]][:10]
        out["cases"].append({"disease": dn[j], "known": int(M[:, j].sum()), "v3_only": int(newd[j]),
                             "top10": [{"lnc": ln[i], "rank": n + 1, "score": round(float(S[i, j]), 4), "in_v3": bool(E[i, j]),
                                        "pmid": sorted(pmid.get((i, j), []))[:3]} for n, i in enumerate(rows)]})
    (ROOT / "results" / f"case_study_{ds}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "cases"}, indent=1))


if __name__ == "__main__":
    main()
