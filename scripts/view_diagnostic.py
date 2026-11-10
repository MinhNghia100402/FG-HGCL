"""How much link information does each hypergraph view carry on its own? (docs/hypergraph_views_d12.md, diagnostic only)

For a lncRNA-side view with incidence H, the propagated score S = G @ M_train (G = HGNN propagator, self loops removed)
ranks a pair (l, d) by how strongly l's hyperedge co-members are linked to d in train; the disease side uses
M_train @ G_d. The association hypergraph (each train disease = a hyperedge over its lncRNAs) gives the 3-hop score.
Evaluated as AP on fold 0's inner validation of the HGCMLDA protocol (same carving as scripts/hgcmlda_protocol.py);
test and independent pairs are never read. Also reports homophily: P(two members of one hyperedge share a train
disease) vs a random lncRNA pair.

Usage: /sda1/nghiavm/research/.venv/bin/python scripts/view_diagnostic.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fghgcl.harness import load_semsim, load_variant  # noqa: E402
from fghgcl.hypergraph import cosine, gmm_hyperedges, knn_hyperedges, propagator  # noqa: E402
from fghgcl.leaky import leaky_similarities  # noqa: E402
from hgcmlda_protocol import SPLITS, mask_of  # noqa: E402


def homophily(H, Mtr, rng, n=400):
    L = H.shape[0]
    same, rand = [], []
    for v in rng.choice(L, min(n, L), replace=False):
        for e in np.nonzero(H[v])[0]:
            mem = [u for u in np.nonzero(H[:, e])[0] if u != v]
            same += [bool((Mtr[u] & Mtr[v]).any()) for u in mem[:20]]
            rand += [bool((Mtr[u] & Mtr[v]).any()) for u in rng.choice(L, min(len(mem), 20))]
    return float(np.mean(same)) if same else float("nan"), float(np.mean(rand)) if rand else float("nan")


def main():
    out = {}
    for ds in (1, 2):
        M, Xl, Xd = load_variant(f"D{ds}")
        dss = load_semsim()
        sp = np.load(SPLITS[ds])
        L, D = M.shape
        ind_pos, ind_neg = mask_of(sp["ind_pos"], M.shape), mask_of(sp["ind_neg"], M.shape)
        tpos, tneg = sp["fold0_test_pos"], sp["fold0_test_neg"]
        train_ones = M & ~mask_of(tpos, M.shape) & ~ind_pos
        train_zeros = ~M & ~ind_neg & ~mask_of(tneg, M.shape)
        rng = np.random.default_rng(2026)
        o, z = np.argwhere(train_ones), np.argwhere(train_zeros)
        vo, vz = o[rng.permutation(len(o))[:len(o) // 8]], z[rng.permutation(len(z))[:len(z) // 8]][:len(o) // 8]
        Mtr = train_ones & ~mask_of(vo, M.shape)                  # inner-train labels only
        Mf = Mtr.astype(np.float64)
        IL, ID = leaky_similarities(M, dss)                       # leaky features of the paper protocol (for reference)
        allL, allD, has = np.ones(L, bool), np.ones(D, bool), np.any(Xl != 0, 1)
        views_l = {
            "KNN nội dung (k 10)": knn_hyperedges(cosine(Xl), 10, allL, has),
            "GMM nội dung (16)": gmm_hyperedges(Xl, 16, allL, has, 2026),
            "KNN trên IL (leaky, k 10)": knn_hyperedges(IL, 10, allL, allL),
            "association (train)": Mtr.astype(np.float32),        # one hyperedge per disease over its train lncRNAs
        }
        views_d = {
            "KNN DSS (k 10)": knn_hyperedges(dss, 10, allD, allD),
            "KNN văn bản (k 10)": knn_hyperedges(cosine(Xd), 10, allD, allD),
            "association (train)": Mtr.T.astype(np.float32),
        }
        res = {"lnc": {}, "dis": {}, "n_val_pos": int(len(vo))}

        def ap(S):
            s = np.r_[S[vo[:, 0], vo[:, 1]], S[vz[:, 0], vz[:, 1]]]
            return float(average_precision_score(np.r_[np.ones(len(vo)), np.zeros(len(vz))], s))
        for name, H in views_l.items():
            G = propagator(H.astype(np.float32)).astype(np.float64)
            np.fill_diagonal(G, 0)
            h, r = homophily(H, Mtr, np.random.default_rng(0))
            res["lnc"][name] = {"ap": ap(G @ Mf), "homophily": h, "random": r, "edges": int(H.shape[1])}
        for name, H in views_d.items():
            G = propagator(H.astype(np.float32)).astype(np.float64)
            np.fill_diagonal(G, 0)
            res["dis"][name] = {"ap": ap(Mf @ G), "edges": int(H.shape[1])}
        res["popularity"] = ap(np.repeat(Mf.sum(0, keepdims=True), L, 0) + 1e-6 * Mf.sum(1, keepdims=True))
        out[f"D{ds}"] = res
        print(f"D{ds}", json.dumps(res, ensure_ascii=False, indent=1))
    p = ROOT / "results" / "hgcmlda_protocol" / "view_diagnostic.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
