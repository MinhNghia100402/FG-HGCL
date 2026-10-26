"""Reproducible split manifest + leakage audit for the both-cold protocol (plan §5, §7 Phase 1, item 5).

For each harness variant: the CDELDA harness node folds (seed 2026, checked against CDELDA's own
snapshot_src/twoside_common.folds), the both-cold blocks of every fold, and the inner-validation split of every
(fold, inner seed). Writes data/splits/<variant>_harness_seed2026.json (index lists + counts + hashes) and prints a
table of positives per block. Fails loudly if any check is violated:
  - lncRNA / disease folds are disjoint and cover every node;
  - test nodes never appear in the train or inner-val node sets; inner-val nodes lie inside the train nodes;
  - every block (inner-train, inner-val, train, test) contains positives.
Usage: /sda1/nghiavm/research/.venv/bin/python scripts/make_manifest.py [--variants l2d5 D1 D2] [--inner-seeds 0 1 2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.path.join(os.environ.get("CDELDA_REPO", "/sda1/nghiavm/research/IHC-LDA/data/raw/CDELDA"),
                                "snapshot_src"))

from fghgcl.harness import SEED, harness_folds, inner_split, load_variant  # noqa: E402


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def manifest(v, inner_seeds):
    M, Xl, Xd = load_variant(v)
    L, D = M.shape
    lf, df = harness_folds(L), harness_folds(D)
    try:                                                    # CDELDA's own fold function, if importable
        from twoside_common import folds as cd_folds
        ok = all(np.array_equal(np.sort(a), np.sort(b)) for a, b in zip(cd_folds(L, 5, seed=SEED), lf)) and \
            all(np.array_equal(np.sort(a), np.sort(b)) for a, b in zip(cd_folds(D, 5, seed=SEED), df))
        assert ok, "folds differ from CDELDA twoside_common.folds"
        checked = True
    except ImportError:
        checked = False
    for fs, n in ((lf, L), (df, D)):
        allidx = np.concatenate(fs)
        assert len(allidx) == n and len(np.unique(allidx)) == n, "folds not a partition"
    out = {"variant": v, "seed": SEED, "n_lnc": L, "n_dis": D, "n_pos": int(M.sum()),
           "checked_against_cdelda_folds": checked,
           "sha256_16": {"M": sha(M.astype(np.uint8)), "lnc_ortho": sha(Xl), "disease_emb": sha(Xd)},
           "lnc_folds": [f.tolist() for f in lf], "dis_folds": [f.tolist() for f in df], "folds": []}
    rows = []
    for k in range(5):
        tl, td = np.ones(L, bool), np.ones(D, bool)
        tl[lf[k]], td[df[k]] = False, False
        te_l, te_d = ~tl, ~td
        fo = {"fold": k, "n_train_lnc": int(tl.sum()), "n_train_dis": int(td.sum()),
              "n_test_lnc": int(te_l.sum()), "n_test_dis": int(te_d.sum()),
              "pos_train_block": int(M[np.ix_(tl, td)].sum()), "pos_test_block": int(M[np.ix_(te_l, te_d)].sum()),
              "pos_cross_blocks_unused": int(M[np.ix_(tl, te_d)].sum() + M[np.ix_(te_l, td)].sum()),
              "test_block_density": float(M[np.ix_(te_l, te_d)].mean()), "inner": []}
        for s in inner_seeds:
            ml, md, vr, vc = inner_split(tl, td, "both", SEED + s)
            assert not (vr & te_l).any() and not (vc & te_d).any() and not (ml & te_l).any() and not (md & te_d).any()
            assert not (vr & ~tl).any() and not (vc & ~td).any() and not (ml & vr).any() and not (md & vc).any()
            inn = {"inner_seed": s, "val_lnc": np.nonzero(vr)[0].tolist(), "val_dis": np.nonzero(vc)[0].tolist(),
                   "pos_inner_train": int(M[np.ix_(ml, md)].sum()), "pos_inner_val": int(M[np.ix_(vr, vc)].sum())}
            assert inn["pos_inner_train"] > 0 and inn["pos_inner_val"] > 0
            fo["inner"].append(inn)
            rows.append((k, s, fo["pos_train_block"], fo["pos_test_block"], inn["pos_inner_train"], inn["pos_inner_val"],
                         int(vr.sum()), int(vc.sum())))
        assert fo["pos_train_block"] > 0 and fo["pos_test_block"] > 0
        out["folds"].append(fo)
    p = ROOT / "data" / "splits" / f"{v}_harness_seed{SEED}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    print(f"\n{v}: {L} lnc x {D} dis, {out['n_pos']} pos, CDELDA folds checked={checked} -> {p.relative_to(ROOT)}")
    print("fold seed  pos_train  pos_test  pos_inner_train  pos_inner_val  val_lnc  val_dis")
    for r in rows:
        print(f"{r[0]:>4} {r[1]:>4} {r[2]:>10} {r[3]:>9} {r[4]:>16} {r[5]:>14} {r[6]:>8} {r[7]:>8}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["l2d5", "D1", "D2"])
    ap.add_argument("--inner-seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    for v in a.variants:
        manifest(v, a.inner_seeds)
