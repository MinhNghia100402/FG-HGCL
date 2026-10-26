"""Map D1 / D2 nodes onto D3 (CDELDA l2d5) nodes, to use D1 / D2 associations as EXTRA training labels for D3.

lncRNA key: HGNC id, else Ensembl gene id, else upper-cased name (IHC-LDA data/processed/<ds>/lnc_map.tsv).
disease key: DOID (D3: resolved incl. alt_ids; D1 / D2: names / synonyms matched to doid.obo with IHC-LDA's matcher in
scripts/extract_disease_text.py), else the normalised name.
Writes data/cross/<src>_to_l2d5.npz: lnc_map[i_src] = D3 index or -1, dis_map[j_src] = D3 index or -1, and prints how many
D1 / D2 positives land on D3 node pairs and how many of those are not already D3 positives.
The extra edges are only ever added to the TRAIN block of a fold (fghgcl/harness.py, cfg.extra_src): any extra edge
touching a held-out D3 node is dropped, so both-cold test nodes still have no training association.
Usage: /sda1/nghiavm/research/.venv/bin/python scripts/cross_dataset_map.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IHC = Path("/sda1/nghiavm/research/IHC-LDA")


def lnc_keys(ds):
    keys = []
    with open(IHC / f"data/processed/{ds}/lnc_map.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            keys.append(r["hgnc_id"] or r["ensembl_gene_id"] or r["name"].strip().upper())
    return keys


def dis_keys(ds):
    sys.path.insert(0, str(IHC))
    sys.path.insert(0, str(IHC / "scripts"))
    import extract_disease_text as edt
    from src.data.datasets import load_dataset
    from src.data.ontology import parse_obo
    terms = parse_obo(edt.OBO)
    live = {k: t for k, t in terms.items() if not t.obsolete}
    d = load_dataset(ds)
    if ds == "D3":
        from src.data.ontology import resolver
        res = resolver(terms)
        return [(n if n in live else (res(n) or n)) for n in d.dis_names]
    by_name = {}
    for t in live.values():
        for n in [t.name, *t.synonyms]:
            by_name.setdefault(edt.norm(n), t)
    out = []
    for name in d.dis_names:
        t = next((by_name[v] for v in edt.variants(name) if v in by_name), None)
        out.append(t.id if t is not None else edt.norm(name))
    return out


def main():
    M4 = np.load(ROOT / "data/harness/data_rd_l2d5/M.npy") > 0
    l4 = {k: i for i, k in enumerate(lnc_keys("D3"))}
    d4 = {k: i for i, k in enumerate(dis_keys("D3"))}
    (ROOT / "data/cross").mkdir(parents=True, exist_ok=True)
    for src in ("D1", "D2"):
        Ms = np.load(ROOT / f"data/harness/data_rd_{src}/M.npy") > 0
        lm = np.array([l4.get(k, -1) for k in lnc_keys(src)])
        dm = np.array([d4.get(k, -1) for k in dis_keys(src)])
        e = np.argwhere(Ms)
        ok = (lm[e[:, 0]] >= 0) & (dm[e[:, 1]] >= 0)
        pairs = np.unique(np.c_[lm[e[ok, 0]], dm[e[ok, 1]]], axis=0)
        new = ~M4[pairs[:, 0], pairs[:, 1]]
        np.savez(ROOT / f"data/cross/{src}_to_l2d5.npz", lnc_map=lm, dis_map=dm, pairs=pairs)
        print(f"{src}: lnc mapped {int((lm >= 0).sum())}/{len(lm)}, dis mapped {int((dm >= 0).sum())}/{len(dm)}; "
              f"positives on D3 pairs {len(pairs)} (of {len(e)}), already D3 positives {int((~new).sum())}, NEW {int(new.sum())}")


if __name__ == "__main__":
    main()
