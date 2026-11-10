"""Copy the label-free knowledge hyperedges of IHC-LDA into the D1 / D2 harness variants.

Source: /sda1/nghiavm/research/IHC-LDA @ 9c88513, features/<ds>/kg_lnc.npz, kg_dis.npz (scripts/build_knowledge.py):
  lncRNA  locus = cytogenetic sub-band of the GENCODE v49 gene midpoint; mirna = one hyperedge per miRNA over the
          lncRNAs with a CLIP-supported site (ENCORI, clipExpNum >= 1)
  disease onto  = one hyperedge per MeSH 2025 tree term over the dataset diseases at or below it
Groups with < 2 members are dropped there. No association label is used. Row order = IHC-LDA data/raw/<ds>/
lnc_names.txt / dis_names.txt = data/harness/data_rd_<ds>/M.npy (checked below).
Select with Config.kg_lnc / kg_dis.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = Path("/sda1/nghiavm/research/IHC-LDA")

for ds in ("D1", "D2"):
    M = np.load(ROOT / "data" / "harness" / f"data_rd_{ds}" / "M.npy")
    raw = np.loadtxt(SRC / "data" / "raw" / ds / "LDA.txt") > 0
    assert raw.shape == M.shape and (raw == (M > 0)).all(), ds
    for side, n in (("lnc", M.shape[0]), ("dis", M.shape[1])):
        z = np.load(SRC / "features" / ds / f"kg_{side}.npz")
        assert z["H"].shape[0] == n
        np.savez_compressed(ROOT / "data" / "harness" / f"data_rd_{ds}" / f"kg_{side}.npz", H=z["H"], src=z["src"], names=z["names"])
        srcs, cnt = np.unique(z["src"], return_counts=True)
        cov = {s: round(float((z["H"][:, z["src"] == s].sum(1) > 0).mean()), 3) for s in srcs}
        print(ds, side, dict(zip(srcs.tolist(), cnt.tolist())), "node coverage", cov)
