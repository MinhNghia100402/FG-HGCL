"""Build CDELDA-harness data directories for every dataset (plan §5, item 6).

data/harness/data_rd_<v>/  (the layout bench.sweep_run.load_variant reads; CDELDA_DATA_ROOT = data/harness)
  l2d5 (= D3): symlinks to IHC-LDA/data/interim/cdelda_rebuild/data/data_rd_l2d5 (CDELDA's own files, unchanged)
  D1, D2 (HGCMLDA datasets), from IHC-LDA data/raw/<ds> and features/<ds>:
    M.npy               LDA.txt (lncRNA x disease, 0/1)
    lnc_ortho.npy       CDELDA-style block concatenation, each block L2-normalised: RNA-FM (rnafm_chunk, 640) ||
                        GTEx (gtex, 68). CDELDA's 8-d ViennaRNA structure block has no D1/D2 counterpart and is omitted.
                        Rows without a value are 0 (CDELDA convention: all-zero row = no content).
    disease_emb.npy     S-BioBERT of "name. DO definition. Synonyms" (distext, 768) = the model CDELDA uses for l2d5
    disease_semsim.npy  DSS.txt (disease semantic similarity; label-free)
    build_report.json   shapes, #positives, content coverage, sha256 of every array, sources
No file is derived from associations.
Usage: /sda1/nghiavm/research/.venv/bin/python scripts/build_harness_data.py
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IHC = Path("/sda1/nghiavm/research/IHC-LDA")
OUT = ROOT / "data" / "harness"
L2D5 = IHC / "data/interim/cdelda_rebuild/data/data_rd_l2d5"


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def l2block(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, np.float64))
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return np.where(n > 0, x / np.maximum(n, 1e-12), 0.0)


def build(ds: str):
    d = OUT / f"data_rd_{ds}"
    d.mkdir(parents=True, exist_ok=True)
    M = (np.loadtxt(IHC / f"data/raw/{ds}/LDA.txt") > 0).astype(np.float32)
    dss = np.loadtxt(IHC / f"data/raw/{ds}/DSS.txt").astype(np.float32)
    rnafm = np.load(IHC / f"features/{ds}/rnafm_chunk.npy")
    gtex = np.load(IHC / f"features/{ds}/gtex.npy")
    Xl = np.concatenate([l2block(rnafm), l2block(gtex)], 1).astype(np.float32)
    Xd = np.load(IHC / f"features/{ds}/distext.npy").astype(np.float32)
    assert Xl.shape[0] == M.shape[0] and Xd.shape[0] == dss.shape[0] == M.shape[1]
    arrays = {"M.npy": M, "lnc_ortho.npy": Xl, "disease_emb.npy": Xd, "disease_semsim.npy": dss}
    for f, a in arrays.items():
        np.save(d / f, a)
    rep = {"dataset": ds, "M_shape": list(M.shape), "n_pos": int(M.sum()),
           "lnc_rows_with_content": int(np.any(Xl != 0, 1).sum()),
           "lnc_blocks": {"rnafm_chunk": [0, 640], "gtex": [640, 640 + gtex.shape[1]]},
           "sources": {"M": f"IHC-LDA/data/raw/{ds}/LDA.txt", "DSS": f"IHC-LDA/data/raw/{ds}/DSS.txt",
                       "lnc": [f"IHC-LDA/features/{ds}/rnafm_chunk.npy", f"IHC-LDA/features/{ds}/gtex.npy"],
                       "dis": f"IHC-LDA/features/{ds}/distext.npy"},
           "sha256_16": {f: sha(a) for f, a in arrays.items()}}
    (d / "build_report.json").write_text(json.dumps(rep, indent=1))
    print(ds, json.dumps({k: rep[k] for k in ("M_shape", "n_pos", "lnc_rows_with_content")}))


def link_l2d5():
    d = OUT / "data_rd_l2d5"
    d.mkdir(parents=True, exist_ok=True)
    for f in L2D5.iterdir():
        t = d / f.name
        if not t.exists():
            os.symlink(f, t)
    M = np.load(L2D5 / "M.npy")
    print("l2d5", json.dumps({"M_shape": list(M.shape), "n_pos": int((M > 0).sum()),
                              "lnc_rows_with_content": int(np.any(np.load(L2D5 / "lnc_ortho.npy") != 0, 1).sum())}))


if __name__ == "__main__":
    link_l2d5()
    for ds in ("D1", "D2"):
        build(ds)
