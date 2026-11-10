"""Alternative lncRNA content matrix: RIBOSPAN-10K-15 (2048) || GTEx (68), each block L2-normalised as in
build_harness_data.py, written next to lnc_ortho.npy as data/harness/data_rd_<ds>/lnc_ribospan10k.npy.
Select it with CCDIFF_LNC_FILE=lnc_ribospan10k.npy (fghgcl.harness.load_variant).
Source features: /sda1/nghiavm/research/IHC-LDA/features/<ds>/{ribospan10k15,gtex}.npy (read-only).
Usage: python3 scripts/build_lnc_ribospan.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_harness_data import IHC, OUT, l2block, sha  # noqa: E402

for ds in ("D1", "D2"):
    rs = np.load(IHC / f"features/{ds}/ribospan10k15.npy")
    gtex = np.load(IHC / f"features/{ds}/gtex.npy")
    Xl = np.concatenate([l2block(rs), l2block(gtex)], 1).astype(np.float32)
    ref = np.load(OUT / f"data_rd_{ds}" / "lnc_ortho.npy")
    assert Xl.shape[0] == ref.shape[0]
    np.save(OUT / f"data_rd_{ds}" / "lnc_ribospan10k.npy", Xl)
    print(json.dumps({"dataset": ds, "shape": list(Xl.shape), "rows_with_content": int(np.any(Xl != 0, 1).sum()),
                      "ref_rows_with_content": int(np.any(ref != 0, 1).sum()), "sha256_16": sha(Xl)}))
