"""Summary of the knowledge-hyperedge runs (scripts/queue_kg.sh) vs FG-HGCL, clean setting, 3 seeds (sample SD)."""
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1] / "results" / "hgcmlda_protocol"
BASE = {1: "D1_clean_B3_final_nce", 2: "D2_clean_B3_final_nceb"}


def row(name):
    js = json.loads((R / f"{name}.json").read_text())
    S = js["seeds"].values()
    v = lambda f: [f(s) for s in S]
    out = {"inner": v(lambda s: s["inner_val_ap"]), "cv_auc": v(lambda s: s["cv"]["auc"]), "cv_aupr": v(lambda s: s["cv"]["aupr"]),
           "ind_auc": v(lambda s: s["independent"]["auc"]), "ind_aupr": v(lambda s: s["independent"]["aupr"])}
    return {k: (float(np.mean(x)), float(np.std(x, ddof=1))) for k, x in out.items()}


if __name__ == "__main__":
    res = {}
    for d in (1, 2):
        b = row(BASE[d])
        res[f"D{d}"] = {"FG-HGCL": b}
        for kg in ("mirna", "locus", "mirna+locus", "onto"):
            p = R / f"{BASE[d]}_kg_{kg}.json"
            if p.exists():
                r = row(p.stem)
                res[f"D{d}"][kg] = r
                print(f"D{d} {kg:12s}", " ".join(f"{k} {r[k][0]:.4f}({r[k][0] - b[k][0]:+.4f})" for k in r))
        print(f"D{d} base        ", " ".join(f"{k} {b[k][0]:.4f}" for k in b))
    (R.parent / "kg_views_summary.json").write_text(json.dumps(res, indent=1))
